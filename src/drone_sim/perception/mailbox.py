"""Storage for position estimates between detector and DMPC.

This module is the *memory* of the perception pipeline. The (separately developed) detector turns a :class:`~drone_sim.perception.camera.CameraView`
into estimated neighbor positions; the perception worker drops them here, and the DMPC bridge picks them up on the main thread before the next solve.
The mailbox itself is **sensor agnostic** — it deliberately imports nothing from ``camera.py`` and only ever sees estimates, never views, so a future
non-camera sensor can feed the same store.

Two time axes meet in a :class:`PositionEstimate` and must not be mixed up:

* ``captured_step`` / ``captured_time`` — *simulation* time of the underlying capture, passed through from the View. This is what the bridge uses for
  finite differences when it derives velocities from :meth:`PerceptionMailbox.history`.
* ``received_time`` — ``time.monotonic()`` at :meth:`PerceptionMailbox.post` time, i.e. process wall clock. It exists for diagnostics and staleness
  checks (the same clock ``AsyncLocalSolver._filter_stale`` runs on) and is **not** a substitute for simulation time.

Consumers **poll**: there are no ``threading.Condition`` objects and no blocking read. One writer (the perception worker) and any number of pollers
(bridge, GUI) is the intended usage.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)

# Kept per (observer, observed) pair. Two entries are the minimum for the bridge's finite-difference velocity estimate.
_DEFAULT_HISTORY_LEN = 5


@dataclass
class PositionEstimate:
   """One estimated neighbor position, as produced by the detector.

   Deliberately **not** frozen: :meth:`PerceptionMailbox.post` stamps ``observer_id`` and ``received_time`` on the way in. Once posted, treat the
   instance as **immutable** — it is shared across threads, and consumers (bridge, GUI) may read it but must never write to it.

   :param observer_id: Identifier of the observing (ego) drone; overwritten by :meth:`PerceptionMailbox.post`.
   :param observed_id: Identifier of the estimated neighbor.
   :param position: ``(3,)`` estimated world position; copied by :meth:`PerceptionMailbox.post`.
   :param captured_step: Simulation step index of the underlying capture.
   :param captured_time: Simulation time in seconds of the underlying capture.
   :param received_time: ``time.monotonic()`` when the estimate was posted; stamped by :meth:`PerceptionMailbox.post`.
   :param sigma: 1-sigma position uncertainty in meters, ``None`` when the detector reports none.
   """

   observer_id: str
   observed_id: str
   position: np.ndarray
   captured_step: int
   captured_time: float
   received_time: float = 0.0
   sigma: float | None = None


class PerceptionMailbox:
   """Thread-safe store for the latest position estimates plus a short history per observed neighbor.

   Thread safety:

   * a ``threading.RLock`` protects the dict/deque structure,
   * one writer (the perception worker thread) and any number of polling readers (main-thread bridge, GUI) are supported,
   * there are **no** Conditions and no blocking read — readers poll, they never wait for new perceptions.

   :meth:`latest` and :meth:`history` hand out copies of the *containers*, not of the records inside them; the records are shared and read-only.

   :param history_len: Number of estimates kept per ``(observer_id, observed_id)`` pair, must be at least 1. Note that ``history_len == 1``
      effectively disables the bridge's velocity estimation, which needs two entries with different ``captured_time`` for its finite difference.
   """

   def __init__(self, history_len: int = _DEFAULT_HISTORY_LEN) -> None:
      if history_len < 1:
         raise ValueError("history_len must be at least 1")
      self._history_len = int(history_len)
      self._lock = threading.RLock()
      self._latest: dict[str, dict[str, PositionEstimate]] = {}
      self._history: dict[tuple[str, str], deque[PositionEstimate]] = {}

   def post(self, observer_id: str, estimates: Iterable[PositionEstimate]) -> None:
      """Store a batch of estimates observed by ``observer_id``.

      This is an **upsert, not a replace**: an empty (or partial) batch deletes nothing, so a neighbor that is no longer visible keeps its last
      estimate. That is what makes a history of two or more entries possible when the camera runs slower than the simulation
      (``camera_rate_steps > 1``); the price is that a neighbor rotated out of the field of view lingers as a ghost. Callers that want forgetting
      semantics call :meth:`clear_observer` — the mailbox does not decide that.

      Every estimate is stamped with ``observer_id`` (the parameter wins, so key and record cannot drift apart) and one shared ``received_time`` for
      the whole batch, and its ``position`` is copied, so an adapter may reuse its buffers. ``captured_step`` / ``captured_time`` are left untouched.

      :param observer_id: Identifier of the observing (ego) drone.
      :param estimates: Estimates to store; may be empty.
      """
      # One batch is one point of reception — sampled once, outside the lock.
      now = time.monotonic()

      posted: list[str] = []
      with self._lock:
         for estimate in estimates:
            estimate.observer_id = observer_id
            estimate.received_time = now
            # Copy: a REST/library adapter may hand us a buffer it reuses for the next detection.
            estimate.position = np.array(estimate.position, dtype=float)
            self._latest.setdefault(observer_id, {})[estimate.observed_id] = estimate
            key = (observer_id, estimate.observed_id)
            if key not in self._history:
               self._history[key] = deque(maxlen=self._history_len)
            self._history[key].append(estimate)
            posted.append(estimate.observed_id)

      # One line per drone per step — guarded so the default INFO log stays clean.
      if _log.isEnabledFor(logging.DEBUG):
         _log.debug("Perception post: observer=%s -> %d estimates %s", observer_id, len(posted), posted)

   def latest(self, observer_id: str) -> dict[str, PositionEstimate]:
      """Most recent estimate per observed neighbor for one observer.

      The returned dict is a shallow copy, so callers cannot corrupt the mailbox structure — the :class:`PositionEstimate` records inside it are
      still shared and must be treated as read-only.

      :param observer_id: Identifier of the observing (ego) drone.
      :return: Mapping ``observed_id -> PositionEstimate``; empty dict for an unknown observer.
      """
      with self._lock:
         if observer_id not in self._latest:
            return {}
         return self._latest[observer_id].copy()

   def history(self, observer_id: str, observed_id: str) -> list[PositionEstimate]:
      """Recent estimates for one ``(observer, observed)`` pair, **oldest first, newest last**.

      The ordering is part of the contract: the DMPC bridge derives velocities from the two newest entries with differing ``captured_time``.
      At most ``history_len`` entries are kept.

      :param observer_id: Identifier of the observing (ego) drone.
      :param observed_id: Identifier of the observed neighbor.
      :return: New list of estimates, oldest first; empty list for an unknown pair.
      """
      with self._lock:
         entries = self._history.get((observer_id, observed_id))
         if entries is None:
            return []
         return list(entries)

   def clear(self) -> None:
      """Drop everything — both the latest estimates and the full history.

      Meant for a complete reset (e.g. a simulation reset). **Do not call this per simulation step:** wiping the history forces every velocity
      estimate of the bridge to zero, because the finite difference has nothing left to difference. The per-step ``clear()`` in the DMPC loop belongs
      to the *trajectory* mailbox, not to this one.
      """
      with self._lock:
         self._latest.clear()
         self._history.clear()

   def clear_observer(self, observer_id: str) -> None:
      """Drop one observer's estimates and history without touching the other observers.

      :param observer_id: Identifier of the observing (ego) drone; an unknown id is a no-op.
      """
      with self._lock:
         self._latest.pop(observer_id, None)
         # Materialize the key list first — deleting during iteration would raise.
         for key in [k for k in self._history if k[0] == observer_id]:
            del self._history[key]