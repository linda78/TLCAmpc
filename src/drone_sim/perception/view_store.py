"""Pickup point for rendered camera views, waiting to be fetched by the external detector.

This is the counterpart to :mod:`drone_sim.perception.mailbox`, for the other direction of the REST exchange: estimates come *in* through the mailbox,
views are handed *out* here. The perception worker renders a capture and drops it off; the detector polls the API layer, which reads from this store.

**Exactly one current view per observing drone.** :meth:`CameraViewStore.put` overwrites unconditionally — there is no queue, no backlog, no history.
A detector that polls slower than the simulation captures will skip captures, and that is the intended behaviour: a stale image is worse than a missed
one, because the estimates derived from it would be fed to the DMPC as if they described the present. The same drop-oldest reasoning governs the
worker's input queue.

Skipping captures costs nothing downstream: :meth:`PerceptionMailbox.post` upserts, so a neighbor simply keeps its last estimate until a fresher one
arrives. The one thing that must survive the round trip is the capture token (``step`` / ``sim_time``) — the detector echoes it back so the DMPC bridge
can tell which simulation instant an estimate belongs to.

Writer/reader model matches the mailbox: one writer (the perception worker thread), any number of polling readers (the HTTP threadpool, the GUI).
Readers never block and never wait for a fresh view.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
   from drone_sim.perception.camera import CameraView

_log = logging.getLogger(__name__)


class CameraViewStore:
   """Thread-safe slot holding the most recent :class:`~drone_sim.perception.camera.CameraView` per observer.

   One slot per observing drone, overwritten on every :meth:`put`. The stored :class:`CameraView` objects are shared with whoever reads them and must
   be treated as **read-only** — the worker builds a fresh View per capture, so nothing is mutated in place, but a reader that wrote into one would
   corrupt what the next reader sees.

   Ordering note: :meth:`put` does not compare capture indices before overwriting. It does not need to — the single writer processes captures
   sequentially off its queue, so views can never arrive out of order. If a second writer is ever added, that assumption has to be revisited here.
   """

   def __init__(self) -> None:
      self._lock = threading.RLock()
      self._views: dict[str, CameraView] = {}

   def put(self, view: CameraView) -> None:
      """Store ``view`` as the current one for its observer, replacing whatever was there.

      :param view: Freshly captured (and, for the REST path, already rendered) View.
      """
      with self._lock:
         replaced = view.observer_id in self._views
         self._views[view.observer_id] = view

      # One line per drone per capture — guarded so the default INFO log stays clean.
      if _log.isEnabledFor(logging.DEBUG):
         _log.debug("View store put: observer=%s step=%d t=%.3f image=%s (%s)", view.observer_id, view.step, view.sim_time,
                    "yes" if view.image_png else "no", "replaced" if replaced else "first")

   def latest(self, observer_id: str) -> CameraView | None:
      """Current view of one observer.

      :param observer_id: Identifier of the observing (ego) drone.
      :return: The stored View, or ``None`` when this observer has never captured. The caller cannot distinguish "not captured yet" from "camera
         disabled" — that is the API layer's job, which knows the config.
      """
      with self._lock:
         return self._views.get(observer_id)

   def observers(self) -> list[str]:
      """Observers that currently hold a view.

      Meant for the cheap poll endpoint, which lists what is available before anyone fetches image bytes.

      :return: New list of observer ids, in insertion order of their first capture.
      """
      with self._lock:
         return list(self._views)

   def clear(self) -> None:
      """Drop every stored view.

      Meant for a full reset (simulation reload). Harmless mid-run — the next capture refills the slots — but pointless, since every ``put``
      already replaces.
      """
      with self._lock:
         self._views.clear()
