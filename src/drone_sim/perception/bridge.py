"""The seam between perception and DMPC: turns estimated neighbor positions into the trajectory messages the solvers already understand.

The distributed MPC solvers never learn that a camera exists. They read neighbor predictions out of a trajectory mailbox — one
:class:`~drone_sim.simulation.distributed.trajectory_exchange.TrajectoryMessage` per neighbor — and it makes no difference to them whether those came
from a neighbor's own ADMM broadcast or from this module. That is the whole point: perception replaces the *source* of neighbor information, not the
control stack.

What arrives from the detector is a bare position. What the solver needs is a horizon of positions plus velocities. The gap is closed by constant
velocity extrapolation: the velocity comes from a finite difference over the two newest estimates in
:meth:`~drone_sim.perception.mailbox.PerceptionMailbox.history` with *different* ``captured_time``, and the horizon is that velocity carried forward,
``p + v · dt · (k+1)`` for ``k in 0..H-1``. The ``k+1`` is not an off-by-one: like
:meth:`~drone_sim.simulation.distributed.local_mpc.LocalMPCSolver._predict_states`, index ``k`` is the state *after* ``k+1`` steps, so a bridge
trajectory and a solver-predicted trajectory line up step for step.

**Routing is deliberately skipped.** Every message goes straight into the receiver's inbox via ``deliver()``, bypassing the NeighborGraph. The camera
already did the routing: a neighbor shows up in an observer's estimates precisely because the observer *saw* it. Running the geometric comm-radius
test on top would only re-decide, on ground truth, what the sensor has already decided.

**Direction of the estimates.** ``perception.latest(receiver_id)`` are the neighbors that *receiver_id observed*, and they are delivered into
*receiver_id*'s own inbox. Each drone therefore avoids what it can see itself — no drone tells another what it sees, and a drone with an empty field
of view gets an empty inbox and plans as if it were alone.

**Latency (risk R7 of the perception plan).** With an asynchronous perception worker the estimates in the mailbox are at least one simulation step
old, and with ``camera_rate_steps > 1`` the *same* estimates are re-extrapolated for several steps in a row, because
:meth:`~drone_sim.perception.mailbox.PerceptionMailbox.post` upserts and never expires anything. The extrapolation compensates for part of that lag —
a neighbor moving at constant velocity is predicted correctly however old the sample is — but not for maneuvers that happened after the capture. The
lag is real; it is the price of a sensor in the loop, and it is why the safety zones matter.

**Timestamps (risk R1).** ``timestamp_mode`` picks which clock the messages are stamped with, and the two are not interchangeable:

* ``"step"`` — the simulation step index, for the synchronous
  :class:`~drone_sim.simulation.distributed.distributed_coordinator.DistributedMPCCoordinator`, which never compares timestamps against a wall clock.
* ``"monotonic"`` — ``time.monotonic()`` sampled *at delivery time*, for the threaded coordinator.
  :meth:`~drone_sim.simulation.distributed.async_local_solver.AsyncLocalSolver._filter_stale` computes ``time.monotonic() - msg.timestamp`` and drops
  everything older than a second. A small integer step index would look like an age of hours, so a "step"-stamped message would be discarded whole,
  and every solver thread would run with an empty inbox.

The bridge runs on the **main thread**, between the perception worker's posts and the solve. It only ever reads the perception mailbox (which is
thread-safe) and writes the trajectory mailbox, so nothing here needs a lock of its own.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
   from drone_sim.perception.mailbox import PerceptionMailbox, PositionEstimate

_log = logging.getLogger(__name__)


def feed_trajectory_mailbox(*, perception: PerceptionMailbox, trajectory_mailbox, receiver_ids: Iterable[str], horizon: int, dt: float,
                            timestamp_mode: Literal["step", "monotonic"], step: int = 0,
                            safety_zone_by_id: dict[str, float] | None = None) -> None:
   """Extrapolate every observer's latest position estimates into trajectory messages and deliver them to that observer's inbox.

   One message per ``(observer, observed)`` pair, delivered under the *observed* drone's id, exactly as a real ADMM broadcast from that neighbor
   would look. Observers without estimates are skipped silently — an empty field of view is a normal state, not an error.

   :param perception: Mailbox holding the detector's estimates; read-only here.
   :param trajectory_mailbox: Target mailbox, either a
      :class:`~drone_sim.simulation.distributed.trajectory_exchange.TrajectoryMailbox` or a
      :class:`~drone_sim.simulation.distributed.threaded_mailbox.ThreadSafeMailbox`. Only ``deliver(receiver_id, message)`` is used, so the two are
      interchangeable from here.
   :param receiver_ids: Observers to feed, normally every drone in the simulation.
   :param horizon: Number of future points ``H`` per trajectory; must match the solver's horizon.
   :param dt: Simulation timestep in seconds, used for the extrapolation spacing (**not** for the finite difference, which uses the estimates' own
      ``captured_time`` and therefore stays correct when the camera runs slower than the simulation).
   :param timestamp_mode: ``"step"`` stamps the messages with ``step``, ``"monotonic"`` with ``time.monotonic()`` at delivery time. See the module
      docstring — the wrong one here empties every async solver's inbox.
   :param step: Simulation step index, used only when ``timestamp_mode == "step"``.
   :param safety_zone_by_id: Optional ``drone_id -> safety zone radius``, passed through best-effort onto the messages. An id that is missing simply
      leaves the field ``None``; no consumer reads it today.
   :raises ValueError: If ``timestamp_mode`` is neither ``"step"`` nor ``"monotonic"``, or ``horizon`` is below 1.
   """
   # Imported here, not at module scope: importing anything from drone_sim.simulation runs that package's __init__, which imports every coordinator
   # -- and the coordinators import this module in turn. A module-scope import would make that a cycle and would additionally drag the whole solver
   # stack (scipy included) into every `import drone_sim.perception`.
   from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage

   if timestamp_mode not in ("step", "monotonic"):
      raise ValueError(f"timestamp_mode must be 'step' or 'monotonic', got {timestamp_mode!r}")
   if horizon < 1:
      raise ValueError(f"horizon must be at least 1, got {horizon}")

   # trajectory[k] = p + v * dt * (k + 1): index k is the state after k+1 steps, matching LocalMPCSolver._predict_states.
   offsets = (dt * np.arange(1, horizon + 1, dtype=float)).reshape((horizon, 1))
   delivered = 0

   for receiver_id in receiver_ids:
      for observed_id, estimate in perception.latest(receiver_id).items():
         position = np.asarray(estimate.position, dtype=float).reshape(3)
         velocity = _finite_difference_velocity(perception.history(receiver_id, observed_id))

         message = TrajectoryMessage(
            drone_id=observed_id,
            trajectory=position + velocity * offsets,
            predicted_velocities=np.tile(velocity, (horizon, 1)),
            # R1: sampled here, at delivery time, so AsyncLocalSolver._filter_stale sees an age of microseconds instead of the whole uptime.
            timestamp=step if timestamp_mode == "step" else time.monotonic(),
            safety_zone_radius=None if safety_zone_by_id is None else safety_zone_by_id.get(observed_id),
         )
         trajectory_mailbox.deliver(receiver_id, message)
         delivered += 1

   if _log.isEnabledFor(logging.DEBUG):
      _log.debug("Perception bridge: %d messages delivered (mode=%s, step=%d, H=%d)", delivered, timestamp_mode, step, horizon)


def _finite_difference_velocity(history: list[PositionEstimate]) -> np.ndarray:
   """Estimate a velocity from the two newest history entries with differing ``captured_time``.

   The history is oldest first, so the newest entry is the reference and the search walks backwards from it. Entries sharing the newest
   ``captured_time`` are skipped rather than treated as an error: two estimates of one capture are a legitimate state (a re-post, or a detector that
   reports twice per frame), and differencing them would divide by zero.

   A single estimate — the first sighting of a neighbor, or a history wiped by a mailbox ``clear()`` — yields a zero velocity, i.e. the neighbor is
   assumed to hover. That is the honest answer: with one sample there is no motion information, and inventing one would put a fabricated trajectory
   into the solver's constraints.

   :param history: Estimates for one ``(observer, observed)`` pair, oldest first, as returned by :meth:`PerceptionMailbox.history`.
   :return: ``(3,)`` velocity in m/s; all zeros when no pair with distinct ``captured_time`` exists.
   """
   if len(history) < 2:
      return np.zeros(3)

   newest = history[-1]
   for older in reversed(history[:-1]):
      delta_t = newest.captured_time - older.captured_time
      if delta_t == 0.0:
         continue
      newest_position = np.asarray(newest.position, dtype=float).reshape(3)
      older_position = np.asarray(older.position, dtype=float).reshape(3)
      return (newest_position - older_position) / delta_t

   return np.zeros(3)
