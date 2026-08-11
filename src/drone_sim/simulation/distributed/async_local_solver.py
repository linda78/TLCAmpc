"""Asynchronous local MPC solver with reactive event loop.

Wraps LocalMPCSolver for asynchronous distributed MPC where each drone
runs its own solve loop in a dedicated thread, reacting to mailbox updates
from neighbors.

Part of v3.0 threaded distributed MPC architecture.
"""

from __future__ import annotations

import time
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage

if TYPE_CHECKING:
    from drone_sim.domain.drone import Drone
    from drone_sim.simulation.distributed.local_mpc import LocalMPCSolver
    from drone_sim.simulation.distributed.threaded_mailbox import ThreadSafeMailbox
    from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph

_log = logging.getLogger(__name__)


class AsyncLocalSolver:
    """Wraps LocalMPCSolver for asynchronous reactive operation.

    Designed for Phase 17's unlimited recalculation vision:
    - React to mailbox updates in real-time
    - Recalculate when neighbor data changes
    - Converge when trajectory stabilizes (placeholder - Phase 17)
    - Support graceful shutdown via stop_event

    Thread safety:
    - Designed to run in a dedicated thread per drone
    - Uses ThreadSafeMailbox for inter-drone communication
    - No internal locks needed (single-threaded per instance)

    Perception mode (``broadcast_enabled=False`` + ``allow_empty_inbox=True``): the inbox is no
    longer a negotiation channel but a snapshot of what this drone's camera saw, delivered once by
    :func:`~drone_sim.perception.bridge.feed_trajectory_mailbox` before the threads are spawned.
    The loop then degenerates to a single solve against that snapshot -- nothing broadcasts, so
    nothing notifies, so ``wait_for_update`` times out on every later pass and the trajectory stays
    put. That stability is what the coordinator's convergence poll observes.
    """

    def __init__(
        self,
        drone: Drone,
        mailbox: ThreadSafeMailbox,
        neighbor_graph: NeighborGraph,
        local_solver: LocalMPCSolver,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-3,
        stale_threshold_sec: float = 1.0,
        u_prev: np.ndarray | None = None,
        lstm_radii: "dict[str, np.ndarray] | None" = None,
        broadcast_enabled: bool = True,
        allow_empty_inbox: bool = False,
    ) -> None:
        """Initialize AsyncLocalSolver.

        :param drone: Drone object with state, route, controller, and physics
        :param mailbox: ThreadSafeMailbox for trajectory exchange
        :param neighbor_graph: NeighborGraph for neighbor lookup
        :param local_solver: LocalMPCSolver for MPC optimization
        :param max_iterations: Maximum solve iterations per run() call
        :param convergence_threshold: Convergence threshold (placeholder for Phase 17)
        :param stale_threshold_sec: Maximum age of neighbor data in seconds
        :param u_prev: Warm-start control sequence from previous timestep (H, 3) or None
        :param lstm_radii: Pre-computed LSTM safety radii snapshot from main thread.
            Dict mapping neighbor_id -> np.ndarray(H,) or None when lstm_provider is None.
            Set once at construction time; solver threads read it as-is (no mutation).
        :param broadcast_enabled: Whether a finished solve is published back to the neighbors.
            ``False`` in perception mode: the inbox is a camera snapshot written once by
            :func:`~drone_sim.perception.bridge.feed_trajectory_mailbox` on the main thread, and
            re-broadcasting an optimized trajectory on top of it would mix negotiated intent into
            what is supposed to be pure observation.
        :param allow_empty_inbox: Whether a solve runs even when no neighbor message is present.
            ``False`` (the default) skips the solve, because in negotiation mode an empty inbox
            just means the neighbors have not spoken yet. ``True`` in perception mode, where an
            empty inbox is a legitimate terminal state -- an empty field of view. Leaving this at
            ``False`` under perception would make such a drone never solve at all, ``traj_prev``
            would stay ``None``, and the coordinator's convergence poll would burn its full
            timeout every single step (risk R6 of the perception plan).
        """
        self.drone = drone
        self.mailbox = mailbox
        self.neighbor_graph = neighbor_graph
        self.solver = local_solver

        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.stale_threshold_sec = stale_threshold_sec
        self.broadcast_enabled = broadcast_enabled
        self.allow_empty_inbox = allow_empty_inbox

        # Warm-start state
        self.u_prev: np.ndarray | None = u_prev
        self.lstm_radii: "dict[str, np.ndarray] | None" = lstm_radii
        self.traj_prev: np.ndarray | None = None

        # Metrics
        self.iteration_count: int = 0

    def run(
        self,
        stop_event: threading.Event,
        obstacles: list[tuple[np.ndarray, np.ndarray]] | None = None,
        room_min: np.ndarray | None = None,
        room_max: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Run reactive solve loop until convergence or max_iterations.

        Thread target function. Waits for mailbox updates, solves MPC,
        broadcasts results, and repeats until converged or stopped.

        Follows Phase 15 convention: stop_event is first argument.

        :param stop_event: Threading event to signal shutdown
        :param obstacles: List of (center, half_extents) static obstacles
        :param room_min: Room lower bounds (3,) or None
        :param room_max: Room upper bounds (3,) or None
        :return: Tuple of (u_opt, traj_opt) from last successful solve, or (None, None)
        """
        obstacles = obstacles or []
        self.iteration_count = 0

        _log.debug("AsyncLocalSolver.run started for %s", self.drone.drone_id)

        # Poll for existing messages at startup (first iteration doesn't wait)
        first_iteration = True

        while not stop_event.is_set() and self.iteration_count < self.max_iterations:
            if not first_iteration:
                # Wait for neighbor update (100ms timeout for responsive shutdown)
                notified = self.mailbox.wait_for_update(
                    receiver_id=self.drone.drone_id,
                    timeout=0.1,
                )

                # Check stop_event after wait
                if stop_event.is_set():
                    break

                # Skip solve if no update received (timeout)
                if not notified:
                    continue
            else:
                first_iteration = False

            # Receive latest messages
            messages = self.mailbox.receive_latest(self.drone.drone_id)

            # Skip if no messages (first iteration with no data). Under perception the drone may
            # legitimately see nothing at all, and then it has to plan alone rather than never plan.
            if not messages and not self.allow_empty_inbox:
                continue

            # Filter stale data
            messages = self._filter_stale(messages)

            # Check convergence (placeholder for Phase 17)
            if self._is_converged():
                _log.debug("Converged at iteration %d for %s", self.iteration_count, self.drone.drone_id)
                break

            # Build neighbor_trajectories dict for solver
            # LocalMPCSolver expects: {neighbor_id: (trajectory, predicted_velocities)}
            neighbor_trajectories = {
                sender_id: (msg.trajectory, msg.predicted_velocities)
                for sender_id, msg in messages.items()
            }

            # Solve MPC
            u_opt, traj_opt, success, vel_opt = self.solver.solve(
                drone=self.drone,
                neighbor_trajectories=neighbor_trajectories,
                obstacles=obstacles,
                room_min=room_min,
                room_max=room_max,
                u_prev=self.u_prev,
                lstm_radii=self.lstm_radii,
            )

            if not success:
                _log.warning("Solver failed for %s at iteration %d", self.drone.drone_id, self.iteration_count)

            # Broadcast trajectory to neighbors (suppressed in perception mode, see broadcast_enabled)
            if self.broadcast_enabled:
                self._broadcast_trajectory(u_opt, traj_opt, vel_opt)

            # Update warm-start state
            self.u_prev = u_opt
            self.traj_prev = traj_opt

            # Increment iteration count
            self.iteration_count += 1

        _log.debug("AsyncLocalSolver.run finished for %s (iterations=%d)", self.drone.drone_id, self.iteration_count)

        return (self.u_prev, self.traj_prev)

    def _filter_stale(
        self,
        messages: dict[str, TrajectoryMessage],
    ) -> dict[str, TrajectoryMessage]:
        """Filter out stale messages based on timestamp age.

        Uses time.monotonic() for current time to match TrajectoryMessage.timestamp
        in async mode (Phase 16+). In sync mode, timestamp is int (timestep), but
        async mode uses float (monotonic time).

        :param messages: Dict mapping sender_id to TrajectoryMessage
        :return: Filtered dict with only fresh messages
        """
        now = time.monotonic()
        filtered = {}

        for sender_id, msg in messages.items():
            # Timestamp is float (monotonic time) in async mode
            age = now - msg.timestamp
            if age <= self.stale_threshold_sec:
                filtered[sender_id] = msg
            else:
                _log.warning(
                    "Filtered stale message from %s to %s (age=%.3fs > threshold=%.3fs)",
                    sender_id, self.drone.drone_id, age, self.stale_threshold_sec,
                )

        return filtered

    def _is_converged(self) -> bool:
        """Check if solver has converged (placeholder for Phase 17).

        Currently always returns False. Phase 17 will add neighbor-aware
        convergence detection using trajectory delta and neighbor updates.

        :return: True if converged, False otherwise
        """
        if self.traj_prev is None:
            # First iteration, not converged
            return False

        # Placeholder: Phase 17 adds neighbor-aware convergence detection
        # Will compare trajectory change and neighbor trajectory updates
        return False

    def _broadcast_trajectory(
        self,
        u_opt: np.ndarray,
        traj_opt: np.ndarray,
        vel_opt: np.ndarray | None = None,
    ) -> None:
        """Broadcast optimized trajectory to neighbors.

        Creates TrajectoryMessage with predicted positions and velocities,
        then broadcasts to all neighbors via ThreadSafeMailbox.

        :param u_opt: Optimized control sequence (H, 3)
        :param traj_opt: Optimized position trajectory (H, 3)
        :param vel_opt: Predicted velocities (H, 3), or None to recompute
        """
        # Get neighbor IDs from neighbor graph
        neighbor_ids = self.neighbor_graph.get_neighbors(self.drone.drone_id)

        # Use provided velocities or recompute if not available
        if vel_opt is None:
            _, vel_opt = self.solver._predict_states(self.drone, u_opt)

        # Create trajectory message with monotonic timestamp
        message = TrajectoryMessage(
            drone_id=self.drone.drone_id,
            trajectory=traj_opt,
            predicted_velocities=vel_opt,
            timestamp=time.monotonic(),
            safety_zone_radius=self.drone.safety_zone,
        )

        # Broadcast to neighbors
        self.mailbox.broadcast(
            sender_id=self.drone.drone_id,
            message=message,
            neighbor_ids=neighbor_ids,
        )
