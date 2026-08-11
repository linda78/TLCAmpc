"""Threaded distributed MPC coordinator with asynchronous convergence.

Implements ThreadedMPCCoordinator: spawns AsyncLocalSolver thread per drone,
monitors global convergence, returns first-step controls when all drones'
trajectories stabilize.

Part of v3.0 threaded distributed MPC architecture (Phase 17).
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from drone_sim.domain.drone import Drone, has_central_cost
from drone_sim.domain.registry import register_coordinator
# Submodule import on purpose -- see the same note in distributed_coordinator.py.
from drone_sim.perception.bridge import feed_trajectory_mailbox
from drone_sim.simulation.distributed.thread_lifecycle import ThreadLifecycleManager
from drone_sim.simulation.distributed.threaded_mailbox import ThreadSafeMailbox
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph
from drone_sim.simulation.distributed.async_local_solver import AsyncLocalSolver
from drone_sim.simulation.distributed.local_mpc import LocalMPCSolver
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


@register_coordinator("dmpc_threaded")
@dataclass
class ThreadedMPCCoordinator:
    """Asynchronous threaded distributed MPC coordinator.

    Spawns AsyncLocalSolver thread per drone, monitors global convergence,
    returns first-step controls when all drones' trajectories stabilize.

    Unlike DistributedMPCCoordinator (synchronous ADMM rounds), this coordinator
    enables truly asynchronous drone-to-drone negotiation via mailbox. Drones
    iterate until convergence (all trajectories stable), not for a fixed number
    of rounds.

    Coordinator role: lifecycle management + convergence detection, NOT
    orchestration. Drones negotiate independently via ThreadSafeMailbox.

    Stateless design: Creates fresh mailbox, lifecycle manager, and solvers
    per solve_controls() call. No persistent state between calls.
    """

    dt: float
    horizon: int = 5
    comm_radius: float | None = None
    convergence_threshold: float = 1e-3
    max_iterations: int = 100  # Per-drone iteration limit (safety cap)
    convergence_timeout_sec: float = 30.0
    gauss_seidel: bool = False  # Optional staggered spawn

    # Internal state (initialized in __post_init__)
    _neighbor_graph: NeighborGraph = field(init=False, repr=False)
    _u_prev: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _last_iteration_count: int = field(init=False, repr=False, default=0)
    _last_converged: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Initialize internal neighbor graph."""
        self._neighbor_graph = NeighborGraph(comm_radius=self.comm_radius)

    def solve_controls(
        self,
        *,
        drones: list[Drone],
        obstacles: list[tuple[np.ndarray, np.ndarray]],
        room_min: np.ndarray | None = None,
        room_max: np.ndarray | None = None,
        lstm_provider: object | None = None,
        perception_mailbox: object | None = None,
    ) -> dict[str, np.ndarray]:
        """Solve for drone controls using asynchronous threaded DMPC.

        Pattern: Spawn threads -> Wait for convergence -> Shutdown -> Extract controls.

        Creates fresh ThreadSafeMailbox, ThreadLifecycleManager, and AsyncLocalSolver
        instances for each call (stateless coordinator).

        **Perception mode.** With a ``perception_mailbox``, the mailbox is not bootstrapped with
        forward-predicted true trajectories but filled from each drone's own camera estimates by
        :func:`~drone_sim.perception.bridge.feed_trajectory_mailbox` — on *this* thread, before any
        solver thread is spawned, so no thread can start against a half-written inbox. The
        timestamps are ``time.monotonic()`` sampled at delivery, the only clock
        :meth:`~drone_sim.simulation.distributed.async_local_solver.AsyncLocalSolver._filter_stale`
        understands; a step index there would age out as "hours old" and silently empty every inbox
        (risk R1). The bridge is called on every ``solve_controls`` for the same reason — those
        stamps expire after ``stale_threshold_sec``.

        The negotiation itself is switched off: solvers run with ``perception_mode=True``, which
        both suppresses the re-broadcast (a camera observation must not be overwritten by a solved
        trajectory) and lets a drone with an empty inbox solve anyway. Each solver is also given
        ``max_iterations=1``, because with nothing broadcasting there is no later pass that could
        see different neighbor data. The threads therefore end themselves after one solve, joining
        them is the completion signal, and ``converged`` is ``True`` by construction — there is
        nothing to converge to. Without the empty-inbox half, a drone whose field of view happens to
        be empty would never solve at all and its ``traj_prev`` would stay ``None`` (risk R6).

        :param drones: List of Drone objects to optimize
        :param obstacles: Static obstacles as list of (center, half_extents) tuples
        :param room_min: Room lower bounds (3,) or None
        :param room_max: Room upper bounds (3,) or None
        :param lstm_provider: Optional provider of per-neighbor LSTM safety radii, or None
        :param perception_mailbox: Optional :class:`~drone_sim.perception.mailbox.PerceptionMailbox`.
            ``None`` (the default) keeps the classic true-state negotiation untouched.
        :return: Dict mapping drone_id to first-step control (3,)
        """
        perception_active = perception_mailbox is not None

        # Filter drones to optimize (must have central_cost interface)
        opt_drones = [d for d in drones if has_central_cost(d.controller)]

        if not opt_drones:
            return {}

        # 1. Update neighbor graph from current positions
        positions = {d.drone_id: np.asarray(d.x[:3], dtype=float) for d in drones}
        self._neighbor_graph.update(positions)

        # Pre-compute LSTM radii in the main thread before spawning solver threads.
        # Threads receive a read-only numpy dict snapshot — no concurrent provider calls.
        all_drones_by_id = {d.drone_id: d for d in drones}
        lstm_radii_by_drone: dict[str, dict[str, np.ndarray]] = {}
        if lstm_provider is not None:
            for drone in opt_drones:
                neighbors = list(self._neighbor_graph.get_neighbors(drone.drone_id))
                if neighbors:
                    r_floor = {nid: all_drones_by_id[nid].safety_zone for nid in neighbors}
                    lstm_radii_by_drone[drone.drone_id] = lstm_provider.compute_neighbor_safety_radii(
                        neighbors, r_floor
                    )

        # 2. Create shared mailbox for this solve
        mailbox = ThreadSafeMailbox()

        # 3. Create thread lifecycle manager
        manager = ThreadLifecycleManager()

        # 4. Create AsyncLocalSolver for each drone
        async_solvers: dict[str, AsyncLocalSolver] = {}

        for drone in opt_drones:
            local_solver = LocalMPCSolver(dt=self.dt, horizon=self.horizon)

            # Pass warm-start from previous timestep if available
            u_prev = self._u_prev.get(drone.drone_id)

            async_solver = AsyncLocalSolver(
                drone=drone,
                mailbox=mailbox,
                neighbor_graph=self._neighbor_graph,
                local_solver=local_solver,
                # One pass is the whole algorithm under perception: nothing broadcasts, so no later pass
                # could ever see different neighbor data. Saying so up front lets the thread exit after
                # its solve instead of idling in wait_for_update until shutdown.
                max_iterations=1 if perception_active else self.max_iterations,
                convergence_threshold=self.convergence_threshold,
                u_prev=u_prev,
                lstm_radii=lstm_radii_by_drone.get(drone.drone_id),
                perception_mode=perception_active,
            )
            async_solvers[drone.drone_id] = async_solver

        # 5. Fill the mailbox before spawning: true-state bootstrap, or the camera snapshot.
        #    The mailbox is fresh per call, so there is nothing stale to clear first.
        if perception_active:
            feed_trajectory_mailbox(
                perception=perception_mailbox,
                trajectory_mailbox=mailbox,
                receiver_ids=list(async_solvers),
                horizon=self.horizon,
                dt=self.dt,
                timestamp_mode="monotonic",
                safety_zone_by_id={d.drone_id: d.safety_zone for d in drones},
            )
        else:
            self._bootstrap_mailbox(mailbox, async_solvers)

        # 6. Spawn threads (optionally staggered for Gauss-Seidel)
        if self.gauss_seidel:
            self._spawn_staggered(manager, async_solvers, obstacles, room_min, room_max)
        else:
            self._spawn_parallel(manager, async_solvers, obstacles, room_min, room_max)

        # 7. Wait for convergence or timeout. Under perception there is nothing to converge to, and the
        #    threads end themselves after their single solve — so the join below *is* the completion
        #    signal. Polling for stability instead would have to observe two identical snapshots first
        #    (2 x 50 ms) and would then still find every thread parked in its 0.1 s condition wait:
        #    ~200 ms of main-thread sleeping per simulation step, on a dt of 0.1 s.
        converged = True if perception_active else self._wait_for_convergence(
            async_solvers,
            timeout_sec=self.convergence_timeout_sec
        )

        # 8. Shutdown threads
        manager.shutdown(timeout=5.0)

        if not converged:
            _log.warning(
                "ThreadedMPCCoordinator: Convergence timeout after %.2fs",
                self.convergence_timeout_sec
            )

        # Track convergence metrics for visualization API
        self._last_converged = converged
        # Iteration count = max iterations across all solvers
        self._last_iteration_count = max(
            (solver.iteration_count for solver in async_solvers.values()),
            default=0
        )

        # 9. Extract first-step controls from final solutions
        controls: dict[str, np.ndarray] = {}
        for drone_id, solver in async_solvers.items():
            if solver.u_prev is not None:
                # Store full control sequence for warm-start
                self._u_prev[drone_id] = solver.u_prev.copy()
                controls[drone_id] = solver.u_prev[0].copy()
            else:
                # Solver failed—return zero control
                _log.error(
                    "ThreadedMPCCoordinator: Solver %s failed, returning zero control",
                    drone_id
                )
                controls[drone_id] = np.zeros(3, dtype=float)

        return controls

    def _bootstrap_mailbox(
        self,
        mailbox: ThreadSafeMailbox,
        async_solvers: dict[str, AsyncLocalSolver],
    ) -> None:
        """Send initial trajectory messages so first iteration doesn't block.

        Pattern: Bootstrap with forward-predicted trajectories from central_initial_guess.
        This shows neighbors approaching (not stationary), creating earlier collision
        constraint pressure for better convergence in collinear scenarios.

        :param mailbox: ThreadSafeMailbox to bootstrap
        :param async_solvers: Dict of drone_id -> AsyncLocalSolver
        """
        for drone_id, solver in async_solvers.items():
            drone = solver.drone
            controller = drone.controller

            # Get initial control guess from controller
            if solver.u_prev is not None:
                # Warm-start: shift previous solution
                u0 = np.concatenate([solver.u_prev[1:], solver.u_prev[-1:]], axis=0)
            else:
                # Cold start: use central_initial_guess
                u0_raw = controller.central_initial_guess(drone)
                u0 = self._pad_or_trim_horizon(u0_raw, self.horizon)

            # Predict forward trajectory from initial controls
            initial_traj, initial_vel = solver.solver._predict_states(drone, u0)

            message = TrajectoryMessage(
                drone_id=drone_id,
                trajectory=initial_traj,
                predicted_velocities=initial_vel,
                timestamp=time.monotonic(),
                safety_zone_radius=drone.safety_zone,
            )

            neighbor_ids = self._neighbor_graph.get_neighbors(drone_id)
            mailbox.broadcast(
                sender_id=drone_id,
                message=message,
                neighbor_ids=neighbor_ids,
            )

    def _pad_or_trim_horizon(self, u: np.ndarray, horizon: int) -> np.ndarray:
        """Pad or trim a control sequence to match the given horizon.

        :param u: Control sequence to adjust
        :param horizon: Target horizon length
        :return: Adjusted control sequence (horizon, 3)
        """
        u = np.asarray(u, dtype=float).reshape((-1, 3))
        if u.shape[0] < horizon:
            pad = np.tile(u[-1:], (horizon - u.shape[0], 1))
            u = np.concatenate([u, pad], axis=0)
        elif u.shape[0] > horizon:
            u = u[:horizon]
        return u

    def _spawn_parallel(
        self,
        manager: ThreadLifecycleManager,
        async_solvers: dict[str, AsyncLocalSolver],
        obstacles: list,
        room_min: np.ndarray | None,
        room_max: np.ndarray | None,
    ) -> None:
        """Spawn all solver threads simultaneously (Jacobi-style).

        :param manager: ThreadLifecycleManager for thread spawning
        :param async_solvers: Dict of drone_id -> AsyncLocalSolver
        :param obstacles: Static obstacles
        :param room_min: Room lower bounds
        :param room_max: Room upper bounds
        """
        for drone_id, solver in async_solvers.items():
            manager.spawn(
                thread_id=drone_id,
                target=solver.run,
                args=(obstacles, room_min, room_max),
            )

    def _spawn_staggered(
        self,
        manager: ThreadLifecycleManager,
        async_solvers: dict[str, AsyncLocalSolver],
        obstacles: list,
        room_min: np.ndarray | None,
        room_max: np.ndarray | None,
    ) -> None:
        """Spawn threads with staggered delays (Gauss-Seidel-style ordering).

        Pattern: Priority-based ordering—most constrained drones first.
        Stagger by 50ms to break symmetry and encourage earlier threads to
        publish before later threads start.

        This is a heuristic for symmetry breaking, NOT true Gauss-Seidel.
        Threading doesn't guarantee execution order.

        :param manager: ThreadLifecycleManager for thread spawning
        :param async_solvers: Dict of drone_id -> AsyncLocalSolver
        :param obstacles: Static obstacles
        :param room_min: Room lower bounds
        :param room_max: Room upper bounds
        """
        # Simple ordering: lexicographic by drone_id
        # Future: priority-based ordering like DistributedMPCCoordinator._compute_priority
        ordered_ids = sorted(async_solvers.keys())

        for i, drone_id in enumerate(ordered_ids):
            solver = async_solvers[drone_id]
            manager.spawn(
                thread_id=drone_id,
                target=solver.run,
                args=(obstacles, room_min, room_max),
            )
            # Stagger by 50ms to break symmetry
            if i < len(ordered_ids) - 1:
                time.sleep(0.05)

    def _wait_for_convergence(
        self,
        async_solvers: dict[str, AsyncLocalSolver],
        timeout_sec: float,
    ) -> bool:
        """Poll for global convergence (all drones' trajectories stable).

        Pattern: Periodic convergence check with timeout escape.

        Convergence criteria:
        1. All solvers have produced at least one solution (traj_prev not None)
        2. All solvers' trajectory change below threshold OR hit max iterations
        3. No timeout exceeded

        Polls every 50ms to check trajectory snapshots. Compares consecutive
        snapshots to detect when all trajectories stabilize.

        :param async_solvers: Dict of drone_id -> AsyncLocalSolver
        :param timeout_sec: Maximum time to wait for convergence
        :return: True if converged, False if timeout
        """
        start_time = time.monotonic()
        prev_trajectories: dict[str, np.ndarray] = {}

        while True:
            # Check timeout first
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                _log.warning(
                    "ThreadedMPCCoordinator: Convergence timeout after %.2fs",
                    timeout_sec
                )
                return False

            # Snapshot current trajectories
            current_trajectories = {
                drone_id: solver.traj_prev.copy()
                for drone_id, solver in async_solvers.items()
                if solver.traj_prev is not None
            }

            # Check if all solvers have produced a solution
            if len(current_trajectories) < len(async_solvers):
                # Some solvers still initializing
                time.sleep(0.05)
                continue

            # Compute trajectory changes
            if prev_trajectories:
                max_change = max(
                    float(np.linalg.norm(current_trajectories[did] - prev_trajectories[did]))
                    for did in current_trajectories.keys()
                )

                if max_change < self.convergence_threshold:
                    # Global convergence achieved
                    _log.info(
                        "ThreadedMPCCoordinator: Converged after %.2fs (max_change=%.6f)",
                        elapsed, max_change
                    )
                    return True

            # Update previous trajectories for next comparison
            prev_trajectories = current_trajectories

            # Sleep before next check
            time.sleep(0.05)

    def get_last_iteration_count(self) -> int:
        """Get the maximum iteration count from the last solve.

        Returns the maximum iteration count across all solver threads.
        Useful for convergence analysis and visualization.
        """
        return self._last_iteration_count

    def get_last_converged(self) -> bool:
        """Check if the last solve converged before timeout.

        Returns True if global convergence was achieved, False if timeout occurred.
        """
        return self._last_converged

    def get_neighbor_pairs(self) -> list[tuple[str, str]]:
        """Get current neighbor pairs for visualization.

        Returns list of (drone_id_1, drone_id_2) tuples for neighbor connections.
        """
        return self._neighbor_graph.get_neighbor_pairs()
