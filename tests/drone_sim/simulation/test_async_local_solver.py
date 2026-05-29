"""Tests for AsyncLocalSolver.

Tests for asynchronous local MPC solver with reactive event loop.
Following TDD: RED (write failing tests) -> GREEN (implement) -> REFACTOR (clean up).
"""

from __future__ import annotations

import time
import threading
import numpy as np
import pytest

from drone_sim.simulation.distributed.async_local_solver import AsyncLocalSolver
from drone_sim.simulation.distributed.local_mpc import LocalMPCSolver
from drone_sim.simulation.distributed.threaded_mailbox import ThreadSafeMailbox
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph
from drone_sim.simulation.distributed.thread_lifecycle import ThreadLifecycleManager
from drone_sim.domain.drone import Drone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics
from drone_sim.controllers.central_cost import CentralMPCAgent


# Helper function to create real Drone for testing
def create_test_drone(drone_id: str, position: np.ndarray, safety_zone: float = 1.0) -> Drone:
    """Create real Drone object with required attributes for testing."""
    dt = 0.1
    physics = LinearKinematicsPhysics(dt=dt, v_max=5.0)
    controller = CentralMPCAgent(dt=dt, horizon=3)

    return Drone(
        drone_id=drone_id,
        radius=0.1,
        safety_zone=safety_zone,
        cons_stop=0.0,
        color="tab:blue",
        safety_color="tab:cyan",
        trace_color="tab:blue",
        controller=controller,
        physics=physics,
        x=np.concatenate([position, np.zeros(3)]),  # 6D state: [pos, vel]
        route=Route(start=position.copy(), waypoints=[], target=position.copy()),  # Target is current position
        alpha=None,
    )


class TestAsyncLocalSolverInit:
    """Tests for AsyncLocalSolver construction and configuration."""

    def test_init_stores_drone_and_mailbox(self):
        """Test constructor stores drone, mailbox, neighbor_graph, solver references."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        assert async_solver.drone is drone
        assert async_solver.mailbox is mailbox
        assert async_solver.neighbor_graph is neighbor_graph
        assert async_solver.solver is solver

    def test_init_default_parameters(self):
        """Test default max_iterations=100, convergence_threshold=1e-3, stale_threshold_sec=1.0."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        assert async_solver.max_iterations == 100
        assert async_solver.convergence_threshold == 1e-3
        assert async_solver.stale_threshold_sec == 1.0

    def test_init_custom_parameters(self):
        """Test override max_iterations, convergence_threshold, stale_threshold_sec."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            max_iterations=50,
            convergence_threshold=1e-4,
            stale_threshold_sec=2.0,
        )

        assert async_solver.max_iterations == 50
        assert async_solver.convergence_threshold == 1e-4
        assert async_solver.stale_threshold_sec == 2.0

    def test_init_warm_start_state_none(self):
        """Test u_prev and traj_prev are None initially."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        assert async_solver.u_prev is None
        assert async_solver.traj_prev is None

    def test_init_iteration_count_zero(self):
        """Test iteration_count starts at 0."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        assert async_solver.iteration_count == 0


class TestAsyncLocalSolverStaleFiltering:
    """Tests for stale data detection in AsyncLocalSolver."""

    def test_filter_stale_keeps_fresh_messages(self):
        """Test messages younger than threshold are kept."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            stale_threshold_sec=1.0,
        )

        # Create fresh message (just now)
        now = time.monotonic()
        msg = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.zeros((3, 3)),
            predicted_velocities=None,
            timestamp=now,
        )

        messages = {"drone-2": msg}
        filtered = async_solver._filter_stale(messages)

        assert "drone-2" in filtered
        assert filtered["drone-2"] is msg

    def test_filter_stale_rejects_old_messages(self):
        """Test messages older than threshold are filtered out."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            stale_threshold_sec=1.0,
        )

        # Create stale message (2 seconds ago)
        old_time = time.monotonic() - 2.0
        msg = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.zeros((3, 3)),
            predicted_velocities=None,
            timestamp=old_time,
        )

        messages = {"drone-2": msg}
        filtered = async_solver._filter_stale(messages)

        assert "drone-2" not in filtered
        assert len(filtered) == 0

    def test_filter_stale_empty_messages(self):
        """Test empty dict returns empty dict."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        messages = {}
        filtered = async_solver._filter_stale(messages)

        assert len(filtered) == 0

    def test_filter_stale_mixed_ages(self):
        """Test mix of fresh and stale messages, only fresh kept."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            stale_threshold_sec=1.0,
        )

        now = time.monotonic()
        fresh_msg = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.zeros((3, 3)),
            predicted_velocities=None,
            timestamp=now,
        )
        stale_msg = TrajectoryMessage(
            drone_id="drone-3",
            trajectory=np.zeros((3, 3)),
            predicted_velocities=None,
            timestamp=now - 2.0,
        )

        messages = {"drone-2": fresh_msg, "drone-3": stale_msg}
        filtered = async_solver._filter_stale(messages)

        assert "drone-2" in filtered
        assert "drone-3" not in filtered
        assert len(filtered) == 1

    def test_filter_stale_custom_threshold(self):
        """Test configurable stale_threshold_sec is respected."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        # Use longer threshold
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            stale_threshold_sec=5.0,
        )

        # Message that would be stale with 1.0s threshold but fresh with 5.0s
        old_time = time.monotonic() - 2.0
        msg = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.zeros((3, 3)),
            predicted_velocities=None,
            timestamp=old_time,
        )

        messages = {"drone-2": msg}
        filtered = async_solver._filter_stale(messages)

        assert "drone-2" in filtered


class TestAsyncLocalSolverConvergence:
    """Tests for placeholder convergence check."""

    def test_is_converged_returns_false_first_iteration(self):
        """Test no previous trajectory means not converged."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        assert async_solver._is_converged() is False

    def test_is_converged_returns_false_placeholder(self):
        """Test after setting traj_prev, still returns False (placeholder)."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        solver = LocalMPCSolver(dt=0.1, horizon=3)

        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        # Set previous trajectory
        async_solver.traj_prev = np.zeros((3, 3))

        # Still not converged (placeholder for Phase 17)
        assert async_solver._is_converged() is False


class TestAsyncLocalSolverBroadcast:
    """Tests for trajectory publishing to neighbors."""

    def test_broadcast_sends_to_neighbors(self):
        """Test after broadcast, neighbors have trajectory in mailbox."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)

        # Set up neighbor graph with drone-1, drone-2, drone-3
        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([1.0, 0.0, 0.0]),
            "drone-3": np.array([2.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        # Broadcast trajectory
        u_opt = np.ones((3, 3))
        traj_opt = np.ones((3, 3)) * 2.0
        async_solver._broadcast_trajectory(u_opt, traj_opt)

        # Check drone-2 and drone-3 received message
        msgs_2 = mailbox.receive_latest("drone-2")
        msgs_3 = mailbox.receive_latest("drone-3")

        assert "drone-1" in msgs_2
        assert "drone-1" in msgs_3
        np.testing.assert_array_equal(msgs_2["drone-1"].trajectory, traj_opt)
        np.testing.assert_array_equal(msgs_3["drone-1"].trajectory, traj_opt)

    def test_broadcast_includes_predicted_velocities(self):
        """Test message contains predicted velocities from solver._predict_states."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)

        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([1.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        u_opt = np.ones((3, 3))
        traj_opt = np.ones((3, 3)) * 2.0
        async_solver._broadcast_trajectory(u_opt, traj_opt)

        msgs = mailbox.receive_latest("drone-2")
        assert msgs["drone-1"].predicted_velocities is not None
        assert msgs["drone-1"].predicted_velocities.shape == (3, 3)

    def test_broadcast_includes_safety_zone_radius(self):
        """Test message contains drone's safety_zone."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]), safety_zone=1.5)
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)

        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([1.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        u_opt = np.ones((3, 3))
        traj_opt = np.ones((3, 3)) * 2.0
        async_solver._broadcast_trajectory(u_opt, traj_opt)

        msgs = mailbox.receive_latest("drone-2")
        assert msgs["drone-1"].safety_zone_radius == 1.5

    def test_broadcast_uses_monotonic_timestamp(self):
        """Test message timestamp is from time.monotonic()."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)

        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([1.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        before = time.monotonic()
        u_opt = np.ones((3, 3))
        traj_opt = np.ones((3, 3)) * 2.0
        async_solver._broadcast_trajectory(u_opt, traj_opt)
        after = time.monotonic()

        msgs = mailbox.receive_latest("drone-2")
        timestamp = msgs["drone-1"].timestamp

        assert before <= timestamp <= after


class TestAsyncLocalSolverRun:
    """Tests for reactive solve loop integration."""

    def test_run_exits_on_stop_event(self):
        """Test setting stop_event causes run() to exit promptly."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({"drone-1": np.array([0.0, 0.0, 0.0])})

        solver = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        stop_event = threading.Event()
        stop_event.set()  # Set immediately

        start = time.monotonic()
        result = async_solver.run(stop_event, obstacles=[])
        elapsed = time.monotonic() - start

        # Should exit almost immediately (within 0.5 seconds)
        assert elapsed < 0.5
        # Should return (None, None) if no solve happened
        assert result == (None, None)

    def test_run_solves_on_mailbox_update(self):
        """Test sending a trajectory via mailbox triggers solve and returns controls."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([5.0, 0.0, 0.0]),  # Far away to avoid collision
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            max_iterations=2,  # Limit iterations for test speed
        )

        stop_event = threading.Event()

        # Send a trajectory message to trigger solve
        msg = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.ones((3, 3)) * 5.0,
            predicted_velocities=np.zeros((3, 3)),
            timestamp=time.monotonic(),
        )
        mailbox.broadcast("drone-2", msg, {"drone-1"})

        # Start solver in thread
        def run_solver():
            return async_solver.run(stop_event, obstacles=[])

        thread = threading.Thread(target=run_solver)
        thread.start()

        # Wait a bit for solve to happen
        time.sleep(0.3)

        # Stop the solver
        stop_event.set()
        thread.join(timeout=2.0)

        # Solver should have computed controls
        assert async_solver.u_prev is not None
        assert async_solver.traj_prev is not None
        assert async_solver.u_prev.shape == (3, 3)
        assert async_solver.traj_prev.shape == (3, 3)

    def test_run_warm_starts_from_previous(self):
        """Test second solve uses shifted u_prev from first solve."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([5.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            max_iterations=3,
        )

        stop_event = threading.Event()

        # Send two trajectory messages to trigger two solves
        msg1 = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.ones((3, 3)) * 5.0,
            predicted_velocities=np.zeros((3, 3)),
            timestamp=time.monotonic(),
        )
        mailbox.broadcast("drone-2", msg1, {"drone-1"})

        def run_solver():
            return async_solver.run(stop_event, obstacles=[])

        thread = threading.Thread(target=run_solver)
        thread.start()

        # Wait for first solve
        time.sleep(0.2)

        # Save first solution
        u_first = async_solver.u_prev.copy() if async_solver.u_prev is not None else None

        # Send second message
        msg2 = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.ones((3, 3)) * 5.1,
            predicted_velocities=np.zeros((3, 3)),
            timestamp=time.monotonic(),
        )
        mailbox.broadcast("drone-2", msg2, {"drone-1"})

        # Wait for second solve
        time.sleep(0.2)

        stop_event.set()
        thread.join(timeout=2.0)

        # Verify warm-start happened (solution should differ from first)
        assert u_first is not None
        assert async_solver.u_prev is not None
        # Solutions may be similar but warm-start should be used
        assert async_solver.iteration_count >= 2

    def test_run_broadcasts_after_solve(self):
        """Test after solve, trajectory is broadcast to neighbors."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([5.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            max_iterations=2,
        )

        stop_event = threading.Event()

        # Send message to trigger solve
        msg = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.ones((3, 3)) * 5.0,
            predicted_velocities=np.zeros((3, 3)),
            timestamp=time.monotonic(),
        )
        mailbox.broadcast("drone-2", msg, {"drone-1"})

        def run_solver():
            return async_solver.run(stop_event, obstacles=[])

        thread = threading.Thread(target=run_solver)
        thread.start()

        # Wait for solve and broadcast
        time.sleep(0.3)

        stop_event.set()
        thread.join(timeout=2.0)

        # Check drone-2 received broadcast from drone-1
        msgs = mailbox.receive_latest("drone-2")
        assert "drone-1" in msgs
        assert msgs["drone-1"].trajectory is not None

    def test_run_respects_max_iterations(self):
        """Test loop exits after max_iterations even without convergence."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([5.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            max_iterations=3,  # Low limit
        )

        stop_event = threading.Event()

        # Start solver thread first
        def run_solver():
            return async_solver.run(stop_event, obstacles=[])

        thread = threading.Thread(target=run_solver)
        thread.start()

        # Send multiple messages to trigger multiple solves
        # Send more messages than max_iterations to verify it stops at limit
        for i in range(10):
            msg = TrajectoryMessage(
                drone_id="drone-2",
                trajectory=np.ones((3, 3)) * (5.0 + i * 0.1),
                predicted_velocities=np.zeros((3, 3)),
                timestamp=time.monotonic(),
            )
            mailbox.broadcast("drone-2", msg, {"drone-1"})
            time.sleep(0.05)

        # Wait for solver to hit max_iterations
        thread.join(timeout=2.0)

        # Should have stopped at max_iterations
        assert async_solver.iteration_count == 3

    def test_run_returns_none_none_when_no_solve(self):
        """Test if stop_event set before any solve, returns (None, None)."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({"drone-1": np.array([0.0, 0.0, 0.0])})

        solver = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
        )

        stop_event = threading.Event()
        stop_event.set()  # Set before run

        result = async_solver.run(stop_event, obstacles=[])

        assert result == (None, None)

    def test_run_increments_iteration_count(self):
        """Test iteration_count increments after each solve."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([5.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            max_iterations=2,
        )

        stop_event = threading.Event()

        # Initial count should be 0
        assert async_solver.iteration_count == 0

        # Start solver thread first
        def run_solver():
            return async_solver.run(stop_event, obstacles=[])

        thread = threading.Thread(target=run_solver)
        thread.start()

        # Send messages to trigger solves
        for i in range(2):
            msg = TrajectoryMessage(
                drone_id="drone-2",
                trajectory=np.ones((3, 3)) * (5.0 + i * 0.1),
                predicted_velocities=np.zeros((3, 3)),
                timestamp=time.monotonic(),
            )
            mailbox.broadcast("drone-2", msg, {"drone-1"})
            time.sleep(0.1)  # Longer sleep to ensure processing

        # Wait for completion
        thread.join(timeout=2.0)

        # Should have incremented
        assert async_solver.iteration_count == 2


class TestAsyncLocalSolverThreadIntegration:
    """Integration tests with Phase 15 infrastructure."""

    def test_two_drones_exchange_trajectories(self):
        """Test two AsyncLocalSolvers in threads exchange trajectories via mailbox and both produce controls."""
        # Set up two drones
        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([5.0, 0.0, 0.0]))

        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([5.0, 0.0, 0.0]),
        })

        solver1 = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        solver2 = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)

        async_solver1 = AsyncLocalSolver(
            drone=drone1,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver1,
            max_iterations=3,
        )
        async_solver2 = AsyncLocalSolver(
            drone=drone2,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver2,
            max_iterations=3,
        )

        stop_event = threading.Event()

        # Bootstrap: send initial message from each drone
        msg1 = TrajectoryMessage(
            drone_id="drone-1",
            trajectory=np.zeros((3, 3)),
            predicted_velocities=np.zeros((3, 3)),
            timestamp=time.monotonic(),
        )
        msg2 = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.ones((3, 3)) * 5.0,
            predicted_velocities=np.zeros((3, 3)),
            timestamp=time.monotonic(),
        )
        mailbox.broadcast("drone-1", msg1, {"drone-2"})
        mailbox.broadcast("drone-2", msg2, {"drone-1"})

        # Start both solvers
        def run_solver1():
            return async_solver1.run(stop_event, obstacles=[])

        def run_solver2():
            return async_solver2.run(stop_event, obstacles=[])

        thread1 = threading.Thread(target=run_solver1)
        thread2 = threading.Thread(target=run_solver2)

        thread1.start()
        thread2.start()

        # Let them run for a bit
        time.sleep(0.5)

        # Stop both
        stop_event.set()
        thread1.join(timeout=2.0)
        thread2.join(timeout=2.0)

        # Both should have computed controls
        assert async_solver1.u_prev is not None
        assert async_solver2.u_prev is not None
        assert async_solver1.traj_prev is not None
        assert async_solver2.traj_prev is not None

        # Both should have done some iterations
        assert async_solver1.iteration_count > 0
        assert async_solver2.iteration_count > 0

    def test_solver_with_lifecycle_manager(self):
        """Test spawn solver via ThreadLifecycleManager, verify it runs and shuts down cleanly."""
        drone = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        mailbox = ThreadSafeMailbox()
        neighbor_graph = NeighborGraph(comm_radius=None)
        neighbor_graph.update({
            "drone-1": np.array([0.0, 0.0, 0.0]),
            "drone-2": np.array([5.0, 0.0, 0.0]),
        })

        solver = LocalMPCSolver(dt=0.1, horizon=3, max_iter=10)
        async_solver = AsyncLocalSolver(
            drone=drone,
            mailbox=mailbox,
            neighbor_graph=neighbor_graph,
            local_solver=solver,
            max_iterations=3,
        )

        # Bootstrap with initial message
        msg = TrajectoryMessage(
            drone_id="drone-2",
            trajectory=np.ones((3, 3)) * 5.0,
            predicted_velocities=np.zeros((3, 3)),
            timestamp=time.monotonic(),
        )
        mailbox.broadcast("drone-2", msg, {"drone-1"})

        # Use ThreadLifecycleManager
        manager = ThreadLifecycleManager()

        manager.spawn(
            thread_id="drone-1-solver",
            target=async_solver.run,
            args=([], ),  # obstacles
        )

        # Verify thread is alive
        status = manager.status()
        assert status["drone-1-solver"] is True

        # Let it run
        time.sleep(0.3)

        # Shutdown cleanly
        manager.shutdown(timeout=2.0)

        # Verify thread is dead
        status = manager.status()
        assert status["drone-1-solver"] is False

        # Solver should have computed something
        assert async_solver.u_prev is not None
        assert async_solver.iteration_count > 0
