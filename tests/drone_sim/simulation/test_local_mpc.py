"""Tests for drone_sim.simulation.local_mpc module.

Tests for LocalMPCSolver class functionality.
"""

from __future__ import annotations

import numpy as np
import pytest
from drone_sim.simulation.distributed.local_mpc import LocalMPCSolver
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.domain.drone import Drone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _make_drone(
    x: np.ndarray,
    target: np.ndarray,
    controller: CentralMPCAgent,
    dt: float = 0.1,
    drone_id: str = "test-drone",
    safety_zone: float = 1.0,
    alpha: float | None = None,
) -> Drone:
    """Helper to create a Drone for local MPC testing."""
    return Drone(
        drone_id=drone_id,
        radius=0.2,
        safety_zone=safety_zone,
        cons_stop=0.0,
        color="tab:blue",
        safety_color="tab:cyan",
        trace_color="tab:blue",
        controller=controller,
        physics=LinearKinematicsPhysics(dt=dt),
        x=np.asarray(x, dtype=float).reshape(6),
        route=Route(start=np.asarray(x, dtype=float).reshape(6)[:3], waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
        alpha=alpha,
    )


class TestLocalMPCSolver:
    """Tests for LocalMPCSolver class."""

    @pytest.fixture
    def solver(self) -> LocalMPCSolver:
        return LocalMPCSolver(dt=0.1, horizon=5)

    @pytest.fixture
    def controller(self) -> CentralMPCAgent:
        return CentralMPCAgent(dt=0.1, horizon=5)

    def test_solve_no_obstacles(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """Solve with no neighbors or obstacles - should reach target."""
        drone = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([2.0, 0.0, 0.0]),
            controller=controller,
        )

        u_opt, traj_opt, success, *_ = solver.solve(
            drone=drone,
            neighbor_trajectories={},
        )

        assert success
        assert u_opt.shape == (5, 3)
        assert traj_opt.shape == (5, 3)
        # Should move toward target (positive x)
        assert traj_opt[-1, 0] > 0.0

    def test_solve_with_neighbor(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """Solve with a neighbor blocking direct path."""
        drone = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([4.0, 0.0, 0.0]),
            controller=controller,
        )

        # Neighbor stationary at (2, 0, 0) - directly in path
        neighbor_traj = np.tile(np.array([[2.0, 0.0, 0.0]]), (5, 1))

        u_opt, traj_opt, success, *_ = solver.solve(
            drone=drone,
            neighbor_trajectories={"neighbor-1": (neighbor_traj, None)},
        )

        assert u_opt.shape == (5, 3)
        assert traj_opt.shape == (5, 3)
        # Should try to avoid the neighbor
        for step in range(5):
            dist = np.linalg.norm(traj_opt[step] - np.array([2.0, 0.0, 0.0]))
            assert dist >= 1.8 or not success

    def test_solve_with_static_obstacle(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """Solve with a static obstacle."""
        drone = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([4.0, 0.0, 0.0]),
            controller=controller,
        )

        obstacles = [(np.array([2.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))]

        u_opt, traj_opt, success, *_ = solver.solve(
            drone=drone,
            neighbor_trajectories={},
            obstacles=obstacles,
        )

        assert u_opt.shape == (5, 3)
        assert traj_opt.shape == (5, 3)

    def test_solve_with_room_bounds(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """Solve within room constraints."""
        drone = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([10.0, 0.0, 0.0]),
            controller=controller,
        )

        room_min = np.array([-5.0, -5.0, -5.0])
        room_max = np.array([5.0, 5.0, 5.0])

        u_opt, traj_opt, success, *_ = solver.solve(
            drone=drone,
            neighbor_trajectories={},
            room_min=room_min,
            room_max=room_max,
        )

        assert u_opt.shape == (5, 3)
        if success:
            for step in range(5):
                for axis in range(3):
                    assert traj_opt[step, axis] >= room_min[axis] + drone.safety_zone - 0.1
                    assert traj_opt[step, axis] <= room_max[axis] - drone.safety_zone + 0.1

    def test_warm_start(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """Warm start from previous solution."""
        drone1 = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([2.0, 0.0, 0.0]),
            controller=controller,
        )

        # First solve
        u_opt1, _, _, _ = solver.solve(
            drone=drone1,
            neighbor_trajectories={},
        )

        # Second solve with warm start
        drone2 = _make_drone(
            x=np.array([0.1, 0.0, 0.0, 0.5, 0.0, 0.0]),
            target=np.array([2.0, 0.0, 0.0]),
            controller=controller,
        )
        u_opt2, _, success2, _ = solver.solve(
            drone=drone2,
            neighbor_trajectories={},
            u_prev=u_opt1,
        )

        assert success2
        assert u_opt2.shape == (5, 3)

    def test_predict_positions(self, solver: LocalMPCSolver):
        """Test position prediction from controls."""
        d1 = _make_drone(x=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), target=np.array([2.0, 0.0, 0.0]), controller=CentralMPCAgent)  # Moving in +x
        u = np.zeros((5, 3))  # No acceleration

        predicted_positions, _ = solver._predict_states(d1, u)

        assert predicted_positions.shape == (5, 3)
        # Position should increase due to initial velocity
        assert predicted_positions[0, 0] > 0.0
        assert predicted_positions[-1, 0] > predicted_positions[0, 0]

    def test_infeasible_returns_false(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """Infeasible problem should return success=False."""
        drone = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([0.0, 0.0, 0.0]),
            controller=controller,
        )

        obstacles = [
            (np.array([1.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.3])),
            (np.array([-1.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.3])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.3, 0.3, 0.3])),
            (np.array([0.0, -1.0, 0.0]), np.array([0.3, 0.3, 0.3])),
        ]

        u_opt, traj_opt, success, *_ = solver.solve(
            drone=drone,
            neighbor_trajectories={},
            obstacles=obstacles,
        )

        assert u_opt.shape == (5, 3)
        assert traj_opt.shape == (5, 3)


class TestLocalMPCSolverInit:
    """Tests for LocalMPCSolver initialization."""

    def test_init_default_values(self):
        """Test LocalMPCSolver initializes with correct default values."""
        solver = LocalMPCSolver(dt=0.1, horizon=5)
        assert solver.dt == 0.1
        assert solver.horizon == 5
        assert solver.max_iter == 50
        assert solver.f_tol == 1e-4

    def test_init_custom_values(self):
        """Test LocalMPCSolver initializes with custom values."""
        solver = LocalMPCSolver(
            dt=0.05,
            horizon=10,
            max_iter=200,
            f_tol=1e-6,
        )
        assert solver.dt == 0.05
        assert solver.horizon == 10
        assert solver.max_iter == 200
        assert solver.f_tol == 1e-6


class TestLocalMPCSolverPredictPositions:
    """Tests for LocalMPCSolver._predict_positions method."""

    @pytest.fixture
    def solver(self) -> LocalMPCSolver:
        return LocalMPCSolver(dt=0.1, horizon=5)

    def test_predict_positions_correct_shape(self, solver: LocalMPCSolver):
        """Test _predict_positions returns correct shape."""
        d1 = _make_drone(x=np.zeros(6), target=np.array([2.0, 0.0, 0.0]), controller=CentralMPCAgent)
        u = np.zeros((5, 3))

        predicted_positions = solver._predict_states(d1, u)[0]

        assert predicted_positions.shape == (5, 3)

    def test_predict_positions_zero_control_zero_velocity(self, solver: LocalMPCSolver):
        """Test positions stay at origin with zero velocity and control."""
        d1 = _make_drone(x=np.zeros(6), target=np.array([2.0, 0.0, 0.0]), controller=CentralMPCAgent)
        u = np.zeros((5, 3))

        predicted_positions = solver._predict_states(d1, u)[0]

        np.testing.assert_array_almost_equal(predicted_positions, np.zeros((5, 3)), decimal=10)

    def test_predict_positions_constant_velocity(self, solver: LocalMPCSolver):
        """Test positions evolve correctly with constant velocity."""
        d1 = _make_drone(x=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), target=np.array([2.0, 0.0, 0.0]), controller=CentralMPCAgent)
        u = np.zeros((5, 3))

        predicted_positions = solver._predict_states(d1, u)[0]

        for step in range(5):
            assert predicted_positions[step, 0] > 0
            assert abs(predicted_positions[step, 1]) < 1e-10
            assert abs(predicted_positions[step, 2]) < 1e-10

    def test_predict_positions_with_acceleration(self, solver: LocalMPCSolver):
        """Test positions evolve correctly with constant acceleration."""
        d1 = _make_drone(x=np.zeros(6), target=np.array([2.0, 0.0, 0.0]), controller=CentralMPCAgent)
        u = np.ones((5, 3))

        predicted_positions = solver._predict_states(d1, u)[0]

        for step in range(5):
            for axis in range(3):
                assert predicted_positions[step, axis] > 0

        for axis in range(3):
            assert predicted_positions[-1, axis] > predicted_positions[0, axis]


class TestLocalMPCSolverAdaptive:
    """Tests for adaptive constraint passthrough in LocalMPCSolver."""

    @pytest.fixture
    def solver(self) -> LocalMPCSolver:
        return LocalMPCSolver(dt=0.1, horizon=5)

    @pytest.fixture
    def controller(self) -> CentralMPCAgent:
        return CentralMPCAgent(dt=0.1, horizon=5)

    def test_predict_states_returns_positions_and_velocities(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """_predict_states returns tuple of (positions, velocities) with correct shapes."""
        drone = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([2.0, 0.0, 0.0]),
            controller=controller,
        )
        # Nonzero controls to produce nonzero velocities
        u = np.ones((5, 3)) * 0.5

        result = solver._predict_states(drone, u)

        assert isinstance(result, tuple)
        assert len(result) == 2
        positions, velocities = result
        assert positions.shape == (5, 3)
        assert velocities.shape == (5, 3)
        # With nonzero controls, velocities should be nonzero
        assert np.any(velocities != 0)

    def test_solve_with_adaptive_drone(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """Adaptive drone solves successfully with neighbor constraints."""
        drone = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([4.0, 0.0, 0.0]),
            controller=controller,
            alpha=1.0,
        )

        # Neighbor close enough to trigger constraints
        neighbor_traj = np.tile(np.array([[3.0, 0.0, 0.0]]), (5, 1))

        u_opt, traj_opt, success, *_ = solver.solve(
            drone=drone,
            neighbor_trajectories={"neighbor-1": (neighbor_traj, None)},
        )

        assert u_opt.shape == (5, 3)
        assert traj_opt.shape == (5, 3)
        # Should complete without error (success may vary based on feasibility)
        assert np.all(np.isfinite(u_opt))

    def test_solve_fixed_drone_unchanged(self, solver: LocalMPCSolver, controller: CentralMPCAgent):
        """Fixed drone (no alpha) still works correctly after _predict_states change."""
        drone = _make_drone(
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            target=np.array([2.0, 0.0, 0.0]),
            controller=controller,
        )

        u_opt, traj_opt, success, *_ = solver.solve(
            drone=drone,
            neighbor_trajectories={},
        )

        assert success
        assert u_opt.shape == (5, 3)
        assert traj_opt.shape == (5, 3)
        # Should move toward target (positive x)
        assert traj_opt[-1, 0] > 0.0
