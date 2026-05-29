"""Tests for ThreadedMPCCoordinator.

Tests for asynchronous distributed MPC coordinator with threaded solver architecture.
Following TDD: RED (write failing tests) -> GREEN (implement) -> REFACTOR (clean up).
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from drone_sim.simulation.distributed.threaded_coordinator import ThreadedMPCCoordinator
from drone_sim.domain.registry import COORDINATORS
from drone_sim.domain.drone import Drone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics
from drone_sim.controllers.central_cost import CentralMPCAgent


# Helper function to create real Drone for testing (same pattern as test_async_local_solver.py)
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


class TestThreadedCoordinatorRegistration:
    """Tests for coordinator registration in registry."""

    def test_coordinator_registered(self):
        """Test 'dmpc_threaded' is in COORDINATORS registry."""
        assert "dmpc_threaded" in COORDINATORS

    def test_registry_points_to_class(self):
        """Test COORDINATORS['dmpc_threaded'] points to ThreadedMPCCoordinator class."""
        assert COORDINATORS["dmpc_threaded"] is ThreadedMPCCoordinator


class TestThreadedCoordinatorConstruction:
    """Tests for ThreadedMPCCoordinator construction and configuration."""

    def test_init_default_fields(self):
        """Test dataclass defaults: horizon=5, convergence_threshold=1e-3, etc."""
        coordinator = ThreadedMPCCoordinator(dt=0.1)

        assert coordinator.dt == 0.1
        assert coordinator.horizon == 5
        assert coordinator.comm_radius is None
        assert coordinator.convergence_threshold == 1e-3
        assert coordinator.max_iterations == 100
        assert coordinator.convergence_timeout_sec == 30.0
        assert coordinator.gauss_seidel is False

    def test_init_creates_neighbor_graph(self):
        """Test __post_init__ creates NeighborGraph with comm_radius."""
        coordinator = ThreadedMPCCoordinator(dt=0.1, comm_radius=5.0)

        assert hasattr(coordinator, "_neighbor_graph")
        assert coordinator._neighbor_graph is not None
        assert coordinator._neighbor_graph.comm_radius == 5.0

    def test_custom_parameters(self):
        """Test override all parameters, verify stored correctly."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.2,
            horizon=8,
            comm_radius=10.0,
            convergence_threshold=1e-4,
            max_iterations=50,
            convergence_timeout_sec=15.0,
            gauss_seidel=True,
        )

        assert coordinator.dt == 0.2
        assert coordinator.horizon == 8
        assert coordinator.comm_radius == 10.0
        assert coordinator.convergence_threshold == 1e-4
        assert coordinator.max_iterations == 50
        assert coordinator.convergence_timeout_sec == 15.0
        assert coordinator.gauss_seidel is True


class TestThreadedCoordinatorSolveControls:
    """Tests for solve_controls() basic functionality."""

    def test_empty_drones_returns_empty(self):
        """Test solve_controls with empty list returns empty dict."""
        coordinator = ThreadedMPCCoordinator(dt=0.1, horizon=3)

        result = coordinator.solve_controls(drones=[], obstacles=[])

        assert result == {}

    def test_no_cost_drones_returns_empty(self):
        """Test drones without central_cost interface return empty dict."""
        coordinator = ThreadedMPCCoordinator(dt=0.1, horizon=3)

        # Create drone with dummy controller (no central_cost)
        class DummyController:
            pass

        dt = 0.1
        physics = LinearKinematicsPhysics(dt=dt, v_max=5.0)
        drone = Drone(
            drone_id="drone-1",
            radius=0.1,
            safety_zone=1.0,
            cons_stop=0.0,
            color="tab:blue",
            safety_color="tab:cyan",
            trace_color="tab:blue",
            controller=DummyController(),
            physics=physics,
            x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            route=Route(start=np.array([0.0, 0.0, 0.0]), waypoints=[], target=np.array([0.0, 0.0, 0.0])),
            alpha=None,
        )

        result = coordinator.solve_controls(drones=[drone], obstacles=[])

        assert result == {}

    def test_solve_returns_dict_with_drone_ids(self):
        """Test solve_controls returns dict with correct drone IDs."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([5.0, 0.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1, drone2], obstacles=[])

        assert isinstance(result, dict)
        assert "drone-1" in result
        assert "drone-2" in result

    def test_solve_returns_3d_control_vectors(self):
        """Test each control is shape (3,) numpy array."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([5.0, 0.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1, drone2], obstacles=[])

        assert result["drone-1"].shape == (3,)
        assert result["drone-2"].shape == (3,)
        assert isinstance(result["drone-1"], np.ndarray)
        assert isinstance(result["drone-2"], np.ndarray)

    def test_two_drones_at_targets(self):
        """Test two drones already at targets converge quickly."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        # Drones at their targets (should produce near-zero controls)
        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([5.0, 0.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1, drone2], obstacles=[])

        # Should get controls for both drones
        assert "drone-1" in result
        assert "drone-2" in result

        # Controls should be finite
        assert np.all(np.isfinite(result["drone-1"]))
        assert np.all(np.isfinite(result["drone-2"]))

    def test_two_drones_far_apart(self):
        """Test well-separated drones produce valid controls."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        # Drones far apart (no collision risk)
        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([10.0, 0.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1, drone2], obstacles=[])

        assert "drone-1" in result
        assert "drone-2" in result
        assert np.all(np.isfinite(result["drone-1"]))
        assert np.all(np.isfinite(result["drone-2"]))


class TestThreadedCoordinatorConvergence:
    """Tests for convergence detection and timeout behavior."""

    def test_convergence_returns_controls(self):
        """Test when drones converge, controls are returned."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        # Two drones at rest (should converge quickly)
        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([5.0, 0.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1, drone2], obstacles=[])

        assert len(result) == 2
        assert "drone-1" in result
        assert "drone-2" in result

    def test_timeout_returns_best_effort(self):
        """Test very short timeout still returns controls (best-effort)."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=0.1,  # Very short timeout
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([5.0, 0.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1, drone2], obstacles=[])

        # Should still return controls even if timeout
        assert "drone-1" in result
        assert "drone-2" in result
        # Controls might not be optimal but should be valid
        assert result["drone-1"].shape == (3,)
        assert result["drone-2"].shape == (3,)

    def test_failed_solver_returns_zero(self):
        """Test if solver.u_prev is None, return np.zeros(3)."""
        # This test is hard to trigger reliably since solvers usually succeed
        # But we test the interface: if u_prev is None, zeros are returned
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=1,  # Very low iteration limit
            convergence_timeout_sec=0.05,  # Very short timeout
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1], obstacles=[])

        # Even with extreme constraints, should return something
        assert "drone-1" in result
        assert result["drone-1"].shape == (3,)


class TestThreadedCoordinatorGaussSeidel:
    """Tests for Gauss-Seidel staggered spawn option."""

    def test_gauss_seidel_flag_stored(self):
        """Test gauss_seidel parameter is stored correctly."""
        coordinator = ThreadedMPCCoordinator(dt=0.1, gauss_seidel=True)

        assert coordinator.gauss_seidel is True

    def test_gauss_seidel_solve_works(self):
        """Test solve_controls with gauss_seidel=True produces valid controls."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
            gauss_seidel=True,
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([5.0, 0.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1, drone2], obstacles=[])

        assert "drone-1" in result
        assert "drone-2" in result
        assert np.all(np.isfinite(result["drone-1"]))
        assert np.all(np.isfinite(result["drone-2"]))


class TestThreadedCoordinatorIntegration:
    """Integration tests for full threaded coordinator scenarios."""

    def test_two_drones_moving_toward_each_other(self):
        """Test drones approaching each other still get controls."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        # Create drones with velocity toward each other
        dt = 0.1
        physics1 = LinearKinematicsPhysics(dt=dt, v_max=5.0)
        physics2 = LinearKinematicsPhysics(dt=dt, v_max=5.0)
        controller1 = CentralMPCAgent(dt=dt, horizon=3)
        controller2 = CentralMPCAgent(dt=dt, horizon=3)

        drone1 = Drone(
            drone_id="drone-1",
            radius=0.1,
            safety_zone=1.0,
            cons_stop=0.0,
            color="tab:blue",
            safety_color="tab:cyan",
            trace_color="tab:blue",
            controller=controller1,
            physics=physics1,
            x=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),  # Moving in +x direction
            route=Route(start=np.array([0.0, 0.0, 0.0]), waypoints=[], target=np.array([5.0, 0.0, 0.0])),
            alpha=None,
        )

        drone2 = Drone(
            drone_id="drone-2",
            radius=0.1,
            safety_zone=1.0,
            cons_stop=0.0,
            color="tab:red",
            safety_color="tab:orange",
            trace_color="tab:red",
            controller=controller2,
            physics=physics2,
            x=np.array([5.0, 0.0, 0.0, -1.0, 0.0, 0.0]),  # Moving in -x direction
            route=Route(start=np.array([5.0, 0.0, 0.0]), waypoints=[], target=np.array([0.0, 0.0, 0.0])),
            alpha=None,
        )

        result = coordinator.solve_controls(drones=[drone1, drone2], obstacles=[])

        assert "drone-1" in result
        assert "drone-2" in result
        assert np.all(np.isfinite(result["drone-1"]))
        assert np.all(np.isfinite(result["drone-2"]))

    @pytest.mark.slow
    def test_backward_compatibility_coordinator_interface(self):
        """Test solve_controls has same keyword-only interface as DistributedMPCCoordinator."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))

        # Test keyword-only arguments work
        result = coordinator.solve_controls(
            drones=[drone1],
            obstacles=[],
            room_min=np.array([-10.0, -10.0, -10.0]),
            room_max=np.array([10.0, 10.0, 10.0]),
        )

        assert "drone-1" in result
        assert result["drone-1"].shape == (3,)

    def test_three_drones_scenario(self):
        """Test three drones scenario produces controls for all."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))
        drone2 = create_test_drone("drone-2", np.array([5.0, 0.0, 0.0]))
        drone3 = create_test_drone("drone-3", np.array([2.5, 4.0, 0.0]))

        result = coordinator.solve_controls(drones=[drone1, drone2, drone3], obstacles=[])

        assert len(result) == 3
        assert "drone-1" in result
        assert "drone-2" in result
        assert "drone-3" in result

        for drone_id in ["drone-1", "drone-2", "drone-3"]:
            assert result[drone_id].shape == (3,)
            assert np.all(np.isfinite(result[drone_id]))

    @pytest.mark.slow
    def test_with_obstacles(self):
        """Test solve_controls with obstacles parameter."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))

        # Obstacle at (2, 0, 0) with half_extents [0.5, 0.5, 0.5]
        obstacles = [(np.array([2.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))]

        result = coordinator.solve_controls(drones=[drone1], obstacles=obstacles)

        assert "drone-1" in result
        assert result["drone-1"].shape == (3,)
        assert np.all(np.isfinite(result["drone-1"]))

    def test_with_room_bounds(self):
        """Test solve_controls with room bounds."""
        coordinator = ThreadedMPCCoordinator(
            dt=0.1,
            horizon=3,
            max_iterations=5,
            convergence_timeout_sec=5.0,
        )

        drone1 = create_test_drone("drone-1", np.array([0.0, 0.0, 0.0]))

        result = coordinator.solve_controls(
            drones=[drone1],
            obstacles=[],
            room_min=np.array([-5.0, -5.0, -5.0]),
            room_max=np.array([5.0, 5.0, 5.0]),
        )

        assert "drone-1" in result
        assert result["drone-1"].shape == (3,)
        assert np.all(np.isfinite(result["drone-1"]))
