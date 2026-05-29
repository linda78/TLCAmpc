"""Tests for drone_sim.simulation.coordinator module.

Tests for:
- CentralMPCGlobalCoordinator initialization
- CentralMPCGlobalCoordinator.solve_controls() public API
- CentralMPCGlobalCoordinator edge cases

Note: Solver-internal tests (predict_states, constraints, constraint integration)
have been moved to test_global_mpc.py alongside GlobalMPCSolver.
"""

from __future__ import annotations

import pytest
import numpy as np

from drone_sim.domain.drone import Drone, Route
from drone_sim.simulation.centralized.coordinator import CentralMPCGlobalCoordinator
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _make_drone(
    drone_id: str,
    x: np.ndarray,
    target: np.ndarray,
    controller: object,
    radius: float = 0.2,
    safety_zone: float = 1.0,
    cons_stop: float = 0.0,
    v_max: float = 5.0,
    u_min: list[float] | None = None,
    u_max: list[float] | None = None,
    dt: float = 0.1,
) -> Drone:
    """Helper to create a Drone object for testing."""
    physics = LinearKinematicsPhysics(dt=dt, v_max=v_max, u_min=u_min, u_max=u_max)
    return Drone(
        drone_id=drone_id,
        radius=radius,
        safety_zone=safety_zone,
        cons_stop=cons_stop,
        color="tab:blue",
        safety_color="tab:cyan",
        trace_color="tab:blue",
        controller=controller,
        physics=physics,
        x=np.asarray(x, dtype=float).reshape(6),
        route=Route(start=np.asarray(x, dtype=float).reshape(6)[:3], waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
    )


class TestCentralMPCGlobalCoordinatorInit:
   """Tests for CentralMPCGlobalCoordinator initialization."""

   def test_init_default_values(self):
      """Test coordinator initializes with correct default values."""
      coord = CentralMPCGlobalCoordinator(dt=0.1)
      assert coord.dt == 0.1
      assert coord.horizon == 5
      assert coord.room_wall_tolerance == 0.0
      assert coord.max_iter == 120
      assert coord.f_tol == 1e-3

   def test_init_custom_values(self):
      """Test coordinator initializes with custom values."""
      coord = CentralMPCGlobalCoordinator(dt=0.05, horizon=10, room_wall_tolerance=0.1, max_iter=200, f_tol=1e-4)
      assert coord.dt == 0.05
      assert coord.horizon == 10
      assert coord.room_wall_tolerance == 0.1
      assert coord.max_iter == 200
      assert coord.f_tol == 1e-4


class TestCentralMPCGlobalCoordinatorSolveControls:
   """Tests for CentralMPCGlobalCoordinator.solve_controls method."""

   def test_solve_controls_multiple_drones(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test solve_controls with multiple drones."""
      controller1 = CentralMPCAgent(dt=sample_coordinator.dt, horizon=sample_coordinator.horizon)
      controller2 = CentralMPCAgent(dt=sample_coordinator.dt, horizon=sample_coordinator.horizon)

      result = sample_coordinator.solve_controls(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller1),
            _make_drone("d2", np.array([10.0, 10.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller2),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0])
      )

      assert isinstance(result, dict)
      assert "d1" in result
      assert result["d1"].shape == (3,)
      assert "d2" in result

   def test_solve_controls_raises_on_infeasible(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test solve_controls raises RuntimeError when optimization is infeasible."""
      controller = CentralMPCAgent(dt=sample_coordinator.dt, horizon=sample_coordinator.horizon)

      with pytest.raises(RuntimeError, match="optimization failed|infeasible"):
         sample_coordinator.solve_controls(
            drones=[
               _make_drone("d1", np.array([-100.0, -100.0, -100.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
            ],
            obstacles=[],
            room_min=np.array([0.0, 0.0, 0.0]),
            room_max=np.array([1.0, 1.0, 1.0])
         )

   def test_solve_controls_respects_bounds(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test solve_controls produces controls within bounds."""
      controller = CentralMPCAgent(
         dt=sample_coordinator.dt,
         horizon=sample_coordinator.horizon,
      )

      result = sample_coordinator.solve_controls(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([100.0, 100.0, 100.0]), controller,
                        u_min=[-2.0, -2.0, -2.0], u_max=[2.0, 2.0, 2.0]),
         ],
         obstacles=[],
         room_min=np.array([-200.0, -200.0, 0.0]),
         room_max=np.array([200.0, 200.0, 200.0])
      )

      u = result["d1"]
      assert np.all(u >= -2.0 - 1e-6)
      assert np.all(u <= 2.0 + 1e-6)


class TestCentralMPCGlobalCoordinatorEdgeCases:
   """Edge case tests for CentralMPCGlobalCoordinator."""

   def test_single_drone_no_collision_constraints(self):
      """Test coordinator with single drone has no drone-drone constraints."""
      coord = CentralMPCGlobalCoordinator(dt=0.1, horizon=3)
      controller = CentralMPCAgent(dt=0.1, horizon=3)

      result = coord.solve_controls(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0])
      )

      assert "d1" in result

   def test_horizon_one(self):
      """Test coordinator with horizon=1."""
      coord = CentralMPCGlobalCoordinator(dt=0.1, horizon=1)
      controller = CentralMPCAgent(dt=0.1, horizon=1)

      result = coord.solve_controls(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0])
      )

      assert "d1" in result
      assert result["d1"].shape == (3,)
