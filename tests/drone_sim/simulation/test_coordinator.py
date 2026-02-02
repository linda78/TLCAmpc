"""Tests for drone_sim.simulation.coordinator module.

Tests for:
- predict_external_const_vel helper function
- CentralMPCGlobalCoordinator._predict_states()
- CentralMPCGlobalCoordinator._constraints()
- CentralMPCGlobalCoordinator.solve_controls()
- CentralMPCGlobalCoordinator observer methods
"""

from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from drone_sim.simulation.coordinator import (predict_external_const_vel, CentralMPCGlobalCoordinator)
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


class TestPredictExternalConstVel:
   """Tests for predict_external_const_vel helper function."""

   def testpredict_external_const_vel_stationary(self):
      """Test prediction for stationary object."""
      p0 = np.array([1.0, 2.0, 3.0])
      v0 = np.array([0.0, 0.0, 0.0])
      dt = 0.1
      horizon = 5

      Ps = predict_external_const_vel(p0, v0, dt, horizon)

      assert Ps.shape == (horizon, 3)
      # All positions should be the same since velocity is zero
      for k in range(horizon):
         assert_array_almost_equal(Ps[k], p0)

   def testpredict_external_const_vel_moving(self):
      """Test prediction for moving object."""
      p0 = np.array([0.0, 0.0, 0.0])
      v0 = np.array([1.0, 0.0, 0.0])  # Moving in x direction
      dt = 0.1
      horizon = 5

      Ps = predict_external_const_vel(p0, v0, dt, horizon)

      assert isinstance(Ps, np.ndarray)
      assert Ps.shape == (horizon, 3)
      # Position at step k+1 should be p0 + (k+1)*dt*v0
      for k in range(horizon):
         expected = p0 + (k + 1) * dt * v0
         assert_array_almost_equal(Ps[k], expected)

   def testpredict_external_const_vel_single_step(self):
      """Test prediction with horizon=1."""
      p0 = np.array([0.0, 0.0, 0.0])
      v0 = np.array([5.0, 5.0, 5.0])
      dt = 0.1
      horizon = 1

      Ps = predict_external_const_vel(p0, v0, dt, horizon)

      assert Ps.shape == (1, 3)
      expected = p0 + dt * v0
      assert_array_almost_equal(Ps[0], expected)


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

   def test_init_creates_physics_model(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test __post_init__ creates internal physics model."""
      assert hasattr(sample_coordinator, "_phys")
      assert sample_coordinator._phys.dt == sample_coordinator.dt


class TestCentralMPCGlobalCoordinatorPredictStates:
   """Tests for CentralMPCGlobalCoordinator._predict_states method."""

   def test_predict_states_shape(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _predict_states returns correct shape."""
      M = 2  # Number of drones
      H = sample_coordinator.horizon
      xs0 = np.zeros((M, 6))
      u = np.zeros((M, H, 3))

      X = sample_coordinator._predict_states(xs0, u)

      assert X.shape == (M, H, 6)

   def test_predict_states_zero_control(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _predict_states with zero control maintains velocity trajectory."""
      M = 1
      H = sample_coordinator.horizon
      xs0 = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])  # Moving in x
      u = np.zeros((M, H, 3))

      X = sample_coordinator._predict_states(xs0, u)

      # Position should increase linearly with velocity
      dt = sample_coordinator.dt
      for k in range(H):
         expected_x = (k + 1) * dt * 1.0
         assert X[0, k, 0] == pytest.approx(expected_x, abs=1e-6)

   def test_predict_states_with_acceleration(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _predict_states with constant acceleration."""
      M = 1
      H = sample_coordinator.horizon
      xs0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # At rest
      u = np.ones((M, H, 3)) * 1.0  # Constant acceleration

      X = sample_coordinator._predict_states(xs0, u)

      # Velocity should increase with each step
      assert X[0, -1, 3] > 0  # Final x velocity positive

   def test_predict_states_multiple_drones(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _predict_states with multiple drones."""
      M = 3
      H = sample_coordinator.horizon
      xs0 = np.array([
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [5.0, 0.0, 0.0, -1.0, 0.0, 0.0],
         [2.5, 5.0, 0.0, 0.0, -1.0, 0.0]
      ])
      u = np.zeros((M, H, 3))

      X = sample_coordinator._predict_states(xs0, u)

      assert X.shape == (M, H, 6)
      assert X[0, -1, 0] > xs0[0, 0]
      assert X[1, -1, 0] < xs0[1, 0]
      assert X[2, -1, 1] < xs0[2, 1]


class TestCentralMPCGlobalCoordinatorConstraints:
   """Tests for CentralMPCGlobalCoordinator._constraints method."""

   def test_constraints_returns_array(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test _constraints returns numpy array."""
      M = 2
      H = sample_coordinator.horizon
      xs0 = np.array([
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      ])
      u_flat = np.zeros(M * H * 3)
      opt_ids = ["d1", "d2"]
      safety_by_id = {"d1": 1.0, "d2": 1.0}
      radii_by_id = {"d1": 0.2, "d2": 0.2}
      cons_stops_by_id = {"d1": 0.0, "d2": 0.0}
      v_max_by_id = {"d1": 5.0, "d2": 5.0}

      g = sample_coordinator._constraints(
         u_flat,
         xs0=xs0,
         opt_ids=opt_ids,
         safety_by_id=safety_by_id,
         radii_by_id=radii_by_id,
         cons_stops_by_id=cons_stops_by_id,
         v_max_by_id=v_max_by_id,
         P_ext={},
         obstacles=[],
         room_min=None,
         room_max=None
      )

      assert isinstance(g, np.ndarray)
      assert len(g) > 0

   def test_constraints_satisfied_when_far_apart(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test constraints are satisfied (>=0) when drones are far apart."""
      M = 2
      H = sample_coordinator.horizon
      xs0 = np.array([
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      ])
      u_flat = np.zeros(M * H * 3)
      opt_ids = ["d1", "d2"]
      safety_by_id = {"d1": 1.0, "d2": 1.0}
      radii_by_id = {"d1": 0.2, "d2": 0.2}
      cons_stops_by_id = {"d1": 0.0, "d2": 0.0}
      v_max_by_id = {"d1": 5.0, "d2": 5.0}

      g = sample_coordinator._constraints(
         u_flat,
         xs0=xs0,
         opt_ids=opt_ids,
         safety_by_id=safety_by_id,
         radii_by_id=radii_by_id,
         cons_stops_by_id=cons_stops_by_id,
         v_max_by_id=v_max_by_id,
         P_ext={},
         obstacles=[],
         room_min=None,
         room_max=None
      )

      assert np.all(g >= 0)

   def test_constraints_violated_when_overlapping(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test constraints are violated (<0) when drones overlap."""
      M = 2
      H = sample_coordinator.horizon
      xs0 = np.array([
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
      ])
      u_flat = np.zeros(M * H * 3)
      opt_ids = ["d1", "d2"]
      safety_by_id = {"d1": 1.0, "d2": 1.0}
      radii_by_id = {"d1": 0.2, "d2": 0.2}
      cons_stops_by_id = {"d1": 0.0, "d2": 0.0}
      v_max_by_id = {"d1": 5.0, "d2": 5.0}

      g = sample_coordinator._constraints(
         u_flat,
         xs0=xs0,
         opt_ids=opt_ids,
         safety_by_id=safety_by_id,
         radii_by_id=radii_by_id,
         cons_stops_by_id=cons_stops_by_id,
         v_max_by_id=v_max_by_id,
         P_ext={},
         obstacles=[],
         room_min=None,
         room_max=None
      )

      assert np.any(g < 0)

   def test_constraints_with_obstacles(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test constraints account for obstacles."""
      M = 1
      H = sample_coordinator.horizon
      xs0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      u_flat = np.zeros(M * H * 3)
      opt_ids = ["d1"]
      safety_by_id = {"d1": 1.0}
      radii_by_id = {"d1": 0.2}
      cons_stops_by_id = {"d1": 0.0}
      v_max_by_id = {"d1": 5.0}
      obstacles = [(np.array([0.5, 0.0, 0.0]), 0.2)]

      g = sample_coordinator._constraints(
         u_flat,
         xs0=xs0,
         opt_ids=opt_ids,
         safety_by_id=safety_by_id,
         radii_by_id=radii_by_id,
         cons_stops_by_id=cons_stops_by_id,
         v_max_by_id=v_max_by_id,
         P_ext={},
         obstacles=obstacles,
         room_min=None,
         room_max=None
      )

      assert np.any(g < 0)

   def test_constraints_with_room_bounds(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test constraints account for room bounds."""
      M = 1
      H = sample_coordinator.horizon
      xs0 = np.array([[-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
      u_flat = np.zeros(M * H * 3)
      opt_ids = ["d1"]
      safety_by_id = {"d1": 1.0}
      radii_by_id = {"d1": 0.2}
      cons_stops_by_id = {"d1": 0.0}
      v_max_by_id = {"d1": 5.0}
      room_min = np.array([0.0, 0.0, 0.0])
      room_max = np.array([10.0, 10.0, 10.0])

      g = sample_coordinator._constraints(
         u_flat,
         xs0=xs0,
         opt_ids=opt_ids,
         safety_by_id=safety_by_id,
         radii_by_id=radii_by_id,
         cons_stops_by_id=cons_stops_by_id,
         v_max_by_id=v_max_by_id,
         P_ext={},
         obstacles=[],
         room_min=room_min,
         room_max=room_max
      )

      assert np.any(g < 0)


class TestCentralMPCGlobalCoordinatorSolveControls:
   """Tests for CentralMPCGlobalCoordinator.solve_controls method."""

   def test_solve_controls_multiple_drones(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test solve_controls with multiple drones."""
      physics = LinearKinematicsPhysics(dt=sample_coordinator.dt)
      controller1 = CentralMPCAgent(dt=sample_coordinator.dt, physics=physics, horizon=sample_coordinator.horizon)
      controller2 = CentralMPCAgent(dt=sample_coordinator.dt, physics=physics, horizon=sample_coordinator.horizon)

      result = sample_coordinator.solve_controls(
         drone_ids=["d1", "d2"],
         xs=[
            np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 5.0, 0.0, 0.0, 0.0]),
         ],
         prefs=[np.array([5.0, 5.0, 5.0]), np.array([5.0, 5.0, 5.0])],
         radii=[0.2, 0.2],
         safety_zones=[1.0, 1.0],
         cons_stops=[0.0, 0.0],
         v_maxs=[5.0, 5.0],
         u_mins=[(-3.0, -3.0, -3.0), (-3.0, -3.0, -3.0)],
         u_maxs=[(3.0, 3.0, 3.0), (3.0, 3.0, 3.0)],
         controllers=[controller1, controller2],
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
      physics = LinearKinematicsPhysics(dt=sample_coordinator.dt)
      controller = CentralMPCAgent(dt=sample_coordinator.dt, physics=physics, horizon=sample_coordinator.horizon)

      with pytest.raises(RuntimeError, match="optimization failed|infeasible"):
         sample_coordinator.solve_controls(
            drone_ids=["d1"],
            xs=[np.array([-100.0, -100.0, -100.0, 0.0, 0.0, 0.0])],
            prefs=[np.array([5.0, 5.0, 5.0])],
            radii=[0.2],
            safety_zones=[1.0],
            cons_stops=[0.0],
            v_maxs=[5.0],
            u_mins=[(-3.0, -3.0, -3.0)],
            u_maxs=[(3.0, 3.0, 3.0)],
            controllers=[controller],
            obstacles=[],
            room_min=np.array([0.0, 0.0, 0.0]),
            room_max=np.array([1.0, 1.0, 1.0])
         )

   def test_solve_controls_respects_bounds(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test solve_controls produces controls within bounds."""
      physics = LinearKinematicsPhysics(dt=sample_coordinator.dt, u_min=[-2.0, -2.0, -2.0], u_max=[2.0, 2.0, 2.0])
      controller = CentralMPCAgent(dt=sample_coordinator.dt, physics=physics, horizon=sample_coordinator.horizon)

      result = sample_coordinator.solve_controls(
         drone_ids=["d1"],
         xs=[np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])],
         prefs=[np.array([100.0, 100.0, 100.0])],
         radii=[0.2],
         safety_zones=[1.0],
         cons_stops=[0.0],
         v_maxs=[5.0],
         u_mins=[(-2.0, -2.0, -2.0)],
         u_maxs=[(2.0, 2.0, 2.0)],
         controllers=[controller],
         obstacles=[],
         room_min=np.array([-200.0, -200.0, 0.0]),
         room_max=np.array([200.0, 200.0, 200.0])
      )

      u = result["d1"]
      assert np.all(u >= -2.0 - 1e-6)
      assert np.all(u <= 2.0 + 1e-6)


class TestCentralMPCGlobalCoordinatorSetExternalPredictions:
   """Tests for CentralMPCGlobalCoordinator.set_external_predictions method."""

   def test_set_external_predictions_empty(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test set_external_predictions with empty input."""
      result = sample_coordinator.set_external_predictions(None)
      assert result == {}

      result = sample_coordinator.set_external_predictions({})
      assert result == {}

   def test_set_external_predictions_reshapes_arrays(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test set_external_predictions reshapes arrays correctly."""
      H = sample_coordinator.horizon
      ext_pred = {"ext1": np.ones((H, 3)) * 5.0, "ext2": np.ones(H * 3) * 3.0}  # Flat array

      result = sample_coordinator.set_external_predictions(ext_pred)

      assert result["ext1"].shape == (H, 3)
      assert result["ext2"].shape == (H, 3)


class TestCentralMPCGlobalCoordinatorObservers:
   """Tests for CentralMPCGlobalCoordinator observer methods."""

   def test_observe_obstacles_adds_constraints(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test observe_obstacles adds constraint values."""
      M = 1
      H = sample_coordinator.horizon
      P_opt = np.zeros((M, H, 3))  # Drone at origin
      obstacles = [(np.array([5.0, 0.0, 0.0]), 0.5)]
      opt_ids = ["d1"]
      safety_by_id = {"d1": 1.0}
      vals: list[float] = []

      sample_coordinator.observe_obstacles(M, P_opt, obstacles, opt_ids, safety_by_id, vals)

      # Should have H constraints (one per timestep)
      assert len(vals) == H
      # Constraints should be positive (satisfied) since obstacle is far
      assert all(v > 0 for v in vals)

   def test_observe_external_predictions_adds_constraints(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test observe_external_predictions adds constraint values."""
      M = 1
      H = sample_coordinator.horizon
      P_opt = np.zeros((M, H, 3))  # Drone at origin
      P_ext = {"ext1": np.ones((H, 3)) * 10.0}  # External drone far away
      opt_ids = ["d1"]
      safety_by_id = {"d1": 1.0, "ext1": 1.0}
      vals: list[float] = []

      sample_coordinator.observe_external_predictions(M, P_ext, P_opt, opt_ids, safety_by_id, vals)

      # Should have H constraints (one per timestep per external drone)
      assert len(vals) == H
      # Constraints should be positive (satisfied) since external is far
      assert all(v > 0 for v in vals)

   def test_observe_no_flying_zone_adds_constraints(self, sample_coordinator: CentralMPCGlobalCoordinator):
      """Test observe_no_flying_zone adds room boundary constraints."""
      M = 1
      H = sample_coordinator.horizon
      P_opt = np.zeros((M, H, 3)) + 5.0  # Drone at (5,5,5)
      opt_ids = ["d1"]
      safety_by_id = {"d1": 1.0}
      room_min = np.array([0.0, 0.0, 0.0])
      room_max = np.array([10.0, 10.0, 10.0])
      vals: list[float] = []
      xs0 = np.array([[5.0, 5.0, 5.0, 0.0, 0.0, 0.0]])  # Current state: position (5,5,5), zero velocity

      sample_coordinator.observe_no_flying_zone(M, P_opt, opt_ids, safety_by_id, room_max, room_min, vals, xs0)

      # Should have H * M * 6 constraints (6 walls per drone per timestep)
      assert len(vals) == H * M * 6
      # Drone is well within room, so constraints should be satisfied
      assert all(v >= 0 for v in vals)


class TestCentralMPCGlobalCoordinatorEdgeCases:
   """Edge case tests for CentralMPCGlobalCoordinator."""

   def test_single_drone_no_collision_constraints(self):
      """Test coordinator with single drone has no drone-drone constraints."""
      coord = CentralMPCGlobalCoordinator(dt=0.1, horizon=3)
      physics = LinearKinematicsPhysics(dt=0.1)
      controller = CentralMPCAgent(dt=0.1, physics=physics, horizon=3)

      result = coord.solve_controls(
         drone_ids=["d1"],
         xs=[np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])],
         prefs=[np.array([5.0, 5.0, 5.0])],
         radii=[0.2],
         safety_zones=[1.0],
         cons_stops=[0.0],
         v_maxs=[5.0],
         u_mins=[(-3.0, -3.0, -3.0)],
         u_maxs=[(3.0, 3.0, 3.0)],
         controllers=[controller],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0])
      )

      assert "d1" in result

   def test_horizon_one(self):
      """Test coordinator with horizon=1."""
      coord = CentralMPCGlobalCoordinator(dt=0.1, horizon=1)
      physics = LinearKinematicsPhysics(dt=0.1)
      controller = CentralMPCAgent(dt=0.1, physics=physics, horizon=1)

      result = coord.solve_controls(
         drone_ids=["d1"],
         xs=[np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])],
         prefs=[np.array([5.0, 5.0, 5.0])],
         radii=[0.2],
         safety_zones=[1.0],
         cons_stops=[0.0],
         v_maxs=[5.0],
         u_mins=[(-3.0, -3.0, -3.0)],
         u_maxs=[(3.0, 3.0, 3.0)],
         controllers=[controller],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0])
      )

      assert "d1" in result
      assert result["d1"].shape == (3,)
