"""Tests for drone_sim.simulation.global_mpc module.

Tests for:
- GlobalMPCSolver._predict_states()
- GlobalMPCSolver._constraints()
- GlobalMPCSolver.solve()
- Constraint class integration via solver
"""

from __future__ import annotations

import pytest
import numpy as np

from drone_sim.domain.drone import Drone, Route
from drone_sim.simulation.centralized.global_mpc import GlobalMPCSolver
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
    alpha: float | None = None,
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
        alpha=alpha,
    )


@pytest.fixture
def solver() -> GlobalMPCSolver:
    """Return a GlobalMPCSolver with standard test settings."""
    return GlobalMPCSolver(dt=0.1, horizon=5)


class TestGlobalMPCSolverPredictStates:
   """Tests for GlobalMPCSolver._predict_states method."""

   def test_predict_states_shape(self, solver: GlobalMPCSolver):
      """Test _predict_states returns correct shape."""
      num_drones = 2
      horizon = solver.horizon
      d1 = _make_drone("d1", np.zeros(6), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      d2 = _make_drone("d2", np.zeros(6), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u = np.zeros((num_drones, horizon, 3))

      predicted_states = solver._predict_states([d1, d2], u)

      assert predicted_states.shape == (num_drones, horizon, 6)

   def test_predict_states_zero_control(self, solver: GlobalMPCSolver):
      """Test _predict_states with zero control maintains velocity trajectory."""
      num_drones = 1
      horizon = solver.horizon
      # Moving in x
      d1 = _make_drone("d1", np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u = np.zeros((num_drones, horizon, 3))

      predicted_states = solver._predict_states([d1], u)

      # Position should increase linearly with velocity
      dt = solver.dt
      for k in range(horizon):
         expected_x = (k + 1) * dt * 1.0
         assert predicted_states[0, k, 0] == pytest.approx(expected_x, abs=1e-6)

   def test_predict_states_with_acceleration(self, solver: GlobalMPCSolver):
      """Test _predict_states with constant acceleration."""
      num_drones = 1
      horizon = solver.horizon
      d1 = _make_drone("d1", np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u = np.ones((num_drones, horizon, 3)) * 1.0  # Constant acceleration

      predicted_states = solver._predict_states([d1], u)

      # Velocity should increase with each step
      assert predicted_states[0, -1, 3] > 0  # Final x velocity positive

   def test_predict_states_multiple_drones(self, solver: GlobalMPCSolver):
      """Test _predict_states with multiple drones."""
      num_drones = 3
      horizon = solver.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      d2 = _make_drone("d2", np.array([5.0, 0.0, 0.0, -1.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      d3 = _make_drone("d2", np.array([2.5, 5.0, 0.0, 0.0, -1.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u = np.zeros((num_drones, horizon, 3))

      predicted_states = solver._predict_states([d1, d2, d3], u)

      assert predicted_states.shape == (num_drones, horizon, 6)
      assert predicted_states[0, -1, 0] > d1.x[0]
      assert predicted_states[1, -1, 0] < d2.x[0]
      assert predicted_states[2, -1, 1] < d3.x[1]


class TestGlobalMPCSolverConstraints:
   """Tests for GlobalMPCSolver._constraints method."""

   def test_constraints_returns_array(self, solver: GlobalMPCSolver):
      """Test _constraints returns numpy array."""
      num_drones = 2
      horizon = solver.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      d2 = _make_drone("d2", np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u_flat = np.zeros(num_drones * horizon * 3)

      g = solver._constraints(
         u_flat,
         drones=[d1, d2],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      assert isinstance(g, np.ndarray)
      assert len(g) > 0

   def test_constraints_satisfied_when_far_apart(self, solver: GlobalMPCSolver):
      """Test constraints are satisfied (>=0) when drones are far apart."""
      num_drones = 2
      horizon = solver.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      d2 = _make_drone("d2", np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u_flat = np.zeros(num_drones * horizon * 3)

      g = solver._constraints(
         u_flat,
         drones=[d1, d2],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      assert np.all(g >= 0)

   def test_constraints_violated_when_overlapping(self, solver: GlobalMPCSolver):
      """Test constraints are violated (<0) when drones overlap."""
      num_drones = 2
      horizon = solver.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      d2 = _make_drone("d2", np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u_flat = np.zeros(num_drones * horizon * 3)

      g = solver._constraints(
         u_flat,
         drones=[d1, d2],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      assert np.any(g < 0)

   def test_constraints_with_obstacles(self, solver: GlobalMPCSolver):
      """Test constraints account for obstacles."""
      num_drones = 1
      horizon = solver.horizon
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u_flat = np.zeros(num_drones * horizon * 3)
      obstacles = [(np.array([0.5, 0.0, 0.0]), np.array([0.2, 0.2, 0.2]))]

      g = solver._constraints(
         u_flat,
         drones=[d1],
         obstacles=obstacles,
         room_min=None,
         room_max=None,
      )

      assert np.any(g < 0)

   def test_constraints_with_room_bounds(self, solver: GlobalMPCSolver):
      """Test constraints account for room bounds."""
      num_drones = 1
      horizon = solver.horizon
      d1 = _make_drone("d1", np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u_flat = np.zeros(num_drones * horizon * 3)
      room_min = np.array([0.0, 0.0, 0.0])
      room_max = np.array([10.0, 10.0, 10.0])

      g = solver._constraints(
         u_flat,
         drones=[d1],
         obstacles=[],
         room_min=room_min,
         room_max=room_max,
      )

      assert np.any(g < 0)

   def test_constraints_velocity_satisfied(self, solver: GlobalMPCSolver):
      """Test velocity constraints are satisfied when velocity is below v_max."""
      num_drones = 1
      horizon = solver.horizon
      # Drone with velocity (1, 1, 1) has magnitude sqrt(3) ~ 1.73 m/s, below v_max=5.0
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      u_flat = np.zeros(num_drones * horizon * 3)  # Zero acceleration maintains velocity

      g = solver._constraints(
         u_flat,
         drones=[d1],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      # All constraints should be satisfied (>= 0)
      assert np.all(g >= 0)

   def test_constraints_velocity_violated_when_exceeding_vmax(self, solver: GlobalMPCSolver):
      """Test velocity constraints report violations when velocity exceeds v_max (no physics clipping)."""
      num_drones = 1
      horizon = solver.horizon
      # Drone with velocity (3, 3, 3) has magnitude sqrt(27) ~ 5.2 m/s
      # With v_max=2.0 and no physics clipping, velocity constraints should be violated.
      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0]), np.zeros(3),
                        CentralMPCAgent(dt=solver.dt), v_max=2.0)
      u_flat = np.zeros(num_drones * horizon * 3)

      g = solver._constraints(
         u_flat,
         drones=[d1],
         obstacles=[],
         room_min=None,
         room_max=None,
      )

      # Velocity constraints should be violated since speed > v_max and step() no longer clips
      assert np.any(g < 0)


class TestGlobalMPCSolverConstraintIntegration:
   """Tests that constraint classes are properly integrated in the solver."""

   def test_obstacles_constraint_via_class(self, solver: GlobalMPCSolver):
      """Test obstacle constraints produce correct values via constraint classes."""
      from drone_sim.domain.constraints import ObstacleAvoidanceConstraints
      horizon = solver.horizon
      d1 = _make_drone("d1", np.zeros(6), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      pred_pos = np.zeros((horizon, 3))
      obstacles = [(np.array([5.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))]

      obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
      result = obstacle_avoidance.evaluate_single(d1, pred_pos, obstacles, np.array([]))

      assert len(result) == horizon
      assert all(v > 0 for v in result)

   def test_room_constraint_via_class(self, solver: GlobalMPCSolver):
      """Test room constraints produce correct per-face values via constraint classes."""
      from drone_sim.domain.constraints import RoomConstraints
      horizon = solver.horizon
      d1 = _make_drone("d1", np.zeros(6), np.zeros(3), CentralMPCAgent(dt=solver.dt))
      pred_pos = np.zeros((horizon, 3)) + 5.0
      room_min = np.array([0.0, 0.0, 0.0])
      room_max = np.array([10.0, 10.0, 10.0])

      room_constraints = RoomConstraints(horizon=horizon)
      result = room_constraints.evaluate_single(d1, pred_pos, room_max, room_min, np.array([]))

      assert len(result) == horizon * 6
      assert all(v >= 0 for v in result)


class TestGlobalMPCSolverSolve:
   """Tests for GlobalMPCSolver.solve() method."""

   def test_solve_returns_controls_and_sequences(self, solver: GlobalMPCSolver):
      """Verify solve() returns correct shape/keys for controls and sequences."""
      controller = CentralMPCAgent(dt=solver.dt, horizon=solver.horizon)

      controls, u_sequences = solver.solve(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0]),
      )

      assert isinstance(controls, dict)
      assert isinstance(u_sequences, dict)
      assert "d1" in controls
      assert "d1" in u_sequences
      assert controls["d1"].shape == (3,)
      assert u_sequences["d1"].shape == (solver.horizon, 3)

   def test_solve_single_drone(self, solver: GlobalMPCSolver):
      """Single drone optimization works and produces valid controls."""
      controller = CentralMPCAgent(dt=solver.dt, horizon=solver.horizon)

      controls, u_sequences = solver.solve(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0]),
      )

      assert "d1" in controls
      # Control should be finite
      assert np.all(np.isfinite(controls["d1"]))
      assert np.all(np.isfinite(u_sequences["d1"]))

   def test_solve_multiple_drones(self, solver: GlobalMPCSolver):
      """Multi-drone optimization works and produces controls for all drones."""
      controller1 = CentralMPCAgent(dt=solver.dt, horizon=solver.horizon)
      controller2 = CentralMPCAgent(dt=solver.dt, horizon=solver.horizon)

      controls, u_sequences = solver.solve(
         drones=[
            _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller1),
            _make_drone("d2", np.array([10.0, 10.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller2),
         ],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0]),
      )

      assert "d1" in controls
      assert "d2" in controls
      assert controls["d1"].shape == (3,)
      assert controls["d2"].shape == (3,)
      assert u_sequences["d1"].shape == (solver.horizon, 3)
      assert u_sequences["d2"].shape == (solver.horizon, 3)

   def test_solve_with_warm_start(self, solver: GlobalMPCSolver):
      """Passing u_prev works as warm-start for the solver."""
      controller = CentralMPCAgent(dt=solver.dt, horizon=solver.horizon)

      drones = [
         _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 5.0, 5.0]), controller),
      ]

      # First solve (no warm-start)
      controls1, u_sequences1 = solver.solve(
         drones=drones,
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0]),
      )

      # Second solve with warm-start from first
      controls2, u_sequences2 = solver.solve(
         drones=drones,
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0]),
         u_prev=u_sequences1,
      )

      assert "d1" in controls2
      assert np.all(np.isfinite(controls2["d1"]))

   def test_solve_empty_when_no_optimizable_drones(self, solver: GlobalMPCSolver):
      """Solve returns empty dicts when no drones have central_cost."""

      class DummyController:
         pass

      controls, u_sequences = solver.solve(
         drones=[
            _make_drone("d1", np.zeros(6), np.zeros(3), DummyController()),
         ],
         obstacles=[],
      )

      assert controls == {}
      assert u_sequences == {}


class TestGlobalMPCSolverAdaptive:
   """Tests for adaptive constraint passthrough in GlobalMPCSolver."""

   @pytest.fixture
   def solver(self) -> GlobalMPCSolver:
      return GlobalMPCSolver(dt=0.1, horizon=5)

   def test_constraints_with_adaptive_drone(self, solver: GlobalMPCSolver):
      """Adaptive drone pair produces tighter constraints than fixed drones at same positions with velocity."""
      controller = CentralMPCAgent(dt=solver.dt, horizon=solver.horizon)

      # Two adaptive drones with velocity toward each other (speed=3 > 2.19 threshold
      # where adaptive radius exceeds fixed safety_zone=1.0)
      d1_adaptive = _make_drone("d1", np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]),
                                controller, safety_zone=1.0, alpha=1.0)
      d2_adaptive = _make_drone("d2", np.array([8.0, 0.0, 0.0, -3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
                                controller, safety_zone=1.0, alpha=1.0)

      # Same positions/velocities but fixed drones (no alpha)
      d1_fixed = _make_drone("d1", np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]),
                             controller, safety_zone=1.0)
      d2_fixed = _make_drone("d2", np.array([8.0, 0.0, 0.0, -3.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),
                             controller, safety_zone=1.0)

      # Use zero control to maintain velocities
      u_flat = np.zeros(2 * solver.horizon * 3)

      g_adaptive = solver._constraints(u_flat, drones=[d1_adaptive, d2_adaptive], obstacles=[], room_min=None, room_max=None)
      g_fixed = solver._constraints(u_flat, drones=[d1_fixed, d2_fixed], obstacles=[], room_min=None, room_max=None)

      # Adaptive radii grow with velocity, so collision constraint margins should be smaller (tighter)
      # Both arrays have same structure; compare the collision constraint portion (first H values)
      assert g_adaptive[:solver.horizon].min() < g_fixed[:solver.horizon].min()

   def test_constraints_fixed_drones_unchanged(self, solver: GlobalMPCSolver):
      """Fixed drones produce identical constraint values regardless of pred_vel passing."""
      controller = CentralMPCAgent(dt=solver.dt, horizon=solver.horizon)

      d1 = _make_drone("d1", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), controller, safety_zone=1.0)
      d2 = _make_drone("d2", np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), controller, safety_zone=1.0)
      u_flat = np.zeros(2 * solver.horizon * 3)

      g = solver._constraints(u_flat, drones=[d1, d2], obstacles=[], room_min=None, room_max=None)

      # All constraints should be satisfied for far-apart stationary drones
      assert np.all(g >= 0)
      # Collision margin: dist=10, threshold=safety_zone_i + safety_zone_j = 2.0, margin=8.0
      assert g[0] == pytest.approx(8.0, abs=0.1)

   def test_solve_with_adaptive_drone(self, solver: GlobalMPCSolver):
      """Solver returns valid controls when one drone is adaptive."""
      controller = CentralMPCAgent(dt=solver.dt, horizon=solver.horizon)

      d1 = _make_drone("d1", np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 0.0, 5.0]),
                        controller, alpha=1.0)
      d2 = _make_drone("d2", np.array([10.0, 0.0, 5.0, 0.0, 0.0, 0.0]), np.array([5.0, 0.0, 5.0]),
                        controller)

      controls, u_sequences = solver.solve(
         drones=[d1, d2],
         obstacles=[],
         room_min=np.array([-10.0, -10.0, 0.0]),
         room_max=np.array([20.0, 20.0, 20.0]),
      )

      assert "d1" in controls
      assert "d2" in controls
      assert np.all(np.isfinite(controls["d1"]))
      assert np.all(np.isfinite(controls["d2"]))
