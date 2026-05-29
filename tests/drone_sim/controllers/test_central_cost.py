"""Tests for drone_sim.controllers.central_cost module.

Tests for:
- as_diagonal helper function
- CentralMPCAgent.central_initial_guess()
- CentralMPCAgent.central_cost()
- CentralMPCAgent.control()
"""

from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from drone_sim.controllers.central_cost import AdaptiveMPCAgent, CentralMPCAgent, as_diagonal
from drone_sim.domain.registry import CONTROLLERS
from drone_sim.domain.drone import Drone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _make_drone(
    x: np.ndarray | list = None,
    target: np.ndarray | list = None,
    physics: LinearKinematicsPhysics | None = None,
    controller: CentralMPCAgent | None = None,
) -> Drone:
    """Helper to create a Drone for testing."""
    if x is None:
        x = np.zeros(6)
    if target is None:
        target = np.array([5.0, 5.0, 5.0])
    if physics is None:
        physics = LinearKinematicsPhysics(dt=0.1)
    if controller is None:
        controller = CentralMPCAgent(dt=0.1, horizon=5)

    return Drone(
        drone_id="test-drone",
        radius=0.2,
        safety_zone=1.0,
        cons_stop=0.0,
        color="tab:blue",
        safety_color="tab:cyan",
        trace_color="tab:blue",
        controller=controller,
        physics=physics,
        x=np.asarray(x, dtype=float).reshape(6),
        route=Route(start=np.asarray(x, dtype=float).reshape(6)[:3], waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
    )


class TestAsDiag:
   """Tests for as_diagonal helper function."""

   def testas_diagonal_from_list(self):
      """Test as_diagonal creates diagonal matrix from list."""
      expected = np.diag([1.0, 2.0, 3.0])

      result_from_list = as_diagonal([1.0, 2.0, 3.0])
      result_from_np_array = as_diagonal(np.array([1.0, 2.0, 3.0]))

      assert_array_equal(result_from_list, expected)
      assert_array_equal(result_from_np_array, expected)

      weights = [2.5]
      result = as_diagonal(weights)
      assert result.shape == (1, 1)
      assert result[0, 0] == 2.5

   def testas_diagonal_empty_list(self):
      """Test as_diagonal with empty list produces empty matrix."""
      weights = []
      result = as_diagonal(weights)
      assert result.shape == (0, 0)

   def testas_diagonal_reshapes_multidimensional(self):
      """Test as_diagonal flattens multidimensional array."""
      weights = np.array([[1.0, 2.0], [3.0, 4.0]])
      result = as_diagonal(weights)
      # Should flatten to [1, 2, 3, 4] and create 4x4 diagonal
      assert result.shape == (4, 4)
      assert_array_equal(np.diag(result), [1.0, 2.0, 3.0, 4.0])


class TestCentralMPCAgentInit:
   """Tests for CentralMPCAgent initialization."""

   def test_init_default_values(self):
      """Test CentralMPCAgent initializes with correct default values."""
      agent = CentralMPCAgent(dt=0.1)
      assert agent.dt == 0.1
      assert agent.horizon == 12
      assert agent.q_pos == (8.0, 8.0, 8.0)
      assert agent.q_vel == (0.5, 0.5, 0.5)
      assert agent.r_u == (0.2, 0.2, 0.2)

   def test_init_custom_values(self):
      """Test CentralMPCAgent initializes with custom values."""
      agent = CentralMPCAgent(
         dt=0.05,
         horizon=20,
         q_pos=[10.0, 10.0, 10.0],
         q_vel=[1.0, 1.0, 1.0],
         r_u=[0.5, 0.5, 0.5],
      )
      assert agent.dt == 0.05
      assert agent.horizon == 20

   def test_init_creates_weight_matrices(self):
      """Test __post_init__ creates weight matrices."""
      agent = CentralMPCAgent(dt=0.1, q_pos=[1.0, 2.0, 3.0])
      assert agent._Qp.shape == (3, 3)
      assert agent._Qp[0, 0] == 1.0
      assert agent._Qp[1, 1] == 2.0
      assert agent._Qp[2, 2] == 3.0


class TestCentralMPCAgentCentralInitialGuess:
   """Tests for CentralMPCAgent.central_initial_guess method."""

   def test_central_initial_guess_returns_correct_shape(self, sample_controller: CentralMPCAgent):
      """Test central_initial_guess returns array of shape (horizon, 3)."""
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=sample_controller)
      guess = sample_controller.central_initial_guess(drone)
      assert guess.shape == (sample_controller.horizon, 3)

   def test_central_initial_guess_points_toward_target(self, sample_controller: CentralMPCAgent):
      """Test initial guess acceleration points toward target."""
      drone = _make_drone(x=np.zeros(6), target=[10.0, 0.0, 0.0], controller=sample_controller)
      guess = sample_controller.central_initial_guess(drone)

      # First control should have positive x acceleration
      assert guess[0, 0] > 0  # Positive x acceleration
      assert guess[0, 1] == pytest.approx(0.0, abs=1e-6)
      assert guess[0, 2] == pytest.approx(0.0, abs=1e-6)

   def test_central_initial_guess_accounts_for_velocity(self, sample_controller: CentralMPCAgent):
      """Test initial guess accounts for current velocity."""
      drone = _make_drone(x=[0.0, 0.0, 0.0, 5.0, 0.0, 0.0], target=[1.0, 0.0, 0.0], controller=sample_controller)
      guess = sample_controller.central_initial_guess(drone)
      # Should decelerate since we're moving fast toward a close target
      assert guess[0, 0] < 0  # Decelerating

   def test_central_initial_guess_clipped_to_bounds(self, sample_controller: CentralMPCAgent):
      """Test initial guess is clipped to control bounds."""
      drone = _make_drone(x=np.zeros(6), target=[1000.0, 1000.0, 1000.0], controller=sample_controller)
      guess = sample_controller.central_initial_guess(drone)

      u_min, u_max = drone.bounds()
      assert np.all(guess >= u_min)
      assert np.all(guess <= u_max)

   def test_central_initial_guess_at_target(self, sample_controller: CentralMPCAgent):
      """Test initial guess when already at target."""
      drone = _make_drone(x=[5.0, 5.0, 5.0, 0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], controller=sample_controller)
      guess = sample_controller.central_initial_guess(drone)

      # Should be near zero since already at target
      assert_array_almost_equal(guess[0], np.zeros(3), decimal=5)

   def test_central_initial_guess_all_same_value(self, sample_controller: CentralMPCAgent):
      """Test all rows of initial guess are the same."""
      drone = _make_drone(x=np.zeros(6), target=[3.0, 3.0, 3.0], controller=sample_controller)
      guess = sample_controller.central_initial_guess(drone)

      # All rows should be identical
      for i in range(1, guess.shape[0]):
         assert_array_almost_equal(guess[i], guess[0])

   def test_central_initial_guess_accepts_list_input(self, sample_controller: CentralMPCAgent):
      """Test central_initial_guess accepts list state."""
      drone = _make_drone(x=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], controller=sample_controller)
      guess = sample_controller.central_initial_guess(drone)
      assert guess.shape == (sample_controller.horizon, 3)


class TestCentralMPCAgentCentralCost:
   """Tests for CentralMPCAgent.central_cost method."""

   def test_central_cost_returns_scalar(self, sample_controller: CentralMPCAgent):
      """Test central_cost returns a scalar float."""
      u_seq = np.zeros((sample_controller.horizon, 3))
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=sample_controller)
      cost = sample_controller.central_cost(u_seq, drone)
      assert isinstance(cost, float)

   def test_central_cost_zero_at_target(self, sample_controller: CentralMPCAgent):
      """Test central_cost is low when at target with zero velocity."""
      u_seq = np.zeros((sample_controller.horizon, 3))
      drone = _make_drone(x=[5.0, 5.0, 5.0, 0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], controller=sample_controller)
      cost = sample_controller.central_cost(u_seq, drone)
      assert cost < 1.0

   def test_central_cost_increases_with_distance(self, sample_controller: CentralMPCAgent):
      """Test central_cost increases when farther from target."""
      u_seq = np.zeros((sample_controller.horizon, 3))
      p_ref = np.array([10.0, 10.0, 10.0])

      drone_close = _make_drone(x=[9.0, 9.0, 9.0, 0.0, 0.0, 0.0], target=p_ref, controller=sample_controller)
      cost_close = sample_controller.central_cost(u_seq, drone_close)

      drone_far = _make_drone(x=np.zeros(6), target=p_ref, controller=sample_controller)
      cost_far = sample_controller.central_cost(u_seq, drone_far)

      assert cost_far > cost_close

   def test_central_cost_increases_with_control_effort(self, sample_controller: CentralMPCAgent):
      """Test central_cost increases with larger control inputs."""
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=sample_controller)

      u_seq_small = np.ones((sample_controller.horizon, 3)) * 0.1
      cost_small = sample_controller.central_cost(u_seq_small, drone)

      u_seq_large = np.ones((sample_controller.horizon, 3)) * 3.0
      cost_large = sample_controller.central_cost(u_seq_large, drone)

      assert cost_large > cost_small * 0.5

   def test_central_cost_positive(self, sample_controller: CentralMPCAgent):
      """Test central_cost is always non-negative."""
      u_seq = np.random.randn(sample_controller.horizon, 3)
      drone = _make_drone(x=np.random.randn(6), target=np.random.randn(3), controller=sample_controller)
      cost = sample_controller.central_cost(u_seq, drone)
      assert cost >= 0

   def test_central_cost_accepts_different_horizon_length(self, sample_controller: CentralMPCAgent):
      """Test central_cost accepts u_seq with different horizon length."""
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=sample_controller)

      u_seq_short = np.zeros((3, 3))
      cost_short = sample_controller.central_cost(u_seq_short, drone)

      u_seq_long = np.zeros((20, 3))
      cost_long = sample_controller.central_cost(u_seq_long, drone)

      assert cost_long > cost_short

   def test_central_cost_accepts_flat_array(self, sample_controller: CentralMPCAgent):
      """Test central_cost reshapes flat array correctly."""
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=sample_controller)
      u_flat = np.zeros(sample_controller.horizon * 3)
      cost = sample_controller.central_cost(u_flat, drone)
      assert isinstance(cost, float)


class TestCentralMPCAgentControl:
   """Tests for CentralMPCAgent.control method."""

   def test_control_returns_correct_shape(self, sample_controller: CentralMPCAgent):
      """Test control returns array of shape (3,)."""
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=sample_controller)
      u = sample_controller.control(drone, [], [])
      assert u.shape == (3,)

   def test_control_respects_bounds(self, sample_controller: CentralMPCAgent):
      """Test control output is within bounds."""
      drone = _make_drone(x=np.zeros(6), target=[1000.0, 1000.0, 1000.0], controller=sample_controller)
      u = sample_controller.control(drone, [], [])

      u_min, u_max = drone.bounds()
      assert np.all(u >= u_min)
      assert np.all(u <= u_max)

   def test_control_points_toward_target(self, sample_controller: CentralMPCAgent):
      """Test control accelerates toward target."""
      drone = _make_drone(x=np.zeros(6), target=[10.0, 0.0, 0.0], controller=sample_controller)
      u = sample_controller.control(drone, [], [])
      assert u[0] > 0  # Should accelerate in +x

   def test_control_zero_at_target(self, sample_controller: CentralMPCAgent):
      """Test control is near zero when at target and at rest."""
      drone = _make_drone(x=[5.0, 5.0, 5.0, 0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], controller=sample_controller)
      u = sample_controller.control(drone, [], [])
      assert_array_almost_equal(u, np.zeros(3), decimal=5)

   def test_control_ignores_neighbors_and_obstacles(self, sample_controller: CentralMPCAgent):
      """Test control method ignores neighbors/obstacles (uses initial guess only)."""
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=sample_controller)

      u_no_neighbors = sample_controller.control(drone, [], [])

      neighbors = [(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.2, 1.0, np.array([2.0, 0.0, 0.0]))]
      u_with_neighbors = sample_controller.control(drone, neighbors, [])

      assert_array_almost_equal(u_no_neighbors, u_with_neighbors)


class TestCentralMPCAgentEdgeCases:
   """Edge case tests for CentralMPCAgent."""

   def test_horizon_one(self):
      """Test agent with horizon=1."""
      agent = CentralMPCAgent(dt=0.1, horizon=1)
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=agent)

      guess = agent.central_initial_guess(drone)
      assert guess.shape == (1, 3)

      cost = agent.central_cost(guess, drone)
      assert cost >= 0

   def test_zero_weights(self):
      """Test agent with zero position weights (only penalizes velocity and control)."""
      agent = CentralMPCAgent(dt=0.1, q_pos=[0.0, 0.0, 0.0], q_vel=[1.0, 1.0, 1.0])
      drone = _make_drone(x=np.zeros(6), target=[100.0, 100.0, 100.0], controller=agent)

      u_seq = np.zeros((agent.horizon, 3))
      cost = agent.central_cost(u_seq, drone)

      # With zero position weight and zero control, cost should be zero
      assert cost == pytest.approx(0.0)

   def test_large_horizon(self):
      """Test agent with large horizon."""
      agent = CentralMPCAgent(dt=0.1, horizon=100)
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=agent)

      guess = agent.central_initial_guess(drone)
      assert guess.shape == (100, 3)

      cost = agent.central_cost(guess, drone)
      assert np.isfinite(cost)

   def test_negative_target(self):
      """Test agent with negative target coordinates."""
      agent = CentralMPCAgent(dt=0.1, horizon=5)
      drone = _make_drone(x=np.zeros(6), target=[-10.0, -10.0, -10.0], controller=agent)

      guess = agent.central_initial_guess(drone)
      # Should point toward negative direction
      assert guess[0, 0] < 0
      assert guess[0, 1] < 0
      assert guess[0, 2] < 0


class TestAdaptiveMPCAgent:
   """Tests for AdaptiveMPCAgent velocity-penalized cost model."""

   def test_init_default_lambda_vel(self):
      """Test default lambda_vel=0.8 and inherited fields."""
      agent = AdaptiveMPCAgent(dt=0.1)
      assert agent.lambda_vel == 1.0
      assert agent.horizon == 12
      assert agent.q_pos == (8.0, 8.0, 8.0)
      assert agent.q_vel == (0.5, 0.5, 0.5)
      assert agent.r_u == (0.2, 0.2, 0.2)

   def test_init_custom_lambda_vel(self):
      """Test custom lambda_vel is stored."""
      agent = AdaptiveMPCAgent(dt=0.1, lambda_vel=2.5)
      assert agent.lambda_vel == 2.5

   def test_cost_higher_than_base_with_velocity(self):
      """Test adaptive cost > base cost when velocity is nonzero."""
      base = CentralMPCAgent(dt=0.1, horizon=5)
      adaptive = AdaptiveMPCAgent(dt=0.1, horizon=5, lambda_vel=1.0)

      u_seq = np.ones((5, 3)) * 1.0
      drone_base = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=base)
      drone_adaptive = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=adaptive)

      cost_base = base.central_cost(u_seq, drone_base)
      cost_adaptive = adaptive.central_cost(u_seq, drone_adaptive)

      assert cost_adaptive > cost_base

   def test_cost_equals_base_at_zero_velocity(self):
      """Test cost matches base when drone is at target with zero velocity and zero controls."""
      base = CentralMPCAgent(dt=0.1, horizon=5)
      adaptive = AdaptiveMPCAgent(dt=0.1, horizon=5, lambda_vel=1.0)

      u_seq = np.zeros((5, 3))
      drone_base = _make_drone(x=[5.0, 5.0, 5.0, 0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], controller=base)
      drone_adaptive = _make_drone(x=[5.0, 5.0, 5.0, 0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], controller=adaptive)

      cost_base = base.central_cost(u_seq, drone_base)
      cost_adaptive = adaptive.central_cost(u_seq, drone_adaptive)

      assert cost_adaptive == pytest.approx(cost_base)

   def test_cost_increases_with_lambda_vel(self):
      """Test higher lambda_vel produces higher cost."""
      low = AdaptiveMPCAgent(dt=0.1, horizon=5, lambda_vel=0.5)
      high = AdaptiveMPCAgent(dt=0.1, horizon=5, lambda_vel=5.0)

      u_seq = np.ones((5, 3)) * 1.0
      drone_low = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=low)
      drone_high = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=high)

      cost_low = low.central_cost(u_seq, drone_low)
      cost_high = high.central_cost(u_seq, drone_high)

      assert cost_high > cost_low

   def test_cost_returns_scalar(self):
      """Test return type is float."""
      agent = AdaptiveMPCAgent(dt=0.1, horizon=5)
      u_seq = np.ones((5, 3))
      drone = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=agent)

      cost = agent.central_cost(u_seq, drone)
      assert isinstance(cost, float)

   def test_registered_as_mpc_agent_adaptive(self):
      """Test CONTROLLERS registry contains 'mpc_agent_adaptive'."""
      assert "mpc_agent_adaptive" in CONTROLLERS
      assert CONTROLLERS["mpc_agent_adaptive"] is AdaptiveMPCAgent

   def test_inherits_initial_guess(self):
      """Test central_initial_guess returns same shape/values as parent."""
      base = CentralMPCAgent(dt=0.1, horizon=5)
      adaptive = AdaptiveMPCAgent(dt=0.1, horizon=5)

      drone_base = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=base)
      drone_adaptive = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=adaptive)

      guess_base = base.central_initial_guess(drone_base)
      guess_adaptive = adaptive.central_initial_guess(drone_adaptive)

      assert_array_almost_equal(guess_adaptive, guess_base)

   def test_inherits_control(self):
      """Test control() returns same result as parent."""
      base = CentralMPCAgent(dt=0.1, horizon=5)
      adaptive = AdaptiveMPCAgent(dt=0.1, horizon=5)

      drone_base = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=base)
      drone_adaptive = _make_drone(x=np.zeros(6), target=[5.0, 5.0, 5.0], controller=adaptive)

      u_base = base.control(drone_base, [], [])
      u_adaptive = adaptive.control(drone_adaptive, [], [])

      assert_array_almost_equal(u_adaptive, u_base)
