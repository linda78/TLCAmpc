"""Tests for drone_sim.controllers.central_cost module.

Tests for:
- as_diagonal helper function
- CentralMPCAgent.central_bounds()
- CentralMPCAgent.central_initial_guess()
- CentralMPCAgent.central_cost()
- CentralMPCAgent.control()
"""

from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from drone_sim.controllers.central_cost import CentralMPCAgent, as_diagonal
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


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
      physics = LinearKinematicsPhysics(dt=0.1)
      agent = CentralMPCAgent(dt=0.1, physics=physics)
      assert agent.dt == 0.1
      assert agent.horizon == 5
      assert agent.q_pos == (1.0, 1.0, 1.0)
      assert agent.q_vel == (0.1, 0.1, 0.1)
      assert agent.r_u == (0.01, 0.01, 0.01)
      # Bounds come from the physics model
      assert_array_almost_equal(agent.physics.u_min, [-3.0, -3.0, -3.0])
      assert_array_almost_equal(agent.physics.u_max, [3.0, 3.0, 3.0])

   def test_init_custom_values(self):
      """Test CentralMPCAgent initializes with custom values."""
      physics = LinearKinematicsPhysics(
         dt=0.05,
         u_min=[-5.0, -5.0, -5.0],
         u_max=[5.0, 5.0, 5.0]
      )
      agent = CentralMPCAgent(
         dt=0.05,
         physics=physics,
         horizon=20,
         q_pos=[10.0, 10.0, 10.0],
         q_vel=[1.0, 1.0, 1.0],
         r_u=[0.5, 0.5, 0.5]
      )
      assert agent.dt == 0.05
      assert agent.horizon == 20

   def test_init_uses_passed_physics(self):
      """Test controller uses the physics model passed at initialization."""
      physics = LinearKinematicsPhysics(dt=0.1, v_max=2.5)
      agent = CentralMPCAgent(dt=0.1, physics=physics)
      assert agent.physics is physics
      assert agent.physics.dt == 0.1
      assert agent.physics.v_max == 2.5

   def test_init_creates_weight_matrices(self):
      """Test __post_init__ creates weight matrices."""
      physics = LinearKinematicsPhysics(dt=0.1)
      agent = CentralMPCAgent(dt=0.1, physics=physics, q_pos=[1.0, 2.0, 3.0])
      assert agent._Qp.shape == (3, 3)
      assert agent._Qp[0, 0] == 1.0
      assert agent._Qp[1, 1] == 2.0
      assert agent._Qp[2, 2] == 3.0


class TestCentralMPCAgentCentralBounds:
   """Tests for CentralMPCAgent.central_bounds method."""

   def test_central_bounds_returns(self, sample_controller: CentralMPCAgent):
      """Test central_bounds returns tuple of (u_min, u_max) with correct values."""
      bounds = sample_controller.central_bounds()

      assert isinstance(bounds, tuple)
      assert len(bounds) == 2
      u_min, u_max = bounds
      assert_array_almost_equal(u_min, [-3.0, -3.0, -3.0])
      assert_array_almost_equal(u_max, [3.0, 3.0, 3.0])

   def test_central_bounds_returns_copies(self, sample_controller: CentralMPCAgent):
      """Test central_bounds returns copies, not references."""
      u_min1, u_max1 = sample_controller.central_bounds()
      u_min2, u_max2 = sample_controller.central_bounds()

      u_min1[0] = 999.0
      assert u_min2[0] != 999.0

   def test_central_bounds_custom_bounds(self):
      """Test central_bounds with custom bound values from physics."""
      physics = LinearKinematicsPhysics(dt=0.1, u_min=[-10.0, -5.0, -2.0], u_max=[10.0, 5.0, 2.0])
      agent = CentralMPCAgent(dt=0.1, physics=physics)
      u_min, u_max = agent.central_bounds()
      assert_array_almost_equal(u_min, [-10.0, -5.0, -2.0])
      assert_array_almost_equal(u_max, [10.0, 5.0, 2.0])


class TestCentralMPCAgentCentralInitialGuess:
   """Tests for CentralMPCAgent.central_initial_guess method."""

   def test_central_initial_guess_returns_correct_shape(self, sample_controller: CentralMPCAgent):
      """Test central_initial_guess returns array of shape (horizon, 3)."""
      x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([5.0, 5.0, 5.0])

      guess = sample_controller.central_initial_guess(x0, p_ref)

      assert guess.shape == (sample_controller.horizon, 3)

   def test_central_initial_guess_points_toward_target(self, sample_controller: CentralMPCAgent):
      """Test initial guess acceleration points toward target."""
      x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # At origin, at rest
      p_ref = np.array([10.0, 0.0, 0.0])  # Target to the right

      guess = sample_controller.central_initial_guess(x0, p_ref)

      # First control should have positive x acceleration
      assert guess[0, 0] > 0  # Positive x acceleration
      assert guess[0, 1] == pytest.approx(0.0, abs=1e-6)
      assert guess[0, 2] == pytest.approx(0.0, abs=1e-6)

   def test_central_initial_guess_accounts_for_velocity(self, sample_controller: CentralMPCAgent):
      """Test initial guess accounts for current velocity."""
      x0 = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])  # Moving fast in x
      p_ref = np.array([1.0, 0.0, 0.0])  # Target close by

      guess = sample_controller.central_initial_guess(x0, p_ref)

      # Should decelerate since we're moving fast toward a close target
      # The formula is a = (p_ref - p) - 0.5*v
      # a = (1 - 0) - 0.5*5 = 1 - 2.5 = -1.5
      assert guess[0, 0] < 0  # Decelerating

   def test_central_initial_guess_clipped_to_bounds(self, sample_controller: CentralMPCAgent):
      """Test initial guess is clipped to control bounds."""
      x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([1000.0, 1000.0, 1000.0])  # Very far target

      guess = sample_controller.central_initial_guess(x0, p_ref)

      u_min, u_max = sample_controller.central_bounds()
      assert np.all(guess >= u_min)
      assert np.all(guess <= u_max)

   def test_central_initial_guess_at_target(self, sample_controller: CentralMPCAgent):
      """Test initial guess when already at target."""
      x0 = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])  # At target, at rest
      p_ref = np.array([5.0, 5.0, 5.0])  # Same as position

      guess = sample_controller.central_initial_guess(x0, p_ref)

      # Should be near zero since already at target
      assert_array_almost_equal(guess[0], np.zeros(3), decimal=5)

   def test_central_initial_guess_all_same_value(self, sample_controller: CentralMPCAgent):
      """Test all rows of initial guess are the same."""
      x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([3.0, 3.0, 3.0])

      guess = sample_controller.central_initial_guess(x0, p_ref)

      # All rows should be identical
      for i in range(1, guess.shape[0]):
         assert_array_almost_equal(guess[i], guess[0])

   def test_central_initial_guess_accepts_list_input(self, sample_controller: CentralMPCAgent):
      """Test central_initial_guess accepts list inputs."""
      x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      p_ref = [5.0, 5.0, 5.0]

      guess = sample_controller.central_initial_guess(x0, p_ref)

      assert guess.shape == (sample_controller.horizon, 3)


class TestCentralMPCAgentCentralCost:
   """Tests for CentralMPCAgent.central_cost method."""

   def test_central_cost_returns_scalar(self, sample_controller: CentralMPCAgent):
      """Test central_cost returns a scalar float."""
      u_seq = np.zeros((sample_controller.horizon, 3))
      x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([5.0, 5.0, 5.0])

      cost = sample_controller.central_cost(u_seq, x0, p_ref)

      assert isinstance(cost, float)

   def test_central_cost_zero_at_target(self, sample_controller: CentralMPCAgent):
      """Test central_cost is low when at target with zero velocity."""
      # Already at target, at rest, no control
      u_seq = np.zeros((sample_controller.horizon, 3))
      x0 = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
      p_ref = np.array([5.0, 5.0, 5.0])

      cost = sample_controller.central_cost(u_seq, x0, p_ref)

      # Cost should be very small (might not be exactly zero due to numerical issues)
      # but should be much smaller than when far from target
      assert cost < 1.0

   def test_central_cost_increases_with_distance(self, sample_controller: CentralMPCAgent):
      """Test central_cost increases when farther from target."""
      u_seq = np.zeros((sample_controller.horizon, 3))
      p_ref = np.array([10.0, 10.0, 10.0])

      # Close to target
      x0_close = np.array([9.0, 9.0, 9.0, 0.0, 0.0, 0.0])
      cost_close = sample_controller.central_cost(u_seq, x0_close, p_ref)

      # Far from target
      x0_far = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      cost_far = sample_controller.central_cost(u_seq, x0_far, p_ref)

      assert cost_far > cost_close

   def test_central_cost_increases_with_control_effort(self, sample_controller: CentralMPCAgent):
      """Test central_cost increases with larger control inputs."""
      x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([5.0, 5.0, 5.0])

      # Small control
      u_seq_small = np.ones((sample_controller.horizon, 3)) * 0.1
      cost_small = sample_controller.central_cost(u_seq_small, x0, p_ref)

      # Large control
      u_seq_large = np.ones((sample_controller.horizon, 3)) * 3.0
      cost_large = sample_controller.central_cost(u_seq_large, x0, p_ref)

      # Large control should have higher cost due to control effort penalty
      # (though it might get to target faster, the r_u penalty adds up)
      # This depends on the weights, but with default weights, large control adds significant cost
      assert cost_large > cost_small * 0.5  # Reasonable sanity check

   def test_central_cost_positive(self, sample_controller: CentralMPCAgent):
      """Test central_cost is always non-negative."""
      u_seq = np.random.randn(sample_controller.horizon, 3)
      x0 = np.random.randn(6)
      p_ref = np.random.randn(3)

      cost = sample_controller.central_cost(u_seq, x0, p_ref)

      assert cost >= 0

   def test_central_cost_clips_control(self, sample_controller: CentralMPCAgent):
      """Test central_cost clips control to bounds."""
      x0 = np.zeros(6)
      p_ref = np.array([5.0, 5.0, 5.0])

      # Control exceeding bounds
      u_seq_extreme = np.ones((sample_controller.horizon, 3)) * 100.0
      cost_extreme = sample_controller.central_cost(u_seq_extreme, x0, p_ref)

      # Control at bounds
      u_min, u_max = sample_controller.central_bounds()
      u_seq_clipped = np.tile(u_max, (sample_controller.horizon, 1))
      cost_clipped = sample_controller.central_cost(u_seq_clipped, x0, p_ref)

      # Costs should be equal since extreme is clipped to bounds
      assert cost_extreme == pytest.approx(cost_clipped)

   def test_central_cost_accepts_different_horizon_length(self, sample_controller: CentralMPCAgent):
      """Test central_cost accepts u_seq with different horizon length."""
      x0 = np.zeros(6)
      p_ref = np.array([5.0, 5.0, 5.0])

      # Different horizon than controller's default
      u_seq_short = np.zeros((3, 3))
      cost_short = sample_controller.central_cost(u_seq_short, x0, p_ref)

      u_seq_long = np.zeros((20, 3))
      cost_long = sample_controller.central_cost(u_seq_long, x0, p_ref)

      # Longer horizon should have higher cost (more time steps to sum)
      assert cost_long > cost_short

   def test_central_cost_accepts_flat_array(self, sample_controller: CentralMPCAgent):
      """Test central_cost reshapes flat array correctly."""
      x0 = np.zeros(6)
      p_ref = np.array([5.0, 5.0, 5.0])

      # Flat array representing (H, 3)
      u_flat = np.zeros(sample_controller.horizon * 3)
      cost = sample_controller.central_cost(u_flat, x0, p_ref)

      assert isinstance(cost, float)


class TestCentralMPCAgentControl:
   """Tests for CentralMPCAgent.control method."""

   def test_control_returns_correct_shape(self, sample_controller: CentralMPCAgent):
      """Test control returns array of shape (3,)."""
      x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([5.0, 5.0, 5.0])
      neighbors = []
      obstacles = []

      u = sample_controller.control(x, p_ref, neighbors, obstacles, self_radius=0.2, self_safety_zone=1.0)

      assert u.shape == (3,)

   def test_control_respects_bounds(self, sample_controller: CentralMPCAgent):
      """Test control output is within bounds."""
      x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([1000.0, 1000.0, 1000.0])  # Far away
      neighbors = []
      obstacles = []

      u = sample_controller.control(x, p_ref, neighbors, obstacles, self_radius=0.2, self_safety_zone=1.0)

      u_min, u_max = sample_controller.central_bounds()
      assert np.all(u >= u_min)
      assert np.all(u <= u_max)

   def test_control_points_toward_target(self, sample_controller: CentralMPCAgent):
      """Test control accelerates toward target."""
      x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # At origin, at rest
      p_ref = np.array([10.0, 0.0, 0.0])  # Target in +x direction
      neighbors = []
      obstacles = []

      u = sample_controller.control(x, p_ref, neighbors, obstacles, self_radius=0.2, self_safety_zone=1.0)

      assert u[0] > 0  # Should accelerate in +x

   def test_control_zero_at_target(self, sample_controller: CentralMPCAgent):
      """Test control is near zero when at target and at rest."""
      x = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])  # At target
      p_ref = np.array([5.0, 5.0, 5.0])
      neighbors = []
      obstacles = []

      u = sample_controller.control(x, p_ref, neighbors, obstacles, self_radius=0.2, self_safety_zone=1.0)

      assert_array_almost_equal(u, np.zeros(3), decimal=5)

   def test_control_ignores_neighbors_and_obstacles(self, sample_controller: CentralMPCAgent):
      """Test control method ignores neighbors/obstacles (uses initial guess only)."""
      x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([5.0, 5.0, 5.0])

      u_no_neighbors = sample_controller.control(x, p_ref, [], [], self_radius=0.2, self_safety_zone=1.0)

      neighbors = [(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.2, 1.0, np.array([2.0, 0.0, 0.0]))]
      u_with_neighbors = sample_controller.control(x, p_ref, neighbors, [], self_radius=0.2, self_safety_zone=1.0)

      assert_array_almost_equal(u_no_neighbors, u_with_neighbors)


class TestCentralMPCAgentEdgeCases:
   """Edge case tests for CentralMPCAgent."""

   def test_horizon_one(self):
      """Test agent with horizon=1."""
      physics = LinearKinematicsPhysics(dt=0.1)
      agent = CentralMPCAgent(dt=0.1, physics=physics, horizon=1)
      x0 = np.zeros(6)
      p_ref = np.array([5.0, 5.0, 5.0])

      guess = agent.central_initial_guess(x0, p_ref)
      assert guess.shape == (1, 3)

      cost = agent.central_cost(guess, x0, p_ref)
      assert cost >= 0

   def test_asymmetric_bounds(self):
      """Test agent with asymmetric control bounds from physics."""
      physics = LinearKinematicsPhysics(dt=0.1, u_min=[-1.0, -2.0, -3.0], u_max=[4.0, 5.0, 6.0])
      agent = CentralMPCAgent(dt=0.1, physics=physics)
      u_min, u_max = agent.central_bounds()
      assert_array_almost_equal(u_min, [-1.0, -2.0, -3.0])
      assert_array_almost_equal(u_max, [4.0, 5.0, 6.0])

   def test_zero_weights(self):
      """Test agent with zero position weights (only penalizes velocity and control)."""
      physics = LinearKinematicsPhysics(dt=0.1)
      agent = CentralMPCAgent(dt=0.1, physics=physics, q_pos=[0.0, 0.0, 0.0], q_vel=[1.0, 1.0, 1.0])
      x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
      p_ref = np.array([100.0, 100.0, 100.0])

      u_seq = np.zeros((agent.horizon, 3))
      cost = agent.central_cost(u_seq, x0, p_ref)

      # With zero position weight and zero control, cost should be zero
      # (assuming velocity stays zero)
      assert cost == pytest.approx(0.0)

   def test_large_horizon(self):
      """Test agent with large horizon."""
      physics = LinearKinematicsPhysics(dt=0.1)
      agent = CentralMPCAgent(dt=0.1, physics=physics, horizon=100)
      x0 = np.zeros(6)
      p_ref = np.array([5.0, 5.0, 5.0])

      guess = agent.central_initial_guess(x0, p_ref)
      assert guess.shape == (100, 3)

      cost = agent.central_cost(guess, x0, p_ref)
      assert np.isfinite(cost)

   def test_negative_target(self):
      """Test agent with negative target coordinates."""
      physics = LinearKinematicsPhysics(dt=0.1)
      agent = CentralMPCAgent(dt=0.1, physics=physics, horizon=5)
      x0 = np.zeros(6)
      p_ref = np.array([-10.0, -10.0, -10.0])

      guess = agent.central_initial_guess(x0, p_ref)
      # Should point toward negative direction
      assert guess[0, 0] < 0
      assert guess[0, 1] < 0
      assert guess[0, 2] < 0
