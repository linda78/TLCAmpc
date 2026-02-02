"""Tests for drone_sim.domain.drone module.

Tests for:
- Route class: current_ref(), advance_if_reached()
- Drone class: position(), velocity()
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from drone_sim.domain.drone import Drone, Route
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


class TestRoute:
   """Tests for Route dataclass."""

   def test_current_ref_returns_first_waypoint(self, sample_route: Route):
      """Test current_ref returns first waypoint when idx=0."""
      ref = sample_route.current_ref()
      assert_array_almost_equal(ref, np.array([1.0, 1.0, 1.0]))

   def test_current_ref_returns_second_waypoint_after_advance(self, sample_route: Route):
      """Test current_ref returns next waypoint after advancing."""
      sample_route.idx = 1
      ref = sample_route.current_ref()
      assert_array_almost_equal(ref, np.array([2.0, 2.0, 2.0]))

   def test_current_ref_returns_target_when_waypoints_exhausted(self, sample_route: Route):
      """Test current_ref returns target when all waypoints visited."""
      sample_route.idx = len(sample_route.waypoints)
      ref = sample_route.current_ref()
      assert_array_almost_equal(ref, np.array([3.0, 3.0, 3.0]))

   def test_current_ref_empty_waypoints_returns_target(self, sample_route_no_waypoints: Route):
      """Test current_ref returns target when no waypoints exist."""
      ref = sample_route_no_waypoints.current_ref()
      assert_array_almost_equal(ref, np.array([5.0, 5.0, 5.0]))

   def test_current_ref_returns_target_when_idx_exceeds_waypoints(self, sample_route: Route):
      """Test current_ref returns target when idx > len(waypoints)."""
      sample_route.idx = 100  # Far beyond waypoints
      ref = sample_route.current_ref()
      assert_array_almost_equal(ref, sample_route.target)

   def test_advance_if_reached_advances_when_within_radius(self, sample_route: Route):
      """Test advance_if_reached increments idx when position is within waypoint_radius."""
      position = np.array([1.0, 1.0, 1.0])  # Exactly at first waypoint
      initial_idx = sample_route.idx
      sample_route.advance_if_reached(position)
      assert sample_route.idx == initial_idx + 1

   def test_advance_if_reached_advances_when_at_boundary(self, sample_route: Route):
      """Test advance_if_reached advances when exactly at waypoint_radius distance."""
      # waypoint_radius is 0.5, so position at distance 0.5 should still advance
      waypoint = sample_route.waypoints[0]
      # Move 0.5 units away in x direction
      position = waypoint + np.array([0.5, 0.0, 0.0])
      initial_idx = sample_route.idx
      sample_route.advance_if_reached(position)
      assert sample_route.idx == initial_idx + 1

   def test_advance_if_reached_does_not_advance_when_outside_radius(self, sample_route: Route):
      """Test advance_if_reached does not advance when position is outside radius."""
      position = np.array([0.0, 0.0, 0.0])  # Far from first waypoint
      initial_idx = sample_route.idx
      sample_route.advance_if_reached(position)
      assert sample_route.idx == initial_idx  # idx unchanged

   def test_advance_if_reached_does_nothing_when_all_waypoints_visited(self, sample_route: Route):
      """Test advance_if_reached does nothing when idx >= len(waypoints)."""
      sample_route.idx = len(sample_route.waypoints)
      position = sample_route.target.copy()  # At target
      initial_idx = sample_route.idx
      sample_route.advance_if_reached(position)
      assert sample_route.idx == initial_idx  # idx unchanged

   def test_advance_if_reached_does_nothing_for_empty_waypoints(self, sample_route_no_waypoints: Route):
      """Test advance_if_reached does nothing when waypoints list is empty."""
      position = sample_route_no_waypoints.target.copy()
      initial_idx = sample_route_no_waypoints.idx
      sample_route_no_waypoints.advance_if_reached(position)
      assert sample_route_no_waypoints.idx == initial_idx

   def test_advance_if_reached_edge_case_very_small_radius(self):
      """Test advance_if_reached with very small waypoint_radius."""
      route = Route(waypoints=[np.array([1.0, 1.0, 1.0])], target=np.array([5.0, 5.0, 5.0]), waypoint_radius=0.001)
      position = np.array([1.0, 1.0, 1.0005])
      route.advance_if_reached(position)
      assert route.idx == 1

   def test_advance_if_reached_edge_case_zero_radius(self):
      """Test advance_if_reached with zero waypoint_radius (must be exactly at point)."""
      route = Route(waypoints=[np.array([1.0, 1.0, 1.0])], target=np.array([5.0, 5.0, 5.0]), waypoint_radius=0.0)
      # Exactly at waypoint
      position = np.array([1.0, 1.0, 1.0])
      route.advance_if_reached(position)
      assert route.idx == 1

      # Reset and try slightly off
      route.idx = 0
      position = np.array([1.0001, 1.0, 1.0])
      route.advance_if_reached(position)
      assert route.idx == 0  # Should not advance

   def test_advance_if_reached_multiple_calls_advance_through_waypoints(self, sample_route: Route):
      """Test multiple advance_if_reached calls progress through all waypoints."""
      # Start at idx=0
      assert sample_route.idx == 0

      # Reach first waypoint
      sample_route.advance_if_reached(sample_route.waypoints[0])
      assert sample_route.idx == 1

      # Reach second waypoint
      sample_route.advance_if_reached(sample_route.waypoints[1])
      assert sample_route.idx == 2

      # At target - should not advance further
      sample_route.advance_if_reached(sample_route.target)
      assert sample_route.idx == 2  # No more waypoints to advance

   def test_route_single_waypoint(self):
      """Test Route with a single waypoint."""
      route = Route(waypoints=[np.array([2.0, 2.0, 2.0])], target=np.array([5.0, 5.0, 5.0]))
      assert route.current_ref()[0] == 2.0
      route.advance_if_reached(np.array([2.0, 2.0, 2.0]))
      assert route.idx == 1
      assert_array_almost_equal(route.current_ref(), route.target)

   def test_route_many_waypoints(self):
      """Test Route with many waypoints."""
      waypoints = [np.array([float(i), float(i), float(i)]) for i in range(100)]
      route = Route(waypoints=waypoints, target=np.array([100.0, 100.0, 100.0]))
      assert len(route.waypoints) == 100
      assert route.current_ref()[0] == 0.0

      # Advance through several waypoints
      for i in range(10):
         route.advance_if_reached(waypoints[i])
      assert route.idx == 10

   def test_destination_reached(self):
      """Test destination_reached."""
      route = Route(waypoints=[np.array([float(i), float(i), float(i)]) for i in range(5)],
                    target=np.array([5.0, 5.0, 5.0]))
      assert not route.target_reached(np.array([5.0, 5.0, 5.01]), thresh=1e-3)
      assert not route.target_reached(np.array([5.0, 5.0, 5.001]), thresh=1e-3)



class TestDrone:
   """Tests for Drone dataclass."""

   def test_position_returns(self, sample_drone: Drone):
      """Test position() returns the first 3 elements of state vector."""
      sample_drone.x = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
      pos = sample_drone.position()
      assert_array_almost_equal(pos, np.array([1.0, 2.0, 3.0]))
      pos[0] = 99.0
      assert sample_drone.x[0] == 99.0  # Original modified

   def test_velocity_returns(self, sample_drone: Drone):
      """Test velocity() returns the last 3 elements of state vector."""
      sample_drone.x = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
      vel = sample_drone.velocity()
      assert_array_almost_equal(vel, np.array([0.1, 0.2, 0.3]))
      vel[0] = 99.0
      assert sample_drone.x[3] == 99.0  # Original modified

   def test_pos_and_vel_zero_state(self, sample_drone: Drone):
      """Test velocity() with zero state."""
      sample_drone.x = np.zeros(6)
      # pos = 0
      pos = sample_drone.position()
      assert_array_equal(pos, np.zeros(3))
      # vel = 0
      vel = sample_drone.velocity()
      assert_array_equal(vel, np.zeros(3))

   def test_drone_attributes(self, sample_drone: Drone):
      """Test Drone dataclass attributes are accessible and cover full state."""
      assert sample_drone.drone_id == "drone-1"
      assert sample_drone.radius == 0.2
      assert sample_drone.safety_zone == 1.0
      assert sample_drone.cons_stop == 0.0
      assert sample_drone.color == "tab:blue"
      pos = sample_drone.position()
      vel = sample_drone.velocity()
      reconstructed = np.concatenate([pos, vel])
      assert_array_equal(reconstructed, sample_drone.x)

   def test_drone_edge_case_large_values(self, sample_controller: CentralMPCAgent, sample_route: Route, sample_physics: LinearKinematicsPhysics):
      """Test Drone with large coordinate values."""
      drone = Drone(
         drone_id="large-drone",
         radius=0.2,
         safety_zone=1.0,
         cons_stop=0.0,
         color="red",
         safety_color="orange",
         trace_color="blue",
         controller=sample_controller,
         x=np.array([1e6, 1e6, 1e6, 1e3, 1e3, 1e3], dtype=float),
         route=sample_route,
         kinematics=sample_physics
      )
      assert drone.position()[0] == 1e6
      assert drone.velocity()[0] == 1e3

   def test_drone_edge_case_negative_values(self, sample_controller: CentralMPCAgent, sample_route: Route, sample_physics: LinearKinematicsPhysics):
      """Test Drone with negative coordinate values."""
      drone = Drone(
         drone_id="negative-drone",
         radius=0.2,
         safety_zone=1.0,
         cons_stop=0.0,
         color="red",
         safety_color="orange",
         trace_color="blue",
         controller=sample_controller,
         x=np.array([-10.0, -20.0, -5.0, -1.0, -2.0, -0.5], dtype=float),
         route=sample_route,
         kinematics=sample_physics
      )
      assert_array_almost_equal(drone.position(), np.array([-10.0, -20.0, -5.0]))
      assert_array_almost_equal(drone.velocity(), np.array([-1.0, -2.0, -0.5]))
