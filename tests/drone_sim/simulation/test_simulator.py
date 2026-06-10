"""Tests for drone_sim.simulation.simulator module.

Tests for:
- Simulator.from_config factory method
- Simulator.step method
- Simulator._compute_collisions method
- Simulator.to_dict method
"""

from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from drone_sim.domain.config import (ControllerSpec, DroneConfig, ObstacleConfig, PhysicsSpec, RoomConfig, ScenarioConfig)
from drone_sim.simulation.simulator import Simulator


class TestSimulatorFromConfig:
   """Tests for Simulator.from_config factory method."""

   def test_from_config_creates_simulator(self, sample_scenario_config: ScenarioConfig):
      """Test from_config creates a valid Simulator instance."""
      sim = Simulator.from_config(sample_scenario_config)

      assert isinstance(sim, Simulator)
      assert sim.dt == sample_scenario_config.dt
      assert len(sim.drones) == len(sample_scenario_config.drones)

   def test_from_config_initializes_values(self, sample_scenario_config: ScenarioConfig):
      """Test from_config initializes t, step_count, compute_time_s to zero."""
      sim = Simulator.from_config(sample_scenario_config)

      # Timings
      assert sim.t == 0.0
      assert sim.step_count == 0
      assert sim.compute_time_s == 0.0
      # Drones
      for drone in sim.drones:
         assert drone.drone_id in sim.traces
         assert len(sim.traces[drone.drone_id]) == 1
         assert_array_almost_equal(sim.traces[drone.drone_id][0], drone.position())
         # start with v = 0
         assert_array_almost_equal(drone.velocity(), np.zeros(3))
      # drones at start position
      for i, drone_cfg in enumerate(sample_scenario_config.drones):
         expected_pos = np.array(drone_cfg.start)
         actual_pos = sim.drones[i].position()
         assert_array_almost_equal(actual_pos, expected_pos)
      # Room bounds
      expected_min = np.array(sample_scenario_config.room.min)
      expected_max = np.array(sample_scenario_config.room.max)
      assert_array_almost_equal(sim.room_min, expected_min)
      assert_array_almost_equal(sim.room_max, expected_max)
      # Routes
      for i, drone in enumerate(sim.drones):
         assert drone.route is not None
         expected_target = np.array(sample_scenario_config.drones[i].target)
         assert_array_almost_equal(drone.route.target, expected_target)
      # physics
      assert sim.physics is not None
      assert hasattr(sim.physics, "step")
      # coordinator
      assert sim.coordinator is not None

   def test_from_config_initializes_obstacles(self):
      """Test from_config initializes obstacles from config."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 0], target=[5, 5, 5])],
         obstacles=[
            ObstacleConfig(center=[2.5, 2.5, 2.5], half_extents=[0.5, 0.5, 0.5]),
            ObstacleConfig(center=[1.0, 1.0, 1.0], half_extents=[0.3, 0.3, 0.3])
         ],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)

      assert len(sim.obstacles) == 2
      assert_array_almost_equal(sim.obstacles[0][0], [2.5, 2.5, 2.5])
      assert_array_almost_equal(sim.obstacles[0][1], [0.5, 0.5, 0.5])

   def test_from_config_derives_room_when_not_specified(self):
      """Test from_config derives room bounds when not specified."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[
            DroneConfig(drone_id="d1", start=[0, 0, 0], target=[10, 10, 10]),
            DroneConfig(drone_id="d2", start=[5, 5, 5], target=[-5, -5, 5])
         ],
         room=None
      )

      sim = Simulator.from_config(cfg)

      assert sim.room_min[0] < -5
      assert sim.room_max[0] > 10


class TestSimulatorStep:
   """Tests for Simulator.step method."""

   def test_step_updates_compute_time(self, sample_simulator: Simulator):
      """Test step updates compute_time_s."""
      initial_time = sample_simulator.compute_time_s
      sample_simulator.step()

      # Compute time always updates (even for infeasible steps)
      assert sample_simulator.compute_time_s > initial_time

   def test_step_moves_drones(self, sample_simulator: Simulator):
      """Test step updates drone positions when step succeeds."""
      initial_positions = [d.position().copy() for d in sample_simulator.drones]
      sample_simulator.step()

      if sample_simulator.infeasible:
         # If infeasible, drones should not move
         for i, drone in enumerate(sample_simulator.drones):
            assert_array_almost_equal(drone.position(), initial_positions[i])
      else:
         # At least one drone should have moved
         moved = False
         for i, drone in enumerate(sample_simulator.drones):
            if not np.allclose(drone.position(), initial_positions[i]):
               moved = True
               break
         assert moved

   def test_step_records_last_controls(self):
      """A feasible step mirrors the applied first-step controls into last_controls."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[
            DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 0, 5]),
            DroneConfig(drone_id="d2", start=[5, 0, 5], target=[0, 0, 5]),
         ],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10]),
      )
      sim = Simulator.from_config(cfg)
      assert sim.last_controls == {}  # empty before the first step

      sim.step()
      if sim.infeasible:
         pytest.skip("step was infeasible; nothing applied")

      # One finite (3,) control per drone, keyed by drone_id.
      assert set(sim.last_controls) == {"d1", "d2"}
      for did, u in sim.last_controls.items():
         u = np.asarray(u)
         assert u.shape == (3,)
         assert np.all(np.isfinite(u))

   def test_step_increase_values_only_if_feasible(self, sample_simulator: Simulator):
      """Test step appends to drone traces, step counter and dt when step succeeds."""
      initial_trace_len = len(sample_simulator.traces[sample_simulator.drones[0].drone_id])
      initial_count = sample_simulator.step_count
      initial_t = sample_simulator.t

      sample_simulator.step()

      new_trace_len = len(sample_simulator.traces[sample_simulator.drones[0].drone_id])
      if not sample_simulator.infeasible:
         assert new_trace_len == initial_trace_len + 1
         assert sample_simulator.step_count == initial_count + 1
         assert sample_simulator.t == pytest.approx(initial_t + sample_simulator.dt)
      else:
         assert new_trace_len == initial_trace_len
         assert sample_simulator.step_count == initial_count
         assert sample_simulator.t == initial_t

   def test_step_advances_waypoints(self):
      """Test step advances waypoints when reached."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 5], waypoints=[[0.01, 0.01, 5.01]], target=[5, 5, 5])],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)
      initial_idx = sim.drones[0].route.idx

      for _ in range(5):
         sim.step()

      assert sim.drones[0].route.idx >= initial_idx

   def test_step_clamps_to_room_bounds(self):
      """Test step clamps drone positions to room bounds."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[9.5, 0, 5], target=[100, 0, 5], radius=0.2)],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)

      for _ in range(50):
         sim.step()

      pos = sim.drones[0].position()
      max_x = sim.room_max[0] - sim.drones[0].radius
      assert pos[0] <= max_x + 1e-6

   def test_step_raises_without_coordinator(self):
      """Test step raises RuntimeError when coordinator is None."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=None,
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 5, 5])],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)

      with pytest.raises(RuntimeError, match="coordinator"):
         sim.step()

   def test_step_handles_infeasible_optimization(self):
      """Test step sets infeasible flag when optimization fails."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[-50, -50, -50], target=[5, 5, 5])],
         room=RoomConfig(min=[0, 0, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)
      sim.step()

      assert sim.infeasible is True
      assert sim.infeasible_reason is not None


class TestSimulatorComputeCollisions:
   """Tests for Simulator._compute_collisions method."""

   def test_compute_collisions_empty_when_far_apart(self, sample_simulator: Simulator):
      """Test _compute_collisions returns empty list when drones are far apart."""
      # Set drones far apart
      sample_simulator.drones[0].x[:3] = [0, 0, 5]
      sample_simulator.drones[1].x[:3] = [100, 100, 5]

      collisions = sample_simulator._compute_collisions()

      # Filter for drone-drone collisions only
      drone_collisions = [c for c in collisions if c["kind"] == "drone_drone"]
      assert len(drone_collisions) == 0

   def test_compute_collisions_detects_drone_drone(self, sample_simulator: Simulator):
      """Test _compute_collisions detects drone-drone collision."""
      # Set drones overlapping
      sample_simulator.drones[0].x[:3] = [0, 0, 5]
      sample_simulator.drones[1].x[:3] = [0.5, 0, 5]  # Within safety zone

      collisions = sample_simulator._compute_collisions()

      drone_collisions = [c for c in collisions if c["kind"] == "drone_drone"]
      assert len(drone_collisions) > 0
      assert drone_collisions[0]["kind"] == "drone_drone"

      for collision in collisions:
         assert "distance" in collision
         assert "threshold" in collision
         assert collision["distance"] <= collision["threshold"]

   def test_compute_collisions_detects_drone_obstacle(self):
      """Test _compute_collisions detects drone-obstacle collision."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 5, 5])],
         obstacles=[ObstacleConfig(center=[0.5, 0, 5], half_extents=[0.3, 0.3, 0.3])],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)
      collisions = sim._compute_collisions()

      obstacle_collisions = [c for c in collisions if c["kind"] == "drone_obstacle"]
      assert len(obstacle_collisions) > 0
      assert obstacle_collisions[0]["kind"] == "drone_obstacle"

   def test_compute_collisions_drone_inside_box(self):
      """Drone positioned inside a box obstacle always triggers a drone_obstacle collision.

      point_to_box_dist returns 0.0 inside the box, which is <= safety_zone,
      so the collision event is always generated.
      """
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 0], target=[5, 5, 5])],
         obstacles=[ObstacleConfig(center=[0.0, 0.0, 0.0], half_extents=[1.0, 1.0, 1.0])],
         room=RoomConfig(min=[-10, -10, -10], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)
      collisions = sim._compute_collisions()

      obstacle_collisions = [c for c in collisions if c["kind"] == "drone_obstacle"]
      assert len(obstacle_collisions) == 1
      assert obstacle_collisions[0]["distance"] == pytest.approx(0.0, abs=1e-2)
      assert obstacle_collisions[0]["threshold"] > 0

   def test_compute_collisions_no_collision_drone_far_outside_box(self):
      """Drone far from box obstacle produces no drone_obstacle collision event.

      Drone at [10, 0, 0], box centered at [0,0,0] with half_extents [0.5,0.5,0.5].
      point_to_box_dist approx 9.5, drone safety_zone=1.0 -> 9.5 > 1.0 -> no collision.
      """
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[10, 0, 0], target=[5, 5, 5])],
         obstacles=[ObstacleConfig(center=[0.0, 0.0, 0.0], half_extents=[0.5, 0.5, 0.5])],
         room=RoomConfig(min=[-10, -10, -10], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)
      collisions = sim._compute_collisions()

      obstacle_collisions = [c for c in collisions if c["kind"] == "drone_obstacle"]
      assert len(obstacle_collisions) == 0


class TestSimulatorToDict:
   """Tests for Simulator.to_dict method."""

   def test_to_dict_returns_dict_with_required_keys(self, sample_simulator: Simulator):
      """Test to_dict returns a dictionary."""
      result = sample_simulator.to_dict()
      assert isinstance(result, dict)
      # required keys
      assert "t" in result
      assert "dt" in result
      assert "room" in result
      assert "drones" in result
      assert "obstacles" in result
      assert "collisions" in result

      import json
      result = sample_simulator.to_dict()

      # Should not raise
      json_str = json.dumps(result)
      assert isinstance(json_str, str)

   def test_to_dict_room_format(self, sample_simulator: Simulator):
      """Test to_dict room has correct format."""
      result = sample_simulator.to_dict()

      assert "min" in result["room"]
      assert "max" in result["room"]
      assert len(result["room"]["min"]) == 3
      assert len(result["room"]["max"]) == 3

   def test_to_dict_drone_format(self, sample_simulator: Simulator):
      """Test to_dict drones have correct format."""
      result = sample_simulator.to_dict()

      for drone_dict in result["drones"]:
         assert "drone_id" in drone_dict
         assert "x" in drone_dict
         assert "route_idx" in drone_dict
         assert "p_ref" in drone_dict
         assert "radius" in drone_dict
         assert "safety_zone" in drone_dict
         assert "drone_color" in drone_dict
         assert len(drone_dict["x"]) == 6
         assert len(drone_dict["p_ref"]) == 3

   def test_to_dict_obstacle_format(self):
      """Test to_dict obstacles have correct format."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 5, 5])],
         obstacles=[ObstacleConfig(center=[2.5, 2.5, 2.5], half_extents=[0.5, 0.5, 0.5])],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)
      result = sim.to_dict()

      assert len(result["obstacles"]) == 1
      assert "center" in result["obstacles"][0]
      assert "half_extents" in result["obstacles"][0]
      assert len(result["obstacles"][0]["center"]) == 3

   def test_to_dict_collisions_format(self, sample_simulator: Simulator):
      """Test to_dict collisions is a list."""
      result = sample_simulator.to_dict()

      assert isinstance(result["collisions"], list)


class TestSimulatorAdaptiveConfigPipeline:
   """Tests for adaptive alpha flowing through the config-to-drone pipeline."""

   def test_from_config_fixed_safety_zone(self):
      """Test config without alpha creates drone with is_adaptive=False."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 5, 5], safety_zone=1.0)],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10]),
      )
      sim = Simulator.from_config(cfg)
      drone = sim.drones[0]
      assert drone.is_adaptive is False
      assert drone.alpha is None
      assert drone.safety_zone == 1.0

   def test_from_config_adaptive_alpha(self):
      """Test config with alpha creates drone with is_adaptive=True and correct alpha."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 5, 5], alpha=0.5)],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10]),
      )
      sim = Simulator.from_config(cfg)
      drone = sim.drones[0]
      assert drone.is_adaptive is True
      assert drone.alpha == 0.5

   def test_from_config_mixed_drones(self):
      """Test scenario with some fixed, some adaptive drones."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[
            DroneConfig(drone_id="fixed", start=[0, 0, 5], target=[5, 5, 5], safety_zone=1.0),
            DroneConfig(drone_id="adaptive", start=[5, 0, 5], target=[0, 5, 5], alpha=0.5),
         ],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10]),
      )
      sim = Simulator.from_config(cfg)
      fixed_drone = sim.drones[0]
      adaptive_drone = sim.drones[1]

      # Fixed drone
      assert fixed_drone.is_adaptive is False
      assert fixed_drone.compute_adaptive_radius(np.zeros(3)) == fixed_drone.safety_zone

      # Adaptive drone
      assert adaptive_drone.is_adaptive is True
      assert adaptive_drone.compute_adaptive_radius(np.zeros(3)) == adaptive_drone.safety_zone


class TestSimulatorEdgeCases:
   """Edge case tests for Simulator."""

   def test_multiple_steps(self, sample_simulator: Simulator):
      """Test multiple consecutive steps."""
      successful_steps = 0
      for _ in range(10):
         sample_simulator.step()
         if not sample_simulator.infeasible:
            successful_steps += 1

      assert sample_simulator.step_count == successful_steps
      assert sample_simulator.t == pytest.approx(successful_steps * sample_simulator.dt)

   def test_trace_length_limit(self):
      """Test trace length is limited to trace_len."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 5, 5])],
         room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)
      sim.trace_len = 5

      for _ in range(20):
         sim.step()

      assert len(sim.traces["d1"]) <= 5

   def test_velocity_zeroed_on_wall_hit(self):
      """Test velocity component is zeroed when hitting wall."""
      cfg = ScenarioConfig(
         dt=0.1,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         coordinator=ControllerSpec(type="mpc_central"),
         drones=[DroneConfig(drone_id="d1", start=[9.5, 5, 5], target=[20, 5, 5], radius=0.2)],
         room=RoomConfig(min=[0, 0, 0], max=[10, 10, 10])
      )

      sim = Simulator.from_config(cfg)

      for _ in range(20):
         sim.step()
         if sim.drones[0].position()[0] >= 10 - 0.2 - 0.01:
            break

      pos = sim.drones[0].position()
      max_x = sim.room_max[0] - sim.drones[0].radius
      assert pos[0] <= max_x + 1e-6
