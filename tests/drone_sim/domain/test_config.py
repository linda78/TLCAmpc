"""Tests for drone_sim.domain.config module.

Tests Pydantic model validation for:
- PhysicsConfig, ControllerSpec
- DroneConfig
- ObstacleConfig, RoomConfig
- ScenarioConfig
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drone_sim.domain.config import (PhysicsConfig, ControllerSpec, DroneConfig, ObstacleConfig, RoomConfig, ScenarioConfig)


class TestPhysicsConfig:
   """Tests for PhysicsConfig model."""

   def test_physics_config_valid(self):
      """Test PhysicsConfig creation with valid inputs."""
      cfg = PhysicsConfig(id="standard", type="linear_kinematics", params={"v_max": 5.0})
      assert cfg.id == "standard"
      assert cfg.type == "linear_kinematics"
      assert cfg.params == {"v_max": 5.0}

   def test_physics_config_default_params(self):
      """Test PhysicsConfig uses empty dict for params by default."""
      cfg = PhysicsConfig(id="default", type="linear_kinematics")
      assert cfg.id == "default"
      assert cfg.params == {}

   def test_physics_config_missing_id_raises(self):
      """Test PhysicsConfig raises ValidationError when id is missing."""
      with pytest.raises(ValidationError, match="id"):
         PhysicsConfig(type="linear_kinematics")  # type: ignore[call-arg]

   def test_physics_config_missing_type_raises(self):
      """Test PhysicsConfig raises ValidationError when type is missing."""
      with pytest.raises(ValidationError, match="type"):
         PhysicsConfig(id="standard")  # type: ignore[call-arg]

class TestControllerSpec:
   """Tests for ControllerSpec model."""

   def test_controller_spec_valid(self):
      """Test ControllerSpec creation with valid inputs."""
      spec = ControllerSpec(type="mpc_agent", params={"horizon": 10})
      assert spec.type == "mpc_agent"
      assert spec.params["horizon"] == 10

   def test_controller_spec_default_params(self):
      """Test ControllerSpec uses empty dict for params by default."""
      spec = ControllerSpec(type="pid")
      assert spec.params == {}

   def test_controller_spec_missing_type_raises(self):
      """Test ControllerSpec raises ValidationError when type is missing."""
      with pytest.raises(ValidationError, match="type"):
         ControllerSpec()  # type: ignore[call-arg]


class TestDroneConfig:
   """Tests for DroneConfig model."""

   def test_drone_config_valid_minimal(self):
      """Test DroneConfig with minimal required fields."""
      cfg = DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])
      assert cfg.drone_id == "d1"
      assert cfg.physics == "standard"
      assert cfg.start == [0.0, 0.0, 0.0]
      assert cfg.target == [5.0, 5.0, 5.0]
      assert cfg.waypoints == []
      assert cfg.radius == 0.2  # default
      assert cfg.safety_zone == 1.0  # default

   def test_drone_config_valid_full(self):
      """Test DroneConfig with all fields specified."""
      cfg = DroneConfig(
         drone_id="drone-alpha",
         physics="fast",
         start=[1.0, 2.0, 3.0],
         waypoints=[[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
         target=[10.0, 10.0, 10.0],
         controller=ControllerSpec(type="mpc_agent"),
         radius=0.3,
         safety_zone=1.5,
         cons_stop=0.1,
         drone_color="red",
         safety_color="orange",
         trace_color="blue"
      )
      assert cfg.drone_id == "drone-alpha"
      assert cfg.physics == "fast"
      assert len(cfg.waypoints) == 2
      assert cfg.radius == 0.3
      assert cfg.cons_stop == 0.1

   def test_drone_config_start_wrong_length_raises(self):
      """Test DroneConfig raises ValidationError for start with wrong length."""
      with pytest.raises(ValidationError, match="start"):
         DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0], target=[5.0, 5.0, 5.0])

   def test_drone_config_target_wrong_length_raises(self):
      """Test DroneConfig raises ValidationError for target with wrong length."""
      with pytest.raises(ValidationError, match="target"):
         DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0])

   def test_drone_config_missing_drone_id_raises(self):
      """Test DroneConfig raises ValidationError when drone_id is missing."""
      with pytest.raises(ValidationError, match="drone_id"):
         DroneConfig(physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])  # type: ignore[call-arg]

   def test_drone_config_missing_physics_raises(self):
      """Test DroneConfig raises ValidationError when physics is missing."""
      with pytest.raises(ValidationError, match="physics"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])  # type: ignore[call-arg]

   def test_drone_config_color_as_rgb_list(self):
      """Test DroneConfig accepts RGB list for colors."""
      cfg = DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], drone_color=[0.5, 0.2, 0.8])
      assert cfg.drone_color == [0.5, 0.2, 0.8]

   def test_drone_config_edge_case_zero_radius(self):
      """Test DroneConfig allows zero radius (edge case)."""
      cfg = DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], radius=0.0)
      assert cfg.radius == 0.0

   def test_drone_config_edge_case_empty_waypoints(self):
      """Test DroneConfig with explicit empty waypoints."""
      cfg = DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], waypoints=[], target=[5.0, 5.0, 5.0])
      assert cfg.waypoints == []


class TestObstacleConfig:
   """Tests for ObstacleConfig model."""

   def test_obstacle_config_valid(self):
      """Test ObstacleConfig creation with valid inputs."""
      obs = ObstacleConfig(center=[2.0, 3.0, 4.0], radius=0.5)
      assert obs.center == [2.0, 3.0, 4.0]
      assert obs.radius == 0.5

   def test_obstacle_config_center_wrong_length_raises(self):
      """Test ObstacleConfig raises ValidationError for center with wrong length."""
      with pytest.raises(ValidationError, match="center"):
         ObstacleConfig(center=[1.0, 2.0], radius=0.5)

   def test_obstacle_config_missing_radius_raises(self):
      """Test ObstacleConfig raises ValidationError when radius is missing."""
      with pytest.raises(ValidationError, match="radius"):
         ObstacleConfig(center=[1.0, 2.0, 3.0])  # type: ignore[call-arg]

   def test_obstacle_config_edge_case_zero_radius(self):
      """Test ObstacleConfig allows zero radius (point obstacle)."""
      obs = ObstacleConfig(center=[0.0, 0.0, 0.0], radius=0.0)
      assert obs.radius == 0.0


class TestRoomConfig:
   """Tests for RoomConfig model."""

   def test_room_config_valid(self):
      """Test RoomConfig creation with valid inputs."""
      room = RoomConfig(min=[-10.0, -10.0, 0.0], max=[10.0, 10.0, 10.0])
      assert room.min == [-10.0, -10.0, 0.0]
      assert room.max == [10.0, 10.0, 10.0]

   def test_room_config_min_wrong_length_raises(self):
      """Test RoomConfig raises ValidationError for min with wrong length."""
      with pytest.raises(ValidationError, match="min"):
         RoomConfig(min=[-10.0, -10.0], max=[10.0, 10.0, 10.0])

   def test_room_config_max_wrong_length_raises(self):
      """Test RoomConfig raises ValidationError for max with wrong length."""
      with pytest.raises(ValidationError, match="max"):
         RoomConfig(min=[-10.0, -10.0, 0.0], max=[10.0, 10.0])


class TestScenarioConfig:
   """Tests for ScenarioConfig model."""

   def test_scenario_config_valid_minimal(self):
      """Test ScenarioConfig with minimal required fields."""
      cfg = ScenarioConfig(
         physics=[PhysicsConfig(id="standard", type="linear_kinematics")],
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
      )
      assert cfg.dt == 0.1
      assert cfg.coordinator is None
      assert cfg.obstacles == []
      assert cfg.room is None

   def test_scenario_config_valid_full(self):
      """Test ScenarioConfig with all fields specified."""
      cfg = ScenarioConfig(
         dt=0.05,
         physics=[
            PhysicsConfig(id="standard", type="linear_kinematics"),
            PhysicsConfig(id="fast", type="linear_kinematics", params={"v_max": 10.0})
         ],
         controller=ControllerSpec(type="mpc_agent", params={"horizon": 10}),
         coordinator=ControllerSpec(type="mpc_central", params={"horizon": 5}),
         drones=[
            DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0]),
            DroneConfig(drone_id="d2", physics="fast", start=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
         ],
         obstacles=[ObstacleConfig(center=[2.5, 2.5, 2.5], radius=0.3)],
         room=RoomConfig(min=[-10.0, -10.0, 0.0], max=[10.0, 10.0, 10.0])
      )
      assert cfg.dt == 0.05
      assert len(cfg.drones) == 2
      assert len(cfg.physics) == 2
      assert len(cfg.obstacles) == 1
      assert cfg.room is not None

   def test_scenario_config_missing_physics_raises(self):
      """Test ScenarioConfig raises ValidationError when physics is missing."""
      with pytest.raises(ValidationError, match="physics"):
         ScenarioConfig(
            controller=ControllerSpec(type="mpc_agent"),
            drones=[DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )  # type: ignore[call-arg]

   def test_scenario_config_missing_controller_raises(self):
      """Test ScenarioConfig raises ValidationError when controller is missing."""
      with pytest.raises(ValidationError, match="controller"):
         ScenarioConfig(
            physics=[PhysicsConfig(id="standard", type="linear_kinematics")],
            drones=[DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )  # type: ignore[call-arg]

   def test_scenario_config_missing_drones_raises(self):
      """Test ScenarioConfig raises ValidationError when drones is missing."""
      with pytest.raises(ValidationError, match="drones"):
         ScenarioConfig(
            physics=[PhysicsConfig(id="standard", type="linear_kinematics")],
            controller=ControllerSpec(type="mpc_agent"),
         )  # type: ignore[call-arg]

   def test_scenario_config_edge_case_empty_drones_list(self):
      """Test ScenarioConfig allows empty drones list (edge case)."""
      cfg = ScenarioConfig(
         physics=[PhysicsConfig(id="standard", type="linear_kinematics")],
         controller=ControllerSpec(type="mpc_agent"),
         drones=[],
      )
      assert cfg.drones == []

   def test_scenario_config_edge_case_zero_dt(self):
      """Test ScenarioConfig allows zero dt (edge case, though impractical)."""
      cfg = ScenarioConfig(
         dt=0.0,
         physics=[PhysicsConfig(id="standard", type="linear_kinematics")],
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
      )
      assert cfg.dt == 0.0

   def test_scenario_config_many_drones(self):
      """Test ScenarioConfig with many drones."""
      drones = [DroneConfig(drone_id=f"d{i}", physics="standard", start=[float(i), 0.0, 0.0], target=[float(i), 10.0, 10.0]) for i in range(100)]
      cfg = ScenarioConfig(
         physics=[PhysicsConfig(id="standard", type="linear_kinematics")],
         controller=ControllerSpec(type="mpc_agent"),
         drones=drones,
      )
      assert len(cfg.drones) == 100

   def test_scenario_config_empty_physics_list_raises(self):
      """Test ScenarioConfig raises ValidationError when physics list is empty."""
      with pytest.raises(ValidationError, match="At least one physics configuration must be defined"):
         ScenarioConfig(
            physics=[],
            controller=ControllerSpec(type="mpc_agent"),
            drones=[DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )

   def test_scenario_config_invalid_physics_reference_raises(self):
      """Test ScenarioConfig raises ValidationError when drone references non-existent physics ID."""
      with pytest.raises(ValidationError, match="references unknown physics ID"):
         ScenarioConfig(
            physics=[PhysicsConfig(id="standard", type="linear_kinematics")],
            controller=ControllerSpec(type="mpc_agent"),
            drones=[DroneConfig(drone_id="d1", physics="nonexistent", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )

   def test_scenario_config_duplicate_physics_ids_raises(self):
      """Test ScenarioConfig raises ValidationError when physics IDs are duplicated."""
      with pytest.raises(ValidationError, match="Physics IDs must be unique"):
         ScenarioConfig(
            physics=[
               PhysicsConfig(id="standard", type="linear_kinematics"),
               PhysicsConfig(id="standard", type="linear_kinematics", params={"v_max": 10.0})
            ],
            controller=ControllerSpec(type="mpc_agent"),
            drones=[DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )

   def test_scenario_config_multiple_physics_configs(self):
      """Test ScenarioConfig with multiple physics configurations."""
      cfg = ScenarioConfig(
         physics=[
            PhysicsConfig(id="standard", type="linear_kinematics", params={"v_max": 5.0}),
            PhysicsConfig(id="fast", type="linear_kinematics", params={"v_max": 10.0}),
            PhysicsConfig(id="slow", type="linear_kinematics", params={"v_max": 2.0})
         ],
         controller=ControllerSpec(type="mpc_agent"),
         drones=[
            DroneConfig(drone_id="d1", physics="standard", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0]),
            DroneConfig(drone_id="d2", physics="fast", start=[1.0, 0.0, 0.0], target=[5.0, 5.0, 5.0]),
            DroneConfig(drone_id="d3", physics="slow", start=[2.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])
         ]
      )
      assert len(cfg.physics) == 3
      assert cfg.drones[0].physics == "standard"
      assert cfg.drones[1].physics == "fast"
      assert cfg.drones[2].physics == "slow"
