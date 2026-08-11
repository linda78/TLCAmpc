"""Tests for drone_sim.domain.config module.

Tests Pydantic model validation for:
- PhysicsSpec, ControllerSpec
- DroneConfig
- ObstacleConfig, RoomConfig
- ScenarioConfig
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from drone_sim.domain.config import (PhysicsSpec, ControllerSpec, DroneConfig, ObstacleConfig, RoomConfig, ScenarioConfig)
from drone_sim.domain.drone_model import DroneModel


class TestPhysicsSpec:
   """Tests for PhysicsSpec model."""

   def test_physics_spec_valid(self):
      """Test PhysicsSpec creation with valid inputs."""
      spec = PhysicsSpec(type="linear_kinematics", params={"dt": 0.1})
      assert spec.type == "linear_kinematics"
      assert spec.params == {"dt": 0.1}

   def test_physics_spec_default_params(self):
      """Test PhysicsSpec uses empty dict for params by default."""
      spec = PhysicsSpec(type="custom_physics")
      assert spec.type == "custom_physics"
      assert spec.params == {}

   def test_physics_spec_missing_type_raises(self):
      """Test PhysicsSpec raises ValidationError when type is missing."""
      with pytest.raises(ValidationError, match="type"):
         PhysicsSpec()  # type: ignore[call-arg]


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
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])
      assert cfg.drone_id == "d1"
      assert cfg.start == [0.0, 0.0, 0.0]
      assert cfg.target == [5.0, 5.0, 5.0]
      assert cfg.waypoints == []
      assert cfg.radius == 0.2  # default
      assert cfg.safety_zone == 1.0  # default

   def test_drone_config_valid_full(self):
      """Test DroneConfig with all fields specified."""
      cfg = DroneConfig(
         drone_id="drone-alpha",
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
      assert len(cfg.waypoints) == 2
      assert cfg.radius == 0.3
      assert cfg.cons_stop == 0.1

   def test_drone_config_start_wrong_length_raises(self):
      """Test DroneConfig raises ValidationError for start with wrong length."""
      with pytest.raises(ValidationError, match="start"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0], target=[5.0, 5.0, 5.0])

   def test_drone_config_target_wrong_length_raises(self):
      """Test DroneConfig raises ValidationError for target with wrong length."""
      with pytest.raises(ValidationError, match="target"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0])

   def test_drone_config_missing_drone_id_raises(self):
      """Test DroneConfig raises ValidationError when drone_id is missing."""
      with pytest.raises(ValidationError, match="drone_id"):
         DroneConfig(start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])  # type: ignore[call-arg]

   def test_drone_config_color_as_rgb_list(self):
      """Test DroneConfig accepts RGB list for colors."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], drone_color=[0.5, 0.2, 0.8])
      assert cfg.drone_color == [0.5, 0.2, 0.8]

   def test_drone_config_edge_case_zero_radius(self):
      """Test DroneConfig allows zero radius (edge case)."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], radius=0.0)
      assert cfg.radius == 0.0

   def test_drone_config_edge_case_empty_waypoints(self):
      """Test DroneConfig with explicit empty waypoints."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], waypoints=[], target=[5.0, 5.0, 5.0])
      assert cfg.waypoints == []

   def test_drone_config_default_alpha_is_none(self):
      """Test default config has alpha=None (fixed mode)."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])
      assert cfg.alpha is None

   def test_drone_config_with_alpha(self):
      """Test setting alpha makes config adaptive."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], alpha=0.5)
      assert cfg.alpha == 0.5

   def test_drone_config_alpha_must_be_positive(self):
      """Test alpha=0 or alpha=-1 raises validation error."""
      with pytest.raises(ValidationError, match="alpha must be positive"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], alpha=0.0)
      with pytest.raises(ValidationError, match="alpha must be positive"):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], alpha=-1.0)

   def test_drone_config_existing_configs_unchanged(self):
      """Test existing config without alpha works exactly as before."""
      cfg = DroneConfig(
         drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
         radius=0.3, safety_zone=1.5, cons_stop=0.1,
      )
      assert cfg.alpha is None
      assert cfg.radius == 0.3
      assert cfg.safety_zone == 1.5
      assert cfg.cons_stop == 0.1


class TestDroneConfigSafetyZoneMode:
   """Tests for DroneConfig.safety_zone_mode field (Phase 23)."""

   def test_default_is_fixed(self):
      """Default safety_zone_mode is 'fixed' — backward compatible."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])
      assert cfg.safety_zone_mode == "fixed"

   def test_accepts_adaptive_mode(self):
      """safety_zone_mode='adaptive' is valid."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
                        safety_zone_mode="adaptive")
      assert cfg.safety_zone_mode == "adaptive"

   def test_accepts_lstm_mode(self):
      """safety_zone_mode='lstm' is valid."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
                        safety_zone_mode="lstm")
      assert cfg.safety_zone_mode == "lstm"

   def test_invalid_mode_raises(self):
      """Unknown safety_zone_mode raises ValidationError."""
      with pytest.raises(ValidationError):
         DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
                     safety_zone_mode="unknown_mode")

   def test_existing_configs_unaffected(self):
      """Existing DroneConfig without safety_zone_mode still works (backward compat)."""
      cfg = DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0],
                        radius=0.3, safety_zone=1.5, cons_stop=0.1, alpha=0.5)
      assert cfg.safety_zone_mode == "fixed"  # default
      assert cfg.alpha == 0.5  # existing field unchanged


class TestObstacleConfig:
   """Tests for ObstacleConfig model."""

   def test_obstacle_config_valid(self):
      """Test ObstacleConfig creation with valid inputs."""
      obs = ObstacleConfig(center=[2.0, 3.0, 4.0], half_extents=[0.5, 0.5, 0.5])
      assert obs.center == [2.0, 3.0, 4.0]
      assert obs.half_extents == [0.5, 0.5, 0.5]

   def test_obstacle_config_center_wrong_length_raises(self):
      """Test ObstacleConfig raises ValidationError for center with wrong length."""
      with pytest.raises(ValidationError, match="center"):
         ObstacleConfig(center=[1.0, 2.0], half_extents=[0.5, 0.5, 0.5])

   def test_obstacle_config_missing_half_extents_raises(self):
      """Test ObstacleConfig raises ValidationError when half_extents is missing."""
      with pytest.raises(ValidationError):
         ObstacleConfig(center=[1.0, 2.0, 3.0])  # missing half_extents

   def test_obstacle_config_edge_case_zero_half_extents(self):
      """Test ObstacleConfig allows zero half_extents (point obstacle)."""
      obs = ObstacleConfig(center=[0.0, 0.0, 0.0], half_extents=[0.0, 0.0, 0.0])
      assert obs.half_extents == [0.0, 0.0, 0.0]


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
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
      )
      assert cfg.dt == 0.1
      assert cfg.coordinator is None
      assert cfg.obstacles == []
      assert cfg.room is None

   def test_scenario_config_valid_full(self):
      """Test ScenarioConfig with all fields specified."""
      cfg = ScenarioConfig(
         dt=0.05,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent", params={"horizon": 10}),
         coordinator=ControllerSpec(type="mpc_central", params={"horizon": 5}),
         drones=[
            DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0]),
            DroneConfig(drone_id="d2", start=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
         ],
         obstacles=[ObstacleConfig(center=[2.5, 2.5, 2.5], half_extents=[0.3, 0.3, 0.3])],
         room=RoomConfig(min=[-10.0, -10.0, 0.0], max=[10.0, 10.0, 10.0])
      )
      assert cfg.dt == 0.05
      assert len(cfg.drones) == 2
      assert len(cfg.obstacles) == 1
      assert cfg.room is not None

   def test_scenario_config_missing_physics_raises(self):
      """Test ScenarioConfig raises ValidationError when physics is missing."""
      with pytest.raises(ValidationError, match="physics"):
         ScenarioConfig(
            controller=ControllerSpec(type="mpc_agent"),
            drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )  # type: ignore[call-arg]

   def test_scenario_config_missing_controller_raises(self):
      """Test ScenarioConfig raises ValidationError when controller is missing."""
      with pytest.raises(ValidationError, match="controller"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
         )  # type: ignore[call-arg]

   def test_scenario_config_missing_drones_raises(self):
      """Test ScenarioConfig raises ValidationError when drones is missing."""
      with pytest.raises(ValidationError, match="drones"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent"),
         )  # type: ignore[call-arg]

   def test_scenario_config_edge_case_empty_drones_list(self):
      """Test ScenarioConfig allows empty drones list (edge case)."""
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[],
      )
      assert cfg.drones == []

   def test_scenario_config_edge_case_zero_dt(self):
      """Test ScenarioConfig allows zero dt (edge case, though impractical)."""
      cfg = ScenarioConfig(
         dt=0.0,
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
      )
      assert cfg.dt == 0.0

   def test_scenario_config_many_drones(self):
      """Test ScenarioConfig with many drones."""
      drones = [DroneConfig(drone_id=f"d{i}", start=[float(i), 0.0, 0.0], target=[float(i), 10.0, 10.0]) for i in range(100)]
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=drones,
      )
      assert len(cfg.drones) == 100

   def test_scenario_config_default_lstm_model_path_is_none(self):
      """Default lstm_model_path is None — backward compatible."""
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])]
      )
      assert cfg.lstm_model_path is None

   def test_scenario_config_accepts_lstm_model_path(self):
      """ScenarioConfig accepts a string lstm_model_path."""
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])],
         lstm_model_path="/path/to/model.pt"
      )
      assert cfg.lstm_model_path == "/path/to/model.pt"


class TestScenarioConfigBoF:
   """Tests for the BoF-related ScenarioConfig fields and validator."""

   def _drone(self):
      return DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])

   def test_defaults(self):
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[self._drone()],
      )
      assert cfg.bof_enabled is False
      assert cfg.bof_backend == "library"
      assert cfg.bof_url is None
      assert cfg.bof_has_velocity is False
      assert cfg.bof_history_size == 100
      assert cfg.bof_horizon == 50
      assert cfg.bof_growth_tau is None

   def test_library_backend_does_not_require_url(self):
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[self._drone()],
         bof_enabled=True,
         bof_backend="library",
      )
      assert cfg.bof_url is None

   def test_rest_backend_requires_url(self):
      with pytest.raises(ValidationError, match="bof_url"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent"),
            drones=[self._drone()],
            bof_enabled=True,
            bof_backend="rest",
         )

   def test_rest_backend_with_url_ok(self):
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[self._drone()],
         bof_enabled=True,
         bof_backend="rest",
         bof_url="http://localhost:5050",
      )
      assert cfg.bof_url == "http://localhost:5050"

   def test_invalid_backend_raises(self):
      with pytest.raises(ValidationError):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent"),
            drones=[self._drone()],
            bof_backend="grpc",  # not in Literal
         )  # type: ignore[arg-type]

   def test_history_size_must_be_positive(self):
      with pytest.raises(ValidationError, match="bof_history_size"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent"),
            drones=[self._drone()],
            bof_history_size=0,
         )

   def test_bof_horizon_must_be_positive(self):
      with pytest.raises(ValidationError, match="bof_horizon"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent"),
            drones=[self._drone()],
            bof_horizon=0,
         )

   def test_bof_growth_tau_accepts_positive(self):
      cfg = ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[self._drone()],
         bof_growth_tau=12.0,
      )
      assert cfg.bof_growth_tau == 12.0

   def test_bof_growth_tau_zero_or_negative_raises(self):
      with pytest.raises(ValidationError, match="bof_growth_tau"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent"),
            drones=[self._drone()],
            bof_growth_tau=0.0,
         )
      with pytest.raises(ValidationError, match="bof_growth_tau"):
         ScenarioConfig(
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent"),
            drones=[self._drone()],
            bof_growth_tau=-1.0,
         )


class TestScenarioConfigCamera:
   """Tests for the camera-related ScenarioConfig fields and validator."""

   def _drone(self):
      return DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])

   def _cfg(self, **kwargs):
      return ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=[self._drone()],
         **kwargs,
      )

   def test_defaults(self):
      cfg = self._cfg()
      assert cfg.camera_enabled is False
      assert cfg.camera_fov_deg == 90.0
      assert cfg.camera_range == 10.0
      assert cfg.camera_backend == "stub"
      assert cfg.camera_api_port == 5006
      assert cfg.camera_expose_truth is False
      assert cfg.camera_noise_sigma == 0.0
      assert cfg.camera_rate_steps == 1
      assert cfg.camera_render_images is False
      assert cfg.camera_async is True
      assert cfg.camera_feeds_dmpc is False

   def test_rest_backend_requires_render_images(self):
      # We serve rendered images for pickup; without them the detector has nothing to fetch.
      with pytest.raises(ValidationError, match="camera_render_images"):
         self._cfg(camera_enabled=True, camera_backend="rest")

   def test_rest_backend_with_render_images_ok(self):
      cfg = self._cfg(camera_enabled=True, camera_backend="rest", camera_render_images=True)
      assert cfg.camera_backend == "rest"
      assert cfg.camera_render_images is True

   def test_stub_backend_does_not_require_render_images(self):
      cfg = self._cfg(camera_enabled=True, camera_backend="stub")
      assert cfg.camera_render_images is False

   def test_api_port_bounds(self):
      with pytest.raises(ValidationError, match="camera_api_port"):
         self._cfg(camera_api_port=0)
      with pytest.raises(ValidationError, match="camera_api_port"):
         self._cfg(camera_api_port=65536)
      cfg = self._cfg(camera_api_port=8080)
      assert cfg.camera_api_port == 8080

   def test_expose_truth_opt_in(self):
      cfg = self._cfg(camera_enabled=True, camera_expose_truth=True)
      assert cfg.camera_expose_truth is True

   def test_invalid_backend_raises(self):
      with pytest.raises(ValidationError):
         self._cfg(camera_backend="grpc")  # type: ignore[arg-type]

   def test_fov_bounds(self):
      with pytest.raises(ValidationError, match="camera_fov_deg"):
         self._cfg(camera_fov_deg=0.0)
      with pytest.raises(ValidationError, match="camera_fov_deg"):
         self._cfg(camera_fov_deg=361.0)
      cfg = self._cfg(camera_fov_deg=360.0)
      assert cfg.camera_fov_deg == 360.0

   def test_range_must_be_positive(self):
      with pytest.raises(ValidationError, match="camera_range"):
         self._cfg(camera_range=0.0)
      with pytest.raises(ValidationError, match="camera_range"):
         self._cfg(camera_range=-1.0)

   def test_rate_steps_must_be_at_least_one(self):
      with pytest.raises(ValidationError, match="camera_rate_steps"):
         self._cfg(camera_rate_steps=0)

   def test_noise_sigma_must_be_nonnegative(self):
      with pytest.raises(ValidationError, match="camera_noise_sigma"):
         self._cfg(camera_noise_sigma=-0.1)
      cfg = self._cfg(camera_noise_sigma=0.0)
      assert cfg.camera_noise_sigma == 0.0

   def test_feeds_dmpc_requires_enabled(self):
      with pytest.raises(ValidationError, match="camera_feeds_dmpc"):
         self._cfg(camera_feeds_dmpc=True)
      cfg = self._cfg(camera_enabled=True, camera_feeds_dmpc=True)
      assert cfg.camera_feeds_dmpc is True

   def test_render_images_requires_enabled(self):
      with pytest.raises(ValidationError, match="camera_render_images"):
         self._cfg(camera_render_images=True)
      cfg = self._cfg(camera_enabled=True, camera_render_images=True)
      assert cfg.camera_render_images is True


class TestScenarioConfigDroneModel:
   """Tests for drone_model / drone_model_path: scenario-wide default plus per-drone override, resolved and checked at load time."""

   BUNDLED = "drone_costum_0_0_5.obj"
   OTHER_BUNDLED = "drone_costum_0_1.obj"

   def _cfg(self, drones, **kwargs):
      return ScenarioConfig(
         physics=PhysicsSpec(type="linear_kinematics"),
         controller=ControllerSpec(type="mpc_agent"),
         drones=drones,
         **kwargs,
      )

   def _drone(self, drone_id="d1", **kwargs):
      return DroneConfig(drone_id=drone_id, start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], **kwargs)

   def test_defaults_to_sphere_without_a_path(self):
      """Test every existing config keeps drawing spheres — nothing to configure, no file to find."""
      cfg = self._cfg([self._drone()])

      assert cfg.drone_model == "sphere"
      assert cfg.drone_model_path is None
      assert cfg.drones[0].drone_model is None
      assert cfg.drones[0].drone_model_path is None
      assert cfg.drone_model_for(cfg.drones[0]) == DroneModel(kind="sphere")

   def test_drone_without_override_inherits_the_scenario_default(self):
      """Test the homogeneous case: set the model once on the scenario, every drone picks it up."""
      cfg = self._cfg([self._drone("d1"), self._drone("d2")], drone_model="obj", drone_model_path=self.BUNDLED)

      models = [cfg.drone_model_for(d) for d in cfg.drones]

      assert [m.kind for m in models] == ["obj", "obj"]
      assert models[0].path == models[1].path
      assert models[0].path.name == self.BUNDLED

   def test_drone_override_wins_over_the_default(self):
      """Test the heterogeneous case: a drone naming its own mesh keeps it while the rest inherit."""
      cfg = self._cfg(
         [self._drone("inherits"), self._drone("overrides", drone_model_path=self.OTHER_BUNDLED)],
         drone_model="obj",
         drone_model_path=self.BUNDLED,
      )

      inherited, overridden = (cfg.drone_model_for(d) for d in cfg.drones)

      assert inherited.path.name == self.BUNDLED
      assert overridden.path.name == self.OTHER_BUNDLED

   def test_kind_and_path_resolve_independently(self):
      """Test a drone can opt out to a sphere inside an obj scenario without also having to unset the path."""
      cfg = self._cfg(
         [self._drone("mesh"), self._drone("ball", drone_model="sphere")],
         drone_model="obj",
         drone_model_path=self.BUNDLED,
      )

      mesh, ball = (cfg.drone_model_for(d) for d in cfg.drones)

      assert mesh.kind == "obj"
      assert ball.kind == "sphere"
      assert ball.path is None

   def test_drone_may_opt_in_to_obj_alone(self):
      """Test a single obj drone in an otherwise spherical scenario needs both fields on itself."""
      cfg = self._cfg([self._drone("ball"), self._drone("mesh", drone_model="obj", drone_model_path=self.BUNDLED)])

      ball, mesh = (cfg.drone_model_for(d) for d in cfg.drones)

      assert ball.kind == "sphere"
      assert mesh.kind == "obj"
      assert mesh.path.is_file()

   def test_missing_file_raises_at_load_time(self):
      """Test a bad path fails while parsing the config, not once per capture on the perception worker thread."""
      with pytest.raises(ValidationError, match="not found"):
         self._cfg([self._drone()], drone_model="obj", drone_model_path="does_not_exist.obj")

   def test_obj_without_path_raises_at_load_time(self):
      """Test selecting the mesh renderer without naming a mesh is rejected."""
      with pytest.raises(ValidationError, match="drone_model_path"):
         self._cfg([self._drone()], drone_model="obj")

   def test_error_names_the_offending_drone(self):
      """Test the message points at the drone to fix — with a heterogeneous fleet the scenario default may be perfectly fine."""
      with pytest.raises(ValidationError, match="broken"):
         self._cfg([self._drone("fine"), self._drone("broken", drone_model="obj", drone_model_path="nope.obj")])

   def test_unusable_scenario_default_is_tolerated_when_every_drone_overrides_it(self):
      """Test validation looks at the resolved per-drone result, not at the raw default in isolation."""
      cfg = self._cfg(
         [self._drone("ball", drone_model="sphere")],
         drone_model="obj",
         drone_model_path="does_not_exist.obj",
      )

      assert cfg.drone_model_for(cfg.drones[0]).kind == "sphere"

   def test_invalid_kind_raises(self):
      """Test the kind is a closed set on both levels."""
      with pytest.raises(ValidationError):
         self._cfg([self._drone()], drone_model="cylinder")  # type: ignore[arg-type]
      with pytest.raises(ValidationError):
         self._drone(drone_model="cylinder")  # type: ignore[arg-type]

   def test_absolute_path_is_accepted(self, tmp_path):
      """Test users can point at a model outside the package."""
      own = tmp_path / "my_drone.obj"
      own.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

      cfg = self._cfg([self._drone()], drone_model="obj", drone_model_path=str(own))

      assert cfg.drone_model_for(cfg.drones[0]).path == own
