"""Shared pytest fixtures for drone_sim tests."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from drone_sim.api import app as app_module
from drone_sim.api.app import app
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.domain.config import (DroneConfig, ScenarioConfig, PhysicsSpec, ControllerSpec, ObstacleConfig, RoomConfig)
from drone_sim.domain.drone import Drone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics
from drone_sim.simulation.centralized.coordinator import CentralMPCGlobalCoordinator
from drone_sim.simulation.simulator import Simulator


@pytest.fixture
def client():
   """Create a test client and reset simulator state."""
   # Reset global simulator state before each test
   app_module._sim = None
   return TestClient(app)


@pytest.fixture
def valid_config():
   """Return a valid scenario configuration dict."""
   return {
      "dt": 0.1,
      "physics": {"id": "default", "type": "linear_kinematics", "params": {"v_max": 5.0, "u_min": [-3.0, -3.0, -3.0], "u_max": [3.0, 3.0, 3.0]}},
      "controller": {"type": "mpc_agent", "params": {"horizon": 5}},
      "coordinator": {"type": "mpc_central", "params": {"horizon": 5}},
      "drones": [
         {
            "drone_id": "d1",
            "start": [0.0, 0.0, 5.0],
            "waypoints": [],
            "target": [5.0, 5.0, 5.0],
            "radius": 0.2,
            "safety_zone": 1.0,
            "physics": "default",
         }
      ],
      "obstacles": [],
      "room": {"min": [-10.0, -10.0, 0.0], "max": [20.0, 20.0, 20.0]}
   }


@pytest.fixture
def configured_client(client, valid_config):
   """Return a test client with a loaded configuration."""
   client.post("/config", json=valid_config)
   return client


@pytest.fixture
def sample_state_vector() -> np.ndarray:
   """Return a 6D state vector [x, y, z, vx, vy, vz]."""
   return np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=float)


@pytest.fixture
def sample_control_vector() -> np.ndarray:
   """Return a 3D control vector [ax, ay, az]."""
   return np.array([0.5, -0.5, 0.1], dtype=float)


@pytest.fixture
def sample_position() -> np.ndarray:
   """Return a 3D position vector."""
   return np.array([1.0, 2.0, 3.0], dtype=float)


@pytest.fixture
def sample_velocity() -> np.ndarray:
   """Return a 3D velocity vector."""
   return np.array([0.1, 0.2, 0.3], dtype=float)


@pytest.fixture
def sample_physics(sample_dt: float) -> LinearKinematicsPhysics:
   """Return an initialized LinearKinematicsPhysics instance."""
   return LinearKinematicsPhysics(dt=sample_dt)


@pytest.fixture
def sample_dt() -> float:
   """Return a standard simulation timestep."""
   return 0.1


@pytest.fixture
def sample_controller(sample_dt: float) -> CentralMPCAgent:
   """Return an initialized CentralMPCAgent instance."""
   return CentralMPCAgent(dt=sample_dt, horizon=5)


@pytest.fixture
def sample_coordinator(sample_dt: float) -> CentralMPCGlobalCoordinator:
   """Return an initialized CentralMPCGlobalCoordinator instance."""
   return CentralMPCGlobalCoordinator(dt=sample_dt, horizon=5)


@pytest.fixture
def sample_route() -> Route:
   """Return a sample Route with waypoints and target."""
   return Route(
      start=np.array([0.0, 0.0, 0.0]),  # matches sample_drone.x[:3]
      waypoints=[np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])],
      target=np.array([3.0, 3.0, 3.0]),
      waypoint_radius=0.5
   )


@pytest.fixture
def sample_route_no_waypoints() -> Route:
   """Return a Route with no waypoints (direct to target)."""
   return Route(
      start=np.array([0.0, 0.0, 0.0]),  # matches sample_drone.x[:3]
      waypoints=[],
      target=np.array([5.0, 5.0, 5.0]),
      waypoint_radius=0.5
   )


@pytest.fixture
def sample_drone(sample_controller: CentralMPCAgent, sample_route: Route) -> Drone:
   """Return a sample Drone instance."""
   return Drone(
      drone_id="drone-1",
      radius=0.2,
      safety_zone=1.0,
      cons_stop=0.0,
      color="tab:blue",
      safety_color="tab:cyan",
      trace_color="tab:blue",
      controller=sample_controller,
      physics=LinearKinematicsPhysics(dt=0.1),
      x=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
      route=sample_route
   )


@pytest.fixture
def sample_drone_config() -> DroneConfig:
   """Return a valid DroneConfig for testing."""
   return DroneConfig(
      drone_id="test-drone-1",
      start=[0.0, 0.0, 5.0],
      waypoints=[[1.0, 1.0, 5.0], [2.0, 2.0, 5.0]],
      target=[5.0, 5.0, 5.0],
      radius=0.2,
      safety_zone=1.0,
      drone_color="tab:blue"
   )


@pytest.fixture
def sample_physics_spec() -> PhysicsSpec:
   """Return a valid PhysicsSpec."""
   return PhysicsSpec(type="linear_kinematics", params={})


@pytest.fixture
def sample_controller_spec() -> ControllerSpec:
   """Return a valid ControllerSpec."""
   return ControllerSpec(type="mpc_agent", params={"horizon": 5})


@pytest.fixture
def sample_coordinator_spec() -> ControllerSpec:
   """Return a valid coordinator ControllerSpec."""
   return ControllerSpec(type="mpc_central", params={"horizon": 5})


@pytest.fixture
def sample_obstacle_config() -> ObstacleConfig:
   """Return a valid ObstacleConfig with box geometry."""
   return ObstacleConfig(center=[2.5, 2.5, 2.5], half_extents=[0.5, 0.5, 0.5])


@pytest.fixture
def sample_room_config() -> RoomConfig:
   """Return a valid RoomConfig."""
   return RoomConfig(min=[-5.0, -5.0, 0.0], max=[10.0, 10.0, 10.0])


@pytest.fixture
def sample_scenario_config(
      sample_drone_config: DroneConfig,
      sample_physics_spec: PhysicsSpec,
      sample_controller_spec: ControllerSpec,
      sample_coordinator_spec: ControllerSpec,
      sample_room_config: RoomConfig
) -> ScenarioConfig:
   """Return a complete ScenarioConfig for testing."""
   drone2 = DroneConfig(
      drone_id="test-drone-2",
      start=[8.0, 8.0, 5.0],
      waypoints=[],
      target=[6.0, 6.0, 5.0],
      radius=0.2,
      safety_zone=1.0,
      drone_color="tab:orange"
   )
   return ScenarioConfig(
      dt=0.1,
      physics=sample_physics_spec,
      controller=sample_controller_spec,
      coordinator=sample_coordinator_spec,
      drones=[sample_drone_config, drone2],
      obstacles=[],
      room=sample_room_config
   )


@pytest.fixture
def sample_scenario_config_single_drone(
      sample_drone_config: DroneConfig,
      sample_physics_spec: PhysicsSpec,
      sample_controller_spec: ControllerSpec,
      sample_coordinator_spec: ControllerSpec,
      sample_room_config: RoomConfig
) -> ScenarioConfig:
   """Return a ScenarioConfig with a single drone."""
   return ScenarioConfig(
      dt=0.1,
      physics=sample_physics_spec,
      controller=sample_controller_spec,
      coordinator=sample_coordinator_spec,
      drones=[sample_drone_config],
      obstacles=[],
      room=sample_room_config
   )


@pytest.fixture
def sample_simulator(sample_scenario_config: ScenarioConfig) -> Simulator:
   """Return an initialized Simulator instance."""
   return Simulator.from_config(sample_scenario_config)


@pytest.fixture
def sample_simulator_single_drone(sample_scenario_config_single_drone: ScenarioConfig) -> Simulator:
   """Return a Simulator with a single drone."""
   return Simulator.from_config(sample_scenario_config_single_drone)


@pytest.fixture
def sample_obstacles() -> list[tuple[np.ndarray, np.ndarray]]:
   """Return a list of obstacles as (center, half_extents) tuples."""
   return [
      (np.array([5.0, 5.0, 5.0]), np.array([0.5, 0.5, 0.5])),
      (np.array([7.0, 3.0, 2.0]), np.array([0.3, 0.3, 0.3]))
   ]


@pytest.fixture
def sample_room_bounds() -> tuple[np.ndarray, np.ndarray]:
   """Return room bounds as (room_min, room_max)."""
   return (
      np.array([-10.0, -10.0, 0.0]),
      np.array([10.0, 10.0, 10.0])
   )
