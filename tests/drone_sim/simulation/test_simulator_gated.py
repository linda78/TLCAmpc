"""Tests for the gated-drone class switch and coordinator compat warning.

Verifies that ``config.type`` drives Drone vs GatedDrone in
``Simulator.from_config``, and that a RuntimeWarning fires when a
GatedDrone runs without the IntersectionDMPCCoordinator.
"""

from __future__ import annotations

import warnings

import pytest

from drone_sim.domain.config import (
   ControllerSpec,
   DroneConfig,
   PhysicsSpec,
   RoomConfig,
   ScenarioConfig,
)
from drone_sim.domain.drone import Drone, GatedDrone
from drone_sim.simulation.simulator import Simulator
# Importing the intersection module registers its coordinator type.
import drone_sim.simulation  # noqa: F401


def _physics_spec() -> PhysicsSpec:
   return PhysicsSpec(
      id="default",
      type="linear_kinematics",
      params={"v_max": 3.0, "u_min": [-3.0, -3.0, -3.0], "u_max": [3.0, 3.0, 3.0]},
   )


def _make_config(
   types: list[str],
   coordinator_type: str = "dmpc_admm_intersection",
) -> ScenarioConfig:
   drones = []
   for i, t in enumerate(types, start=1):
      drones.append(
         DroneConfig(
            drone_id=f"d{i}",
            start=[float(i), 0.0, 0.0],
            target=[float(i), 5.0, 0.0],
            radius=0.2,
            safety_zone=0.5,
            type=t,  # type: ignore[arg-type]
            physics="default",
         )
      )
   return ScenarioConfig(
      dt=0.1,
      physics=_physics_spec(),
      controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
      coordinator=ControllerSpec(
         type=coordinator_type,
         params={"horizon": 4, "intersection_enabled": True,
                 "intersection_sphere_radius": 0.6,
                 "intersection_resolution_margin": 1.1}
                 if coordinator_type == "dmpc_admm_intersection"
                 else {"horizon": 4},
      ),
      drones=drones,
      room=RoomConfig(min=[-5, -5, -5], max=[5, 5, 5]),
   )


def test_default_config_creates_plain_drone():
   cfg = _make_config(["default"])
   sim = Simulator.from_config(cfg)
   assert type(sim.drones[0]) is Drone


def test_gated_config_creates_gated_drone():
   cfg = _make_config(["gated"])
   sim = Simulator.from_config(cfg)
   assert isinstance(sim.drones[0], GatedDrone)
   assert sim.drones[0].is_waiting is False


def test_mixed_config_creates_mixed_drones():
   cfg = _make_config(["default", "gated", "default", "gated"])
   sim = Simulator.from_config(cfg)
   assert type(sim.drones[0]) is Drone
   assert isinstance(sim.drones[1], GatedDrone)
   assert type(sim.drones[2]) is Drone
   assert isinstance(sim.drones[3], GatedDrone)


def test_missing_type_defaults_to_drone():
   """Backwards-compat: configs without ``type`` produce plain Drone."""
   cfg_dict = {
      "dt": 0.1,
      "physics": {"id": "default", "type": "linear_kinematics",
                  "params": {"v_max": 3.0, "u_min": [-3, -3, -3], "u_max": [3, 3, 3]}},
      "controller": {"type": "mpc_agent", "params": {"horizon": 4}},
      "coordinator": {"type": "dmpc_admm", "params": {"horizon": 4}},
      "drones": [{"drone_id": "d1", "start": [0, 0, 0], "target": [1, 0, 0]}],
      "room": {"min": [-5, -5, -5], "max": [5, 5, 5]},
   }
   cfg = ScenarioConfig(**cfg_dict)
   sim = Simulator.from_config(cfg)
   assert type(sim.drones[0]) is Drone


def test_gated_with_intersection_coordinator_no_warning():
   cfg = _make_config(["gated"], coordinator_type="dmpc_admm_intersection")
   with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      Simulator.from_config(cfg)
   compat = [w for w in caught if "IntersectionDMPCCoordinator" in str(w.message)]
   assert compat == []


def test_gated_with_dmpc_admm_emits_runtime_warning():
   cfg = _make_config(["gated"], coordinator_type="dmpc_admm")
   with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      Simulator.from_config(cfg)
   compat = [w for w in caught if issubclass(w.category, RuntimeWarning)
             and "IntersectionDMPCCoordinator" in str(w.message)]
   assert len(compat) == 1


def test_plain_drone_with_dmpc_admm_no_warning():
   cfg = _make_config(["default"], coordinator_type="dmpc_admm")
   with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      Simulator.from_config(cfg)
   compat = [w for w in caught if "IntersectionDMPCCoordinator" in str(w.message)]
   assert compat == []
