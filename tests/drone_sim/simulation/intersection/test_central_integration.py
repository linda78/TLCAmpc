"""Smoke tests for ``IntersectionCentralCoordinator``.

Mirrors ``test_dmpc_integration.py`` for the central MPC substrate. The
sequencing logic itself is covered by ``test_intersection_state.py``; these
tests verify the central coordinator wires the shared mixin up correctly and
honors the GatedDrone wait-state exactly like the DMPC variant.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from drone_sim.controllers.braking import BrakingMPCAgent
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.domain.config import ScenarioConfig
from drone_sim.domain.drone import Drone, GatedDrone, Route
from drone_sim.domain.registry import COORDINATORS, create_coordinator
from drone_sim.domain.sphere_obstacle import SphereObstacle
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics
from drone_sim.simulation.centralized.coordinator import (
   CentralMPCGlobalCoordinator,
)
from drone_sim.simulation.intersection import (
   IntersectionCentralCoordinator,
   IntersectionMixin,
)
from drone_sim.simulation.simulator import Simulator

_CONFIG = (
   Path(__file__).resolve().parents[4]
   / "configs" / "franck_intersection" / "4drone_central_intersection.json"
)


def _make_drone(
   drone_id: str, x: np.ndarray, target: np.ndarray, dt: float = 0.1,
   *, gated: bool = False,
) -> Drone:
   physics = LinearKinematicsPhysics(dt=dt, v_max=3.0)
   controller = CentralMPCAgent(dt=dt, horizon=5)
   cls = GatedDrone if gated else Drone
   return cls(
      drone_id=drone_id,
      radius=0.1,
      safety_zone=0.5,
      cons_stop=0.0,
      color="tab:blue",
      safety_color="tab:cyan",
      trace_color="tab:blue",
      controller=controller,
      physics=physics,
      x=np.asarray(x, dtype=float).reshape(6),
      route=Route(waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
   )


def test_central_intersection_coordinator_is_registered():
   from drone_sim.simulation import intersection as _  # noqa: F401

   assert "mpc_central_intersection" in COORDINATORS


def test_central_intersection_built_from_config_params():
   coord = create_coordinator({
      "type": "mpc_central_intersection",
      "params": {
         "dt": 0.1,
         "horizon": 5,
         "max_iter": 50,
         "intersection_enabled": True,
         "intersection_sphere_radius": 0.7,
      },
   })
   assert isinstance(coord, IntersectionCentralCoordinator)
   assert coord.intersection_sphere_radius == 0.7
   assert coord.horizon == 5
   # Mixes the sequencing mixin onto the central MPC base.
   assert isinstance(coord, IntersectionMixin)
   assert isinstance(coord, CentralMPCGlobalCoordinator)


def test_central_wrapper_disabled_passes_through_without_spheres():
   coord = IntersectionCentralCoordinator(
      dt=0.1, horizon=5, max_iter=50,
      intersection_enabled=False,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 0.5, 0, 0]), np.array([3, 0, 0])),
      _make_drone("d2", np.array([3, 0, 0, -0.5, 0, 0]), np.array([0, 0, 0])),
   ]
   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      result = coord.solve_controls(drones=drones, obstacles=[])
   assert "d1" in result and "d2" in result
   assert coord.active_spheres() == {}


def test_central_wrapper_enabled_creates_sphere_and_parks_gated_loser():
   coord = IntersectionCentralCoordinator(
      dt=0.1, horizon=5, max_iter=50,
      intersection_enabled=True,
      intersection_sphere_radius=1.2,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 2.0, 0, 0]), np.array([3, 0, 0]), gated=True),
      _make_drone("d2", np.array([3, 0, 0, -1.0, 0, 0]), np.array([0, 0, 0]), gated=True),
   ]
   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      coord.solve_controls(drones=drones, obstacles=[])
   active = coord.active_spheres()
   pair = frozenset({"d1", "d2"})
   assert pair in active
   record = active[pair]
   assert isinstance(record.sphere, SphereObstacle)
   assert record.priority in {"d1", "d2"}
   loser_id = next(iter(pair - {record.priority}))
   loser = next(d for d in drones if d.drone_id == loser_id)
   assert isinstance(loser, GatedDrone)
   assert loser.is_waiting is True
   assert loser.is_stopping is True
   assert loser.is_parked is False
   assert loser.waiting_for == record.priority_root
   assert isinstance(loser.controller, BrakingMPCAgent)


def test_central_stopping_drone_stays_in_optimisation():
   """A stopping GatedDrone has central_cost — the central MPC keeps optimizing it.

   While braking, the loser stays opt-eligible so the priority drone still sees
   its trajectory and collision constraints. This is the realistic state during
   sequencing: the state store transitions a loser STOPPING -> WAITING -> free
   within a single store step (``_release_losers``), so a drone is never left in
   the WAITING state at solve time — it is always FREE or STOPPING, both of which
   carry central_cost.
   """
   coord = IntersectionCentralCoordinator(
      dt=0.1, horizon=5, max_iter=50,
      intersection_enabled=True,
      intersection_sphere_radius=1.2,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 2.0, 0, 0]), np.array([3, 0, 0]), gated=True),
      _make_drone("d2", np.array([3, 0, 0, -1.0, 0, 0]), np.array([0, 0, 0]), gated=True),
   ]
   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      u_by_id = coord.solve_controls(drones=drones, obstacles=[])
   record = coord.active_spheres()[frozenset({"d1", "d2"})]
   loser_id = next(iter(record.pair - {record.priority}))
   loser = next(d for d in drones if d.drone_id == loser_id)
   # Loser is STOPPING -> BrakingMPCAgent -> has central_cost -> stays optimized.
   assert isinstance(loser.controller, BrakingMPCAgent)
   assert loser_id in u_by_id


def test_central_plain_drone_pair_tracks_sphere_but_does_not_wait():
   coord = IntersectionCentralCoordinator(
      dt=0.1, horizon=5, max_iter=50,
      intersection_enabled=True,
      intersection_sphere_radius=1.2,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 2.0, 0, 0]), np.array([3, 0, 0])),
      _make_drone("d2", np.array([3, 0, 0, -1.0, 0, 0]), np.array([0, 0, 0])),
   ]
   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      coord.solve_controls(drones=drones, obstacles=[])
   active = coord.active_spheres()
   assert frozenset({"d1", "d2"}) in active
   for d in drones:
      assert not isinstance(d, GatedDrone)


def test_central_config_builds_without_gated_warning():
   """The 4-drone central config builds and does NOT trigger the gated-drone warning."""
   cfg = ScenarioConfig.model_validate(json.loads(_CONFIG.read_text()))
   with warnings.catch_warnings():
      warnings.simplefilter("error")  # any warning here fails the test
      sim = Simulator.from_config(cfg)
   assert isinstance(sim.coordinator, IntersectionCentralCoordinator)


@pytest.mark.slow
def test_central_config_runs_two_steps():
   cfg = ScenarioConfig.model_validate(json.loads(_CONFIG.read_text()))
   sim = Simulator.from_config(cfg)
   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      for _ in range(2):
         sim.step()
   assert sim.step_count == 2


def test_central_goal_params_reach_state_store():
   """Goal guard / tolerance knobs are forwarded from the coordinator to the store."""
   coord = IntersectionCentralCoordinator(
      dt=0.1, horizon=5, intersection_sphere_radius=2.0,
      intersection_goal_radius=0.7, intersection_goal_tolerance=0.05,
   )
   assert coord._state_store.goal_radius == 0.7
   assert coord._state_store.effective_goal_radius == 0.7
   assert coord._state_store.goal_tolerance == 0.05
   # Default: guard falls back to radius * margin.
   coord = IntersectionCentralCoordinator(dt=0.1, horizon=5, intersection_sphere_radius=2.0)
   assert coord._state_store.goal_radius is None
   assert coord._state_store.effective_goal_radius == pytest.approx(2.2)


@pytest.mark.slow
def test_central_passer_by_is_not_parked_behind_drone_at_goal():
   """Regression: a gated drone parked at its goal must not block a passer-by.

   drone-1 sits at its target; drone-2 flies past it within the detection
   threshold. drone-1 wins the TTC tie-break, so without the goal guard it
   got a sphere it never exits and drone-2 waited forever.
   """
   cfg = ScenarioConfig.model_validate({
      "dt": 0.1,
      "room": {"min": [-5, -5, -5], "max": [5, 5, 5]},
      "physics": {"id": "default", "type": "linear_kinematics",
                  "params": {"v_max": 3.0, "u_min": [-3, -3, -3], "u_max": [3, 3, 3]}},
      "comm_radius": None,
      "controller": {"type": "mpc_agent",
                     "params": {"horizon": 8, "q_pos": [3, 3, 3], "r_u": [0.1, 0.1, 0.1]}},
      "coordinator": {"type": "mpc_central_intersection",
                      "params": {"horizon": 8, "intersection_enabled": True,
                                 "intersection_sphere_radius": 2.0,
                                 "intersection_resolution_margin": 1.1}},
      "drones": [
         {"drone_id": "drone-1", "type": "gated", "start": [0, 0, 0], "target": [0, 0, 0],
          "radius": 0.2, "safety_zone": 0.6, "cons_stop": 0.0, "drone_color": "#00FFFF",
          "physics": "default", "safety_zone_mode": "fixed"},
         {"drone_id": "drone-2", "type": "gated", "start": [-3.5, 0.3, 0], "target": [3.5, 0.3, 0],
          "radius": 0.2, "safety_zone": 0.6, "cons_stop": 0.0, "drone_color": "#800080",
          "physics": "default", "safety_zone_mode": "fixed"},
      ],
   })
   sim = Simulator.from_config(cfg)
   _, d2 = sim.drones
   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      for _ in range(120):
         sim.step()
         assert not d2.is_waiting, "drone-2 was parked behind a drone that sits at its goal"
         assert sim.coordinator.active_spheres() == {}
         if d2.route.target_reached(d2.position(), thresh=0.1):
            break
   assert d2.route.target_reached(d2.position(), thresh=0.1)
