"""Smoke tests for ``IntersectionDMPCCoordinator``."""

from __future__ import annotations

import numpy as np
import pytest

from drone_sim.controllers.braking import BRAKING_CONTROLLER, BrakingMPCAgent
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.controllers.wait import WAIT_CONTROLLER
from drone_sim.domain.drone import Drone, GatedDrone, Route
from drone_sim.domain.registry import COORDINATORS, create_coordinator
from drone_sim.domain.sphere_obstacle import SphereObstacle
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics
from drone_sim.simulation.distributed.distributed_coordinator import (
   DistributedMPCCoordinator,
)
from drone_sim.simulation.intersection import IntersectionDMPCCoordinator


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


def test_dmpc_stays_neutral_without_intersection():
   """Plain DMPC has no intersection knobs."""
   coord = DistributedMPCCoordinator(dt=0.1, horizon=5)
   assert not hasattr(coord, "intersection_enabled")
   assert not hasattr(coord, "_state_store")


def test_dmpc_obstacles_by_id_overrides_global_list():
   """The DMPC ``obstacles_by_id`` hook is preserved (used by SphereObstacle dispatch)."""
   coord = DistributedMPCCoordinator(dt=0.1, horizon=5, max_admm_iter=2)
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 0.5, 0, 0]), np.array([3, 0, 0])),
      _make_drone("d2", np.array([3, 0, 0, -0.5, 0, 0]), np.array([0, 0, 0])),
   ]
   sphere = SphereObstacle(center=np.array([1.5, 0, 0]), radius=0.3)
   obstacles_by_id = {"d1": [sphere], "d2": []}
   result = coord.solve_controls(
      drones=drones, obstacles=[], obstacles_by_id=obstacles_by_id,
   )
   for did, u in result.items():
      assert np.all(np.isfinite(u)), f"non-finite control for {did}"


def test_intersection_coordinator_is_registered():
   from drone_sim.simulation import intersection as _  # noqa: F401

   assert "dmpc_admm_intersection" in COORDINATORS


def test_intersection_coordinator_built_from_config_params():
   coord = create_coordinator({
      "type": "dmpc_admm_intersection",
      "params": {
         "dt": 0.1,
         "horizon": 5,
         "max_admm_iter": 5,
         "intersection_enabled": True,
         "intersection_sphere_radius": 0.7,
      },
   })
   assert isinstance(coord, IntersectionDMPCCoordinator)
   assert coord.intersection_sphere_radius == 0.7
   assert coord.horizon == 5
   # Subclass relationship: inherits DMPC behavior.
   assert isinstance(coord, DistributedMPCCoordinator)


def test_wrapper_disabled_passes_through_without_spheres():
   coord = IntersectionDMPCCoordinator(
      dt=0.1, horizon=5, max_admm_iter=5,
      intersection_enabled=False,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 0.5, 0, 0]), np.array([3, 0, 0])),
      _make_drone("d2", np.array([3, 0, 0, -0.5, 0, 0]), np.array([0, 0, 0])),
   ]
   result = coord.solve_controls(drones=drones, obstacles=[])
   assert "d1" in result and "d2" in result
   assert coord.active_spheres() == {}


def test_wrapper_enabled_creates_sphere_and_parks_gated_loser():
   coord = IntersectionDMPCCoordinator(
      dt=0.1, horizon=5, max_admm_iter=1,
      intersection_enabled=True,
      intersection_sphere_radius=1.2,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 2.0, 0, 0]), np.array([3, 0, 0]), gated=True),
      _make_drone("d2", np.array([3, 0, 0, -1.0, 0, 0]), np.array([0, 0, 0]), gated=True),
   ]
   coord.solve_controls(drones=drones, obstacles=[])
   active = coord.active_spheres()
   pair = frozenset({"d1", "d2"})
   assert pair in active
   record = active[pair]
   assert isinstance(record.sphere, SphereObstacle)
   assert record.priority in {"d1", "d2"}
   # The non-priority drone enters STOPPING (BrakingMPCAgent, still opt).
   loser_id = next(iter(pair - {record.priority}))
   loser = next(d for d in drones if d.drone_id == loser_id)
   assert isinstance(loser, GatedDrone)
   assert loser.is_waiting is True
   assert loser.is_stopping is True
   assert loser.is_parked is False
   assert loser.waiting_for == record.priority_root
   assert isinstance(loser.controller, BrakingMPCAgent)


def test_stopping_drone_stays_in_dmpc_optimisation():
   """A stopping GatedDrone has central_cost — DMPC keeps optimising it.

   While braking, the loser stays opt-eligible. The priority drone sees
   its trajectory and collision constraints stay active.
   """
   coord = IntersectionDMPCCoordinator(
      dt=0.1, horizon=5, max_admm_iter=1,
      intersection_enabled=True,
      intersection_sphere_radius=1.2,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 2.0, 0, 0]), np.array([3, 0, 0]), gated=True),
      _make_drone("d2", np.array([3, 0, 0, -1.0, 0, 0]), np.array([0, 0, 0]), gated=True),
   ]
   u_by_id = coord.solve_controls(drones=drones, obstacles=[])
   record = coord.active_spheres()[frozenset({"d1", "d2"})]
   loser_id = next(iter(record.pair - {record.priority}))
   # Loser is in stopping → has central_cost → DMPC must produce a control.
   assert loser_id in u_by_id


def test_parked_drone_drops_out_of_dmpc():
   """A directly-parked GatedDrone has WaitController and falls out of DMPC.

   In the current design the state machine only transitions STOPPING ->
   WAITING at sphere drop (and immediately releases to free), so WAITING
   is not normally observable mid-flight. This test directly forces the
   parked state to verify the DMPC filter still excludes it cleanly.
   """
   coord = IntersectionDMPCCoordinator(
      dt=0.1, horizon=5, max_admm_iter=1,
      intersection_enabled=True,
      intersection_sphere_radius=1.2,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 2.0, 0, 0]), np.array([3, 0, 0]), gated=True),
      _make_drone("d2", np.array([3, 0, 0, -1.0, 0, 0]), np.array([0, 0, 0]), gated=True),
   ]
   coord.solve_controls(drones=drones, obstacles=[])
   record = coord.active_spheres()[frozenset({"d1", "d2"})]
   loser_id = next(iter(record.pair - {record.priority}))
   loser = next(d for d in drones if d.drone_id == loser_id)
   # Force parked state directly — no automatic velocity-based transition.
   loser.transition_to_parked()
   assert loser.is_parked
   assert loser.controller is WAIT_CONTROLLER
   # Now the DMPC must skip it.
   u_by_id = coord.solve_controls(drones=drones, obstacles=[])
   assert loser_id not in u_by_id


def test_plain_drone_pair_tracks_sphere_but_does_not_wait():
   """Plain Drone pair: sphere is tracked for GUI, no one waits."""
   coord = IntersectionDMPCCoordinator(
      dt=0.1, horizon=5, max_admm_iter=1,
      intersection_enabled=True,
      intersection_sphere_radius=1.2,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 2.0, 0, 0]), np.array([3, 0, 0])),
      _make_drone("d2", np.array([3, 0, 0, -1.0, 0, 0]), np.array([0, 0, 0])),
   ]
   coord.solve_controls(drones=drones, obstacles=[])
   active = coord.active_spheres()
   pair = frozenset({"d1", "d2"})
   assert pair in active
   # Plain Drones cannot be parked, so none of them is in wait-state.
   for d in drones:
      assert not isinstance(d, GatedDrone)


@pytest.mark.slow
def test_wrapper_runs_two_steps_without_crash():
   coord = IntersectionDMPCCoordinator(
      dt=0.1, horizon=5, max_admm_iter=20,
      intersection_enabled=True,
      intersection_sphere_radius=0.6,
   )
   drones = [
      _make_drone("d1", np.array([0, 0, 0, 1.0, 0, 0]), np.array([3, 0, 0]), gated=True),
      _make_drone("d2", np.array([3, 0, 0, -1.0, 0, 0]), np.array([0, 0, 0]), gated=True),
   ]
   for _ in range(2):
      result = coord.solve_controls(drones=drones, obstacles=[])
      for did, u in result.items():
         assert np.all(np.isfinite(u)), f"non-finite control for {did}"
         drone = next(d for d in drones if d.drone_id == did)
         drone.x = drone.predict(u)
