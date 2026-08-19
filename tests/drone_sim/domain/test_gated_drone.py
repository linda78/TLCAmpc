"""Tests for GatedDrone wait-state and module helper has_waiters."""

from __future__ import annotations

import numpy as np

from drone_sim.controllers.braking import BRAKING_CONTROLLER, BrakingMPCAgent
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.controllers.wait import WAIT_CONTROLLER
from drone_sim.domain.drone import Drone, GatedDrone, Route, has_central_cost, has_waiters
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _make_physics(dt: float = 0.1) -> LinearKinematicsPhysics:
   return LinearKinematicsPhysics(
      dt=dt, v_max=3.0, u_min=[-3.0, -3.0, -3.0], u_max=[3.0, 3.0, 3.0]
   )


def _make_gated(drone_id: str = "a") -> GatedDrone:
   physics = _make_physics()
   return GatedDrone(
      drone_id=drone_id,
      radius=0.2,
      safety_zone=0.5,
      cons_stop=0.0,
      color="tab:blue",
      safety_color="tab:blue",
      trace_color="tab:blue",
      controller=CentralMPCAgent(dt=0.1, horizon=4),
      physics=physics,
      x=np.zeros(6),
      route=Route(waypoints=[], target=np.array([5.0, 0.0, 0.0])),
   )


def _make_drone(drone_id: str = "x") -> Drone:
   physics = _make_physics()
   return Drone(
      drone_id=drone_id,
      radius=0.2,
      safety_zone=0.5,
      cons_stop=0.0,
      color="tab:blue",
      safety_color="tab:blue",
      trace_color="tab:blue",
      controller=CentralMPCAgent(dt=0.1, horizon=4),
      physics=physics,
      x=np.zeros(6),
      route=Route(waypoints=[], target=np.array([5.0, 0.0, 0.0])),
   )


def test_default_not_waiting():
   d = _make_gated()
   assert d.waiting_for is None
   assert d.is_waiting is False
   assert d.is_stopping is False
   assert d.is_parked is False
   assert d.wait_state == "free"


def test_start_waiting_enters_stopping_state():
   d = _make_gated()
   original = d.controller
   d.start_waiting("b")
   assert d.waiting_for == "b"
   assert d.wait_state == "stopping"
   assert d.is_waiting is True
   assert d.is_stopping is True
   assert d.is_parked is False
   assert isinstance(d.controller, BrakingMPCAgent)
   # BrakingMPCAgent stays in opt_drones (DMPC brakes it with collision constraints).
   assert has_central_cost(d.controller) is True
   assert d._original_controller is original


def test_transition_to_parked_swaps_to_wait_controller():
   d = _make_gated()
   d.start_waiting("b")
   d.transition_to_parked()
   assert d.wait_state == "waiting"
   assert d.is_waiting is True
   assert d.is_stopping is False
   assert d.is_parked is True
   assert d.controller is WAIT_CONTROLLER
   # WaitController makes the drone non-opt for DMPC.
   assert has_central_cost(d.controller) is False
   # waiting_for is still set so priority logic knows who we're behind.
   assert d.waiting_for == "b"


def test_transition_to_parked_noop_when_not_stopping():
   d = _make_gated()
   # In free state.
   d.transition_to_parked()
   assert d.wait_state == "free"
   # After full park, calling again is a no-op.
   d.start_waiting("b")
   d.transition_to_parked()
   ctrl = d.controller
   d.transition_to_parked()
   assert d.controller is ctrl
   assert d.wait_state == "waiting"


def test_stop_waiting_from_stopping_restores_controller():
   d = _make_gated()
   original = d.controller
   d.start_waiting("b")
   d.stop_waiting()
   assert d.wait_state == "free"
   assert d.waiting_for is None
   assert d.controller is original
   assert d._original_controller is None


def test_stop_waiting_from_parked_restores_controller():
   d = _make_gated()
   original = d.controller
   d.start_waiting("b")
   d.transition_to_parked()
   d.stop_waiting()
   assert d.wait_state == "free"
   assert d.waiting_for is None
   assert d.controller is original


def test_start_waiting_same_target_is_noop():
   d = _make_gated()
   d.start_waiting("b")
   first_ctrl = d.controller
   d.start_waiting("b")
   assert d.waiting_for == "b"
   assert d.controller is first_ctrl
   assert d.wait_state == "stopping"


def test_start_waiting_other_target_is_noop_while_waiting():
   """First wait wins — a second start_waiting is a no-op."""
   d = _make_gated()
   d.start_waiting("b")
   d.start_waiting("c")
   assert d.waiting_for == "b"


def test_start_waiting_idempotent_in_parked_state():
   d = _make_gated()
   d.start_waiting("b")
   d.transition_to_parked()
   ctrl = d.controller
   d.start_waiting("c")  # already parked → no-op
   assert d.waiting_for == "b"
   assert d.controller is ctrl
   assert d.wait_state == "waiting"


def test_stop_waiting_noop_when_not_waiting():
   d = _make_gated()
   d.stop_waiting()
   assert d.waiting_for is None


def test_has_waiters_true_when_someone_waits_on_target():
   a = _make_gated("a")
   b = _make_gated("b")
   a.start_waiting("b")
   assert has_waiters([a, b], "b") is True


def test_has_waiters_false_when_no_one_waits():
   a = _make_gated("a")
   b = _make_gated("b")
   assert has_waiters([a, b], "b") is False


def test_has_waiters_ignores_plain_drone():
   plain = _make_drone("p")
   # A plain Drone has no waiting_for attribute at all and must be skipped.
   assert has_waiters([plain], "p") is False


def test_gated_drone_is_subtype_of_drone():
   d = _make_gated()
   assert isinstance(d, Drone)
