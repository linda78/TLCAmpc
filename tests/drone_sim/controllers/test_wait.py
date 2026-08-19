"""Tests for the WaitController used by GatedDrone."""

from __future__ import annotations

import numpy as np
import pytest

from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.controllers.wait import WAIT_CONTROLLER, WaitController
from drone_sim.domain.drone import Drone, Route, has_central_cost
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _make_drone(velocity: np.ndarray, dt: float = 0.1) -> Drone:
   x = np.zeros(6)
   x[3:6] = velocity
   physics = LinearKinematicsPhysics(
      dt=dt, v_max=3.0, u_min=[-3.0, -3.0, -3.0], u_max=[3.0, 3.0, 3.0]
   )
   return Drone(
      drone_id="d",
      radius=0.2,
      safety_zone=0.5,
      cons_stop=0.0,
      color="tab:blue",
      safety_color="tab:blue",
      trace_color="tab:blue",
      controller=WAIT_CONTROLLER,
      physics=physics,
      x=x,
      route=Route(waypoints=[], target=np.array([10.0, 0.0, 0.0])),
   )


def test_wait_controller_brakes_positive_velocity():
   drone = _make_drone(np.array([1.0, 0.5, -0.5]))
   u = WAIT_CONTROLLER.control(drone, neighbors=[], obstacles=[])
   # -v/dt = [-10, -5, 5] -> clipped to actuator bounds [-3, 3]
   np.testing.assert_array_equal(u, np.array([-3.0, -3.0, 3.0]))


def test_wait_controller_zero_velocity_zero_control():
   drone = _make_drone(np.zeros(3))
   u = WAIT_CONTROLLER.control(drone, neighbors=[], obstacles=[])
   np.testing.assert_array_equal(u, np.zeros(3))


def test_wait_controller_within_bounds_unclipped():
   # Velocity small enough that -v/dt stays inside bounds.
   drone = _make_drone(np.array([0.1, -0.05, 0.02]))
   u = WAIT_CONTROLLER.control(drone, neighbors=[], obstacles=[])
   np.testing.assert_allclose(u, np.array([-1.0, 0.5, -0.2]))


def test_wait_controller_is_not_a_central_cost_provider():
   # The DMPC filter must reject the WaitController so wait-state drones
   # automatically fall into the non-optimized path.
   assert has_central_cost(WAIT_CONTROLLER) is False


def test_wait_controller_singleton_is_stateless():
   # Two drones may share the same singleton without interference.
   a = _make_drone(np.array([2.0, 0.0, 0.0]))
   b = _make_drone(np.array([0.0, 2.0, 0.0]))
   ua = WAIT_CONTROLLER.control(a, [], [])
   ub = WAIT_CONTROLLER.control(b, [], [])
   np.testing.assert_array_equal(ua, np.array([-3.0, 0.0, 0.0]))
   np.testing.assert_array_equal(ub, np.array([0.0, -3.0, 0.0]))


def test_central_cost_controller_passes_filter_for_contrast():
   ctrl = CentralMPCAgent(dt=0.1, horizon=4)
   assert has_central_cost(ctrl) is True
