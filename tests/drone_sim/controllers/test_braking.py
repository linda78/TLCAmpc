"""Tests for the BrakingMPCAgent used in the GatedDrone STOPPING state."""

from __future__ import annotations

import numpy as np

from drone_sim.controllers.braking import BRAKING_CONTROLLER, BrakingMPCAgent
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
      controller=BRAKING_CONTROLLER,
      physics=physics,
      x=x,
      route=Route(waypoints=[], target=np.array([10.0, 0.0, 0.0])),
   )


def test_braking_controller_implements_central_cost_protocol():
   # Unlike WaitController, BrakingMPCAgent stays in DMPC's opt_drones.
   assert has_central_cost(BRAKING_CONTROLLER) is True
   assert hasattr(BRAKING_CONTROLLER, "_Qp")
   assert hasattr(BRAKING_CONTROLLER, "_Qv")
   assert hasattr(BRAKING_CONTROLLER, "_R")


def test_braking_initial_guess_is_brake_step():
   drone = _make_drone(np.array([2.0, -1.0, 0.5]))
   u_seq = BRAKING_CONTROLLER.central_initial_guess(drone)
   # All steps identical: clip(-v/dt, [-3, 3]) = clip([-20, 10, -5], [-3, 3])
   expected = np.array([-3.0, 3.0, -3.0])
   np.testing.assert_allclose(u_seq[0], expected)
   # Same brake repeated across the horizon.
   for step in u_seq:
      np.testing.assert_allclose(step, expected)


def test_braking_cost_is_zero_at_rest():
   drone = _make_drone(np.zeros(3))
   u_seq = np.zeros((4, 3))
   cost = BRAKING_CONTROLLER.central_cost(u_seq, drone)
   assert cost == 0.0


def test_braking_cost_penalizes_velocity():
   drone = _make_drone(np.array([1.0, 0.0, 0.0]))
   # Apply zero control: velocity stays at 1 across horizon (linear kinematics).
   u_seq = np.zeros((4, 3))
   cost = BRAKING_CONTROLLER.central_cost(u_seq, drone)
   # qv * sum(v²) over 4 steps with v_x = 1 = 4 * q_vel[0].
   assert cost > 0


def test_braking_q_pos_is_zero():
   # No position-reference penalty — MPC must not pull drone to its route target.
   qp = np.diag(BRAKING_CONTROLLER._Qp)
   np.testing.assert_array_equal(qp, np.zeros(3))


def test_braking_control_falls_back_to_brake_step():
   drone = _make_drone(np.array([2.0, 0.0, 0.0]))
   u = BRAKING_CONTROLLER.control(drone, neighbors=[], obstacles=[])
   np.testing.assert_allclose(u, np.array([-3.0, 0.0, 0.0]))
