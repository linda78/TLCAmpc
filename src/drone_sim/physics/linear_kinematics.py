from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_sim.domain.registry import register_physics
from drone_sim.physics.base import PhysicsModel


@register_physics("linear_kinematics")
@dataclass
class LinearKinematicsPhysics(PhysicsModel):
   """Discrete-time constant-acceleration model.

      State:
          x = [x, y, z, v_x, v_y, v_z]^T
      Control:
          u = [a_x, a_y, a_z]^T

      Update:
          x_{k+1} = A x_k + B u_k

      A = State Transition Matrix
            [[I, dt*I],
           [0,   I]]

      B = Control Input Matrix
            [[0.5*dt^2*I],
           [  dt*I  ]]

      v_max: Maximum velocity magnitude (m/s) for trajectory optimization constraint.
      u_min: Minimum control input (acceleration) per axis (m/s^2). (break acceleration)
      u_max: Maximum control input (acceleration) per axis (m/s^2).
   """

   # Maximum velocity magnitude (m/s) for trajectory optimization constraint.
   v_max: float = 5.0

   # Control input (acceleration) limits per axis (m/s^2).
   u_min: tuple[float, float, float] | list[float] | np.ndarray = (-3.0, -3.0, -3.0) # acceleration in negative direction (brake the drone)
   u_max: tuple[float, float, float] | list[float] | np.ndarray = (3.0, 3.0, 3.0) # acceleration in positive direction

   def __post_init__(self) -> None:
      self.u_min = np.asarray(self.u_min, dtype=float)
      self.u_max = np.asarray(self.u_max, dtype=float)
      self.A = np.block([[np.eye(3), self.dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
      self.B = np.block([[0.5 * self.dt ** 2 * np.eye(3)], [self.dt * np.eye(3)]])

   def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
      x = np.asarray(x, dtype=float).reshape(6)
      u = np.asarray(u, dtype=float).reshape(3)
      result_x = self.A @ x + self.B @ u
      return self.clip_velocity(result_x)

   def clip_velocity(self, x: np.ndarray) -> np.ndarray:
      vel = x[3:6]
      vel_mag = np.linalg.norm(vel)
      if vel_mag > self.v_max:
         x[3:6] = vel * (self.v_max / vel_mag)
      return x
