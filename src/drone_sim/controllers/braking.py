"""Velocity-penalty MPC cost — brakes a drone to zero velocity.

Implements ``central_cost`` so a stopping drone stays in DMPC's
``opt_drones``, keeping collision constraints active for neighbors. The
``_Qp/_Qv/_R`` fields are required because :class:`LocalMPCSolver` reads
the cost inline from them rather than calling :meth:`central_cost`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from drone_sim.controllers.central_cost import as_diagonal

if TYPE_CHECKING:
   from drone_sim.domain.drone import Drone


@dataclass
class BrakingMPCAgent:
   """Velocity-penalty MPC cost — pure brake-to-zero, ignores route reference."""

   horizon: int = 4
   q_pos: list[float] = (0.0, 0.0, 0.0)
   q_vel: list[float] = (10.0, 10.0, 10.0)
   r_u: list[float] = (0.01, 0.01, 0.01)

   def __post_init__(self) -> None:
      self._Qp = as_diagonal(self.q_pos)
      self._Qv = as_diagonal(self.q_vel)
      self._R = as_diagonal(self.r_u)

   def central_initial_guess(self, drone: "Drone") -> np.ndarray:
      v = np.asarray(drone.x, dtype=float)[3:6]
      u_min, u_max = drone.bounds()
      brake = np.clip(-v / drone.physics.dt, u_min, u_max)
      return np.tile(brake.reshape(1, 3), (self.horizon, 1))

   def central_cost(self, u_seq: np.ndarray, drone: "Drone") -> float:
      u_seq = np.asarray(u_seq, dtype=float).reshape((-1, 3))
      _, velocities = drone.physics.predict_trajectory(drone.x, u_seq)
      qv = np.diag(self._Qv)
      r = np.diag(self._R)
      return float(np.sum(velocities ** 2 * qv) + np.sum(u_seq ** 2 * r))

   def control(self, drone: "Drone", neighbors: list[tuple[np.ndarray, np.ndarray, float, float, np.ndarray]],
         obstacles: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
      return self.central_initial_guess(drone)[0]


BRAKING_CONTROLLER = BrakingMPCAgent()

__all__ = ("BrakingMPCAgent", "BRAKING_CONTROLLER")
