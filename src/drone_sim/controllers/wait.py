"""Stateless brake-to-zero controller for parked GatedDrones.

Has no ``central_cost`` protocol — failing ``has_central_cost`` makes a
parked drone fall out of DMPC's ``opt_drones`` set, and the simulator's
per-drone fallback loop picks up the brake command from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
   from drone_sim.domain.drone import Drone


class WaitController:
   """Brake-to-zero controller — stateless, shared across all parked drones."""

   def control(self, drone: "Drone", neighbors: list[tuple[np.ndarray, np.ndarray, float, float, np.ndarray]],
         obstacles: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
      v = np.asarray(drone.x, dtype=float)[3:6]
      u_min, u_max = drone.bounds()
      return np.clip(-v / drone.physics.dt, u_min, u_max)


WAIT_CONTROLLER = WaitController()

__all__ = ("WaitController", "WAIT_CONTROLLER")
