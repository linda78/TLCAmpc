from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_sim.domain.drone import Drone
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.centralized.global_mpc import GlobalMPCSolver


@register_coordinator("mpc_central")
@dataclass
class CentralMPCGlobalCoordinator:
   """Central coordinator for mixed controllers.

   Thin wrapper around GlobalMPCSolver that manages warm-start state
   and exposes the solve_controls() interface expected by the Simulator.

   Optimizes only drones whose controller implements the central-cost interface.

   Safety constraints use owner-only rule:
       dist(owner, other) >= owner.safety_zone + other.safety_buffer
   """

   dt: float
   horizon: int = 10
   room_wall_tolerance: float = 0.0

   # Small acceleration used for warm-start symmetry breaking.
   # Helps SLSQP escape head-on collinear deadlocks where the distance
   # constraint gradient is zero in lateral directions.
   symmetry_break_accel: float = 0.05

   max_iter: int = 120
   f_tol: float = 1e-3

   def __post_init__(self) -> None:
      self._u_prev: dict[str, np.ndarray] = {}
      self._solver = GlobalMPCSolver(
         dt=self.dt, horizon=self.horizon,
         max_iter=self.max_iter, f_tol=self.f_tol,
         room_wall_tolerance=self.room_wall_tolerance,
         symmetry_break_accel=self.symmetry_break_accel,
      )

   def solve_controls(self, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, np.ndarray]], room_min: np.ndarray | None = None,
         room_max: np.ndarray | None = None, lstm_provider: object | None = None) -> dict[str, np.ndarray]:

      # Compute lstm_radii from provider if available (once before SLSQP).
      # Central MPC uses evaluate_multi() which takes a flat dict keyed by drone_id.
      # LSTMSafetyZoneProvider.compute_neighbor_safety_radii() returns exactly that format.
      lstm_radii: dict | None = None
      if lstm_provider is not None:
         all_ids = [d.drone_id for d in drones]
         r_floor_by_id = {d.drone_id: d.safety_zone for d in drones}
         lstm_radii = lstm_provider.compute_neighbor_safety_radii(all_ids, r_floor_by_id)

      controls, u_sequences = self._solver.solve(
         drones=drones, obstacles=obstacles,
         room_min=room_min, room_max=room_max,
         u_prev=self._u_prev,
         lstm_radii=lstm_radii,
      )

      # Store returned u_sequences for warm-start next call
      self._u_prev = u_sequences

      return controls
