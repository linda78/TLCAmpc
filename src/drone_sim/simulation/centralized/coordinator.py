from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.drone import Drone
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.centralized.global_mpc import GlobalMPCSolver
from drone_sim.simulation.utils.cpu_costs import CpuCostLedger


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
   horizon: int = 5
   room_wall_tolerance: float = 0.0

   # Small acceleration used for warm-start symmetry breaking.
   # Helps SLSQP escape head-on collinear deadlocks where the distance
   # constraint gradient is zero in lateral directions.
   symmetry_break_accel: float = 0.05

   max_iter: int = 120
   f_tol: float = 1e-3

   # Shared CPU-time ledger, see drone_sim.simulation.utils.cpu_costs
   costs: CpuCostLedger = field(default_factory=CpuCostLedger, repr=False)

   def __post_init__(self) -> None:
      self._u_prev: dict[str, np.ndarray] = {}
      self._solver = GlobalMPCSolver(
         dt=self.dt, horizon=self.horizon, costs=self.costs,
         max_iter=self.max_iter, f_tol=self.f_tol,
         room_wall_tolerance=self.room_wall_tolerance,
         symmetry_break_accel=self.symmetry_break_accel,
      )

   def solve_controls(self, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, np.ndarray]], room_min: np.ndarray | None = None,
         room_max: np.ndarray | None = None) -> dict[str, np.ndarray]:

      controls, u_sequences = self._solver.solve(
         drones=drones, obstacles=obstacles,
         room_min=room_min, room_max=room_max,
         u_prev=self._u_prev,
      )

      # Store returned u_sequences for warm-start next call
      self._u_prev = u_sequences

      return controls
