"""Central MPC coordinator augmented with the intersection-zone state machine.

Registered as ``"mpc_central_intersection"``. Combines
:class:`IntersectionMixin` with :class:`CentralMPCGlobalCoordinator`: the
sequencing logic lives in the mixin, the central MPC source itself is untouched.

The yielding mechanism is identical to the DMPC variant — gated losers are parked
via :meth:`GatedDrone.start_waiting`, which drops them out of the central
optimization automatically. Conflict detection is seeded from the same
MPC-initial-guess / warm-start prediction the central solver itself uses, so the
central variant detects the same conflicts as the DMPC variant (plain
constant-velocity seeding is too coarse to catch conflicts that only appear once
drones steer toward their goals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from drone_sim.domain.drone import Drone, GatedDrone, has_central_cost
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.centralized.coordinator import (CentralMPCGlobalCoordinator)
from drone_sim.simulation.distributed.local_mpc import _pad_or_trim_horizon
from drone_sim.simulation.intersection.intersection_mixin import IntersectionMixin

_log = logging.getLogger(__name__)


@register_coordinator("mpc_central_intersection")
@dataclass
class IntersectionCentralCoordinator(IntersectionMixin, CentralMPCGlobalCoordinator):
   """Central MPC coordinator with the intersection-zone state store.

   Per step: build predicted trajectories, run detection, step the state
   store (which parks gated losers), then delegate to the central MPC.
   """

   def solve_controls(self, *, drones: list[Drone], obstacles: list, room_min: np.ndarray | None = None, room_max: np.ndarray | None = None,
         lstm_provider: object | None = None) -> dict[str, np.ndarray]:
      """Step the intersection state then delegate to the central MPC."""
      self._run_intersection_step(drones)
      result = super().solve_controls(drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max, lstm_provider=lstm_provider)
      if _log.isEnabledFor(logging.DEBUG):
         opt_ids = [d.drone_id for d in drones if has_central_cost(d.controller)]
         waiting = [(d.drone_id, d.waiting_for) for d in drones if isinstance(d, GatedDrone) and d.is_waiting]
         _log.debug("IntersectionCentralCoordinator.solve_controls: central_opt=%s u_by_id_keys=%s waiting=%s active_spheres=%s", opt_ids, sorted(result.keys()),
               waiting, [sorted(p) for p in self._state_store.active_pairs()])
      return result

   def _seed_trajectories(self, drones: list[Drone]) -> dict[str, np.ndarray]:
      """(H, 3) per drone — predicted from the central MPC warm-start / initial guess.

      Mirrors the fidelity of the DMPC ``init_trajectories`` seed by reusing the
      central solver's :meth:`GlobalMPCSolver._predict_states`: each drone's
      trajectory is rolled out from the shifted previous solution
      (:attr:`_u_prev`) or, on the first step, the controller's
      ``central_initial_guess``. Drones without a central cost (e.g. parked
      waiters) roll out under zero control, i.e. constant velocity.
      """
      n = len(drones)
      u = np.zeros((n, self.horizon, 3), dtype=float)
      for i, drone in enumerate(drones):
         u_prev = self._u_prev.get(drone.drone_id)
         if u_prev is not None:
            shifted = np.concatenate([u_prev[1:], u_prev[-1:]], axis=0)
            u[i] = _pad_or_trim_horizon(shifted, self.horizon)
         elif has_central_cost(drone.controller):
            u[i] = _pad_or_trim_horizon(drone.controller.central_initial_guess(drone), self.horizon)
      states = self._solver._predict_states(drones, u)
      return {drone.drone_id: states[i, :, :3] for i, drone in enumerate(drones)}


__all__ = ("IntersectionCentralCoordinator",)