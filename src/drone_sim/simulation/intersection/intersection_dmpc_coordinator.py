"""DMPC coordinator augmented with the intersection-zone state machine.

Registered as ``"dmpc_admm_intersection"``. Subclasses
:class:`DistributedMPCCoordinator` and adds the intersection knobs + a
state store. The inherited DMPC source itself is untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.drone import Drone, GatedDrone, has_central_cost
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.distributed.distributed_coordinator import (DistributedMPCCoordinator)
from drone_sim.simulation.intersection.intersection_state import (IntersectionStateStore, SphereRecord)
from drone_sim.simulation.utils.stateless_conflict_detection import (compute_threshold, constant_velocity_trajectory, detect_intersections)

_log = logging.getLogger(__name__)


@register_coordinator("dmpc_admm_intersection")
@dataclass
class IntersectionDMPCCoordinator(DistributedMPCCoordinator):
   """DMPC coordinator with the intersection-zone state store.

   Per step: build predicted trajectories, run detection, step the state
   store (which parks gated losers), then delegate to the inherited DMPC.
   """

   intersection_enabled: bool = True
   intersection_sphere_radius: float = 1.2
   intersection_resolution_margin: float = 1.1

   _state_store: IntersectionStateStore = field(init=False)

   def __post_init__(self) -> None:
      super().__post_init__()
      self._state_store = IntersectionStateStore(radius=self.intersection_sphere_radius, margin=self.intersection_resolution_margin)

   def solve_controls(self, *, drones: list[Drone], obstacles: list, room_min: np.ndarray | None = None, room_max: np.ndarray | None = None,
         lstm_provider: object | None = None, obstacles_by_id: dict[str, list] | None = None) -> dict[str, np.ndarray]:
      """Step the intersection state then delegate to the inherited DMPC."""
      if self.intersection_enabled:
         predicted_trajs = self._predicted_trajectories(drones)
         threshold = compute_threshold(drones)
         events = detect_intersections(drones, predicted_trajs, threshold, self.dt)
         self._state_store.step(events, drones, threshold)
      result = super().solve_controls(drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max, lstm_provider=lstm_provider,
            obstacles_by_id=obstacles_by_id)
      if _log.isEnabledFor(logging.DEBUG):
         opt_ids = [d.drone_id for d in drones if has_central_cost(d.controller)]
         waiting = [(d.drone_id, d.waiting_for) for d in drones if isinstance(d, GatedDrone) and d.is_waiting]
         _log.debug("IntersectionDMPCCoordinator.solve_controls: dmpc_opt=%s u_by_id_keys=%s waiting=%s active_spheres=%s", opt_ids, sorted(result.keys()),
               waiting, [sorted(p) for p in self._state_store.active_pairs()])
      return result

   def active_spheres(self) -> dict[frozenset[str], SphereRecord]:
      """Active sphere snapshot — consumed by the GUI to render zones."""
      return self._state_store.active_pairs()

   def _predicted_trajectories(self, drones: list[Drone]) -> dict[str, np.ndarray]:
      """(H, 3) per drone — DMPC warm-start with constant-velocity fallback."""
      seeded, _ = self.init_trajectories(drones)
      horizon = self.horizon
      out: dict[str, np.ndarray] = {}
      for drone in drones:
         traj = seeded.get(drone.drone_id)
         if traj is None:
            traj = constant_velocity_trajectory(position=np.asarray(drone.x, dtype=float)[:3], velocity=np.asarray(drone.x, dtype=float)[3:6], horizon=horizon,
                  dt=self.dt)
         out[drone.drone_id] = traj
      return out


__all__ = ("IntersectionDMPCCoordinator",)
