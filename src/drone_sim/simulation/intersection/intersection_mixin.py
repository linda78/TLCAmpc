"""Coordinator-agnostic intersection-zone sequencing.

:class:`IntersectionMixin` adds the intersection-sphere state machine to any MPC
coordinator. Per step it builds predicted trajectories, runs stateless conflict
detection, and advances :class:`IntersectionStateStore` (which parks gated losers
via :meth:`GatedDrone.start_waiting`). The actual yielding is done by the
GatedDrone wait-state, which any coordinator honors automatically, so the mixin is
independent of the MPC underneath.

A coordinator mixes it in *before* its base, e.g.
``class C(IntersectionMixin, DistributedMPCCoordinator)``, and calls
:meth:`_run_intersection_step` at the top of its ``solve_controls`` before
delegating to ``super().solve_controls``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.drone import Drone
from drone_sim.simulation.intersection.intersection_state import (IntersectionStateStore, SphereRecord)
from drone_sim.simulation.utils.stateless_conflict_detection import (compute_threshold, constant_velocity_trajectory, detect_intersections)


@dataclass
class IntersectionMixin:
   """Intersection-sphere sequencing for any MPC coordinator.

   :param intersection_enabled: Master switch. When ``False`` the mixin is a
          no-op and no spheres are placed.
   :param intersection_sphere_radius: Radius of a placed conflict sphere.
   :param intersection_resolution_margin: Anti-flapping factor for sphere
          dissolution (see :class:`IntersectionStateStore`).
   """

   intersection_enabled: bool = True
   intersection_sphere_radius: float = 1.2
   intersection_resolution_margin: float = 1.1

   _state_store: IntersectionStateStore = field(init=False)

   def __post_init__(self) -> None:
      super().__post_init__()
      self._state_store = IntersectionStateStore(radius=self.intersection_sphere_radius, margin=self.intersection_resolution_margin)

   def _run_intersection_step(self, drones: list[Drone]) -> None:
      """Detect conflicts and advance the sphere store (parks gated losers).

      No-op when :attr:`intersection_enabled` is ``False``. Call this at the
      top of ``solve_controls`` before delegating to the base coordinator.
      """
      if not self.intersection_enabled:
         return
      predicted_trajs = self._seed_trajectories(drones)
      threshold = compute_threshold(drones)
      events = detect_intersections(drones, predicted_trajs, threshold, self.dt)
      self._state_store.step(events, drones, threshold)

   def active_spheres(self) -> dict[frozenset[str], SphereRecord]:
      """Active sphere snapshot — consumed by the GUI to render zones."""
      return self._state_store.active_pairs()

   def _seed_trajectories(self, drones: list[Drone]) -> dict[str, np.ndarray]:
      """(H, 3) predicted trajectory per drone for conflict detection.

      Default: constant-velocity extrapolation. Coordinators with a richer
      warm-start (e.g. the DMPC ``init_trajectories``) override this.
      """
      out: dict[str, np.ndarray] = {}
      for drone in drones:
         x = np.asarray(drone.x, dtype=float)
         out[drone.drone_id] = constant_velocity_trajectory(position=x[:3], velocity=x[3:6], horizon=self.horizon, dt=self.dt)
      return out


__all__ = ("IntersectionMixin",)