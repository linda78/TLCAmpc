from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence, TypeAlias

import numpy as np

from drone_sim.controllers.base import Controller
from drone_sim.physics.base import PhysicsModel

_log = logging.getLogger(__name__)

Color: TypeAlias = str | tuple[float, float, float]


def has_central_cost(ctrl: object) -> bool:
   """Check if controller implements the central_cost interface."""
   return all(hasattr(ctrl, name) for name in ("central_cost", "central_initial_guess", "horizon"))


def has_waiters(drones: Sequence["Drone"], drone_id: str) -> bool:
   """Whether any GatedDrone is currently waiting on ``drone_id``."""
   for d in drones:
      if isinstance(d, GatedDrone) and d.waiting_for == drone_id:
         return True
   return False


@dataclass
class Route:
   waypoints: list[np.ndarray]
   target: np.ndarray
   waypoint_radius: float = 0.5
   idx: int = 0

   def current_ref(self) -> np.ndarray:
      if self.idx < len(self.waypoints):
         return self.waypoints[self.idx]
      return self.target

   def advance_if_reached(self, position: np.ndarray) -> None:
      if self.idx < len(self.waypoints):
         if np.linalg.norm(position - self.waypoints[self.idx]) <= self.waypoint_radius:
            self.idx += 1

   def target_reached(self, position: np.ndarray, thresh: float = 1e-3) -> bool:
      return np.linalg.norm(position - self.target) < thresh


@dataclass
class Drone:
   drone_id: str
   radius: float
   safety_zone: float
   cons_stop: float

   color: Color
   safety_color: Color
   trace_color: Color

   controller: Controller
   physics: PhysicsModel
   x: np.ndarray  # [x,y,z,vx,vy,vz]
   route: Route

   # Adaptive safety zone parameter. When set, the drone uses a velocity-dependent
   # safety radius instead of the fixed safety_zone.
   alpha: float | None = None

   # Cached u_max_scalar for adaptive radius computation (set lazily)
   _cached_u_max_scalar: float | None = field(default=None, init=False, repr=False)

   # Safety zone mode: "fixed", "adaptive", or "lstm".
   # Drives _safety_radius() branch selection in constraint evaluation.
   # TODO: adaptive is not enough, we have velocity dependent safety zones and uncertainty safety zones ... switch that behavior,
   #  so all adaptive safety_zone are "adaptive" and then be able to check if drone is velocity dependent or uncertainty dependent. Maybe return a type, instead of a bool.
   safety_zone_mode: str = "fixed"

   @property
   def is_adaptive(self) -> bool:
      """Whether this drone uses velocity-dependent adaptive safety zones."""
      # TODO: adaptive is not enough, we have velocity dependent safety zones and uncertainty safety zones ... switch that behavior,
      #  so all adaptive safety_zone are "adaptive" and then be able to check if drone is velocity dependent or uncertainty dependent. Maybe return a type, instead of a bool.
      return self.alpha is not None

   @property
   def v_max(self) -> float:
      return self.physics.v_max()

   def compute_adaptive_radius(self, velocity: np.ndarray) -> float:
      """Compute safety radius based on current velocity.

      :param velocity: 3D velocity vector [vx, vy, vz].
      :return: safety radius (fixed safety_zone or adaptive r_min + alpha * s_stop).
      """
      if not self.is_adaptive:
         return self.safety_zone
      return self._compute_adaptive_radius_impl(float(np.linalg.norm(velocity)))

   def compute_max_adaptive_radius(self) -> float:
      """Return the maximum adaptive radius for the drone.

      :return: maximum adaptive radius.
      """
      if not self.is_adaptive:
         return self.safety_zone
      return self._compute_adaptive_radius_impl(self.v_max)

   def _compute_adaptive_radius_impl(self, speed: float) -> float:
      """Compute adaptive radius from the speed (velocity magnitude).

      :param speed: scalar speed (norm of velocity vector).
      :return: adaptive safety radius.
      """
      if self._cached_u_max_scalar is None:
         _, u_max = self.bounds()
         self._cached_u_max_scalar = float(np.max(np.abs(u_max)))
      v_norm_sq = speed * speed
      s_stop = v_norm_sq / (2.0 * self._cached_u_max_scalar)
      return self.safety_zone + self.alpha * s_stop

   def predict(self, u: np.ndarray) -> np.ndarray:
      return self.physics.step(self.x, u)

   def bounds(self) -> tuple[list[float], list[float]]:
      return self.physics.central_bounds()

   def position(self) -> np.ndarray:
      return self.x[:3]

   def velocity(self) -> np.ndarray:
      return self.x[3:]


@dataclass
class GatedDrone(Drone):
   """Drone with a wait-state for the intersection-zone coordinator.

   States (``wait_state``):

   - ``"free"`` — original controller, normal flight.
   - ``"stopping"`` — controller is :data:`BRAKING_CONTROLLER`. Stays in
     DMPC's ``opt_drones``; collision constraints remain active.
   - ``"waiting"`` — controller is :data:`WAIT_CONTROLLER`. Falls out of
     ``opt_drones`` (no ``central_cost``).

   :param waiting_for: drone_id this drone is waiting on, or ``None``.
   """

   waiting_for: str | None = None
   wait_state: str = "free"
   _original_controller: Controller | None = field(default=None, init=False, repr=False)

   @property
   def is_waiting(self) -> bool:
      return self.wait_state != "free"

   @property
   def is_stopping(self) -> bool:
      return self.wait_state == "stopping"

   @property
   def is_parked(self) -> bool:
      return self.wait_state == "waiting"

   def start_waiting(self, target_id: str) -> None:
      """Enter STOPPING — swap to :data:`BRAKING_CONTROLLER`.

      Idempotent: no-op if already in a non-free state (first wait wins).
      """
      if self.is_waiting:
         _log.debug(
            "GatedDrone.start_waiting: %s already in state=%s on %s, requested target=%s — no-op",
            self.drone_id, self.wait_state, self.waiting_for, target_id,
         )
         return
      from drone_sim.controllers.braking import BRAKING_CONTROLLER

      self._original_controller = self.controller
      self.controller = BRAKING_CONTROLLER
      self.waiting_for = target_id
      self.wait_state = "stopping"
      _log.info(
         "GatedDrone.start_waiting: %s -> STOPPING on %s",
         self.drone_id, target_id,
      )

   def transition_to_parked(self) -> None:
      """Move STOPPING → WAITING — swap to :data:`WAIT_CONTROLLER`."""
      if self.wait_state != "stopping":
         return
      from drone_sim.controllers.wait import WAIT_CONTROLLER

      self.controller = WAIT_CONTROLLER
      self.wait_state = "waiting"
      _log.info(
         "GatedDrone.transition_to_parked: %s STOPPING -> WAITING",
         self.drone_id,
      )

   def stop_waiting(self) -> None:
      """Release this drone — restore the original controller."""
      if not self.is_waiting:
         return
      released_target = self.waiting_for
      released_state = self.wait_state
      if self._original_controller is not None:
         self.controller = self._original_controller
      self._original_controller = None
      self.waiting_for = None
      self.wait_state = "free"
      _log.info(
         "GatedDrone.stop_waiting: %s released from %s (was waiting on %s)",
         self.drone_id, released_state, released_target,
      )
