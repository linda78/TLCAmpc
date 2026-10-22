from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from drone_sim.controllers.base import Controller
from drone_sim.physics.base import PhysicsModel

Color: TypeAlias = str | tuple[float, float, float]


def has_central_cost(ctrl: object) -> bool:
   """Check if controller implements the central_cost interface."""
   return all(hasattr(ctrl, name) for name in ("central_cost", "central_initial_guess", "horizon"))


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

   # Cached u_max_scalar for the stopping distance (adaptive and terminal radius, set lazily)
   _cached_u_max_scalar: float | None = field(default=None, init=False, repr=False)

   @property
   def is_adaptive(self) -> bool:
      """Whether this drone uses velocity-dependent adaptive safety zones."""
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

   def compute_terminal_radius(self, velocity: np.ndarray) -> float:
      """Compute the safety radius for the last step of the MPC horizon.

      The step radius ``r_min + alpha * s_stop`` is enough inside the horizon, where the
      planner still sees and resolves conflicts. At the last step nothing beyond the
      horizon is checked, so the sphere there has to contain the full stopping distance
      plus the distance travelled during half a sampling step:

          r_term = r_min + ||v||^2 / (2 U_max) + 0.5 * dt * ||v||

      With this terminal radius "brake to a standstill" is always a feasible continuation
      of the current plan, which keeps the next MPC problem feasible. The base radius is
      ``safety_zone`` for every drone: r_min for adaptive drones, the fixed radius r_s for
      static drones. Both get the same braking allowance on top.

      :param velocity: 3D velocity vector [vx, vy, vz] at the last horizon step.
      :return: terminal safety radius.
      """
      speed = float(np.linalg.norm(velocity))
      return self.safety_zone + self._stopping_distance(speed) + 0.5 * self.physics.dt * speed

   def _stopping_distance(self, speed: float) -> float:
      """Minimum stopping distance s_stop = ||v||^2 / (2 U_max) under maximum deceleration."""
      if self._cached_u_max_scalar is None:
         _, u_max = self.bounds()
         self._cached_u_max_scalar = float(np.max(np.abs(u_max)))
      return speed * speed / (2.0 * self._cached_u_max_scalar)

   def _compute_adaptive_radius_impl(self, speed: float) -> float:
      """Compute adaptive radius from the speed (velocity magnitude).

      :param speed: scalar speed (norm of velocity vector).
      :return: adaptive safety radius.
      """
      return self.safety_zone + self.alpha * self._stopping_distance(speed)

   def predict(self, u: np.ndarray) -> np.ndarray:
      return self.physics.step(self.x, u)

   def bounds(self) -> tuple[list[float], list[float]]:
      return self.physics.central_bounds()

   def position(self) -> np.ndarray:
      return self.x[:3]

   def velocity(self) -> np.ndarray:
      return self.x[3:]
