from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from drone_sim.controllers.base import Controller

if TYPE_CHECKING:
   from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics

Color: TypeAlias = str | tuple[float, float, float]


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
   x: np.ndarray  # [x,y,z,vx,vy,vz]
   route: Route

   # Per-drone kinematics model containing v_max constraint.
   kinematics: "LinearKinematicsPhysics"

   def position(self) -> np.ndarray:
      return self.x[:3]

   def velocity(self) -> np.ndarray:
      return self.x[3:]

   @property
   def v_max(self) -> float:
      """Maximum velocity magnitude from kinematics model."""
      return self.kinematics.v_max

   @property
   def u_min(self) -> np.ndarray:
      """Minimum control input (acceleration) per axis from kinematics model."""
      return self.kinematics.u_min

   @property
   def u_max(self) -> np.ndarray:
      """Maximum control input (acceleration) per axis from kinematics model."""
      return self.kinematics.u_max
