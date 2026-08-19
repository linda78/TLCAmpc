from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SphereObstacle:
   """Spherical static obstacle.

   Used by the intersection-zone coordinator to mark the predicted meeting
   point of two drones. Plain dataclass so it can live next to
   ``(center, half_extents)`` tuples in the obstacles list consumed by
   :class:`ObstacleAvoidanceConstraints`.
   """

   center: np.ndarray
   radius: float
   meta: dict[str, Any] = field(default_factory=dict)

   def __post_init__(self) -> None:
      self.center = np.asarray(self.center, dtype=float).reshape(3)
      self.radius = float(self.radius)

   def to_aabb(self) -> tuple[np.ndarray, np.ndarray]:
      """Conservative axis-aligned bounding box (half_extents = radius).

      Useful as a fallback path for code that still expects the legacy
      ``(center, half_extents)`` tuple form.
      """
      half_extents = np.full(3, self.radius, dtype=float)
      return self.center.copy(), half_extents
