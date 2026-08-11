"""Geometric camera model for the perception pipeline.

This module answers one question and nothing else: *which neighbors does the camera of one ego drone see right now?* No image processing happens here —
the result is a plain :class:`CameraView` describing pose, view cone and the ground truth of the neighbors inside it. The (separately developed) detector
consumes that View through the perception adapter and returns position estimates; the geometry below only decides what is offered to it.

Drones carry no orientation: the state is ``x = [px, py, pz, vx, vy, vz]`` (``domain/drone.py``), so the camera axis is derived from the velocity with a
documented fallback chain (see :meth:`CameraModel.view_direction`).

Visibility test (no ``arccos``, pure dot product)::

    dist <= range_m  and  dot(view_dir, d) >= dist * cos(fov_deg / 2)

with ``d`` the vector from the ego position to the neighbor. ``fov_deg`` is the *full* opening angle of the cone.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
   from drone_sim.domain.drone import Drone

_log = logging.getLogger(__name__)

# Below this speed the velocity carries no usable heading information.
_HEADING_EPS = 1e-6

# Neighbors closer than this to the ego position have no meaningful direction;
# they are reported as visible regardless of the view cone.
_COINCIDENT_EPS = 1e-9

# Last-resort view direction when neither velocity, cached heading nor route give one.
_FALLBACK_VIEW_DIR = np.array([1.0, 0.0, 0.0])


@dataclass
class VisibleDrone:
   """One neighbor inside the ego drone's view cone (ground truth of the visibility test).

   The arrays are **copies** of the neighbor's state, so a :class:`CameraView` can be handed to a worker thread while the simulation keeps writing
   ``drone.x``.

   :param drone_id: Identifier of the observed neighbor.
   :param position: ``(3,)`` world position at capture time.
   :param velocity: ``(3,)`` world velocity at capture time.
   :param radius: Physical radius of the neighbor.
   """

   drone_id: str
   position: np.ndarray
   velocity: np.ndarray
   radius: float


@dataclass
class CameraView:
   """What one ego drone sees at one point in time.

   Deliberately **not** frozen: the perception worker attaches an optional rendered pseudo-FPV image to ``image_png`` before handing the View to the
   detector. The field layout mirrors the REST payload of the perception adapter (``pose`` / ``camera`` / ``visible`` / ``image_png_base64``).

   :param observer_id: Identifier of the observing (ego) drone.
   :param step: Simulation step index at capture time.
   :param sim_time: Simulation time in seconds at capture time.
   :param position: ``(3,)`` copy of the ego position.
   :param view_dir: ``(3,)`` unit vector of the camera axis.
   :param fov_deg: Full opening angle of the view cone in degrees.
   :param range_m: Maximum sensing distance in meters.
   :param visible: Neighbors inside the cone, nearest-first order not guaranteed.
   :param image_png: Optional rendered PNG bytes, attached later by the worker.
   """

   observer_id: str
   step: int
   sim_time: float
   position: np.ndarray
   view_dir: np.ndarray
   fov_deg: float
   range_m: float
   visible: list[VisibleDrone] = field(default_factory=list)
   image_png: bytes | None = None


class CameraModel:
   """Forward-facing camera cone attached to a drone.

   ``fov_deg`` is the *full* opening angle; the half angle used in the visibility test is ``fov_deg / 2``. At ``fov_deg == 360`` the cosine of the half
   angle is ``-1`` and — because ``|dot(view_dir, d)| <= dist`` — every neighbor within ``range_m`` is visible (spherical vision).

   The model is **stateful**: it caches the last non-zero heading per drone id for the zero-velocity fallback. :meth:`capture` is therefore meant to be
   called from the simulator's main thread only, not from the perception worker.

   :param fov_deg: Full field of view in degrees, must be in ``(0, 360]``.
   :param range_m: Maximum sensing distance in meters, must be positive.
   """

   def __init__(self, fov_deg: float = 90.0, range_m: float = 10.0) -> None:
      # Mirrors ScenarioConfig._validate_camera — a CameraModel is also constructed directly (tests, scripts), not only from a validated config.
      if not 0 < fov_deg <= 360:
         raise ValueError("fov_deg must be in (0, 360]")
      if range_m <= 0:
         raise ValueError("range_m must be positive")
      self.fov_deg = float(fov_deg)
      self.range_m = float(range_m)
      self._cos_half = math.cos(math.radians(self.fov_deg / 2.0))
      self._last_heading: dict[str, np.ndarray] = {}

   def view_direction(self, drone: Drone) -> np.ndarray:
      """Unit vector the camera of ``drone`` points along.

      Drones have no orientation state, so the heading is resolved in this order:

      1. the normalized velocity, when ``||v|| >= 1e-6`` (also cached for this drone id),
      2. the cached last non-zero heading of this drone,
      3. the direction to the current route reference (``route.current_ref() - position``),
      4. ``+x`` as a last resort.

      Only observed headings (case 1) are cached — the route direction is a guess, not a measurement.

      :param drone: Ego drone.
      :return: ``(3,)`` unit vector.
      """
      velocity = np.asarray(drone.velocity(), dtype=float)
      speed = float(np.linalg.norm(velocity))
      if speed >= _HEADING_EPS:
         heading = velocity / speed
         self._last_heading[drone.drone_id] = heading
         return heading.copy()

      cached = self._last_heading.get(drone.drone_id)
      if cached is not None:
         return cached.copy()

      to_ref = np.asarray(drone.route.current_ref(), dtype=float) - np.asarray(drone.position(), dtype=float)
      dist = float(np.linalg.norm(to_ref))
      if dist >= _HEADING_EPS:
         return to_ref / dist

      return _FALLBACK_VIEW_DIR.copy()

   def capture(self, ego: Drone, others: Iterable[Drone], *, step: int, sim_time: float) -> CameraView:
      """Build the :class:`CameraView` of ``ego`` at the current simulation state.

      The ego drone itself is skipped (by identity *and* by id), so the full drone list can be passed as ``others``. Neighbors are copied out of their
      drones, so the returned View is safe to hand to another thread. Apart from the heading cache, nothing is mutated.

      :param ego: Observing drone.
      :param others: Candidate neighbors; may include ``ego``.
      :param step: Simulation step index, copied into the View.
      :param sim_time: Simulation time in seconds, copied into the View.
      :return: The View, with ``visible`` holding every neighbor inside cone and range.
      """
      origin = np.asarray(ego.position(), dtype=float)
      view_dir = self.view_direction(ego)

      visible: list[VisibleDrone] = []
      for other in others:
         if other is ego or other.drone_id == ego.drone_id:
            continue
         position = np.asarray(other.position(), dtype=float)
         offset = position - origin
         dist = float(np.linalg.norm(offset))
         if dist > self.range_m:
            continue
         # Coincident positions have no direction to test against the cone.
         if dist > _COINCIDENT_EPS and float(np.dot(view_dir, offset)) < dist * self._cos_half:
            continue
         visible.append(VisibleDrone(drone_id=other.drone_id, position=np.array(position, dtype=float),
                                     velocity=np.array(other.velocity(), dtype=float), radius=float(other.radius)))

      # One line per drone per step — guarded so the default INFO log stays clean.
      if _log.isEnabledFor(logging.DEBUG):
         _log.debug("Camera capture: observer=%s step=%d t=%.3f fov=%.1f range=%.1f dir=(%+.3f, %+.3f, %+.3f) -> %d visible %s", ego.drone_id, step,
                    sim_time, self.fov_deg, self.range_m, view_dir[0], view_dir[1], view_dir[2], len(visible), [v.drone_id for v in visible])

      return CameraView(observer_id=ego.drone_id, step=step, sim_time=sim_time, position=np.array(origin, dtype=float), view_dir=view_dir,
                        fov_deg=self.fov_deg, range_m=self.range_m, visible=visible)