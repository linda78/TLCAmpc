"""Tests for drone_sim.perception.camera module.

Covers the purely geometric camera model:
- view_direction() and its fallback chain (velocity -> cached heading -> route -> +x)
- capture() field of view and range tests
- bookkeeping (self exclusion, passed-through fields, array copies)
- CameraModel constructor validation
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drone_sim.domain.drone import Drone, Route
from drone_sim.perception.camera import CameraModel
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


def _drone(drone_id: str, pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), target=(10.0, 0.0, 0.0), radius=0.1) -> Drone:
   """Helper to create a Drone object for testing. The camera never touches the controller."""
   return Drone(
      drone_id=drone_id,
      radius=radius,
      safety_zone=0.5,
      cons_stop=0.0,
      color="tab:blue",
      safety_color="tab:cyan",
      trace_color="tab:blue",
      controller=None,  # type: ignore[arg-type]
      physics=LinearKinematicsPhysics(dt=0.1, v_max=5.0),
      x=np.array([*pos, *vel], dtype=float),
      route=Route(waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
   )


def _at_angle(deg: float, dist: float = 5.0) -> tuple[float, float, float]:
   """Position at ``deg`` degrees off the +x axis in the xy-plane, ``dist`` away from the origin."""
   rad = math.radians(deg)
   return (dist * math.cos(rad), dist * math.sin(rad), 0.0)


class TestViewDirection:
   """Tests for CameraModel.view_direction and its fallback chain."""

   def test_uses_normalized_velocity(self):
      """Test the view direction is the normalized velocity."""
      cam = CameraModel()
      direction = cam.view_direction(_drone("ego", vel=(2.0, 0.0, 0.0)))

      np.testing.assert_allclose(direction, np.array([1.0, 0.0, 0.0]))
      assert float(np.linalg.norm(direction)) == pytest.approx(1.0)

   def test_cached_heading_fallback(self):
      """Test the last observed heading is reused once the drone stops."""
      cam = CameraModel()
      drone = _drone("ego", vel=(0.0, 3.0, 0.0))
      cam.view_direction(drone)

      drone.x[3:] = 0.0

      np.testing.assert_allclose(cam.view_direction(drone), np.array([0.0, 1.0, 0.0]))

   def test_route_fallback(self):
      """Test a standing drone with an empty heading cache looks toward its route reference."""
      cam = CameraModel()
      direction = cam.view_direction(_drone("ego", pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), target=(0.0, 0.0, 5.0)))

      np.testing.assert_allclose(direction, np.array([0.0, 0.0, 1.0]))

   def test_x_axis_fallback(self):
      """Test +x is the last resort when velocity is zero and the target equals the position."""
      cam = CameraModel()
      direction = cam.view_direction(_drone("ego", pos=(1.0, 2.0, 3.0), vel=(0.0, 0.0, 0.0), target=(1.0, 2.0, 3.0)))

      np.testing.assert_allclose(direction, np.array([1.0, 0.0, 0.0]))

   def test_route_fallback_not_cached(self):
      """Test the route direction is not cached - it is a guess, not an observed heading."""
      cam = CameraModel()
      cam.view_direction(_drone("ego", pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), target=(0.0, 0.0, 5.0)))

      assert cam._last_heading == {}


class TestCaptureFov:
   """Tests for the view cone in CameraModel.capture (ego at the origin, looking +x)."""

   @pytest.fixture
   def cam(self) -> CameraModel:
      """Camera with a 90 degree full opening angle and 10 m range."""
      return CameraModel(fov_deg=90.0, range_m=10.0)

   @pytest.fixture
   def ego(self) -> Drone:
      """Ego drone at the origin moving along +x."""
      return _drone("ego", pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))

   def test_straight_ahead_visible(self, cam: CameraModel, ego: Drone):
      """Test a neighbor straight ahead is visible."""
      view = cam.capture(ego, [_drone("a", pos=(5.0, 0.0, 0.0))], step=0, sim_time=0.0)

      assert [v.drone_id for v in view.visible] == ["a"]

   def test_behind_not_visible(self, cam: CameraModel, ego: Drone):
      """Test a neighbor behind the ego drone is not visible."""
      view = cam.capture(ego, [_drone("a", pos=(-5.0, 0.0, 0.0))], step=0, sim_time=0.0)

      assert view.visible == []

   def test_just_inside_cone_visible(self, cam: CameraModel, ego: Drone):
      """Test a neighbor just inside the 45 degree half angle is visible.

      The comparison is ``>=``, so the exact boundary counts as visible - but floating point equality right at 45.0
      degrees would be flaky, so inclusivity is probed at 44.999 degrees instead.
      """
      view = cam.capture(ego, [_drone("a", pos=_at_angle(44.999))], step=0, sim_time=0.0)

      assert [v.drone_id for v in view.visible] == ["a"]

   def test_just_outside_cone_not_visible(self, cam: CameraModel, ego: Drone):
      """Test a neighbor just outside the 45 degree half angle is not visible."""
      view = cam.capture(ego, [_drone("a", pos=_at_angle(46.0))], step=0, sim_time=0.0)

      assert view.visible == []

   def test_fov_360_sees_everything_in_range(self, ego: Drone):
      """Test fov_deg=360 gives spherical vision - only range_m limits it."""
      cam = CameraModel(fov_deg=360.0, range_m=10.0)
      view = cam.capture(ego, [_drone("a", pos=(-5.0, 0.0, 0.0))], step=0, sim_time=0.0)

      assert [v.drone_id for v in view.visible] == ["a"]

   def test_coincident_position_visible(self, cam: CameraModel, ego: Drone):
      """Test a neighbor sitting on the ego position is visible - no direction to test against the cone."""
      view = cam.capture(ego, [_drone("a", pos=(0.0, 0.0, 0.0))], step=0, sim_time=0.0)

      assert [v.drone_id for v in view.visible] == ["a"]


class TestCaptureRange:
   """Tests for the range limit in CameraModel.capture."""

   @pytest.fixture
   def cam(self) -> CameraModel:
      """Camera with a 90 degree full opening angle and 10 m range."""
      return CameraModel(fov_deg=90.0, range_m=10.0)

   @pytest.fixture
   def ego(self) -> Drone:
      """Ego drone at the origin moving along +x."""
      return _drone("ego", pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))

   def test_beyond_range_not_visible(self, cam: CameraModel, ego: Drone):
      """Test a neighbor beyond range_m is not visible."""
      view = cam.capture(ego, [_drone("a", pos=(11.0, 0.0, 0.0))], step=0, sim_time=0.0)

      assert view.visible == []

   def test_exactly_at_range_visible(self, cam: CameraModel, ego: Drone):
      """Test a neighbor exactly at range_m is visible - the border is inclusive."""
      view = cam.capture(ego, [_drone("a", pos=(10.0, 0.0, 0.0))], step=0, sim_time=0.0)

      assert [v.drone_id for v in view.visible] == ["a"]


class TestCaptureBookkeeping:
   """Tests for the fields and copies of the returned CameraView."""

   @pytest.fixture
   def cam(self) -> CameraModel:
      """Camera with a 90 degree full opening angle and 10 m range."""
      return CameraModel(fov_deg=90.0, range_m=10.0)

   @pytest.fixture
   def ego(self) -> Drone:
      """Ego drone at the origin moving along +x."""
      return _drone("ego", pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))

   def test_self_excluded(self, cam: CameraModel, ego: Drone):
      """Test the ego drone is skipped even when the full drone list is passed as others."""
      view = cam.capture(ego, [ego, _drone("a", pos=(5.0, 0.0, 0.0))], step=0, sim_time=0.0)

      assert [v.drone_id for v in view.visible] == ["a"]

   def test_view_fields(self, cam: CameraModel, ego: Drone):
      """Test observer_id, step, sim_time, fov_deg and range_m are passed through; image_png stays None."""
      view = cam.capture(ego, [], step=7, sim_time=0.7)

      assert view.observer_id == "ego"
      assert view.step == 7
      assert view.sim_time == 0.7
      assert view.fov_deg == 90.0
      assert view.range_m == 10.0
      assert view.image_png is None

   def test_visible_drone_carries_velocity_and_radius(self, cam: CameraModel, ego: Drone):
      """Test a VisibleDrone carries the neighbor's velocity and radius."""
      view = cam.capture(ego, [_drone("a", pos=(5.0, 0.0, 0.0), vel=(0.0, 2.0, 0.0), radius=0.25)],
                         step=0, sim_time=0.0)

      np.testing.assert_allclose(view.visible[0].velocity, np.array([0.0, 2.0, 0.0]))
      assert view.visible[0].radius == 0.25

   def test_arrays_are_copies(self, cam: CameraModel, ego: Drone):
      """Test the View survives the simulation writing on drone.x - required for handing it to a worker thread."""
      other = _drone("a", pos=(5.0, 0.0, 0.0))
      view = cam.capture(ego, [other], step=0, sim_time=0.0)

      ego.x[:] = 99.0
      other.x[:] = 99.0

      np.testing.assert_allclose(view.position, np.array([0.0, 0.0, 0.0]))
      np.testing.assert_allclose(view.visible[0].position, np.array([5.0, 0.0, 0.0]))

   def test_empty_others_gives_empty_visible(self, cam: CameraModel, ego: Drone):
      """Test capturing without neighbors yields an empty visible list."""
      view = cam.capture(ego, [], step=0, sim_time=0.0)

      assert view.visible == []


class TestCameraModelValidation:
   """Tests for CameraModel constructor validation."""

   @pytest.mark.parametrize("fov_deg", [0.0, 361.0])
   def test_invalid_fov_raises(self, fov_deg: float):
      """Test fov_deg outside (0, 360] raises ValueError."""
      with pytest.raises(ValueError, match="fov_deg"):
         CameraModel(fov_deg=fov_deg)

   @pytest.mark.parametrize("range_m", [0.0, -1.0])
   def test_invalid_range_raises(self, range_m: float):
      """Test a non-positive range_m raises ValueError."""
      with pytest.raises(ValueError, match="range_m"):
         CameraModel(range_m=range_m)

   def test_fov_360_is_valid(self):
      """Test fov_deg=360 is accepted (spherical vision)."""
      cam = CameraModel(fov_deg=360.0)

      assert cam.fov_deg == 360.0