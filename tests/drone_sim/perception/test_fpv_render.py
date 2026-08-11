"""Tests for drone_sim.perception.fpv_render module.

A pinhole camera can be tested exactly, which its Axes3D predecessor could not. Two layers:

- **the projection**, pure functions with hand-computable answers: centre pixel, frame edge, behind the camera, an orthonormal roll-free basis
- **the image**, checked against the three failures that motivated the rewrite: apparent area must fall off as 1/depth², a neighbor behind the drone
  must not appear, and a near neighbor must occlude a far one

The image tests measure ink in pixel counts rather than comparing to reference PNGs — a golden image would break on every Pillow rasterisation
change without telling us anything about the geometry.
"""

from __future__ import annotations

import io
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drone_sim.domain.drone_model import DroneModel, resolve_drone_model
from drone_sim.domain.mesh import SPHERE_RESOLUTION, sphere_grid
from drone_sim.perception.camera import CameraView, VisibleDrone
from drone_sim.perception.fpv_render import (_MAX_FOV_DEG, _Camera, _box_mesh, _camera_basis, _face_groups, _focal_length_px, _neighbor_mesh,
                                             _obj_mesh, _project, _shade, _sphere_mesh, _warn_unknown_model, _warn_wide_fov, render_fpv_png)

SIZE = (320, 240)
EYE = np.array([5.0, 5.0, 2.0])

OBJ_MODEL = resolve_drone_model("obj", "drone_costum_0_0_5.obj")
OTHER_OBJ_MODEL = resolve_drone_model("obj", "drone_costum_0_1.obj")
SPHERE_MODEL = DroneModel()


def _view(*, position=(5.0, 5.0, 2.0), view_dir=(1.0, 0.0, 0.0), fov_deg=90.0, range_m=10.0,
          visible=(("d1", (8.0, 5.0, 2.0), 0.2),)) -> CameraView:
   """Build a CameraView from (drone_id, position, radius) triples."""
   return CameraView(observer_id="ego", step=1, sim_time=0.1, position=np.array(position, dtype=float),
                     view_dir=np.array(view_dir, dtype=float), fov_deg=fov_deg, range_m=range_m,
                     visible=[VisibleDrone(drone_id=did, position=np.array(pos, dtype=float), velocity=np.zeros(3), radius=radius)
                              for did, pos, radius in visible])


def _pixels(png: bytes) -> np.ndarray:
   """Decode to an (H, W, 3) uint8 array."""
   from PIL import Image

   with Image.open(io.BytesIO(png)) as img:
      return np.asarray(img.convert("RGB"))


def _png_size(png: bytes) -> tuple[int, int]:
   """Decode just enough of the PNG to read its pixel dimensions."""
   from PIL import Image

   with Image.open(io.BytesIO(png)) as img:
      return img.size


def _ink_mask(png: bytes) -> np.ndarray:
   """Boolean mask of the pixels that are not background white."""
   return (_pixels(png) < 250).any(axis=2)


def _ink(png: bytes) -> int:
   """Number of non-background pixels — the apparent area of everything drawn."""
   return int(_ink_mask(png).sum())


class TestCameraBasis:
   """Tests for the orthonormal camera basis — the replacement for matplotlib's orbit angles."""

   def test_basis_is_orthonormal(self):
      """Test the three axes are unit length and mutually perpendicular for an arbitrary heading."""
      right, up, forward = _camera_basis(np.array([0.4, -0.7, 0.3]))

      for axis in (right, up, forward):
         assert float(np.linalg.norm(axis)) == pytest.approx(1.0)
      assert float(np.dot(right, up)) == pytest.approx(0.0, abs=1e-12)
      assert float(np.dot(right, forward)) == pytest.approx(0.0, abs=1e-12)
      assert float(np.dot(up, forward)) == pytest.approx(0.0, abs=1e-12)

   def test_forward_is_the_view_direction(self):
      """Test the camera looks where the drone looks — the sign error that made the old renderer look backwards."""
      _, _, forward = _camera_basis(np.array([3.0, 0.0, 0.0]))

      np.testing.assert_allclose(forward, [1.0, 0.0, 0.0])

   def test_camera_has_no_roll(self):
      """Test the horizon stays level for a tilted heading: right is horizontal and up keeps its world-up component.

      Up cannot simply *be* world up — it is orthogonalised against the heading and therefore tilts with it. Roll-free means the tilt happens
      only in the vertical plane containing the heading, which is exactly "right has no z component".
      """
      right, up, forward = _camera_basis(np.array([1.0, 1.0, 0.3]))

      assert float(right[2]) == pytest.approx(0.0, abs=1e-12)
      assert float(up[2]) > 0.0
      # up lies in the plane spanned by forward and world up.
      assert float(np.dot(up, np.cross(forward, [0.0, 0.0, 1.0]))) == pytest.approx(0.0, abs=1e-12)

   def test_looking_along_x_puts_right_on_minus_y(self):
      """Test the handedness: standing at the origin looking down +x with +z up, your right hand points at -y."""
      right, up, _ = _camera_basis(np.array([1.0, 0.0, 0.0]))

      np.testing.assert_allclose(right, [0.0, -1.0, 0.0], atol=1e-12)
      np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-12)

   def test_straight_up_uses_the_fallback_reference(self):
      """Test a vertical heading does not produce a zero-length or NaN basis, where world up is useless as a reference."""
      right, up, forward = _camera_basis(np.array([0.0, 0.0, 1.0]))

      np.testing.assert_allclose(forward, [0.0, 0.0, 1.0])
      assert np.isfinite(right).all() and np.isfinite(up).all()
      assert float(np.linalg.norm(right)) == pytest.approx(1.0)
      assert float(np.dot(right, forward)) == pytest.approx(0.0, abs=1e-12)

   def test_straight_down_uses_the_fallback_reference(self):
      """Test the vertical case is symmetric."""
      right, up, forward = _camera_basis(np.array([0.0, 0.0, -1.0]))

      np.testing.assert_allclose(forward, [0.0, 0.0, -1.0])
      assert float(np.linalg.norm(np.cross(right, up))) == pytest.approx(1.0)

   def test_unnormalized_direction_is_handled(self):
      """Test the length of the direction vector does not change the basis."""
      long_dir = _camera_basis(np.array([3.0, 0.0, 0.0]))
      unit_dir = _camera_basis(np.array([1.0, 0.0, 0.0]))

      for a, b in zip(long_dir, unit_dir):
         np.testing.assert_allclose(a, b)

   def test_zero_direction_falls_back_to_plus_x(self):
      """Test a degenerate heading does not produce NaNs."""
      for a, b in zip(_camera_basis(np.zeros(3)), _camera_basis(np.array([1.0, 0.0, 0.0]))):
         np.testing.assert_allclose(a, b)


class TestFocalLength:
   """Tests for the FOV -> focal length mapping, now in pixels rather than matplotlib units."""

   def test_focal_length_follows_the_pinhole_formula(self):
      """Test f_px = (width/2) / tan(fov/2): at 90 degrees the half-width equals the focal length."""
      assert _focal_length_px(90.0, 320) == pytest.approx(160.0)
      assert _focal_length_px(60.0, 320) == pytest.approx(160.0 / math.tan(math.radians(30.0)))

   def test_wider_fov_shortens_the_focal_length(self):
      """Test a wider cone means a shorter focal length, i.e. more world per pixel."""
      assert _focal_length_px(120.0, 320) < _focal_length_px(90.0, 320) < _focal_length_px(60.0, 320)

   def test_focal_length_scales_with_frame_width(self):
      """Test the same angle on a wider frame means proportionally more pixels."""
      assert _focal_length_px(90.0, 640) == pytest.approx(2.0 * _focal_length_px(90.0, 320))

   def test_spherical_fov_is_clamped_instead_of_diverging(self):
      """Test 180 degrees and beyond do not blow up: tan(90 deg) is infinite, but camera_fov_deg=360 is a valid config."""
      clamped = _focal_length_px(_MAX_FOV_DEG, 320)

      assert _focal_length_px(180.0, 320) == pytest.approx(clamped)
      assert _focal_length_px(360.0, 320) == pytest.approx(clamped)
      assert clamped > 0.0


class TestProjection:
   """Tests for the world -> pixel mapping, computed by hand."""

   CAM = _Camera(eye=EYE, basis=_camera_basis(np.array([1.0, 0.0, 0.0])), f_px=_focal_length_px(90.0, 320), width=320, height=240)
   F_PX = CAM.f_px
   CX, CY = CAM.cx, CAM.cy

   def _project(self, *points):
      return _project(np.array(points, dtype=float), self.CAM)

   def test_principal_point_is_the_frame_centre(self):
      """Test the camera derives cx/cy from the frame rather than storing them — nothing here models a shifted optical axis."""
      assert (self.CAM.cx, self.CAM.cy) == (160.0, 120.0)

   def test_point_straight_ahead_lands_in_the_centre(self):
      """Test the optical axis maps to the principal point, whatever the distance."""
      pixels, depth = self._project(EYE + [1.0, 0.0, 0.0], EYE + [7.0, 0.0, 0.0])

      np.testing.assert_allclose(pixels[0], [self.CX, self.CY])
      np.testing.assert_allclose(pixels[1], [self.CX, self.CY])
      np.testing.assert_allclose(depth, [1.0, 7.0])

   def test_point_at_the_fov_edge_lands_on_the_frame_edge(self):
      """Test the half-FOV direction maps exactly to the frame border — this is what f_px is defined by."""
      pixels, _ = self._project(EYE + [1.0, -1.0, 0.0], EYE + [1.0, 1.0, 0.0])

      assert pixels[0][0] == pytest.approx(320.0)
      assert pixels[1][0] == pytest.approx(0.0)

   def test_sideways_and_upward_points_land_where_the_formula_says(self):
      """Test an off-axis point at a known angle: 1 m up and 2 m ahead is f_px/2 pixels above the centre."""
      pixels, depth = self._project(EYE + [2.0, 0.0, 1.0], EYE + [2.0, 1.0, 0.0])

      assert depth[0] == pytest.approx(2.0)
      np.testing.assert_allclose(pixels[0], [self.CX, self.CY - self.F_PX / 2.0])
      # +y is to the camera's left, so it lands left of centre.
      np.testing.assert_allclose(pixels[1], [self.CX - self.F_PX / 2.0, self.CY])

   def test_image_y_is_flipped(self):
      """Test a point above the axis gets a *smaller* v — the image origin is top left, the world's z points up."""
      pixels, _ = self._project(EYE + [2.0, 0.0, 1.0], EYE + [2.0, 0.0, -1.0])

      assert pixels[0][1] < self.CY < pixels[1][1]

   def test_apparent_offset_halves_when_the_distance_doubles(self):
      """Test the perspective divide itself: same lateral offset twice as far away is half as far from the centre."""
      pixels, _ = self._project(EYE + [2.0, 1.0, 0.0], EYE + [4.0, 1.0, 0.0])

      assert (self.CX - pixels[0][0]) == pytest.approx(2.0 * (self.CX - pixels[1][0]))

   def test_points_behind_the_camera_are_reported_with_non_positive_depth(self):
      """Test the one check that fixes the old renderer's worst bug: behind-the-drone points are rejectable.

      The predecessor drew a neighbor 3 m *behind* the drone on top of the one in front of it.
      """
      _, depth = self._project(EYE + [-1.0, 0.0, 0.0], EYE, EYE + [-0.001, 0.0, 0.0])

      assert (depth <= 0.0).all()

   def test_projection_of_a_point_behind_the_camera_stays_finite(self):
      """Test rejected points do not produce NaN or a division warning that could poison a whole frame."""
      pixels, _ = self._project(EYE + [-1.0, 5.0, 0.0])

      assert np.isfinite(pixels).all()


class TestMeshes:
   """Tests for the geometry that is fed into the projection."""

   def test_sphere_mesh_is_a_unit_sphere(self):
      """Test every vertex sits on the unit sphere, so scaling by the radius gives the right size."""
      mesh = _sphere_mesh()

      np.testing.assert_allclose(np.linalg.norm(mesh.verts, axis=1), 1.0, atol=1e-12)
      assert len(mesh.faces) > 0
      assert all(max(face) < len(mesh.verts) for face in mesh.faces)

   def test_sphere_mesh_uses_the_same_parametrisation_as_the_overview(self):
      """Test the camera's sphere and ``draw_sphere_wireframe``'s sphere are the same object, not two lookalikes.

      Both go through ``domain.mesh.sphere_grid``; before that shared helper existed the two copies could drift apart silently.
      """
      x, y, z = sphere_grid(SPHERE_RESOLUTION)

      np.testing.assert_array_equal(_sphere_mesh().verts, np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1))

   def test_sphere_mesh_is_cached(self):
      """Test the grid is built once, not per drone per frame."""
      assert _sphere_mesh() is _sphere_mesh()

   def test_obj_mesh_is_normalised_to_a_unit_longest_axis(self):
      """Test the cached mesh is size-neutral; the per-drone scale is applied afterwards."""
      mesh = _obj_mesh(str(OBJ_MODEL.path))

      extents = mesh.verts.max(axis=0) - mesh.verts.min(axis=0)
      assert float(extents.max()) == pytest.approx(1.0)
      assert len(mesh.faces) > 0

   def test_obj_mesh_is_cached_per_path(self):
      """Test a heterogeneous fleet holds one mesh per model, not the first model twice (nor one parse per frame)."""
      first = _obj_mesh(str(OBJ_MODEL.path))
      again = _obj_mesh(str(OBJ_MODEL.path))
      other = _obj_mesh(str(OTHER_OBJ_MODEL.path))

      assert first is again
      assert other is not first
      assert not np.array_equal(other.verts, first.verts)

   def test_obj_mesh_cache_is_unbounded(self):
      """Test the cache cannot evict.

      A bounded cache would be worse than none for the case this is built for: a fleet with more distinct models than slots would evict and
      re-parse the .obj for every neighbor of every frame — 2.5 ms for the small asset, 17 ms for the full-resolution one.
      """
      assert _obj_mesh.cache_info().maxsize is None

   def test_obj_mesh_without_faces_is_none(self, tmp_path):
      """Test a file with no geometry is reported as unusable instead of crashing the frame."""
      empty = tmp_path / "empty.obj"
      empty.write_text("# no geometry here\n")

      assert _obj_mesh(str(empty)) is None

   def test_neighbor_sphere_has_the_drone_radius(self):
      """Test a sphere neighbor is drawn at exactly its physical radius."""
      mesh, scale = _neighbor_mesh(SPHERE_MODEL, 0.35)

      assert float(np.linalg.norm(mesh.verts * scale, axis=1).max()) == pytest.approx(0.35)

   def test_neighbor_obj_longest_axis_is_the_drone_diameter(self):
      """Test the absolute model scale, which the 1/depth² test cannot see (R18).

      The FPV renders physically: longest axis == 2 * radius. Drawing at the ``radius * 5`` that tools/live_view.py uses would make every
      distance estimate come out 2.5x too close, and only a test that nails the absolute size would notice.
      """
      mesh, scale = _neighbor_mesh(OBJ_MODEL, 0.2)
      verts = mesh.verts * scale

      extents = verts.max(axis=0) - verts.min(axis=0)
      assert float(extents.max()) == pytest.approx(0.4)

   def test_neighbors_of_different_size_share_one_mesh(self):
      """Test the per-drone size is a scale factor, not a second copy of the geometry."""
      small, _ = _neighbor_mesh(OBJ_MODEL, 0.2)
      large, _ = _neighbor_mesh(OBJ_MODEL, 0.9)

      assert small is large

   def test_box_mesh_matches_the_half_extents(self):
      """Test an obstacle box spans exactly twice its half extents in each axis, with six faces."""
      mesh = _box_mesh((0.5, 1.0, 2.0))

      np.testing.assert_allclose(mesh.verts.max(axis=0) - mesh.verts.min(axis=0), [1.0, 2.0, 4.0])
      np.testing.assert_allclose((mesh.verts.max(axis=0) + mesh.verts.min(axis=0)) / 2, [0.0, 0.0, 0.0])
      assert len(mesh.faces) == 6


class TestFaceBatching:
   """Tests for the per-mesh batching that lets a few hundred faces be shaded in one numpy pass."""

   def test_triangles_and_quads_are_bucketed_separately(self):
      """Test a mixed mesh yields one rectangular index array per vertex count — numpy cannot hold ragged faces."""
      groups = _face_groups(((0, 1, 2), (0, 1, 2, 3), (1, 2, 3)))

      assert sorted(group.shape for group in groups) == [(1, 4), (2, 3)]

   def test_degenerate_faces_are_dropped(self):
      """Test edges and points enclose no area and never reach the rasteriser."""
      groups = _face_groups(((0, 1), (0,), (0, 1, 2)))

      assert [group.shape for group in groups] == [(1, 3)]

   def test_buckets_and_colors_are_computed_once_per_mesh(self):
      """Test the derived draw data is memoised on the mesh, not rebuilt per drone per frame.

      Shading survives the per-drone transform because a mesh is only ever uniformly scaled and translated, and neither changes a face normal's
      direction — so it can be computed here rather than on every frame's world-space vertices.
      """
      mesh = _sphere_mesh()

      assert mesh.groups is mesh.groups

   def test_every_face_gets_a_color(self):
      """Test the colour list lines up with the index array of its bucket, since the two are zipped when drawing."""
      for idx, colors in _obj_mesh(str(OBJ_MODEL.path)).groups:
         assert len(colors) == len(idx)
         assert all(len(color) == 3 for color in colors)

   def test_shading_brightens_a_face_turned_toward_the_light(self):
      """Test the flat shading actually varies with orientation — without it a mesh is a flat silhouette."""
      lit = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])       # normal along +z, near the light
      edge_on = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])   # normal along x, across the light

      assert max(_shade((200, 200, 200), lit)[0]) > max(_shade((200, 200, 200), edge_on)[0])

   def test_shading_ignores_face_winding(self):
      """Test a reversed face is shaded the same.

      .obj files in the wild wind their faces inconsistently; taking the signed dot product would turn half a mesh black.
      """
      face = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
      flipped = face[:, ::-1]

      assert _shade((200, 200, 200), face) == _shade((200, 200, 200), flipped)

   def test_shading_is_unchanged_by_uniform_scale_and_translation(self):
      """Test the invariant the per-mesh colour cache rests on: placing a mesh in the world cannot change its shading.

      If this ever stopped holding, cached colours would be silently wrong for every drone away from the origin.
      """
      face = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
      placed = face * 3.7 + np.array([12.0, -4.0, 5.5])

      assert _shade((200, 100, 50), face) == _shade((200, 100, 50), placed)

   def test_shading_stays_within_the_base_color(self):
      """Test shading only darkens: the ambient floor keeps a face visible, the base color is the ceiling."""
      faces = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])  # degenerate: ambient only

      shaded = _shade((200, 100, 50), faces)

      assert all(0 <= channel for color in shaded for channel in color)
      assert all(channel <= ceiling for color in shaded for channel, ceiling in zip(color, (200, 100, 50)))
      assert all(color[0] >= 200 * 0.3 for color in shaded)


class TestDepthCues:
   """The regressions this rewrite exists for: the image has to carry distance."""

   def test_apparent_area_falls_off_as_one_over_depth_squared(self):
      """Test a neighbor at 1 m covers roughly 64x the area of the same neighbor at 8 m.

      This is the core test. The Axes3D predecessor measured 1.17 here — an eye 66 m behind the drone flattens the perspective almost completely,
      leaving the detector nothing to estimate distance from. The tolerance is wide because a sphere's silhouette is not exactly 1/z² and the far
      one is only a few pixels across.
      """
      near = _ink(render_fpv_png(_view(visible=(("d", (6.0, 5.0, 2.0), 0.5),)), [], size=SIZE))
      far = _ink(render_fpv_png(_view(visible=(("d", (13.0, 5.0, 2.0), 0.5),)), [], size=SIZE))

      assert near / far == pytest.approx(64.0, rel=0.35)

   def test_neighbor_behind_the_drone_is_not_drawn(self):
      """Test the second half of the old bug: a neighbor behind the camera produced pixels.

      A 360 degree cone reports every neighbor in range as visible, including those behind — ``view.visible`` is the cone test, not a frustum
      test, so the renderer has to drop them itself.
      """
      behind = _view(visible=(("d", (2.0, 5.0, 2.0), 0.5),), fov_deg=360.0)

      assert _ink(render_fpv_png(behind, [], size=SIZE)) == 0

   def test_the_rectangular_frame_crops_what_the_circular_cone_reports(self):
      """Test the documented cone-vs-rectangle divergence, on a neighbor chosen to sit exactly in the gap.

      ``fov_deg`` is the horizontal field of view; at 320x240 the vertical one is only ~74 degrees. A neighbor 2 m ahead and 1.8 m up is 42
      degrees off axis — inside the 90 degree cone that ``CameraModel`` tests, above the top edge of the image. This is a compromise, not a bug:
      the renderer is not a second implementation of the visibility test.
      """
      above = _view(visible=(("d", (7.0, 5.0, 3.8), 0.15),))

      assert _ink(render_fpv_png(above, [], size=SIZE)) == 0
      # Same neighbor moved into the vertical frame: the geometry is otherwise identical, only the angle changed.
      inside = _view(visible=(("d", (7.0, 5.0, 3.0), 0.15),))
      assert _ink(render_fpv_png(inside, [], size=SIZE)) > 0

   def test_a_near_neighbor_occludes_a_far_one(self):
      """Test the painter's algorithm: the pixels of the near drone are exactly what it looks like alone.

      Two drones on the optical axis, the near one large enough to cover the far one. If the sort order were reversed, the far drone's shading
      would paint over the near one's inside the overlap.
      """
      near_only = _view(visible=(("near", (6.0, 5.0, 2.0), 0.5),))
      both = _view(visible=(("far", (9.0, 5.0, 2.0), 0.5), ("near", (6.0, 5.0, 2.0), 0.5)))

      near_png = render_fpv_png(near_only, [], size=SIZE)
      both_png = render_fpv_png(both, [], size=SIZE)
      mask = _ink_mask(near_png)

      assert mask.sum() > 0
      np.testing.assert_array_equal(_pixels(both_png)[mask], _pixels(near_png)[mask])
      # The far drone is fully hidden: nothing outside the near silhouette lights up.
      assert _ink(both_png) == _ink(near_png)

   def test_a_partly_hidden_neighbor_still_contributes(self):
      """Test the occlusion is geometric, not "the first drone wins".

      The far drone is placed so its 40 px silhouette straddles the edge of the near drone's 160 px one: part of it is swallowed, part peeks out.
      Total ink must therefore land strictly between "near alone" and "both drawn without any occlusion".
      """
      near = ("near", (6.0, 5.0, 2.0), 0.5)
      far = ("far", (9.0, 6.75, 2.0), 0.5)

      both = _ink(render_fpv_png(_view(visible=(far, near)), [], size=SIZE))
      near_alone = _ink(render_fpv_png(_view(visible=(near,)), [], size=SIZE))
      far_alone = _ink(render_fpv_png(_view(visible=(far,)), [], size=SIZE))

      assert near_alone < both < near_alone + far_alone


class TestModelSelection:
   """Tests for models: which mesh is drawn for which neighbor."""

   def _one(self, models, radius=0.5, drone_id="d"):
      return render_fpv_png(_view(visible=((drone_id, (6.0, 5.0, 2.0), radius),)), [], models=models, size=SIZE)

   def test_sphere_and_obj_produce_different_images(self):
      """Test the configured model actually reaches the rasteriser."""
      sphere = self._one({"d": SPHERE_MODEL})
      obj = self._one({"d": OBJ_MODEL})

      assert sphere != obj

   def test_swapping_the_model_of_the_same_neighbor_changes_the_image(self):
      """Test two different .obj files are two different silhouettes, not one mesh reused."""
      assert self._one({"d": OBJ_MODEL}) != self._one({"d": OTHER_OBJ_MODEL})

   def test_heterogeneous_fleet_draws_both_models_in_one_frame(self):
      """Test a sphere and a mesh at the same distance render side by side — the reason models is a mapping at all."""
      view = _view(visible=(("ball", (6.0, 4.6, 2.0), 0.3), ("mesh", (6.0, 5.4, 2.0), 0.3)))
      models = {"ball": SPHERE_MODEL, "mesh": OBJ_MODEL}

      mask = _ink_mask(render_fpv_png(view, [], models=models, size=SIZE))
      left, right = mask[:, : SIZE[0] // 2], mask[:, SIZE[0] // 2:]

      assert left.sum() > 0 and right.sum() > 0
      # Same radius, same distance, different mesh: the two halves cannot be mirror images of one another.
      assert left.sum() != right.sum()

   def test_no_models_argument_means_spheres(self):
      """Test the default keeps working for callers that have no model map (and for every current config)."""
      assert self._one(None) == self._one({"d": SPHERE_MODEL})

   def test_missing_entry_falls_back_to_a_sphere_and_warns(self, caplog):
      """Test a gap in a configured fleet is survivable but noisy — silently drawing the wrong shape would skew the detector."""
      _warn_unknown_model.cache_clear()

      with caplog.at_level(logging.WARNING):
         png = self._one({"someone_else": OBJ_MODEL}, drone_id="unmapped")

      assert png == self._one({"unmapped": SPHERE_MODEL})
      assert "unmapped" in caplog.text

   def test_missing_entry_warns_once_per_drone_not_once_per_frame(self, caplog):
      """Test the warning does not spam the log at capture rate."""
      _warn_unknown_model.cache_clear()

      with caplog.at_level(logging.WARNING):
         for _ in range(5):
            self._one({}, drone_id="repeat")

      assert caplog.text.count("repeat") == 1

   def test_obj_model_without_a_path_falls_back_to_a_sphere(self):
      """Test a hand-built model that the config layer would have rejected does not kill the worker thread.

      ``resolve_drone_model`` cannot produce this state — ``ScenarioConfig`` raises at load time instead (see test_config.py) — so the renderer
      only has to survive it.
      """
      assert self._one({"d": DroneModel(kind="obj", path=None)}) == self._one({"d": SPHERE_MODEL})


class TestRenderOutput:
   """Tests for the bytes handed to the detector."""

   def test_returns_png_bytes(self):
      """Test the output carries the PNG magic number."""
      assert render_fpv_png(_view(), [], size=SIZE).startswith(b"\x89PNG\r\n\x1a\n")

   def test_dimensions_match_requested_size(self):
      """Test the frame is exactly the requested pixel size."""
      assert _png_size(render_fpv_png(_view(), [], size=(320, 240))) == (320, 240)

   def test_custom_size_is_honored(self):
      """Test a non-default size comes out at that size."""
      assert _png_size(render_fpv_png(_view(), [], size=(160, 160))) == (160, 160)

   def test_dimensions_are_stable_across_frames(self):
      """Test scene content does not change the output shape — a detector assumes a fixed input shape."""
      empty = render_fpv_png(_view(visible=()), [], size=SIZE)
      crowded = render_fpv_png(_view(visible=(("d1", (8.0, 5.0, 2.0), 0.2), ("d2", (6.0, 5.5, 2.5), 0.3))), [], size=SIZE)

      assert _png_size(empty) == _png_size(crowded)

   def test_empty_view_renders_a_blank_frame(self):
      """Test a drone that sees nothing still produces an image — the detector gets one either way."""
      png = render_fpv_png(_view(visible=()), [], size=SIZE)

      assert png.startswith(b"\x89PNG\r\n\x1a\n")
      assert _ink(png) == 0

   def test_obstacles_render(self):
      """Test obstacles are accepted in the simulator's (center, half_extents) form and land in the frame."""
      obstacles = [(np.array([8.0, 5.0, 2.0]), np.array([0.5, 0.5, 1.0]))]

      assert _ink(render_fpv_png(_view(visible=()), obstacles, size=SIZE)) > 0

   def test_obstacle_behind_the_drone_is_not_drawn(self):
      """Test obstacles go through the same near-plane rejection as neighbors."""
      obstacles = [(np.array([2.0, 5.0, 2.0]), np.array([0.5, 0.5, 1.0]))]

      assert _ink(render_fpv_png(_view(visible=()), obstacles, size=SIZE)) == 0

   def test_spherical_fov_renders_and_warns_once(self, caplog):
      """Test an angle no pinhole can image is clamped rather than raising, and says so once."""
      _warn_wide_fov.cache_clear()

      with caplog.at_level(logging.WARNING):
         first = render_fpv_png(_view(fov_deg=360.0), [], size=SIZE)
         render_fpv_png(_view(fov_deg=360.0), [], size=SIZE)

      assert first.startswith(b"\x89PNG\r\n\x1a\n")
      assert caplog.text.count("clamping") == 1

   def test_wider_fov_shrinks_the_same_neighbor(self):
      """Test the FOV reaches the image: a wider lens spreads the same world over the same pixels, so objects get smaller."""
      narrow = _ink(render_fpv_png(_view(fov_deg=60.0, visible=(("d", (6.0, 5.0, 2.0), 0.5),)), [], size=SIZE))
      wide = _ink(render_fpv_png(_view(fov_deg=120.0, visible=(("d", (6.0, 5.0, 2.0), 0.5),)), [], size=SIZE))

      assert wide < narrow

   def test_zero_view_dir_renders(self):
      """Test a degenerate heading renders via the +x fallback."""
      assert render_fpv_png(_view(view_dir=(0.0, 0.0, 0.0)), [], size=SIZE).startswith(b"\x89PNG\r\n\x1a\n")

   def test_straight_up_view_renders(self):
      """Test the vertical heading, where the up reference has to switch, produces a frame with the neighbor in it."""
      view = _view(view_dir=(0.0, 0.0, 1.0), visible=(("d", (5.0, 5.0, 5.0), 0.5),))

      assert _ink(render_fpv_png(view, [], size=SIZE)) > 0

   def test_view_not_mutated(self):
      """Test rendering never writes into the View, in particular not image_png — the caller assigns that."""
      view = _view()
      position_before = np.array(view.visible[0].position)

      render_fpv_png(view, [], size=SIZE)

      assert view.image_png is None
      np.testing.assert_array_equal(view.visible[0].position, position_before)

   def test_render_is_deterministic(self):
      """Test the same view twice gives the same bytes — a detector's training set should not depend on dict iteration luck."""
      view = _view(visible=(("a", (7.0, 5.0, 2.0), 0.3), ("b", (6.0, 5.3, 2.2), 0.3)))
      models = {"a": OBJ_MODEL, "b": SPHERE_MODEL}

      assert render_fpv_png(view, [], models=models, size=SIZE) == render_fpv_png(view, [], models=models, size=SIZE)


class TestLazyImport:
   """Guards the perception package against dragging matplotlib into the simulation core."""

   def test_importing_the_package_does_not_pull_matplotlib(self):
      """Test `import drone_sim.perception` stays matplotlib-free.

      Not even ``simulation.simulator`` pulls matplotlib — the core is deliberately light, and only the render entry points pay for it. The mesh
      geometry is shared with the matplotlib overview through ``domain.mesh``, which is numpy-only precisely so that this holds. Runs in a
      subprocess, because matplotlib is long since loaded by the time this test executes.
      """
      src = Path(__file__).resolve().parents[3] / "src"
      probe = "import sys, drone_sim.perception; print('matplotlib' in sys.modules)"

      result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONPATH": str(src)})

      assert result.stdout.strip() == "False"
