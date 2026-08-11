"""Tests for drone_sim.api.utils.obj_loader — the geometry the overview renderer and the FPV camera share.

Two things matter here beyond "does it parse": the module must stay matplotlib-free (otherwise importing it from the perception package drags
matplotlib into the simulation core), and ``normalize_mesh`` must keep producing exactly what ``draw_obj_mesh`` produced before the extraction —
a scale or rotation change would silently move every drone in both renderers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drone_sim.api.utils.obj_loader import BLENDER_TO_SIM_ROTATION_DEG, load_obj, normalize_mesh, rotation_matrix

ASSETS_DIR = Path(__file__).resolve().parents[3] / "src" / "drone_sim" / "resources" / "assets"


def _write_obj(tmp_path: Path, body: str, name: str = "mesh.obj") -> str:
   path = tmp_path / name
   path.write_text(body)
   return str(path)


class TestLoadObj:
   """Tests for load_obj."""

   def test_parses_vertices_and_triangle_faces(self, tmp_path):
      """Test vertices come back as an (V, 3) array and face indices are 0-based."""
      path = _write_obj(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

      verts, faces = load_obj(path)

      np.testing.assert_allclose(verts, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
      assert faces == [(0, 1, 2)]

   def test_parses_quads_and_index_forms(self, tmp_path):
      """Test faces may be quads and may carry texture/normal indices (v, v/vt, v//vn, v/vt/vn)."""
      body = ("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
              "f 1/1 2/2/2 3//3 4\n")
      path = _write_obj(tmp_path, body)

      _, faces = load_obj(path)

      assert faces == [(0, 1, 2, 3)]

   def test_ignores_blank_lines_and_other_directives(self, tmp_path):
      """Test comments, material references and empty lines do not become geometry."""
      body = ("# a comment\n\nmtllib drone.mtl\nusemtl body\n"
              "v 0 0 0\nvn 0 0 1\nvt 0.5 0.5\nv 1 0 0\nv 0 1 0\n\ns off\nf 1 2 3\n")
      path = _write_obj(tmp_path, body)

      verts, faces = load_obj(path)

      assert verts.shape == (3, 3)
      assert faces == [(0, 1, 2)]

   def test_file_without_geometry_yields_empty_arrays(self, tmp_path):
      """Test an OBJ with no v/f lines returns empties rather than raising — callers check and skip."""
      path = _write_obj(tmp_path, "# nothing here\n")

      verts, faces = load_obj(path)

      assert verts.size == 0
      assert faces == []

   def test_result_is_cached_per_path(self, tmp_path):
      """Test the same path is parsed once — the FPV loads meshes per frame and must not re-read the file each time."""
      path = _write_obj(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

      first_verts, first_faces = load_obj(path)
      second_verts, second_faces = load_obj(path)

      assert first_verts is second_verts
      assert first_faces is second_faces

   def test_different_paths_are_cached_separately(self, tmp_path):
      """Test a heterogeneous fleet gets one mesh per model, not the first model twice."""
      small = _write_obj(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", name="small.obj")
      large = _write_obj(tmp_path, "v 0 0 0\nv 4 0 0\nv 0 4 0\nf 1 2 3\n", name="large.obj")

      small_verts, _ = load_obj(small)
      large_verts, _ = load_obj(large)

      assert not np.allclose(small_verts, large_verts)

   def test_loads_a_real_asset(self):
      """Test the bundled drone model parses — guards the asset path and the parser against each other."""
      verts, faces = load_obj(str(ASSETS_DIR / "drone_costum_0_0_5.obj"))

      assert verts.shape[1] == 3
      assert len(verts) > 0
      assert len(faces) > 0
      assert all(max(face) < len(verts) for face in faces)


class TestRotationMatrix:
   """Tests for rotation_matrix."""

   def test_zero_angles_give_identity(self):
      """Test no rotation is the identity."""
      np.testing.assert_allclose(rotation_matrix(0.0, 0.0, 0.0), np.eye(3), atol=1e-12)

   def test_matrix_is_orthonormal(self):
      """Test the result is a pure rotation: R @ R.T == I and det == +1 (no scaling, no mirroring)."""
      R = rotation_matrix(23.0, -47.0, 91.0)

      np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
      assert float(np.linalg.det(R)) == pytest.approx(1.0)

   def test_blender_rotation_takes_the_model_out_of_y_up(self):
      """Test the default (-90, 0, 0) swaps the model's Y and Z axes, which is what converts Blender's Y-up to the sim's Z-up."""
      R = rotation_matrix(*BLENDER_TO_SIM_ROTATION_DEG)

      np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-12)
      np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0], atol=1e-12)
      np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)

   def test_axes_are_applied_x_then_y_then_z(self):
      """Test the composition order is Rz @ Ry @ Rx, not the reverse — the two differ for non-commuting angles."""
      expected = rotation_matrix(0.0, 0.0, 90.0) @ rotation_matrix(90.0, 0.0, 0.0)

      np.testing.assert_allclose(rotation_matrix(90.0, 0.0, 90.0), expected, atol=1e-12)


class TestNormalizeMesh:
   """Tests for normalize_mesh — this is the shared source of truth for model size and orientation."""

   def test_longest_axis_spans_scale(self):
      """Test the bounding box's longest axis measures exactly *scale* afterwards.

      This is the contract the FPV depends on for depth: it passes ``2 * radius`` so apparent size encodes distance.
      """
      verts = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])

      normed = normalize_mesh(verts, scale=3.0)

      extents = normed.max(axis=0) - normed.min(axis=0)
      assert float(extents.max()) == pytest.approx(3.0)

   def test_shorter_axes_keep_their_proportions(self):
      """Test normalisation is uniform: the model is resized, not squashed into a cube."""
      verts = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])

      normed = normalize_mesh(verts, scale=4.0, rotation_deg=(0.0, 0.0, 0.0))

      extents = normed.max(axis=0) - normed.min(axis=0)
      np.testing.assert_allclose(extents, [4.0, 1.0, 2.0])

   def test_mesh_is_centred_on_the_origin(self):
      """Test the bounding box centre lands at (0, 0, 0) — callers translate it to the drone position themselves."""
      verts = np.array([[10.0, 20.0, 30.0], [14.0, 21.0, 32.0]])

      normed = normalize_mesh(verts, scale=1.0)

      np.testing.assert_allclose((normed.max(axis=0) + normed.min(axis=0)) / 2, [0.0, 0.0, 0.0], atol=1e-12)

   def test_rotation_is_applied(self):
      """Test the default rotation reorients the mesh: a Y-extended model comes out extended in Z."""
      verts = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])

      normed = normalize_mesh(verts, scale=4.0)

      extents = normed.max(axis=0) - normed.min(axis=0)
      np.testing.assert_allclose(extents, [0.0, 0.0, 4.0], atol=1e-12)

   def test_empty_mesh_returns_none(self):
      """Test an empty vertex array is "nothing to draw", not a crash — a malformed asset must not kill a render."""
      assert normalize_mesh(np.zeros((0, 3)), scale=1.0) is None

   def test_degenerate_mesh_returns_none(self):
      """Test coincident vertices return None instead of dividing by a ~zero extent."""
      assert normalize_mesh(np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]), scale=1.0) is None

   def test_matches_the_transform_draw_obj_mesh_used_before_extraction(self):
      """Test the extracted helper reproduces the inlined maths byte for byte on a real asset.

      The old code centred, divided by the largest bbox extent, multiplied by *scale* and rotated — in that order. Recomputed here by hand so a
      refactor of normalize_mesh cannot quietly move every drone in the GUI and in ``/render``.
      """
      verts, _ = load_obj(str(ASSETS_DIR / "drone_costum_0_0_5.obj"))
      scale = 0.7

      bbox_min, bbox_max = verts.min(axis=0), verts.max(axis=0)
      inlined = (verts - (bbox_min + bbox_max) / 2) / (bbox_max - bbox_min).max() * scale
      inlined = (rotation_matrix(-90.0, 0.0, 0.0) @ inlined.T).T

      np.testing.assert_allclose(normalize_mesh(verts, scale=scale), inlined)


class TestNoMatplotlibDependency:
   """The reason this module exists at all: the perception package must be able to import it."""

   def test_importing_obj_loader_does_not_pull_matplotlib(self):
      """Test the module stays numpy-only.

      Runs in a subprocess because matplotlib is long since imported by the time this test executes. If this fails, the FPV renderer can no longer
      share geometry with the overview without reintroducing the matplotlib import into ``drone_sim.perception``.
      """
      src = Path(__file__).resolve().parents[3] / "src"
      probe = "import sys, drone_sim.api.utils.obj_loader; print('matplotlib' in sys.modules)"

      result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONPATH": str(src)})

      assert result.stdout.strip() == "False"
