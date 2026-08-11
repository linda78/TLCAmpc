"""Wavefront OBJ parsing and mesh normalisation — deliberately free of matplotlib.

Two renderers need this geometry: the matplotlib overview in :mod:`drone_sim.api.utils.render_helper` (GUI, ``/render``, ``tools/live_view.py``) and
the pinhole FPV camera in :mod:`drone_sim.perception.fpv_render`, which rasterises with PIL. They must agree on *what* the object is — same vertices,
same centring, same Blender-Y-up-to-sim-Z-up rotation — or the camera would show a different drone than the overview.

That shared source cannot live in ``render_helper``: it imports ``mpl_toolkits`` at module level, and pulling matplotlib into
``drone_sim.perception`` is exactly the regression the perception package guards against in its lazy-import test. Hence this module — numpy only.

Note that the two renderers deliberately differ in *size*: the FPV scales a drone to ``2 * radius`` because apparent size is its main depth cue,
while the overview treats model scale as cosmetics. That divergence is a documented decision, not drift.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# Blender exports Y-up; the simulation is Z-up. Applied around the model's own centre.
BLENDER_TO_SIM_ROTATION_DEG = (-90.0, 0.0, 0.0)

# A bounding box shorter than this in every axis carries no usable scale — normalising it would divide by ~0.
_MIN_EXTENT = 1e-12


@lru_cache(maxsize=8)
def load_obj(path: str) -> tuple[np.ndarray, list[tuple[int, ...]]]:
   """Load vertices and face indices from a Wavefront .obj file.

   Cached by path: the same model asked for twice is parsed once. Callers must treat the returned arrays as read-only — they are shared.

   Returns
   -------
   vertices : ndarray, shape (V, 3)
   faces    : list of tuples of 0-based vertex indices (triangles or quads)
   """
   verts: list[list[float]] = []
   faces: list[tuple[int, ...]] = []
   with open(path, "r") as fh:
      for line in fh:
         parts = line.strip().split()
         if not parts:
            continue
         if parts[0] == "v":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
         elif parts[0] == "f":
            # face indices can be v, v/vt, v/vt/vn, or v//vn
            idx = tuple(int(p.split("/")[0]) - 1 for p in parts[1:])
            faces.append(idx)
   return np.array(verts, dtype=float), faces


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
   """Build a 3x3 rotation matrix from Euler angles (degrees), applied X -> Y -> Z."""
   rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
   cx, sx = np.cos(rx), np.sin(rx)
   cy, sy = np.cos(ry), np.sin(ry)
   cz, sz = np.cos(rz), np.sin(rz)
   Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
   Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
   Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
   return Rz @ Ry @ Rx


def normalize_mesh(verts: np.ndarray, *, scale: float = 1.0, rotation_deg: tuple[float, float, float] = BLENDER_TO_SIM_ROTATION_DEG) -> np.ndarray | None:
   """Centre *verts* on the origin, resize so the longest axis spans *scale*, then rotate by *rotation_deg*.

   The result is still centred on the origin; translating it to a world position is the caller's job. ``None`` is returned for an empty or degenerate
   mesh (all vertices coincident), which callers treat as "nothing to draw" rather than an error — a malformed asset should not kill a render.

   :param verts: ``(V, 3)`` vertices as loaded from the file.
   :param scale: length of the longest bounding-box axis after normalisation, in world units.
   :param rotation_deg: Euler angles (rx, ry, rz) in degrees; the default converts Blender's Y-up to the simulation's Z-up.
   """
   verts = np.asarray(verts, dtype=float)
   if verts.size == 0:
      return None

   bbox_min = verts.min(axis=0)
   bbox_max = verts.max(axis=0)
   max_extent = float((bbox_max - bbox_min).max())
   if max_extent < _MIN_EXTENT:
      return None

   normed = (verts - (bbox_min + bbox_max) / 2) / max_extent * scale
   return (rotation_matrix(*rotation_deg) @ normed.T).T
