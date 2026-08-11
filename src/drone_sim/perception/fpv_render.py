"""FPV rendering: turns a :class:`~drone_sim.perception.camera.CameraView` into PNG bytes with a real pinhole camera.

This is what the external detector actually gets to look at, and the detector's job is to recover 3D positions from it. That is only possible if the
image carries the depth information a real lens carries, which means one thing above all: **the eye sits on the drone**, and apparent size falls off
with 1/distance. The predecessor of this module drew the scene with matplotlib's ``Axes3D``, which places the eye outside the data box at a distance
tied to the box size — measured on one scene, 66 m behind a drone that could see 8 m. Objects 1 m and 8 m ahead came out at an area ratio of 1.17
instead of 64, and neighbors *behind* the drone were drawn as if in front of it. matplotlib's 3D axes is a plot, not a camera; hence the rewrite.

The model here is the textbook one. World points go into camera coordinates against an orthonormal basis, points at or behind the image plane are
discarded, and the rest divide by depth::

    d = p - eye;  x = d·right,  y = d·up,  z = d·forward
    f_px = (width / 2) / tan(fov / 2)
    u = cx + f_px · x/z,   v = cy - f_px · y/z

Everything — neighbor spheres, .obj meshes, obstacle boxes — goes through the same path: mesh → world → camera → pixels → filled polygon, drawn
back to front (painter's algorithm). Occlusion between and inside objects therefore comes for free: a near neighbor hides a far one, as a camera
would show it.

What this is *not*, deliberately:

- **Not a photograph.** No lens distortion, no motion blur, no texture, no UVDAR-style blinking markers; lighting is a single fixed directional term
  so that surface structure is visible at all. A detector tuned only against these images will need retuning against real footage.
- **Not a second visibility test.** ``CameraModel`` tests against a *circular* cone of ``fov_deg``; an image is *rectangular*. ``fov_deg`` is read
  here as the **horizontal** field of view and the vertical one follows from the aspect ratio. Near the frame edges the two disagree — a neighbor
  can pass the cone test and still fall just outside the image. ``view.visible`` remains the ground truth; this module only draws.
- **Not able to show a full sphere.** A pinhole cannot image 180° or more (``tan`` diverges), but ``camera_fov_deg = 360`` is a valid config. Such
  values are clamped to 179° with a one-time warning rather than raising: the simulation should not die over a sensing-range setting. The image then
  shows only the forward sector while ``view.visible`` still reports the full sphere.
- **Not a z-buffer.** Painter's algorithm sorts whole faces by mean depth, which is wrong for interpenetrating geometry. Separate drones and boxes
  are not such a case.
- **Not near-plane clipped.** A face with any vertex at or behind the eye is dropped whole instead of being cut. At a few hundred faces per drone the
  error is a sliver at the frame edge; real clipping would be much more code for it.

No room wireframe is drawn (deliberate): fewer straight lines that a detector could mistake for object edges.

Threading: a fresh ``Image`` per call and no module-level mutable state beyond caches, so this is safe on the perception worker thread. Pillow is
imported inside the function for the same reason matplotlib was — ``import drone_sim.perception`` stays cheap.
"""
from __future__ import annotations

import io
import logging
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from drone_sim.domain.drone_model import DroneModel

if TYPE_CHECKING:
   from drone_sim.perception.camera import CameraView

_log = logging.getLogger(__name__)

# Output size in pixels. PIL rasterises directly, so there is no dpi to translate through.
_DEFAULT_SIZE = (320, 240)

# Below this length a view direction carries no heading; fall back to +x like CameraModel does.
_DIR_EPS = 1e-9

# Fallback heading, mirrors CameraModel._FALLBACK_VIEW_DIR.
_FALLBACK_VIEW_DIR = np.array([1.0, 0.0, 0.0])

# Up reference for the camera roll. Fixed to world up, i.e. the camera is treated as gimbal-stabilised — drones carry no orientation in their state
# (x = [px,py,pz,vx,vy,vz]), so there is no roll to derive from anywhere.
_WORLD_UP = np.array([0.0, 0.0, 1.0])

# When the view direction is this close to vertical, world up is useless as a reference and +x takes over.
_UP_PARALLEL_EPS = 1e-3

# A point must be at least this far in front of the image plane to be projected; at or behind it, the division by depth is meaningless.
_NEAR_EPS = 1e-4

# Widest angle a pinhole can still image. Anything wider is clamped to this (see the module docstring).
_MAX_FOV_DEG = 179.0

# Latitude/longitude resolution of the neighbor sphere. Same parametrisation as draw_sphere_wireframe, so the sphere is the same object in the
# overview and in the camera.
_SPHERE_RESOLUTION = 24

# Surfaces are flat-shaded against one fixed world-space light plus an ambient term, purely so that a mesh reads as a shape instead of a silhouette.
# The light is world-fixed rather than attached to the camera, and carries no distance information — apparent size is the depth cue, not brightness.
_LIGHT_DIR = np.array([0.3, 0.4, 1.0]) / np.linalg.norm([0.3, 0.4, 1.0])
_AMBIENT = 0.35

_BACKGROUND = (255, 255, 255)
_DRONE_COLOR = (58, 102, 168)
_OBSTACLE_COLOR = (188, 72, 60)


@lru_cache(maxsize=None)
def _warn_unknown_model(drone_id: str) -> None:
   """Warn once per drone id about a neighbor missing from the model map, rather than once per frame."""
   _log.warning("FPV render: no model configured for neighbor %r, drawing a sphere", drone_id)


@lru_cache(maxsize=None)
def _warn_unusable_model(path: str) -> None:
   """Warn once per path about an "obj" model that cannot be drawn, rather than once per frame."""
   _log.warning("FPV render: drone model %s has no usable geometry, drawing a sphere", path)


@lru_cache(maxsize=None)
def _warn_wide_fov(fov_deg: float) -> None:
   """Warn once per distinct angle that the cone is too wide for a pinhole and has been clamped."""
   _log.warning("FPV render: camera_fov_deg=%.1f exceeds what a pinhole can image, clamping the picture to %.1f degrees; visibility is unaffected",
                fov_deg, _MAX_FOV_DEG)


def _camera_basis(view_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   """Build the orthonormal camera basis ``(right, up, forward)`` for a look direction.

   The camera has **no roll**: up is world up orthogonalised against the view direction, which is what a gimbal-stabilised camera gives and the only
   defensible choice while drones carry no orientation. Looking (near) straight up or down leaves world up unusable as a reference, so ``+x`` takes
   over there — the resulting roll about the vertical axis is arbitrary but continuous within a run.

   :param view_dir: ``(3,)`` direction the drone is looking; need not be normalised. A zero vector falls back to ``+x``.
   :return: ``(right, up, forward)``, each a ``(3,)`` unit vector, right-handed as ``right = forward × up_ref``.
   """
   direction = np.asarray(view_dir, dtype=float).reshape(3)
   norm = float(np.linalg.norm(direction))
   forward = direction / norm if norm >= _DIR_EPS else _FALLBACK_VIEW_DIR.copy()

   up_ref = _WORLD_UP
   if abs(float(np.dot(forward, up_ref))) > 1.0 - _UP_PARALLEL_EPS:
      up_ref = _FALLBACK_VIEW_DIR

   right = np.cross(forward, up_ref)
   right = right / float(np.linalg.norm(right))
   up = np.cross(right, forward)
   return right, up, forward


def _focal_length_px(fov_deg: float, width: int) -> float:
   """Pinhole focal length in **pixels** for a horizontal field of view.

   :param fov_deg: Full horizontal opening angle in degrees; values at or beyond :data:`_MAX_FOV_DEG` are clamped (see the module docstring).
   :param width: Frame width in pixels.
   :return: Focal length in pixels, i.e. ``(width / 2) / tan(fov / 2)``.
   """
   clamped = min(float(fov_deg), _MAX_FOV_DEG)
   return (width / 2.0) / math.tan(math.radians(clamped / 2.0))


def _project(points_world: np.ndarray, eye: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray], f_px: float, cx: float,
             cy: float) -> tuple[np.ndarray, np.ndarray]:
   """Project world points to pixel coordinates.

   Points at or behind the image plane (``depth <= _NEAR_EPS``) get meaningless pixel values; callers must filter on the returned depth. That check
   is what keeps neighbors behind the drone out of the frame — with a 360° cone ``view.visible`` contains them, and the old renderer drew them.

   :param points_world: ``(N, 3)`` world positions.
   :param eye: ``(3,)`` camera position.
   :param basis: ``(right, up, forward)`` from :func:`_camera_basis`.
   :param f_px: Focal length in pixels.
   :param cx: Principal point x, normally ``width / 2``.
   :param cy: Principal point y, normally ``height / 2``.
   :return: ``(pixels (N, 2), depth (N,))``; v is flipped because the image origin is top left.
   """
   right, up, forward = basis
   offset = np.asarray(points_world, dtype=float).reshape(-1, 3) - np.asarray(eye, dtype=float).reshape(3)

   x = offset @ right
   y = offset @ up
   depth = offset @ forward

   safe = np.where(depth > _NEAR_EPS, depth, 1.0)
   u = cx + f_px * x / safe
   v = cy - f_px * y / safe
   return np.stack([u, v], axis=1), depth


@lru_cache(maxsize=1)
def _unit_sphere_mesh() -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
   """Unit-radius sphere as vertices plus quad faces, in the lat/long parametrisation of ``draw_sphere_wireframe``."""
   res = _SPHERE_RESOLUTION
   u = np.linspace(0.0, 2.0 * np.pi, res)
   v = np.linspace(0.0, np.pi, res)

   x = np.outer(np.cos(u), np.sin(v))
   y = np.outer(np.sin(u), np.sin(v))
   z = np.outer(np.ones_like(u), np.cos(v))
   verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

   faces = tuple((i * res + j, (i + 1) * res + j, (i + 1) * res + j + 1, i * res + j + 1) for i in range(res - 1) for j in range(res - 1))
   return verts, faces


@lru_cache(maxsize=8)
def _unit_obj_mesh(path: str) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]] | None:
   """Load an .obj and normalise it to a longest axis of 1, centred on the origin and rotated into the simulation's Z-up frame.

   Keyed by path, so a heterogeneous fleet holds one parsed mesh per model rather than one per drone. The per-drone size is a multiplication at
   draw time. ``None`` for a mesh with no faces or no extent.
   """
   from drone_sim.api.utils.obj_loader import load_obj, normalize_mesh

   verts, faces = load_obj(path)
   if not faces:
      return None

   normed = normalize_mesh(verts, scale=1.0)
   if normed is None:
      return None
   return normed, tuple(faces)


def _neighbor_mesh(model: DroneModel, radius: float) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
   """Mesh of one neighbor in world-sized units, still centred on the origin.

   An .obj is scaled so its **longest axis equals the drone's diameter**. That is a physical size, not a cosmetic one: the detector reads distance
   out of apparent size, so drawing the model at the ``radius * 5`` that ``tools/live_view.py`` uses for its overview would bias every distance
   estimate by a factor of 2.5 — and no test of relative size would notice. The FPV image and the overview therefore show the same object at
   different sizes on purpose (see R18 in .claude/plans/fpv-pinhole-render.md).

   Falls back to a sphere for an "obj" model that carries no path or no usable geometry. A model resolved through
   :func:`~drone_sim.domain.drone_model.resolve_drone_model` cannot be in that state — the config layer rejects it at load time — so this only
   catches a hand-built :class:`DroneModel`, and it warns rather than raising: a render on the perception worker thread is the wrong place to die.
   """
   if model.kind == "obj":
      mesh = _unit_obj_mesh(str(model.path)) if model.path is not None else None
      if mesh is not None:
         verts, faces = mesh
         return verts * (2.0 * radius), faces
      _warn_unusable_model(str(model.path))

   verts, faces = _unit_sphere_mesh()
   return verts * radius, faces


def _box_mesh(half_extents: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
   """Axis-aligned box as 8 corners and 6 quad faces, centred on the origin."""
   hx, hy, hz = (float(h) for h in np.asarray(half_extents, dtype=float).reshape(3))
   verts = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                     [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]])
   faces = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))
   return verts, faces


@lru_cache(maxsize=16)
def _face_groups(faces: tuple[tuple[int, ...], ...]) -> tuple[np.ndarray, ...]:
   """Bucket faces by vertex count into ``(F, k)`` index arrays — the rectangular shape numpy needs to shade a whole mesh in one go.

   Meshes mix triangles and quads, and the buckets depend only on the mesh, so they are cached alongside it. Faces with fewer than three vertices
   are dropped: they enclose no area.
   """
   by_arity: dict[int, list[tuple[int, ...]]] = {}
   for face in faces:
      if len(face) >= 3:
         by_arity.setdefault(len(face), []).append(face)
   return tuple(np.array(group, dtype=int) for _arity, group in sorted(by_arity.items()))


def _shade(color: tuple[int, int, int], face_verts: np.ndarray) -> np.ndarray:
   """Flat-shade a whole batch of faces against the fixed light.

   The dot product is taken in absolute value: .obj files in the wild wind their faces inconsistently, and a normal pointing the wrong way would
   otherwise turn a visible surface black.

   :param color: Base RGB of the object.
   :param face_verts: ``(F, k, 3)`` world vertices, one row per face.
   :return: ``(F, 3)`` integer RGB.
   """
   normals = np.cross(face_verts[:, 1] - face_verts[:, 0], face_verts[:, 2] - face_verts[:, 0])
   lengths = np.linalg.norm(normals, axis=1)
   lit = lengths > _DIR_EPS

   lambert = np.abs((normals / np.where(lit, lengths, 1.0)[:, None]) @ _LIGHT_DIR)
   intensity = np.where(lit, _AMBIENT + (1.0 - _AMBIENT) * lambert, _AMBIENT)
   return np.rint(np.asarray(color, dtype=float) * intensity[:, None]).astype(int)


def _collect_faces(verts_world: np.ndarray, faces: Sequence[tuple[int, ...]], color: tuple[int, int, int], eye: np.ndarray,
                   basis: tuple[np.ndarray, np.ndarray, np.ndarray], f_px: float, cx: float, cy: float, width: int,
                   height: int) -> list[tuple[float, list[float], tuple[int, int, int]]]:
   """Project one object and turn its faces into drawable ``(depth, flat polygon, color)`` entries.

   Faces are dropped when any vertex is at or behind the image plane, and when the projected polygon misses the frame entirely. Everything up to
   that point is done per mesh rather than per face — a drone is a few hundred faces and this runs once per drone per capture.
   """
   pixels, depth = _project(verts_world, eye, basis, f_px, cx, cy)

   collected: list[tuple[float, list[float], tuple[int, int, int]]] = []
   for idx in _face_groups(tuple(faces)):
      face_depth = depth[idx]
      poly = pixels[idx]
      u, v = poly[..., 0], poly[..., 1]

      keep = ((face_depth.min(axis=1) > _NEAR_EPS) & (u.max(axis=1) >= 0.0) & (u.min(axis=1) <= width) & (v.max(axis=1) >= 0.0)
              & (v.min(axis=1) <= height))
      if not keep.any():
         continue

      kept_idx = idx[keep]
      # Flat [x0, y0, x1, y1, ...] is what ImageDraw.polygon wants, and .tolist() builds it in one C-level pass.
      polygons = poly[keep].reshape(int(keep.sum()), -1).tolist()
      depths = face_depth[keep].mean(axis=1).tolist()
      colors = _shade(color, verts_world[kept_idx]).tolist()

      collected.extend(zip(depths, polygons, (tuple(c) for c in colors)))
   return collected


def render_fpv_png(view: CameraView, obstacles: list[tuple[np.ndarray, np.ndarray]], *, models: Mapping[str, DroneModel] | None = None,
                   size: tuple[int, int] = _DEFAULT_SIZE) -> bytes:
   """Render what one drone sees through a pinhole camera at its own position, as PNG bytes.

   Drawn: one neighbor per entry in ``view.visible`` — a sphere of its radius, or the .obj mesh configured for it — plus the obstacle boxes. The
   output is **exactly** ``size`` pixels every call, so a detector may assume a fixed input shape.

   ``view.visible`` decides *which* neighbors are candidates; the projection decides which of those actually land in the frame. Neighbors behind the
   camera (possible with a wide cone) are dropped here, and near the frame edges the rectangular image and the circular cone disagree slightly — see
   the module docstring.

   :param view: Capture to render. Read-only; the caller assigns the result to ``view.image_png``.
   :param obstacles: ``(center, half_extents)`` pairs, as carried by the simulator.
   :param models: ``drone_id -> DroneModel``, already resolved at config load. ``None``, or a neighbor missing from it, means a sphere; a missing
      entry is warned about once per drone id, because a configured fleet should not have gaps.
   :param size: ``(width, height)`` of the output in pixels.
   :return: PNG bytes.
   """
   # Pillow is imported here rather than at module level for the same reason matplotlib was: `import drone_sim.perception` must stay cheap enough
   # for the simulation core, which never renders anything.
   from PIL import Image, ImageDraw

   width, height = int(size[0]), int(size[1])
   eye = np.asarray(view.position, dtype=float).reshape(3)
   basis = _camera_basis(view.view_dir)
   f_px = _focal_length_px(float(view.fov_deg), width)
   cx, cy = width / 2.0, height / 2.0

   if float(view.fov_deg) > _MAX_FOV_DEG:
      _warn_wide_fov(float(view.fov_deg))

   faces: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []

   for visible in view.visible:
      model = (models or {}).get(visible.drone_id)
      if models is not None and model is None:
         _warn_unknown_model(visible.drone_id)
      verts, mesh_faces = _neighbor_mesh(model or DroneModel(), float(visible.radius))
      position = np.asarray(visible.position, dtype=float).reshape(3)
      faces.extend(_collect_faces(verts + position, mesh_faces, _DRONE_COLOR, eye, basis, f_px, cx, cy, width, height))

   for center, half_extents in obstacles:
      verts, box_faces = _box_mesh(half_extents)
      center = np.asarray(center, dtype=float).reshape(3)
      faces.extend(_collect_faces(verts + center, box_faces, _OBSTACLE_COLOR, eye, basis, f_px, cx, cy, width, height))

   image = Image.new("RGB", (width, height), _BACKGROUND)
   draw = ImageDraw.Draw(image)
   # Painter's algorithm: far faces first, so nearer ones paint over them. outline == fill closes the hairline seams PIL leaves between adjacent
   # polygons, without turning the object into a wireframe.
   for _depth, polygon, color in sorted(faces, key=lambda entry: entry[0], reverse=True):
      draw.polygon(polygon, fill=color, outline=color)

   buf = io.BytesIO()
   image.save(buf, format="PNG")
   png = buf.getvalue()

   if _log.isEnabledFor(logging.DEBUG):
      _log.debug("FPV render: observer=%s step=%d fov=%.1f f_px=%.1f | %d visible, %d faces -> %d bytes %dx%d", view.observer_id, view.step,
                 view.fov_deg, f_px, len(view.visible), len(faces), len(png), width, height)
   return png
