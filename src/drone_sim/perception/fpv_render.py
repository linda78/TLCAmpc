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
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import compress
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import numpy as np

from drone_sim.domain.drone_model import DroneModel
from drone_sim.domain.mesh import SPHERE_RESOLUTION, load_obj, normalize_mesh, sphere_grid

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

# Surfaces are flat-shaded against one fixed world-space light plus an ambient term, purely so that a mesh reads as a shape instead of a silhouette.
# The light is world-fixed rather than attached to the camera, and carries no distance information — apparent size is the depth cue, not brightness.
_LIGHT_DIR = np.array([0.3, 0.4, 1.0]) / np.linalg.norm([0.3, 0.4, 1.0])
_AMBIENT = 0.35

_BACKGROUND = (255, 255, 255)
_DRONE_COLOR = (58, 102, 168)
_OBSTACLE_COLOR = (188, 72, 60)

Color: TypeAlias = tuple[int, int, int]

# One drawable face: mean depth for the painter's sort, the polygon as the flat [x0, y0, x1, y1, ...] that ImageDraw.polygon wants, and its colour.
_Face: TypeAlias = tuple[float, list[float], Color]


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


class _Camera(NamedTuple):
   """The pose and optics of one frame, resolved once in :func:`render_fpv_png` and passed around as a unit.

   Bundling these avoids a long positional tail of interchangeable floats and ints through the projection helpers. The principal point is derived
   rather than stored: nothing in this renderer models a shifted optical axis.

   :param eye: ``(3,)`` camera position — on the drone, which is the whole point of this module.
   :param basis: ``(right, up, forward)`` from :func:`_camera_basis`.
   :param f_px: Focal length in pixels.
   :param width: Frame width in pixels.
   :param height: Frame height in pixels.
   """

   eye: np.ndarray
   basis: tuple[np.ndarray, np.ndarray, np.ndarray]
   f_px: float
   width: int
   height: int

   @property
   def cx(self) -> float:
      return self.width / 2.0

   @property
   def cy(self) -> float:
      return self.height / 2.0


def _project(points_world: np.ndarray, cam: _Camera) -> tuple[np.ndarray, np.ndarray]:
   """Project world points to pixel coordinates.

   Points at or behind the image plane (``depth <= _NEAR_EPS``) get meaningless pixel values; callers must filter on the returned depth. That check
   is what keeps neighbors behind the drone out of the frame — with a 360° cone ``view.visible`` contains them, and the old renderer drew them.

   :param points_world: ``(N, 3)`` world positions.
   :param cam: Camera of this frame.
   :return: ``(pixels (N, 2), depth (N,))``; v is flipped because the image origin is top left.
   """
   right, up, forward = cam.basis
   offset = np.asarray(points_world, dtype=float).reshape(-1, 3) - np.asarray(cam.eye, dtype=float).reshape(3)

   x = offset @ right
   y = offset @ up
   depth = offset @ forward

   safe = np.where(depth > _NEAR_EPS, depth, 1.0)
   u = cam.cx + cam.f_px * x / safe
   v = cam.cy - cam.f_px * y / safe
   return np.stack([u, v], axis=1), depth


def _face_groups(faces: Sequence[tuple[int, ...]]) -> tuple[np.ndarray, ...]:
   """Bucket faces by vertex count into ``(F, k)`` index arrays — the rectangular shape numpy needs to shade a whole mesh in one go.

   Meshes mix triangles and quads. Faces with fewer than three vertices are dropped: they enclose no area.
   """
   by_arity: dict[int, list[tuple[int, ...]]] = {}
   for face in faces:
      if len(face) >= 3:
         by_arity.setdefault(len(face), []).append(face)
   return tuple(np.array(group, dtype=int) for _arity, group in sorted(by_arity.items()))


def _shade(color: Color, face_verts: np.ndarray) -> list[Color]:
   """Flat-shade a whole batch of faces against the fixed light.

   The dot product is taken in absolute value: .obj files in the wild wind their faces inconsistently, and a normal pointing the wrong way would
   otherwise turn a visible surface black.

   :param color: Base RGB of the object.
   :param face_verts: ``(F, k, 3)`` vertices, one row per face.
   :return: One RGB tuple per face.
   """
   normals = np.cross(face_verts[:, 1] - face_verts[:, 0], face_verts[:, 2] - face_verts[:, 0])
   lengths = np.linalg.norm(normals, axis=1)
   lit = lengths > _DIR_EPS

   lambert = np.abs((normals / np.where(lit, lengths, 1.0)[:, None]) @ _LIGHT_DIR)
   intensity = np.where(lit, _AMBIENT + (1.0 - _AMBIENT) * lambert, _AMBIENT)
   return [tuple(rgb) for rgb in np.rint(np.asarray(color, dtype=float) * intensity[:, None]).astype(int).tolist()]


# eq=False keeps identity comparison and hashing: the fields include a numpy array, for which the generated __eq__/__hash__ would blow up.
@dataclass(frozen=True, eq=False)
class _Mesh:
   """A mesh prepared for drawing once and reused for every frame and every drone that shares it.

   The face buckets and the per-face colours are derived from the vertices alone, and — crucially — survive the per-drone transform: a mesh is only
   ever uniformly scaled and translated into world space, which leaves every face *normal direction* unchanged. So the shading can be computed here,
   on the untransformed mesh, instead of per face per frame.

   :param verts: ``(V, 3)`` vertices, centred on the origin, at the module's reference size (unit radius / unit longest axis).
   :param color: Base RGB every face is shaded from.
   """

   verts: np.ndarray
   faces: tuple[tuple[int, ...], ...]
   color: Color

   @cached_property
   def groups(self) -> tuple[tuple[np.ndarray, list[Color]], ...]:
      """Per face-arity bucket: the ``(F, k)`` vertex indices and the shaded colour of each of those faces."""
      return tuple((idx, _shade(self.color, self.verts[idx])) for idx in _face_groups(self.faces))


@lru_cache(maxsize=None)
def _sphere_mesh() -> _Mesh:
   """Unit-radius sphere as vertices plus quad faces, in the lat/long parametrisation shared with ``draw_sphere_wireframe``."""
   res = SPHERE_RESOLUTION
   x, y, z = sphere_grid(res)
   verts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

   faces = tuple((i * res + j, (i + 1) * res + j, (i + 1) * res + j + 1, i * res + j + 1) for i in range(res - 1) for j in range(res - 1))
   return _Mesh(verts=verts, faces=faces, color=_DRONE_COLOR)


@lru_cache(maxsize=None)
def _obj_mesh(path: str) -> _Mesh | None:
   """Load an .obj and normalise it to a longest axis of 1, centred on the origin and rotated into the simulation's Z-up frame.

   Keyed by path and unbounded: the key space is the set of models named in the scenario, resolved and validated at config load, so it cannot grow
   with time. A bounded cache would be worse than none here — a fleet with more distinct models than slots would evict and re-parse the file for
   every neighbor of every frame, which is exactly the heterogeneous case this is meant to serve. ``None`` for a mesh with no faces or no extent.
   """
   verts, faces = load_obj(path)
   if not faces:
      return None

   normed = normalize_mesh(verts, scale=1.0)
   if normed is None:
      return None
   return _Mesh(verts=normed, faces=tuple(faces), color=_DRONE_COLOR)


@lru_cache(maxsize=None)
def _box_mesh(half_extents: tuple[float, float, float]) -> _Mesh:
   """Axis-aligned box as 8 corners and 6 quad faces, centred on the origin. Cached because obstacles are static over a run."""
   hx, hy, hz = half_extents
   verts = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                     [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]])
   faces = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4))
   return _Mesh(verts=verts, faces=faces, color=_OBSTACLE_COLOR)


def _neighbor_mesh(model: DroneModel, radius: float) -> tuple[_Mesh, float]:
   """The shared mesh for one neighbor, plus the factor that scales it to that neighbor's physical size.

   An .obj is scaled so its **longest axis equals the drone's diameter**. That is a physical size, not a cosmetic one: the detector reads distance
   out of apparent size, so drawing the model at the ``radius * 5`` that ``tools/live_view.py`` uses for its overview would bias every distance
   estimate by a factor of 2.5 — and no test of relative size would notice. The FPV image and the overview therefore show the same object at
   different sizes on purpose (see R18 in .claude/plans/fpv-pinhole-render.md).

   Falls back to a sphere for an "obj" model that carries no path or no usable geometry. A model resolved through
   :func:`~drone_sim.domain.drone_model.resolve_drone_model` cannot be in that state — the config layer rejects it at load time — so this only
   catches a hand-built :class:`DroneModel`, and it warns rather than raising: a render on the perception worker thread is the wrong place to die.

   :return: ``(mesh, scale)``; the drawable vertices are ``mesh.verts * scale`` plus the neighbor position.
   """
   if model.kind == "obj":
      mesh = _obj_mesh(str(model.path)) if model.path is not None else None
      if mesh is not None:
         return mesh, 2.0 * radius
      _warn_unusable_model(str(model.path))

   return _sphere_mesh(), radius


def _collect_faces(mesh: _Mesh, scale: float, offset: np.ndarray, cam: _Camera) -> list[_Face]:
   """Place one mesh in the world, project it, and turn its faces into drawable entries.

   Faces are dropped when any vertex is at or behind the image plane, and when the projected polygon misses the frame entirely. Everything is done
   per arity bucket rather than per face — a drone is a few hundred faces and this runs once per drone per capture.

   :param mesh: Prepared mesh, carrying its own face buckets and colours.
   :param scale: Uniform factor applied to ``mesh.verts`` before translation.
   :param offset: ``(3,)`` world position the mesh is centred on.
   :param cam: Camera of this frame.
   """
   verts_world = mesh.verts * scale + np.asarray(offset, dtype=float).reshape(3)
   pixels, depth = _project(verts_world, cam)

   collected: list[_Face] = []
   for idx, colors in mesh.groups:
      face_depth = depth[idx]
      poly = pixels[idx]
      u, v = poly[..., 0], poly[..., 1]

      keep = ((face_depth.min(axis=1) > _NEAR_EPS) & (u.max(axis=1) >= 0.0) & (u.min(axis=1) <= cam.width) & (v.max(axis=1) >= 0.0)
              & (v.min(axis=1) <= cam.height))
      if not keep.any():
         continue

      # Flat [x0, y0, x1, y1, ...] is what ImageDraw.polygon wants, and .tolist() builds it in one C-level pass.
      kept = keep.tolist()
      polygons = poly[keep].reshape(int(keep.sum()), -1).tolist()
      depths = face_depth[keep].mean(axis=1).tolist()

      collected.extend(zip(depths, polygons, compress(colors, kept)))
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
   cam = _Camera(eye=np.asarray(view.position, dtype=float).reshape(3), basis=_camera_basis(view.view_dir),
                 f_px=_focal_length_px(float(view.fov_deg), width), width=width, height=height)

   if float(view.fov_deg) > _MAX_FOV_DEG:
      _warn_wide_fov(float(view.fov_deg))

   faces: list[_Face] = []

   for visible in view.visible:
      model = (models or {}).get(visible.drone_id)
      if models is not None and model is None:
         _warn_unknown_model(visible.drone_id)
      mesh, scale = _neighbor_mesh(model or DroneModel(), float(visible.radius))
      faces.extend(_collect_faces(mesh, scale, visible.position, cam))

   for center, half_extents in obstacles:
      half = np.asarray(half_extents, dtype=float).reshape(3)
      faces.extend(_collect_faces(_box_mesh((float(half[0]), float(half[1]), float(half[2]))), 1.0, center, cam))

   image = Image.new("RGB", (width, height), _BACKGROUND)
   draw = ImageDraw.Draw(image)
   # Painter's algorithm: far faces first, so nearer ones paint over them. outline == fill closes the hairline seams PIL leaves between adjacent
   # polygons, without turning the object into a wireframe.
   for _depth, polygon, color in sorted(faces, key=operator.itemgetter(0), reverse=True):
      draw.polygon(polygon, fill=color, outline=color)

   buf = io.BytesIO()
   image.save(buf, format="PNG")
   png = buf.getvalue()

   if _log.isEnabledFor(logging.DEBUG):
      _log.debug("FPV render: observer=%s step=%d fov=%.1f f_px=%.1f | %d visible, %d faces -> %d bytes %dx%d", view.observer_id, view.step,
                 view.fov_deg, cam.f_px, len(view.visible), len(faces), len(png), width, height)
   return png
