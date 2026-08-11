"""What a drone looks like to a renderer — a sphere, or a mesh loaded from a Wavefront .obj file.

The point of resolving this here rather than in the renderer is *when* it happens. A scenario may name a model file that does not exist; the FPV
camera renders on the perception worker thread, once per drone per capture, where a missing file would surface as a log line per frame and nothing
else. So the path is resolved and checked while the config is being parsed, and the renderer receives only :class:`DroneModel` objects that are
already known to be loadable.

The model is configurable per drone with a scenario-wide default (see :class:`~drone_sim.domain.config.ScenarioConfig`), which is what allows a
heterogeneous fleet — several drone types in one run — without repeating the same entry N times for the homogeneous case.

Deliberately free of numpy, pydantic and any renderer import: both the config layer and the perception layer depend on this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DroneModelKind = Literal["sphere", "obj"]

# Bundled .obj/.mtl assets; a relative drone_model_path is looked up here first.
ASSETS_DIR = Path(__file__).resolve().parent.parent / "resources" / "assets"


@dataclass(frozen=True)
class DroneModel:
   """How a drone is drawn — resolved from the scenario default and the per-drone override.

   :param kind: ``"sphere"`` draws a parametric sphere of the drone's radius, ``"obj"`` the mesh at :attr:`path`.
   :param path: absolute path to an existing .obj file; always ``None`` for ``kind == "sphere"``.
   """

   kind: DroneModelKind = "sphere"
   path: Path | None = None


def find_asset(path: str | Path) -> Path | None:
   """Locate a model file: absolute paths as given, relative ones under :data:`ASSETS_DIR`, else relative to the working directory.

   Returns ``None`` when nothing exists at any of those, so callers can phrase their own error.
   """
   candidate = Path(path)
   if candidate.is_absolute():
      return candidate if candidate.is_file() else None

   in_assets = ASSETS_DIR / candidate
   if in_assets.is_file():
      return in_assets

   return candidate.resolve() if candidate.is_file() else None


def resolve_drone_model(kind: DroneModelKind, path: str | Path | None = None) -> DroneModel:
   """Turn a (kind, path) pair from the config into a checked :class:`DroneModel`.

   :raises ValueError: for an unknown kind, for ``"obj"`` without a path, or when the file cannot be found.
   """
   if kind == "sphere":
      # A path alongside kind="sphere" is dropped rather than rejected: it lets a drone opt out of a scenario-wide .obj default by naming only
      # the kind, and it keeps the scenario's default path harmless for every drone that stays a sphere.
      return DroneModel(kind="sphere")

   if kind != "obj":
      raise ValueError(f"unknown drone model kind {kind!r}, expected 'sphere' or 'obj'")

   if not path:
      raise ValueError("drone_model 'obj' requires drone_model_path")

   resolved = find_asset(path)
   if resolved is None:
      raise ValueError(f"drone_model_path {str(path)!r} not found (looked under {ASSETS_DIR} and relative to the working directory)")

   return DroneModel(kind="obj", path=resolved)
