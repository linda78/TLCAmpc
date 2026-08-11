"""Tests for drone_sim.domain.drone_model — resolving "which mesh, and does it exist" before anything tries to render it.

The behaviour under test is mostly about failing early and in the right place: the FPV camera renders on a worker thread once per drone per capture,
so a bad path must be caught while the config is parsed, not there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drone_sim.domain.drone_model import ASSETS_DIR, DroneModel, find_asset, resolve_drone_model

BUNDLED_ASSET = "drone_costum_0_0_5.obj"


class TestDroneModelDefaults:
   """Tests for the DroneModel dataclass itself."""

   def test_default_is_a_pathless_sphere(self):
      """Test the no-argument model is what a drone gets when nothing is configured."""
      model = DroneModel()

      assert model.kind == "sphere"
      assert model.path is None

   def test_is_frozen(self):
      """Test the model cannot be mutated after resolution — it is shared across frames and drones."""
      model = DroneModel()

      with pytest.raises(Exception):
         model.kind = "obj"  # type: ignore[misc]

   def test_equality_is_by_value(self):
      """Test two models resolved from the same config compare equal, so tests and caches can key on them."""
      assert DroneModel(kind="obj", path=Path("/a/b.obj")) == DroneModel(kind="obj", path=Path("/a/b.obj"))
      assert DroneModel(kind="obj", path=Path("/a/b.obj")) != DroneModel(kind="obj", path=Path("/a/c.obj"))


class TestFindAsset:
   """Tests for find_asset path lookup."""

   def test_relative_name_resolves_under_assets_dir(self):
      """Test a bare file name is looked up in resources/assets — the documented way to name a bundled model."""
      found = find_asset(BUNDLED_ASSET)

      assert found == ASSETS_DIR / BUNDLED_ASSET
      assert found.is_file()

   def test_absolute_path_is_used_as_given(self, tmp_path):
      """Test an absolute path outside the package is accepted, so users can point at their own models."""
      own = tmp_path / "my_drone.obj"
      own.write_text("v 0 0 0\n")

      assert find_asset(own) == own

   def test_relative_path_falls_back_to_the_working_directory(self, tmp_path, monkeypatch):
      """Test a relative path that is not a bundled asset is tried against the working directory before giving up."""
      nested = tmp_path / "models"
      nested.mkdir()
      (nested / "local.obj").write_text("v 0 0 0\n")
      monkeypatch.chdir(tmp_path)

      assert find_asset("models/local.obj") == (nested / "local.obj").resolve()

   def test_missing_file_returns_none(self):
      """Test a name that exists nowhere returns None rather than a path that would fail later."""
      assert find_asset("definitely_not_a_model.obj") is None

   def test_directory_is_not_an_asset(self):
      """Test a directory that happens to match the name is rejected — only files are models."""
      assert find_asset(str(ASSETS_DIR)) is None


class TestResolveDroneModel:
   """Tests for resolve_drone_model."""

   def test_sphere_needs_no_path(self):
      """Test the default kind resolves without touching the filesystem."""
      model = resolve_drone_model("sphere")

      assert model == DroneModel(kind="sphere", path=None)

   def test_sphere_drops_a_leftover_path(self):
      """Test a path is ignored for spheres.

      That is what lets a single drone opt out of a scenario-wide .obj default by naming only ``drone_model: "sphere"``, and it keeps the
      scenario's default path harmless for every drone that stays a sphere.
      """
      model = resolve_drone_model("sphere", BUNDLED_ASSET)

      assert model.kind == "sphere"
      assert model.path is None

   def test_obj_resolves_to_an_absolute_existing_path(self):
      """Test the renderer receives a path it can open directly, with no further lookup."""
      model = resolve_drone_model("obj", BUNDLED_ASSET)

      assert model.kind == "obj"
      assert model.path.is_absolute()
      assert model.path.is_file()

   def test_obj_without_path_raises(self):
      """Test "obj" is meaningless without a file and says so."""
      with pytest.raises(ValueError, match="drone_model_path"):
         resolve_drone_model("obj")

   def test_obj_with_empty_path_raises(self):
      """Test an empty string is treated as missing, not as a path to the working directory."""
      with pytest.raises(ValueError, match="drone_model_path"):
         resolve_drone_model("obj", "")

   def test_obj_with_missing_file_raises_and_names_the_search_locations(self):
      """Test the error mentions the offending name and where it was looked for — the user has to fix a path from this message alone."""
      with pytest.raises(ValueError, match="nope.obj") as excinfo:
         resolve_drone_model("obj", "nope.obj")

      assert str(ASSETS_DIR) in str(excinfo.value)

   def test_unknown_kind_raises(self):
      """Test an unexpected kind fails loudly instead of silently drawing nothing."""
      with pytest.raises(ValueError, match="unknown drone model kind"):
         resolve_drone_model("cylinder")  # type: ignore[arg-type]
