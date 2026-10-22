"""Tests for tools.live_view.

Only the ``--obj-name`` plumbing is covered here: the rest of the module steps a simulator and drives a matplotlib window, which is not worth
pinning down in a unit test. The flag is worth it because it has to do two things at once — behave exactly as before for the configs that exist
today, and stay out of the way of a scenario that deliberately mixes drone models.
"""

from __future__ import annotations

import pytest

from drone_sim.domain.config import ControllerSpec, DroneConfig, PhysicsSpec, ScenarioConfig
from tools.live_view import apply_obj_name

BUNDLED = "drone_costum_0_0_5.obj"
OTHER_BUNDLED = "drone_costum_0_1.obj"


def _scenario(drones=None, **kwargs) -> ScenarioConfig:
   return ScenarioConfig(
      physics=PhysicsSpec(type="linear_kinematics"),
      controller=ControllerSpec(type="mpc_agent"),
      drones=drones or [DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])],
      **kwargs,
   )


class TestApplyObjName:
   """Tests for apply_obj_name."""

   def test_without_a_name_the_scenario_is_untouched(self):
      """Test the common case costs nothing: no flag, no change, spheres as configured."""
      scenario = _scenario()

      assert apply_obj_name(scenario, None) is scenario
      assert apply_obj_name(scenario, "") is scenario

   def test_the_name_becomes_the_scenario_default(self):
      """Test the flag reaches every drone that has no model of its own — the behaviour existing configs rely on."""
      updated = apply_obj_name(_scenario(), BUNDLED)

      assert updated.drone_model == "obj"
      assert updated.drone_model_path == BUNDLED
      assert updated.drone_model_for(updated.drones[0]).path.name == BUNDLED

   def test_a_drone_with_its_own_model_keeps_it(self):
      """Test a heterogeneous scenario is not flattened by the command line.

      This is the whole reason the flag sets the *scenario* field rather than writing into the drones.
      """
      scenario = _scenario([
         DroneConfig(drone_id="inherits", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0]),
         DroneConfig(drone_id="own", start=[1.0, 1.0, 1.0], target=[5.0, 5.0, 5.0], drone_model="obj", drone_model_path=OTHER_BUNDLED),
      ])

      updated = apply_obj_name(scenario, BUNDLED)
      inherits, own = (updated.drone_model_for(d) for d in updated.drones)

      assert inherits.path.name == BUNDLED
      assert own.path.name == OTHER_BUNDLED

   def test_a_drone_configured_as_a_sphere_stays_a_sphere(self):
      """Test an explicit opt-out survives the flag too."""
      scenario = _scenario([DroneConfig(drone_id="ball", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0], drone_model="sphere")])

      updated = apply_obj_name(scenario, BUNDLED)

      assert updated.drone_model_for(updated.drones[0]).kind == "sphere"

   def test_the_original_scenario_is_not_mutated(self):
      """Test the update is a copy; the caller's parsed config keeps whatever it said."""
      scenario = _scenario()

      apply_obj_name(scenario, BUNDLED)

      assert scenario.drone_model == "sphere"
      assert scenario.drone_model_path is None

   def test_an_absolute_path_is_accepted(self, tmp_path):
      """Test a model outside the package can be named on the command line."""
      own = tmp_path / "my_drone.obj"
      own.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

      updated = apply_obj_name(_scenario(), str(own))

      assert updated.drone_model_for(updated.drones[0]).path == own

   def test_an_unknown_name_exits_with_a_message(self, capsys):
      """Test a typo is caught here rather than inside the first render, where it would be a log line per frame."""
      with pytest.raises(SystemExit):
         apply_obj_name(_scenario(), "definitely_not_a_model.obj")

      assert "--obj-name" in capsys.readouterr().err
