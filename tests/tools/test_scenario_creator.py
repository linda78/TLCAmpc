"""Tests for tools.utility.scenario_creator.create_scenario.

The density experiment hands comm_radius and max_admm_iter through create_scenario; the central MPC coordinator takes no comm
radius, so it must stay None there, and the defaults must stay what the volume campaigns ran with.
"""

from __future__ import annotations

import pytest

from tools.utility.scenario_creator import create_scenario

_WAYPOINTS = [[[1.0, 1.0, 1.0], [6.0, 6.0, 6.0]], [[6.0, 1.0, 1.0], [1.0, 6.0, 6.0]]]


def _scenario(coordinator_type: str, room_size: float = 8.0, **kwargs):
   return create_scenario(horizon=4, controller_type="mpc_agent_adaptive", coordinator_type=coordinator_type, room_size=room_size, n_drones=2,
                          waypoints=_WAYPOINTS, **kwargs)


class TestCreateScenario:
   """Tests for the DMPC parameters of create_scenario."""

   def test_defaults_are_the_volume_campaign_values(self):
      cfg = _scenario("dmpc_admm")

      assert cfg.comm_radius is None
      assert cfg.coordinator.params["max_admm_iter"] == 20

   def test_distributed_takes_comm_radius_and_rounds(self):
      cfg = _scenario("dmpc_admm", comm_radius=7.04, max_admm_iter=3)

      assert cfg.comm_radius == pytest.approx(7.04)
      assert cfg.coordinator.params["max_admm_iter"] == 3
      assert cfg.coordinator.params["horizon"] == 4

   def test_central_ignores_both(self):
      cfg = _scenario("mpc_central", comm_radius=7.04, max_admm_iter=3)

      assert cfg.comm_radius is None
      assert cfg.coordinator.params == {"horizon": 4, "terminal_constraint": True}

   def test_terminal_constraint_defaults_on_and_reaches_both_coordinators(self):
      assert _scenario("dmpc_admm").coordinator.params["terminal_constraint"] is True
      assert _scenario("mpc_central").coordinator.params["terminal_constraint"] is True
      assert _scenario("dmpc_admm", terminal_constraint=False).coordinator.params["terminal_constraint"] is False
      assert _scenario("mpc_central", terminal_constraint=False).coordinator.params["terminal_constraint"] is False

   def test_room_size_sets_the_bounds(self):
      cfg = _scenario("dmpc_admm", room_size=12.7)

      assert cfg.room.min == pytest.approx([-6.35] * 3)
      assert cfg.room.max == pytest.approx([6.35] * 3)
