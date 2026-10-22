"""Tests for the placement seed of the distributed worker.

The worker seeds numpy with ``placement_seed(run_index, n_drones)`` right before ``create_scenario``, so every coordinator/controller
combination of one (run, N) pair draws the same start and target positions. Only that contract is covered here: the worker loop itself
needs a database.
"""

from __future__ import annotations

import numpy as np
import pytest

from paper2_tools.distributed.config import placement_seed
from tools.utility.scenario_creator import create_scenario

_SCENARIO = dict(horizon=4, dt=0.1, physics_u=3.0, physics_v=2.5, room_size=8.0, drones_radius=0.4)


def _positions(seed: int, coordinator: str, controller: str, n_drones: int) -> list[tuple[list[float], list[float]]]:
   np.random.seed(seed)
   adaptive = controller == "mpc_agent_adaptive"
   cfg = create_scenario(controller_type=controller, coordinator_type=coordinator, n_drones=n_drones,
                         alpha=0.5 if adaptive else None, safety_zone=1.0 if adaptive else 1.52, **_SCENARIO)
   return [(d.start, d.target) for d in cfg.drones]


class TestPlacementSeed:
   """Tests for placement_seed and its effect on create_scenario."""

   def test_unique_per_run_and_n(self):
      seeds = [placement_seed(run, n) for run in range(20) for n in range(2, 37)]
      assert len(set(seeds)) == len(seeds)

   @pytest.mark.parametrize("controller", ["mpc_agent_adaptive", "mpc_agent"])
   def test_same_seed_gives_same_positions_across_coordinators(self, controller):
      seed = placement_seed(3, 8)
      central = _positions(seed, "mpc_central", controller, 8)
      distributed = _positions(seed, "dmpc_admm", controller, 8)
      assert central == distributed

   def test_other_run_gives_other_positions(self):
      first = _positions(placement_seed(3, 8), "mpc_central", "mpc_agent_adaptive", 8)
      second = _positions(placement_seed(4, 8), "mpc_central", "mpc_agent_adaptive", 8)
      assert first != second
