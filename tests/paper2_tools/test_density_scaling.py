"""Tests for paper2_tools.review_work.density_scaling (todo.md block B).

The experiment is defined by a few formulas and a task list that the Hetzner workers take as they are, so the numbers in the
rows have to be exactly the ones the paper will cite: room side per N, comm radius, rounds per cell, timeout, runs. The
queue CLI (create_queue --experiment density) is tested in test_create_queue.
"""

from __future__ import annotations

import pytest

from paper2_tools.distributed import db
from paper2_tools.distributed.config import placement_seed
from paper2_tools.review_work.density_scaling import density


class TestFormulas:
   """Tests for room_size_for, comm_radius_for and variant_for."""

   def test_room_side_follows_the_cube_root_of_n(self):
      assert density.room_size_for(10) == pytest.approx(8.0)
      assert density.room_size_for(20) == pytest.approx(10.08, abs=0.01)
      assert density.room_size_for(40) == pytest.approx(12.70, abs=0.01)
      assert density.room_size_for(80) == pytest.approx(16.0)

   @pytest.mark.parametrize("n", [10, 20, 40, 80])
   def test_density_stays_at_the_reference(self, n):
      assert n / density.room_size_for(n) ** 3 == pytest.approx(10 / 8.0 ** 3)

   def test_comm_radius_is_two_safety_zones_plus_the_horizon_travel(self):
      assert density.comm_radius_for() == pytest.approx(7.04)
      assert density.comm_radius_for(horizon=4) == pytest.approx(5.04)
      assert density.comm_radius_for(horizon=7) == pytest.approx(6.54)

   def test_variant_names_the_rounds_only_for_the_distributed_mpc(self):
      assert density.variant_for("dmpc_admm", 1) == "k1"
      assert density.variant_for("dmpc_admm", 3) == "k3"
      assert density.variant_for("mpc_central", None) == ""


class TestBuildTasks:
   """Tests for build_tasks."""

   def test_cell_layout_matches_the_todo(self):
      tasks = density.build_tasks()

      assert len(tasks) == 2 * 4 * 10 + 3 * 10
      cells = dict(density.cells(tasks))
      assert set(cells) == {("dmpc_admm", "k1", n) for n in (10, 20, 40, 80)} | {("dmpc_admm", "k3", n) for n in (10, 20, 40, 80)} | \
             {("mpc_central", "", n) for n in (10, 20, 40)}
      assert set(cells.values()) == {10}

   def test_every_task_carries_its_settings(self):
      tasks = density.build_tasks()

      assert all(t["experiment"] == "density" for t in tasks)
      assert all(t["controller"] == "mpc_agent_adaptive" for t in tasks)
      assert all(t["timeout_s"] == 4 * 3600 for t in tasks)
      assert all(t["room_size"] == pytest.approx(density.room_size_for(t["n_drones"])) for t in tasks)
      assert set(tasks[0]) == set(db._TASK_COLUMNS)

   def test_dmpc_tasks_limit_neighbourhood_and_rounds(self):
      dmpc = [t for t in density.build_tasks() if t["coordinator"] == "dmpc_admm"]

      assert all(t["comm_radius"] == pytest.approx(7.04) for t in dmpc)
      assert {t["max_admm_iter"] for t in dmpc} == {1, 3}
      assert all(t["variant"] == f"k{t['max_admm_iter']}" for t in dmpc)

   def test_central_tasks_couple_every_pair_and_have_no_rounds(self):
      central = [t for t in density.build_tasks() if t["coordinator"] == "mpc_central"]

      assert all(t["comm_radius"] is None and t["max_admm_iter"] is None and t["variant"] == "" for t in central)
      assert {t["n_drones"] for t in central} == {10, 20, 40}

   def test_both_architectures_share_the_placement_seed_per_cell(self):
      tasks = density.build_tasks()
      seeds = {(t["coordinator"], t["variant"], t["n_drones"], t["run_index"]): placement_seed(t["run_index"], t["n_drones"]) for t in tasks}

      for n in (10, 20, 40):
         for run in range(10):
            assert seeds[("mpc_central", "", n, run)] == seeds[("dmpc_admm", "k1", n, run)] == seeds[("dmpc_admm", "k3", n, run)]

   def test_overrides_change_the_grid(self):
      tasks = density.build_tasks(runs=2, dmpc_drones=(10,), central_drones=(), admm_iters=(5,), timeout_s=60, comm_radius=3.0)

      assert len(tasks) == 2
      assert all(t["max_admm_iter"] == 5 and t["variant"] == "k5" and t["timeout_s"] == 60 and t["comm_radius"] == 3.0 for t in tasks)
