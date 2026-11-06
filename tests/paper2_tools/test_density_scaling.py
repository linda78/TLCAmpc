"""Tests for paper2_tools.distributed.density and review_work.density_scaling.plot_density (todo.md block B).

The experiment is defined by a few formulas and a task list that the Hetzner workers take as they are, so the numbers in the
rows have to be exactly the ones the paper will cite: room side per N, comm radius, rounds per cell, timeout, runs. The
queue CLI (create_queue --experiment density) is tested in test_create_queue.
"""

from __future__ import annotations

import pandas as pd
import pytest

from paper2_tools.distributed import db
from paper2_tools.distributed.config import placement_seed
from paper2_tools.distributed import density
from paper2_tools.review_work.density_scaling import plot_density


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
      assert all(t["timeout_s"] == 12 * 3600 for t in tasks)
      assert all(t["room_size"] == pytest.approx(density.room_size_for(t["n_drones"])) for t in tasks)
      assert all(t["travel_distance"] == 4.0 for t in tasks)
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
      tasks = density.build_tasks(runs=2, dmpc_drones=(10,), central_drones=(), admm_iters=(5,), timeout_s=60, comm_radius=3.0, travel_distance=2.5)

      assert len(tasks) == 2
      assert all(t["travel_distance"] == 2.5 for t in tasks)
      assert all(t["max_admm_iter"] == 5 and t["variant"] == "k5" and t["timeout_s"] == 60 and t["comm_radius"] == 3.0 for t in tasks)


def _run(n: int, status: str = "finished", steps: int = 100, cpu: float = 100.0, travel: float | None = 4.0) -> dict:
   """One metrics row of a DMPC k=1 run; local solves take all the CPU time and there is one round per step."""
   return {"coordinator": "dmpc_admm", "max_admm_iter": 1, "num_drones": n, "room_size": density.room_size_for(n), "status": status, "steps": steps,
           "cpu_total_s": cpu, "wall_time_s": cpu * 2, "mean_step_time_s": cpu * 2 / max(steps, 1), "admm_iterations": steps, "local_solves": steps * n,
           "local_solve_cpu_s": cpu, "parallel_step_cpu_s": cpu / n, "travel_distance": travel}


class TestScalingTables:
   """Tests for cell_table and scaling_factors of plot_density."""

   def test_every_cell_is_a_row_even_if_nothing_finished(self):
      df = plot_density.add_derived(pd.DataFrame([_run(10), _run(10, "deadlock"), _run(20, "timeout", steps=7)]))

      table = plot_density.cell_table(df).set_index("num_drones")

      assert list(table.index) == [10, 20]
      assert table.loc[10, ["runs", "finished", "deadlock", "timeout"]].tolist() == [2, 1, 1, 0]
      assert table.loc[20, ["runs", "finished", "deadlock", "timeout"]].tolist() == [1, 0, 0, 1]
      assert pd.isna(table.loc[20, "run_time_cpu_s"])
      assert table.loc[20, "step_time_s"] == pytest.approx(100.0 / 7)

   def test_run_time_only_counts_finished_runs(self):
      df = plot_density.add_derived(pd.DataFrame([_run(10, cpu=100.0), _run(10, "timeout", cpu=9000.0)]))

      assert plot_density.cell_table(df)["run_time_cpu_s"].tolist() == [100.0]

   def test_factors_recover_the_exponent_of_a_power_law(self):
      """Test time ~ N^2: exponent 2, factor 4 per doubling, x40 = 16 against N = 10."""
      df = plot_density.add_derived(pd.DataFrame([_run(n, cpu=float(n) ** 2) for n in (10, 20, 40)]))

      factors = plot_density.scaling_factors(plot_density.cell_table(df)).set_index("quantity")

      row = factors.loc["CPU time of a finished run"]
      assert row["exponent"] == pytest.approx(2.0)
      assert row["per_doubling"] == pytest.approx(4.0)
      assert (row["x10"], row["x20"], row["x40"]) == pytest.approx((1.0, 4.0, 16.0))

   def test_flat_time_has_exponent_zero(self):
      df = plot_density.add_derived(pd.DataFrame([_run(n, cpu=float(n)) for n in (10, 20, 40, 80)]))

      factors = plot_density.scaling_factors(plot_density.cell_table(df)).set_index("quantity")

      assert factors.loc["CPU time per local solve", "exponent"] == pytest.approx(0.0, abs=1e-9)

   def test_a_series_with_one_cell_has_no_factor(self):
      df = plot_density.add_derived(pd.DataFrame([_run(10)]))

      assert plot_density.scaling_factors(plot_density.cell_table(df)).empty

   def test_rows_without_travel_distance_are_reported_as_mirrored(self):
      rows = pd.DataFrame([_run(10), _run(10, travel=None)])

      assert plot_density.mirrored_rows(rows) == 1
      assert plot_density.mirrored_rows(rows.drop(columns="travel_distance")) == 2
