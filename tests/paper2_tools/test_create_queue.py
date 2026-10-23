"""Tests for paper2_tools.distributed.create_queue and the metric columns of paper2_tools.distributed.db.

The queue of the paper3 cost campaign is built from two calls with different drone ranges and controllers, and the
constant-density experiment goes into the same queue with --experiment density, so the CLI has to turn its arguments into
exactly the combinations the user named. --sql must produce the same rows as a script for the database console without a
connection. The DB layer keeps its own list of cost columns, which must match the cost row the runner produces.
"""

from __future__ import annotations

import pytest

from paper2_tools.distributed import create_queue, db
from paper2_tools.distributed.config import CONTROLLERS, COORDINATORS
from tools.utility.printer import COST_FIELDNAMES


class TestParseDroneRange:
   """Tests for parse_drone_range."""

   def test_inclusive_range(self):
      assert create_queue.parse_drone_range("2-10") == range(2, 11)
      assert create_queue.parse_drone_range("11-25") == range(11, 26)
      assert create_queue.parse_drone_range("7-7") == range(7, 8)

   @pytest.mark.parametrize("text", ["10-2", "0-5", "abc", "5", "2-"])
   def test_rejects_malformed_ranges(self, text):
      with pytest.raises(Exception):
         create_queue.parse_drone_range(text)


@pytest.fixture
def calls(monkeypatch):
   """Stub the three DB calls of main and record what reached them."""
   calls: dict = {}
   monkeypatch.setattr(create_queue, "ensure_schema", lambda: calls.setdefault("schema", True))
   monkeypatch.setattr(create_queue, "insert_tasks", lambda tasks: calls.update(tasks=tasks) or len(tasks))
   monkeypatch.setattr(create_queue, "count_by_status", lambda experiment=None: calls.update(status_of=experiment) or {})
   return calls


@pytest.fixture
def no_db(monkeypatch):
   """Fail the test if main touches the database."""
   monkeypatch.setattr(create_queue, "ensure_schema", lambda: pytest.fail("ensure_schema must not run on --dry-run or --sql"))
   monkeypatch.setattr(create_queue, "insert_tasks", lambda tasks: pytest.fail("insert_tasks must not run on --dry-run or --sql"))
   monkeypatch.setattr(create_queue, "count_by_status", lambda experiment=None: pytest.fail("count_by_status must not run on --dry-run or --sql"))


class TestMain:
   """Tests for the CLI wiring of main with the default (volume) experiment."""

   def test_first_campaign_call_takes_every_combination(self, calls):
      create_queue.main(["--runs", "20", "--drones", "2-10"])

      tasks = calls["tasks"]
      assert calls["schema"] is True
      assert len(tasks) == 20 * 9 * len(COORDINATORS) * len(CONTROLLERS)
      assert {t["run_index"] for t in tasks} == set(range(20))
      assert {t["n_drones"] for t in tasks} == set(range(2, 11))
      assert {t["coordinator"] for t in tasks} == set(COORDINATORS)
      assert {t["controller"] for t in tasks} == set(CONTROLLERS)
      assert all(t["experiment"] == "volume" and t["variant"] == "" for t in tasks)
      assert all(t["room_size"] is None and t["comm_radius"] is None and t["max_admm_iter"] is None and t["timeout_s"] is None for t in tasks)
      assert calls["status_of"] == "volume"

   def test_second_campaign_call_restricts_the_controller(self, calls):
      create_queue.main(["--runs", "10", "--drones", "11-25", "--controllers", "mpc_agent_adaptive"])

      tasks = calls["tasks"]
      assert len(tasks) == 10 * 15 * len(COORDINATORS)
      assert {t["n_drones"] for t in tasks} == set(range(11, 26))
      assert {t["coordinator"] for t in tasks} == set(COORDINATORS)
      assert {t["controller"] for t in tasks} == {"mpc_agent_adaptive"}

   def test_legacy_n_crit_still_means_two_to_n_crit_plus_one(self, calls):
      create_queue.main(["--runs", "3", "--n-crit", "35"])

      assert {t["n_drones"] for t in calls["tasks"]} == set(range(2, 37))

   def test_drones_and_n_crit_exclude_each_other(self, calls):
      with pytest.raises(SystemExit):
         create_queue.main(["--runs", "3", "--n-crit", "35", "--drones", "2-10"])

   def test_volume_needs_runs_and_a_drone_range(self, calls):
      with pytest.raises(SystemExit):
         create_queue.main(["--drones", "2-10"])
      with pytest.raises(SystemExit):
         create_queue.main(["--runs", "3"])

   def test_density_options_are_rejected_for_the_volume_experiment(self, calls):
      with pytest.raises(SystemExit):
         create_queue.main(["--runs", "3", "--drones", "2-10", "--admm-iters", "1"])

   def test_dry_run_prints_the_cells_and_touches_no_database(self, capsys, no_db):
      create_queue.main(["--runs", "2", "--drones", "2-3", "--dry-run"])

      out = capsys.readouterr().out
      assert "16 tasks" in out
      assert "mpc_central" in out and "dmpc_admm" in out
      assert "Inserted" not in out


class TestDensityExperiment:
   """Tests for --experiment density; the grid itself is tested in test_density_scaling."""

   def test_dry_run_prints_the_cells_and_touches_no_database(self, capsys, no_db):
      create_queue.main(["--experiment", "density", "--dry-run"])

      out = capsys.readouterr().out
      assert "110 tasks" in out
      assert "k1" in out and "k3" in out
      assert "16.00" in out  # room of N = 80

   def test_main_inserts_the_tasks_after_migrating(self, calls, capsys):
      create_queue.main(["--experiment", "density"])

      assert calls["schema"] is True
      assert len(calls["tasks"]) == 110
      assert all(t["experiment"] == "density" for t in calls["tasks"])
      assert calls["status_of"] == "density"
      assert "Inserted 110 new tasks" in capsys.readouterr().out

   def test_grid_options_reach_build_tasks(self, calls):
      create_queue.main(["--experiment", "density", "--runs", "3", "--dmpc-drones", "10,20", "--central-drones", "10",
                         "--admm-iters", "2", "--timeout-h", "0.5", "--comm-radius", "5.5"])

      tasks = calls["tasks"]
      assert len(tasks) == 3 * 2 + 3
      assert all(t["timeout_s"] == 1800 for t in tasks)
      assert {t["variant"] for t in tasks} == {"k2", ""}
      assert {t["comm_radius"] for t in tasks if t["coordinator"] == "dmpc_admm"} == {5.5}

   def test_volume_options_are_rejected_for_the_density_experiment(self, calls):
      with pytest.raises(SystemExit):
         create_queue.main(["--experiment", "density", "--drones", "2-10"])

   @pytest.mark.parametrize("text", ["", "a,b", "0,10", "-5"])
   def test_rejects_malformed_lists(self, text):
      with pytest.raises(SystemExit):
         create_queue.build_parser().parse_args(["--dmpc-drones", text])

   def test_tolerates_a_trailing_comma(self):
      assert create_queue.build_parser().parse_args(["--dmpc-drones", "10,20,"]).dmpc_drones == (10, 20)


class TestSqlScript:
   """Tests for --sql, which replaces the database round trip by a script for the console."""

   def test_sql_to_stdout_touches_no_database(self, capsys, no_db):
      create_queue.main(["--runs", "2", "--drones", "2-3", "--sql"])

      out, err = capsys.readouterr()
      assert "16 tasks" in err  # the cell table leaves stdout to the script
      assert "-- 16 tasks" in out and "-- Generated by: python -m paper2_tools.distributed.create_queue --runs 2 --drones 2-3 --sql" in out
      assert out.count("\n    (") == 16
      assert "INSERT INTO scenario_runs (" + ", ".join(db._TASK_COLUMNS) + ")\nVALUES\n" in out
      assert f"ON CONFLICT ON CONSTRAINT {db.TASK_KEY_CONSTRAINT} DO NOTHING;" in out
      assert "    (0, 2, 'mpc_central', 'mpc_agent', 'volume', '', NULL, NULL, NULL, NULL)," in out
      assert "    (1, 3, 'dmpc_admm', 'mpc_agent_adaptive', 'volume', '', NULL, NULL, NULL, NULL)\nON CONFLICT" in out
      assert out.index("BEGIN;") < out.index("INSERT INTO") < out.index("COMMIT;") < out.index(
         "SELECT status, count(*) FROM scenario_runs WHERE experiment = 'volume' GROUP BY status ORDER BY status;")
      assert "Inserted" not in out

   def test_sql_to_file_carries_the_density_settings(self, tmp_path, capsys, no_db):
      path = tmp_path / "queue.sql"
      create_queue.main(["--experiment", "density", "--runs", "1", "--dmpc-drones", "10", "--central-drones", "10", "--admm-iters", "3",
                         "--sql", str(path)])

      text = path.read_text()
      assert text.count("\n    (") == 2
      assert "    (0, 10, 'dmpc_admm', 'mpc_agent_adaptive', 'density', 'k3', 8.0, 7.04, 3, 14400)," in text
      assert "    (0, 10, 'mpc_central', 'mpc_agent_adaptive', 'density', '', 8.0, NULL, NULL, 14400)\nON CONFLICT" in text
      assert text.endswith("WHERE experiment = 'density' GROUP BY status ORDER BY status;\n")
      out, err = capsys.readouterr()
      assert "2 tasks" in out and str(path) in out and err == ""

   def test_sql_and_dry_run_exclude_each_other(self, no_db):
      with pytest.raises(SystemExit):
         create_queue.main(["--runs", "1", "--drones", "2-2", "--dry-run", "--sql"])

   def test_sql_still_validates_the_options(self, no_db):
      with pytest.raises(SystemExit):
         create_queue.main(["--drones", "2-10", "--sql"])


class TestInsertTasksSql:
   """Tests for the literal SQL of db.insert_tasks_sql and its helpers."""

   def test_literals(self):
      assert db.sql_literal(None) == "NULL"
      assert db.sql_literal(3) == "3"
      assert db.sql_literal(16.0) == "16.0"
      assert db.sql_literal(7.04) == "7.04"
      assert db.sql_literal("k1") == "'k1'"
      assert db.sql_literal("o'k") == "'o''k'"
      with pytest.raises(TypeError):
         db.sql_literal([1])

   def test_rows_follow_the_task_columns(self):
      task = db.make_task(1, 5, "dmpc_admm", "mpc_agent_adaptive", experiment="density", variant="k3", room_size=16.0, comm_radius=7.04,
                          max_admm_iter=3, timeout_s=14400)

      sql = db.insert_tasks_sql([task])

      assert sql.startswith("INSERT INTO scenario_runs (" + ", ".join(db._TASK_COLUMNS) + ")\nVALUES\n")
      assert "\n    (1, 5, 'dmpc_admm', 'mpc_agent_adaptive', 'density', 'k3', 16.0, 7.04, 3, 14400)\n" in sql
      assert sql.endswith(f"ON CONFLICT ON CONSTRAINT {db.TASK_KEY_CONSTRAINT} DO NOTHING;")

   def test_one_statement_per_page_like_execute_values(self):
      tasks = [db.make_task(run, 2, "dmpc_admm", "mpc_agent") for run in range(db._INSERT_PAGE_SIZE + 1)]

      sql = db.insert_tasks_sql(tasks)

      assert sql.count("INSERT INTO scenario_runs") == 2
      assert sql.count("DO NOTHING;") == 2
      assert sql.count("\n    (") == db._INSERT_PAGE_SIZE + 1
      assert db.insert_tasks_sql([]) == ""

   def test_status_query_matches_the_parameterised_one(self):
      assert db.count_by_status_sql() == "SELECT status, count(*) FROM scenario_runs GROUP BY status ORDER BY status;"
      assert db.count_by_status_sql("density") == "SELECT status, count(*) FROM scenario_runs WHERE experiment = 'density' GROUP BY status ORDER BY status;"
      assert db._COUNT_BY_STATUS_SQL.format(where=" WHERE experiment = %s") + ";" == db.count_by_status_sql("x").replace("'x'", "%s")


class TestMetricColumns:
   """Tests for the cost columns of the metrics table."""

   def test_cost_columns_match_the_runner_cost_row(self):
      assert [name for name, _ in db._COST_COLUMNS] == COST_FIELDNAMES

   def test_export_starts_with_the_insert_columns(self):
      """Test the export keeps the metric columns in insert order and appends the run settings and the task identity."""
      assert db._EXPORT_FIELDNAMES[:len(db._METRIC_COLUMNS)] == db._METRIC_COLUMNS
      assert db._EXPORT_FIELDNAMES[len(db._METRIC_COLUMNS):] == ["room_size", "comm_radius", "max_admm_iter", "run_index", "experiment", "variant"]
      assert db._METRIC_COLUMNS[-len(COST_FIELDNAMES):] == COST_FIELDNAMES

   def test_migration_adds_every_cost_column(self):
      statements = db._MIGRATION_SQL.splitlines()

      assert len(statements) == len(COST_FIELDNAMES)
      assert all(s.startswith("ALTER TABLE metrics ADD COLUMN IF NOT EXISTS ") for s in statements)
      assert [s.split("EXISTS ")[1].split(" ")[0] for s in statements] == COST_FIELDNAMES

   def test_migration_adds_the_task_columns(self):
      assert [s.split("EXISTS ")[1].split(" ")[0] for s in db._TASK_PARAM_MIGRATION_SQL.splitlines()] == db._TASK_PARAM_NAMES
      run_columns = [s.split("EXISTS ")[1].split(" ")[0] for s in db._RUNS_MIGRATION_SQL.splitlines()]
      assert run_columns == ["experiment", "variant", "room_size", "comm_radius", "max_admm_iter", "timeout_s"]
      assert "DROP CONSTRAINT IF EXISTS scenario_runs_run_index_n_drones_coordinator_controller_key" in db._TASK_KEY_MIGRATION_SQL
      assert "UNIQUE (run_index, n_drones, coordinator, controller, experiment, variant)" in db._TASK_KEY_MIGRATION_SQL

   def test_claim_sql_binds_the_hostname_where_needed(self):
      assert db._claim_sql(db._PACKAGE_OWNED_BY_HOST).count("%s") == 2
      assert db._claim_sql(db._PACKAGE_UNOWNED).count("%s") == 1

   def test_claim_returns_the_task_settings_and_packages_by_experiment(self):
      sql = db._claim_sql(db._PACKAGE_OWNED_BY_HOST)

      assert "RETURNING id, " + ", ".join(db._TASK_COLUMNS) in sql
      for package_filter in (db._PACKAGE_OWNED_BY_HOST, db._PACKAGE_UNOWNED):
         assert "o.experiment = r.experiment AND o.variant = r.variant" in package_filter


class TestTasks:
   """Tests for make_task and insert_runs."""

   def test_make_task_defaults_to_the_volume_campaign(self):
      task = db.make_task(3, 12, "dmpc_admm", "mpc_agent")

      assert list(task) == db._TASK_COLUMNS
      assert task["experiment"] == "volume" and task["variant"] == ""
      assert all(task[name] is None for name in ("room_size", "comm_radius", "max_admm_iter", "timeout_s"))

   def test_insert_runs_builds_every_combination(self, monkeypatch):
      captured: dict = {}
      monkeypatch.setattr(db, "insert_tasks", lambda tasks: captured.update(tasks=tasks) or len(tasks))

      inserted = db.insert_runs(range(2), range(2, 4), ["mpc_central", "dmpc_admm"], ["mpc_agent"])

      assert inserted == 2 * 2 * 2
      assert {(t["run_index"], t["n_drones"], t["coordinator"]) for t in captured["tasks"]} == \
             {(r, n, c) for r in range(2) for n in (2, 3) for c in ("mpc_central", "dmpc_admm")}
