"""Tests for paper2_tools.distributed.create_queue and the metric columns of paper2_tools.distributed.db.

The queue of the paper3 cost campaign is built from two calls with different drone ranges and controllers, so the CLI has to
turn its arguments into exactly the combinations the user named. The DB layer keeps its own list of cost columns, which must
match the cost row the runner produces.
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


class TestMain:
   """Tests for the CLI wiring of main (DB calls are stubbed)."""

   @pytest.fixture
   def calls(self, monkeypatch):
      calls: dict = {}
      monkeypatch.setattr(create_queue, "ensure_schema", lambda: calls.setdefault("schema", True))
      monkeypatch.setattr(create_queue, "insert_runs", lambda runs, drones, coords, ctrls: calls.update(runs=runs, drones=drones, coords=coords, ctrls=ctrls) or 0)
      monkeypatch.setattr(create_queue, "count_by_status", lambda: {})
      return calls

   def test_first_campaign_call_takes_every_combination(self, calls):
      create_queue.main(["--runs", "20", "--drones", "2-10"])

      assert calls["schema"] is True
      assert calls["runs"] == range(20)
      assert calls["drones"] == range(2, 11)
      assert calls["coords"] == COORDINATORS
      assert calls["ctrls"] == CONTROLLERS

   def test_second_campaign_call_restricts_the_controller(self, calls):
      create_queue.main(["--runs", "10", "--drones", "11-25", "--controllers", "mpc_agent_adaptive"])

      assert calls["runs"] == range(10)
      assert calls["drones"] == range(11, 26)
      assert calls["coords"] == COORDINATORS
      assert calls["ctrls"] == ["mpc_agent_adaptive"]

   def test_legacy_n_crit_still_means_two_to_n_crit_plus_one(self, calls):
      create_queue.main(["--runs", "3", "--n-crit", "35"])

      assert calls["drones"] == range(2, 37)

   def test_drones_and_n_crit_exclude_each_other(self, calls):
      with pytest.raises(SystemExit):
         create_queue.main(["--runs", "3", "--n-crit", "35", "--drones", "2-10"])


class TestMetricColumns:
   """Tests for the cost columns of the metrics table."""

   def test_cost_columns_match_the_runner_cost_row(self):
      assert [name for name, _ in db._COST_COLUMNS] == COST_FIELDNAMES

   def test_export_and_insert_use_the_same_columns(self):
      assert db._EXPORT_FIELDNAMES == db._METRIC_COLUMNS
      assert db._METRIC_COLUMNS[-len(COST_FIELDNAMES):] == COST_FIELDNAMES

   def test_migration_adds_every_cost_column(self):
      statements = db._MIGRATION_SQL.splitlines()

      assert len(statements) == len(COST_FIELDNAMES)
      assert all(s.startswith("ALTER TABLE metrics ADD COLUMN IF NOT EXISTS cpu_") or "admm_iterations" in s or "_evals" in s for s in statements)

   def test_claim_sql_binds_the_hostname_where_needed(self):
      assert db._claim_sql(db._PACKAGE_OWNED_BY_HOST).count("%s") == 2
      assert db._claim_sql(db._PACKAGE_UNOWNED).count("%s") == 1
