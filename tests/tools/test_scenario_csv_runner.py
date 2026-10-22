"""Tests for tools.utility.scenario_csv_runner.

The queue CSV carries the run index since the revision runs, because the worker derives the placement seed from (run, N).
"""

from __future__ import annotations

from tools.utility.scenario_csv_runner import QueueStatus, claim_row, create_queue


class TestCreateQueue:
   """Tests for create_queue and claim_row."""

   def test_rows_carry_run_index(self, tmp_path):
      queue = tmp_path / "queue.csv"
      create_queue(queue, runs=range(2), drones=range(2, 4), coordinators=["mpc_central"], controllers=["mpc_agent"])

      claimed = [claim_row(queue, queue.with_suffix(".lock")) for _ in range(4)]

      assert [(int(r["run"]), int(r["n_drones"])) for r in claimed] == [(0, 2), (1, 2), (0, 3), (1, 3)]
      assert all(r["status"] == QueueStatus.IN_PROCESS for r in claimed)
      assert claim_row(queue, queue.with_suffix(".lock")) is None
