"""Tests for paper2_tools.distributed.worker.resolve_task_settings.

The worker reads room size, comm radius, ADMM rounds and timeout from the task row and falls back to its command line, so the
density experiment and the volume campaign can share the same workers. Rows from before the settings existed carry none of the
keys and must behave exactly as before.
"""

from __future__ import annotations

import pytest

from paper2_tools.distributed import db
from paper2_tools.distributed.config import MAX_ADMM_ITER, TIMEOUT_MS
from paper2_tools.distributed.worker import TaskSettings, resolve_task_settings


class TestResolveTaskSettings:
   """Tests for resolve_task_settings."""

   def test_legacy_row_without_settings_uses_the_worker_defaults(self):
      settings = resolve_task_settings({"coordinator": "dmpc_admm"}, room_size=8.0)

      assert settings == TaskSettings(room_size=8.0, comm_radius=None, max_admm_iter=MAX_ADMM_ITER, timeout_s=float(TIMEOUT_MS))

   def test_null_settings_mean_the_defaults_too(self):
      task = {"coordinator": "dmpc_admm", "room_size": None, "comm_radius": None, "max_admm_iter": None, "timeout_s": None}

      assert resolve_task_settings(task, room_size=8.0) == resolve_task_settings({"coordinator": "dmpc_admm"}, room_size=8.0)

   def test_task_values_win_over_the_defaults(self):
      task = {"coordinator": "dmpc_admm", "room_size": 12.7, "comm_radius": 7.04, "max_admm_iter": 1, "timeout_s": 14400}

      settings = resolve_task_settings(task, room_size=8.0)

      assert settings == TaskSettings(room_size=12.7, comm_radius=7.04, max_admm_iter=1, timeout_s=14400.0)

   def test_central_mpc_ignores_a_comm_radius(self):
      task = {"coordinator": "mpc_central", "room_size": 12.7, "comm_radius": 7.04, "max_admm_iter": None, "timeout_s": 14400}

      settings = resolve_task_settings(task, room_size=8.0)

      assert settings.comm_radius is None
      assert settings.room_size == pytest.approx(12.7)

   def test_metrics_view_matches_the_db_columns(self):
      settings = resolve_task_settings({"coordinator": "dmpc_admm", "comm_radius": 7.04, "max_admm_iter": 3}, room_size=8.0)

      assert list(settings.as_metrics()) == db._TASK_PARAM_NAMES
      assert settings.as_metrics() == {"room_size": 8.0, "comm_radius": 7.04, "max_admm_iter": 3}
