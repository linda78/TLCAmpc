"""Tests for tools.utility.helper.

Only the DeadlockDetector is covered here: pairwise_distances and piecewise_linear_loss_3d are plain numerics that the scenario
runs exercise anyway. The detector decides when a distributed run is cut short, so its counting has to be exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.utility.helper import DeadlockDetector


def _positions(**by_id: tuple[float, float, float]) -> dict[str, np.ndarray]:
   return {drone_id: np.asarray(p, dtype=float) for drone_id, p in by_id.items()}


class TestDeadlockDetector:
   """Tests for DeadlockDetector."""

   def test_disabled_window_never_flags(self):
      detector = DeadlockDetector(window=None, min_movement=1.0)

      assert all(not detector.update(_positions(a=(0, 0, 0))) for _ in range(10))
      assert detector.quiet_steps == 0

   def test_first_update_is_only_the_baseline(self):
      detector = DeadlockDetector(window=1, min_movement=0.1)

      assert detector.update(_positions(a=(0, 0, 0))) is False
      assert detector.quiet_steps == 0

   def test_flags_after_window_quiet_steps(self):
      detector = DeadlockDetector(window=3, min_movement=0.1)
      detector.update(_positions(a=(0, 0, 0), b=(1, 1, 1)))

      flags = [detector.update(_positions(a=(0, 0, 0), b=(1, 1, 1))) for _ in range(3)]

      assert flags == [False, False, True]

   def test_movement_below_threshold_counts_as_quiet(self):
      detector = DeadlockDetector(window=2, min_movement=0.1)
      detector.update(_positions(a=(0, 0, 0)))

      assert detector.update(_positions(a=(0.05, 0, 0))) is False
      assert detector.update(_positions(a=(0.1 - 1e-9, 0, 0))) is True

   def test_one_moving_drone_resets_the_run(self):
      """Test a single drone that still makes progress keeps the whole swarm out of the deadlock state."""
      detector = DeadlockDetector(window=2, min_movement=0.1)
      detector.update(_positions(a=(0, 0, 0), b=(1, 1, 1)))
      detector.update(_positions(a=(0, 0, 0), b=(1, 1, 1)))
      assert detector.quiet_steps == 1

      assert detector.update(_positions(a=(0, 0, 0), b=(1, 1.5, 1))) is False
      assert detector.quiet_steps == 0

      assert detector.update(_positions(a=(0, 0, 0), b=(1, 1.5, 1))) is False
      assert detector.update(_positions(a=(0, 0, 0), b=(1, 1.5, 1))) is True

   def test_keeps_copies_of_the_positions(self):
      """Test the baseline does not alias the caller's array, which the simulator may update in place."""
      detector = DeadlockDetector(window=1, min_movement=0.1)
      pos = np.zeros(3)
      detector.update({"a": pos})

      pos += 5.0

      assert detector.update({"a": pos}) is False

   @pytest.mark.parametrize("window, min_movement", [(0, 0.1), (-1, 0.1), (3, -0.1)])
   def test_rejects_invalid_settings(self, window, min_movement):
      with pytest.raises(ValueError):
         DeadlockDetector(window=window, min_movement=min_movement)
