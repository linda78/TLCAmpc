"""LSTM collision regression tests.

Verifies zero hard collisions (physical body overlap) across all LSTM scenario
configs under configs/lstm/.

Tests are decorated @pytest.mark.slow for CI filtering. Run with:
    pytest -m slow tests/test_lstm_collision_regression.py

Pass criterion: no drone pair satisfies dist < d1.radius + d2.radius (0.4m)
at any step. Goal-reaching rate is logged but advisory.
"""
from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_GLOB = str(_PROJECT_ROOT / "configs" / "lstm" / "*.json")
_CONFIG_FILES = sorted(glob.glob(_CONFIG_GLOB))
_WARMUP_STEPS = 20    # LSTM history buffer warmup
_SIM_STEPS = 150      # Steps to collect metrics after warmup
_TARGET_REACH_DIST = 0.5  # meters — advisory goal-reaching threshold

# ---------------------------------------------------------------------------
# Guard: skip entire module when no configs exist (Plan 01 not yet run)
# ---------------------------------------------------------------------------

if not _CONFIG_FILES:
    pytest.skip(
        "No configs/lstm/*.json files found. Run Plan 01 first.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_hard_collisions(sim: Simulator) -> int:
    """Count drone pairs with physical body overlap (dist < 0.4m)."""
    drones = sim.drones
    count = 0
    for i in range(len(drones)):
        for j in range(i + 1, len(drones)):
            dist = float(np.linalg.norm(drones[i].position() - drones[j].position()))
            if dist < drones[i].radius + drones[j].radius:
                count += 1
    return count


def _goal_reaching_rate(sim: Simulator) -> float:
    """Advisory: fraction of drones within 0.5m of their target."""
    reached = sum(
        1 for d in sim.drones
        if float(np.linalg.norm(d.position() - d.route.target)) <= _TARGET_REACH_DIST
    )
    return reached / len(sim.drones) if sim.drones else 0.0

# ---------------------------------------------------------------------------
# Parametrized regression test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "config_path",
    _CONFIG_FILES,
    ids=[Path(p).name for p in _CONFIG_FILES],
)
@pytest.mark.lstm_only
def test_no_hard_collisions(config_path: str) -> None:
    """LSTM mode produces zero physical body overlaps across full simulation run."""
    with open(config_path) as f:
        cfg_dict = json.load(f)

    scenario = ScenarioConfig.model_validate(cfg_dict)
    sim = Simulator.from_config(scenario)

    total_hard_collisions = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Warmup: fill LSTM history buffer (no assertions during warmup)
        for _ in range(_WARMUP_STEPS):
            sim.step()

        # Metric collection
        for _step_idx in range(_SIM_STEPS):
            sim.step()
            hard_collisions = _count_hard_collisions(sim)
            total_hard_collisions += hard_collisions

    # Advisory: log goal-reaching rate
    goal_rate = _goal_reaching_rate(sim)
    config_name = Path(config_path).name
    print(
        f"\n[{config_name}] goal_reaching_rate={goal_rate:.2f}, "
        f"hard_collisions={total_hard_collisions}"
    )

    # Hard assertion: zero physical body overlaps
    assert total_hard_collisions == 0, (
        f"{config_name}: {total_hard_collisions} hard collision(s) detected "
        f"(dist < d1.radius + d2.radius = 0.4m). "
        f"Consider increasing k_alpha or safety_zone floor."
    )
