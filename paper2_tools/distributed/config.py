"""Shared configuration for the distributed scenario runner."""

from __future__ import annotations

import os

DATABASE_URL_ENV = "DRONESIM_DATABASE_URL"

COORDINATORS = ['mpc_central', 'dmpc_admm']
CONTROLLERS = ['mpc_agent', 'mpc_agent_adaptive']

# Experiment names in the scenario_runs queue (create_queue --experiment). The campaigns in the fixed 8 m room (paper 2
# revision, paper 3 costs) are VOLUME_EXPERIMENT, the constant-density scaling experiment of
# paper2_tools/review_work/density_scaling is DENSITY_EXPERIMENT. Renaming one means changing it here and running
# UPDATE scenario_runs SET experiment = '<new>' WHERE experiment = '<old>' on databases that already hold rows.
VOLUME_EXPERIMENT = "volume"
DENSITY_EXPERIMENT = "density"
EXPERIMENTS = (VOLUME_EXPERIMENT, DENSITY_EXPERIMENT)

HORIZON: int = 8
DT: float = 0.1
ALPHA: float = 0.5
TERMINAL_CONSTRAINT: bool = True  # braking-distance terminal radius at the last horizon step (room walls); False reproduces the runs before 22.10.2026

MAX_STEPS: int = 10000
TRACE_LEN: int = 10000
TIMEOUT_MS: int = 86400  # seconds per run, despite the name; a task's timeout_s overrides it
MAX_ADMM_ITER: int = 20  # DMPC rounds per step of the volume campaigns; a task's max_admm_iter overrides it

# Deadlock check: a run ends with status "deadlock" once every drone moved less than
# DEADLOCK_MIN_MOVEMENT metres per step for DEADLOCK_STEPS consecutive steps. With DT = 0.1 s
# that is 0.1 m/s for 10 s of simulated time. None disables the check.
DEADLOCK_STEPS: int | None = 100
DEADLOCK_MIN_MOVEMENT: float = 0.01


def placement_seed(run_index: int, n_drones: int) -> int:
    """Seed for the start/target placement of one (run, N) pair.

    Every coordinator/controller combination of the same pair uses this seed, so
    they draw identical start and target positions as long as their placement
    constraints (min pair distance, wall margin) agree.
    """
    return 1000 * run_index + n_drones


def get_database_url() -> str:
    url = os.environ.get(DATABASE_URL_ENV)
    if not url:
        raise RuntimeError(
            f"Environment variable {DATABASE_URL_ENV} is not set. "
            f"Example: export {DATABASE_URL_ENV}='postgresql://user:pass@host:5432/dronesim'"
        )
    return url
