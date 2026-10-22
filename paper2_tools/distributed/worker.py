"""CLI: Worker that claims tasks from DB, runs simulations, writes results back.

Replicates the logic of _run_worker() in scenarios.py but uses PostgreSQL
instead of CSV file-locking.

Usage:
    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.worker \
        --v-max 2.5 --u-max 3.0 --room-size 8.0 \
        --static-safety-zone 1.52 --adaptive-safety-zone 1.0 --r-min 0.4 \
        --deadlock-steps 100 --deadlock-min-movement 0.01

The command line holds the campaign defaults. A task row may override room_size, comm_radius, max_admm_iter and timeout_s
(see resolve_task_settings), which is how the constant-density scaling experiment runs on the same workers.
"""

from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Any

import numpy as np

from paper2_tools.distributed.config import (HORIZON, DT, ALPHA, TERMINAL_CONSTRAINT, MAX_STEPS, TRACE_LEN, TIMEOUT_MS, MAX_ADMM_ITER,
                                             DEADLOCK_STEPS, DEADLOCK_MIN_MOVEMENT, VOLUME_EXPERIMENT, placement_seed)
from paper2_tools.distributed.db import claim_task, ensure_schema, insert_metrics, mark_done, mark_error, count_by_status

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSettings:
    """Effective per-run settings after applying the task row on top of the worker defaults."""

    room_size: float
    comm_radius: float | None
    max_admm_iter: int
    timeout_s: float

    def as_metrics(self) -> dict[str, Any]:
        """The columns stored next to the metrics row (db._TASK_PARAM_COLUMNS)."""
        return {"room_size": self.room_size, "comm_radius": self.comm_radius, "max_admm_iter": self.max_admm_iter}


def resolve_task_settings(task: Mapping[str, Any], *, room_size: float, timeout_s: float = TIMEOUT_MS,
                          max_admm_iter: int = MAX_ADMM_ITER) -> TaskSettings:
    """Combine a claimed task row with the worker defaults.

    A task value that is present and not None wins; None (or a row from before the settings existed) means the default.
    comm_radius only applies to the distributed coordinator, the central one couples every pair regardless.
    """
    task_room = task.get("room_size")
    task_timeout = task.get("timeout_s")
    task_iter = task.get("max_admm_iter")
    comm_radius = task.get("comm_radius") if task.get("coordinator") == "dmpc_admm" else None
    return TaskSettings(
        room_size=float(task_room) if task_room is not None else float(room_size),
        comm_radius=float(comm_radius) if comm_radius is not None else None,
        max_admm_iter=int(task_iter) if task_iter is not None else int(max_admm_iter),
        timeout_s=float(task_timeout) if task_timeout is not None else float(timeout_s),
    )


def _run_one_task(v_max: float, u_max: float, room_size: float, alpha: float,
                  static_safety_zone: float, adaptive_safety_zone: float, r_min: float,
                  deadlock_steps: int | None, deadlock_min_movement: float) -> bool:
    """Claim and execute a single task. Returns True if a task was processed, False if queue empty."""
    # Import simulation code inside the worker process to avoid PySide6 issues in main process
    from tools.utility import DronePlacementError, Status
    from tools.utility.scenario_creator import create_scenario
    from tools.horizon_live_view_grid import run_single_scenario, print_results_prep

    task = claim_task()
    if task is None:
        return False

    task_id = task["id"]
    run_index = task["run_index"]
    n_drones = task["n_drones"]
    coordinator = task["coordinator"]
    controller = task["controller"]
    seed = placement_seed(run_index, n_drones)
    settings = resolve_task_settings(task, room_size=room_size)

    logger.info("Claimed task %d: run=%d, n_drones=%d, %s/%s, seed=%d, experiment=%s/%s, room=%.2f, comm_radius=%s, max_admm_iter=%d, timeout=%.0fs",
                task_id, run_index, n_drones, coordinator, controller, seed, task.get("experiment", VOLUME_EXPERIMENT), task.get("variant", ""),
                settings.room_size, settings.comm_radius, settings.max_admm_iter, settings.timeout_s)

    try:
        # Same seed for all coordinator/controller combinations of this (run, N):
        # create_scenario draws the start/target positions from np.random.
        np.random.seed(seed)
        cfg = create_scenario(
            horizon=HORIZON, dt=DT,
            controller_type=controller, coordinator_type=coordinator,
            physics_u=u_max, physics_v=v_max,
            room_size=settings.room_size, n_drones=n_drones,
            alpha=alpha if controller == "mpc_agent_adaptive" else None,
            drones_radius=r_min,
            safety_zone=adaptive_safety_zone if controller == "mpc_agent_adaptive" else static_safety_zone,
            comm_radius=settings.comm_radius, max_admm_iter=settings.max_admm_iter,
            terminal_constraint=TERMINAL_CONSTRAINT,
        )

        status, wall_time, jerk_3d_value, step_durations, step_mean_pair_dists, all_pair_dists, _frames, costs = \
            run_single_scenario(scenario=cfg, max_steps=MAX_STEPS, trace_len=TRACE_LEN,
                                timeout=settings.timeout_s, store_gif=False,
                                deadlock_steps=deadlock_steps, deadlock_min_movement=deadlock_min_movement)

        _fieldnames, row = print_results_prep(
            all_pair_dists=all_pair_dists, jerk_3d_value=jerk_3d_value,
            num_drones=n_drones, status=status,
            step_durations=step_durations, step_mean_pair_dists=step_mean_pair_dists,
            wall_time=wall_time, coordinator_type=coordinator, controller_type=controller,
            horizon=HORIZON, costs=costs,
        )

        insert_metrics(task_id, row, task_params=settings.as_metrics())
        mark_done(task_id)
        logger.info("Task %d completed: status=%s, wall_time=%.1fs, cpu=%.1fs (solver %.1f, constraints %.1f, cost %.1f, setup %.1f), "
                    "local_solves=%d, admm_iterations=%d",
                    task_id, status, wall_time, costs["cpu_total_s"], costs["cpu_solver_s"], costs["cpu_constraints_s"], costs["cpu_cost_s"],
                    costs["cpu_setup_s"], costs["local_solves"], costs["admm_iterations"])

    except DronePlacementError as e:
        logger.exception("Task %d infeasible: %s", task_id, e)
        _, row = print_results_prep(all_pair_dists=[0.0], jerk_3d_value=0.0, num_drones=n_drones, status=Status.NO_POSSIBLE_PLACEMENT, step_durations=[0.0],
              step_mean_pair_dists=[0.0], wall_time=0.0, coordinator_type=coordinator, controller_type=controller, horizon=HORIZON)
        insert_metrics(task_id, row, task_params=settings.as_metrics())
        mark_done(task_id)
    except Exception as e:
        logger.exception("Task %d failed: %s", task_id, e)
        mark_error(task_id, str(e)[:500])
    finally:
        gc.collect()

    return True


def _worker_loop(v_max: float, u_max: float, room_size: float, alpha: float,
                 static_safety_zone: float, adaptive_safety_zone: float, r_min: float,
                 deadlock_steps: int | None, deadlock_min_movement: float) -> None:
    """Keep claiming tasks until the queue is empty."""
    while _run_one_task(v_max, u_max, room_size, alpha, static_safety_zone, adaptive_safety_zone, r_min,
                        deadlock_steps, deadlock_min_movement):
        pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Distributed scenario worker")
    parser.add_argument("--n-workers", type=int, default=-1, help="Number of worker processes (-1 = cpu count)")
    parser.add_argument("--v-max", type=float, default=2.5)
    parser.add_argument("--u-max", type=float, default=3.0)
    parser.add_argument("--room-size", type=float, default=8.0)
    parser.add_argument("--static-safety-zone", type=float, default=1.52)
    parser.add_argument("--adaptive-safety-zone", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=0.4)
    parser.add_argument("--deadlock-steps", type=int, default=DEADLOCK_STEPS or 0,
                        help="Consecutive steps in which no drone moves before the run counts as a deadlock (0 disables)")
    parser.add_argument("--deadlock-min-movement", type=float, default=DEADLOCK_MIN_MOVEMENT,
                        help="Movement per step (metres) below which a drone counts as not moving")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%d.%m %H:%M:%S",
    )

    ensure_schema()  # a worker from a newer branch may meet a database from an older campaign

    n_workers = args.n_workers if args.n_workers > 0 else multiprocessing.cpu_count()
    logger.info("Starting %d worker processes", n_workers)

    deadlock_steps = args.deadlock_steps if args.deadlock_steps > 0 else None
    if deadlock_steps is None:
        logger.info("Deadlock check disabled")
    else:
        logger.info("Deadlock check: %d steps with movement < %g m per step", deadlock_steps, args.deadlock_min_movement)

    worker_args = (args.v_max, args.u_max, args.room_size, ALPHA,
                   args.static_safety_zone, args.adaptive_safety_zone, args.r_min,
                   deadlock_steps, args.deadlock_min_movement)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = set()
        for _ in range(n_workers):
            futures.add(executor.submit(_worker_loop, *worker_args))

        done, _ = wait(futures)
        for f in done:
            try:
                f.result()
            except Exception as e:
                logger.error("Worker process failed: %s", e)

    status_counts = count_by_status()
    logger.info("All workers done. Queue status: %s", dict(status_counts))


if __name__ == "__main__":
    main()
