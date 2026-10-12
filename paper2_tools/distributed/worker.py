"""CLI: Worker that claims tasks from DB, runs simulations, writes results back.

Replicates the logic of _run_worker() in scenarios.py but uses PostgreSQL
instead of CSV file-locking.

Usage:
    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.worker \
        --v-max 2.5 --u-max 3.0 --room-size 8.0 \
        --static-safety-zone 1.52 --adaptive-safety-zone 1.0 --r-min 0.4
"""

from __future__ import annotations

import argparse
import gc
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

from paper2_tools.distributed.config import HORIZON, DT, ALPHA, MAX_STEPS, TRACE_LEN, TIMEOUT_MS, placement_seed
from paper2_tools.distributed.db import claim_task, insert_metrics, mark_done, mark_error, count_by_status

logger = logging.getLogger(__name__)


def _run_one_task(v_max: float, u_max: float, room_size: float, alpha: float,
                  static_safety_zone: float, adaptive_safety_zone: float, r_min: float) -> bool:
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

    logger.info("Claimed task %d: run=%d, n_drones=%d, %s/%s, seed=%d",
                task_id, run_index, n_drones, coordinator, controller, seed)

    try:
        # Same seed for all coordinator/controller combinations of this (run, N):
        # create_scenario draws the start/target positions from np.random.
        np.random.seed(seed)
        cfg = create_scenario(
            horizon=HORIZON, dt=DT,
            controller_type=controller, coordinator_type=coordinator,
            physics_u=u_max, physics_v=v_max,
            room_size=room_size, n_drones=n_drones,
            alpha=alpha if controller == "mpc_agent_adaptive" else None,
            drones_radius=r_min,
            safety_zone=adaptive_safety_zone if controller == "mpc_agent_adaptive" else static_safety_zone,
        )

        status, wall_time, jerk_3d_value, step_durations, step_mean_pair_dists, all_pair_dists, _frames = \
            run_single_scenario(scenario=cfg, max_steps=MAX_STEPS, trace_len=TRACE_LEN,
                                timeout=TIMEOUT_MS, store_gif=False)

        _fieldnames, row = print_results_prep(
            all_pair_dists=all_pair_dists, jerk_3d_value=jerk_3d_value,
            num_drones=n_drones, status=status,
            step_durations=step_durations, step_mean_pair_dists=step_mean_pair_dists,
            wall_time=wall_time, coordinator_type=coordinator, controller_type=controller,
            horizon=HORIZON,
        )

        insert_metrics(task_id, row)
        mark_done(task_id)
        logger.info("Task %d completed: status=%s, wall_time=%.1fs", task_id, status, wall_time)

    except DronePlacementError as e:
        logger.exception("Task %d infeasible: %s", task_id, e)
        _, row = print_results_prep(all_pair_dists=[0.0], jerk_3d_value=0.0, num_drones=n_drones, status=Status.NO_POSSIBLE_PLACEMENT, step_durations=[0.0],
              step_mean_pair_dists=[0.0], wall_time=0.0, coordinator_type=coordinator, controller_type=controller, horizon=HORIZON)
        insert_metrics(task_id, row)
        mark_done(task_id)
    except Exception as e:
        logger.exception("Task %d failed: %s", task_id, e)
        mark_error(task_id, str(e)[:500])
    finally:
        gc.collect()

    return True


def _worker_loop(v_max: float, u_max: float, room_size: float, alpha: float,
                 static_safety_zone: float, adaptive_safety_zone: float, r_min: float) -> None:
    """Keep claiming tasks until the queue is empty."""
    while _run_one_task(v_max, u_max, room_size, alpha, static_safety_zone, adaptive_safety_zone, r_min):
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
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%d.%m %H:%M:%S",
    )

    n_workers = args.n_workers if args.n_workers > 0 else multiprocessing.cpu_count()
    logger.info("Starting %d worker processes", n_workers)

    worker_args = (args.v_max, args.u_max, args.room_size, ALPHA,
                   args.static_safety_zone, args.adaptive_safety_zone, args.r_min)

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
