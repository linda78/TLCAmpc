"""CLI: Populate the scenario_runs queue in PostgreSQL.

Every call adds the tasks of one experiment. Rows that already exist (same run, N, coordinator, controller, experiment, variant)
are left alone, so a campaign is built from several calls and every call is repeatable. ``--dry-run`` prints the cells and
touches no database. The workers take any pending row, so after filling the queue restart ``sim-worker`` on servers whose
worker already stopped on an empty queue.

Volume experiment (the default, fixed 8 m room, worker defaults for everything else). The paper3 cost campaign:

    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.create_queue --runs 20 --drones 2-10
    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.create_queue --runs 10 --drones 11-25 --controllers mpc_agent_adaptive

The legacy form ``--n-crit 35`` still means drones 2..n_crit+1 with all combinations.

Constant-density scaling experiment (grid and formulas in paper2_tools/review_work/density_scaling/density.py):

    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.create_queue --experiment density --dry-run
    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.create_queue --experiment density
"""

from __future__ import annotations

import argparse
from typing import Any

from paper2_tools.distributed.config import CONTROLLERS, COORDINATORS, DENSITY_EXPERIMENT, EXPERIMENTS, VOLUME_EXPERIMENT
from paper2_tools.distributed.db import count_by_status, ensure_schema, insert_tasks, make_task
from paper2_tools.review_work.density_scaling import density

# Options that belong to one experiment only; naming one for the other experiment is an error, not silently ignored.
_VOLUME_OPTIONS = ("drones", "n_crit", "coordinators", "controllers")
_DENSITY_OPTIONS = ("dmpc_drones", "central_drones", "admm_iters", "timeout_h", "comm_radius")


def parse_drone_range(text: str) -> range:
    """Parse an inclusive drone-count range such as ``2-10`` into range(2, 11)."""
    try:
        first, last = (int(part) for part in text.split("-", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected FROM-TO, e.g. 2-10, got {text!r}") from None
    if first < 1 or last < first:
        raise argparse.ArgumentTypeError(f"expected 1 <= FROM <= TO, got {text!r}")
    return range(first, last + 1)


def parse_int_list(text: str) -> tuple[int, ...]:
    """Parse a comma separated list of positive integers such as ``10,20,40``."""
    try:
        values = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a comma separated list of integers, got {text!r}") from None
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Populate the distributed scenario queue")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=VOLUME_EXPERIMENT,
                        help=f"which experiment's tasks to add (default: {VOLUME_EXPERIMENT})")
    parser.add_argument("--runs", type=int, default=None,
                        help=f"runs (0..runs-1) per cell; required for {VOLUME_EXPERIMENT}, default {density.RUNS} for {DENSITY_EXPERIMENT}")
    parser.add_argument("--dry-run", action="store_true", help="only print the cells, touch no database")

    volume = parser.add_argument_group(f"{VOLUME_EXPERIMENT} experiment")
    drones = volume.add_mutually_exclusive_group()
    drones.add_argument("--drones", type=parse_drone_range, help="inclusive drone-count range, e.g. 2-10")
    drones.add_argument("--n-crit", type=int, help="legacy: drones range 2..n_crit+1")
    volume.add_argument("--coordinators", nargs="+", choices=COORDINATORS, default=None, help="default: all")
    volume.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=None, help="default: all")

    dens = parser.add_argument_group(f"{DENSITY_EXPERIMENT} experiment")
    dens.add_argument("--dmpc-drones", type=parse_int_list, default=None,
                      help=f"N values of the distributed MPC (default {','.join(map(str, density.DMPC_DRONES))})")
    dens.add_argument("--central-drones", type=parse_int_list, default=None,
                      help=f"N values of the central MPC (default {','.join(map(str, density.CENTRAL_DRONES))})")
    dens.add_argument("--admm-iters", type=parse_int_list, default=None,
                      help=f"DMPC rounds per step, one cell each (default {','.join(map(str, density.ADMM_ITERS))})")
    dens.add_argument("--timeout-h", type=float, default=None, help=f"timeout per run in hours (default {density.TIMEOUT_S / 3600:g})")
    dens.add_argument("--comm-radius", type=float, default=None, help=f"override r_com (default {density.comm_radius_for():.2f} m)")
    return parser


def _reject_foreign_options(parser: argparse.ArgumentParser, args: argparse.Namespace, names: tuple[str, ...], owner: str) -> None:
    given = [name for name in names if getattr(args, name) is not None]
    if given:
        flags = ", ".join("--" + name.replace("_", "-") for name in given)
        parser.error(f"{flags}: only for --experiment {owner}, not for {args.experiment}")


def tasks_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[dict[str, Any]]:
    """The queue rows the arguments describe (see db.make_task)."""
    if args.experiment == VOLUME_EXPERIMENT:
        _reject_foreign_options(parser, args, _DENSITY_OPTIONS, DENSITY_EXPERIMENT)
        if args.runs is None:
            parser.error(f"--runs is required for --experiment {VOLUME_EXPERIMENT}")
        if args.drones is None and args.n_crit is None:
            parser.error(f"--drones or --n-crit is required for --experiment {VOLUME_EXPERIMENT}")
        drone_range = args.drones if args.drones is not None else range(2, args.n_crit + 2)
        coordinators = args.coordinators or COORDINATORS
        controllers = args.controllers or CONTROLLERS
        return [make_task(run, n, coordinator, controller)
                for run in range(args.runs) for n in drone_range for coordinator in coordinators for controller in controllers]

    _reject_foreign_options(parser, args, _VOLUME_OPTIONS, VOLUME_EXPERIMENT)
    return density.build_tasks(
        runs=density.RUNS if args.runs is None else args.runs,
        dmpc_drones=density.DMPC_DRONES if args.dmpc_drones is None else args.dmpc_drones,
        central_drones=density.CENTRAL_DRONES if args.central_drones is None else args.central_drones,
        admm_iters=density.ADMM_ITERS if args.admm_iters is None else args.admm_iters,
        timeout_s=density.TIMEOUT_S if args.timeout_h is None else int(round(args.timeout_h * 3600)),
        comm_radius=args.comm_radius,
    )


def describe(tasks: list[dict[str, Any]]) -> str:
    """Cell table of the tasks, one line per (experiment, coordinator, controller, variant, N). '-' is the worker default."""
    cells: dict[tuple, list] = {}
    for t in tasks:
        key = (t["experiment"], t["coordinator"], t["controller"], t["variant"], t["n_drones"])
        cells.setdefault(key, [t, 0])[1] += 1

    lines = [f"{'experiment':<10} {'coordinator':<12} {'controller':<20} {'var':<4} {'N':>3} {'room [m]':>8} {'r_com [m]':>9} {'k':>3} "
             f"{'timeout':>8} {'runs':>4}"]
    for (experiment, coordinator, controller, variant, n), (t, count) in cells.items():
        room = "-" if t["room_size"] is None else f"{t['room_size']:.2f}"
        r_com = "-" if t["comm_radius"] is None else f"{t['comm_radius']:.2f}"
        k = "-" if t["max_admm_iter"] is None else str(t["max_admm_iter"])
        timeout = "-" if t["timeout_s"] is None else f"{t['timeout_s'] / 3600:.1f}h"
        lines.append(f"{experiment:<10} {coordinator:<12} {controller:<20} {variant or '-':<4} {n:>3} {room:>8} {r_com:>9} {k:>3} "
                     f"{timeout:>8} {count:>4}")
    lines.append(f"{len(tasks)} tasks ('-' = worker default)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    tasks = tasks_from_args(parser, args)
    print(describe(tasks))
    if args.dry_run:
        return

    ensure_schema()
    inserted = insert_tasks(tasks)
    print(f"Inserted {inserted} new tasks (experiment '{args.experiment}').")
    print(f"Queue status of experiment '{args.experiment}': {dict(count_by_status(args.experiment))}")


if __name__ == "__main__":
    main()
