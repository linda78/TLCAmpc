"""CLI: Populate the scenario_runs queue in PostgreSQL.

Every call adds the runs 0..runs-1 for each (n_drones, coordinator, controller) combination it selects; rows that already exist
are left alone, so a campaign with different settings per drone range is built from several calls. The paper3 cost campaign:

    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.create_queue --runs 20 --drones 2-10
    DRONESIM_DATABASE_URL="..." python -m paper2_tools.distributed.create_queue --runs 10 --drones 11-25 --controllers mpc_agent_adaptive

The legacy form ``--n-crit 35`` still means drones 2..n_crit+1 with all combinations.
"""

from __future__ import annotations

import argparse

from paper2_tools.distributed.config import COORDINATORS, CONTROLLERS
from paper2_tools.distributed.db import ensure_schema, insert_runs, count_by_status


def parse_drone_range(text: str) -> range:
    """Parse an inclusive drone-count range such as ``2-10`` into range(2, 11)."""
    try:
        first, last = (int(part) for part in text.split("-", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected FROM-TO, e.g. 2-10, got {text!r}") from None
    if first < 1 or last < first:
        raise argparse.ArgumentTypeError(f"expected 1 <= FROM <= TO, got {text!r}")
    return range(first, last + 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Populate the distributed scenario queue")
    parser.add_argument("--runs", type=int, required=True, help="Number of runs (0..runs-1) per combination")
    drones = parser.add_mutually_exclusive_group(required=True)
    drones.add_argument("--drones", type=parse_drone_range, help="Inclusive drone-count range, e.g. 2-10")
    drones.add_argument("--n-crit", type=int, help="Legacy: drones range 2..n_crit+1")
    parser.add_argument("--coordinators", nargs="+", choices=COORDINATORS, default=COORDINATORS, help="Default: all")
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=CONTROLLERS, help="Default: all")
    args = parser.parse_args(argv)

    ensure_schema()

    drone_range = args.drones if args.drones is not None else range(2, args.n_crit + 2)
    runs = range(args.runs)

    inserted = insert_runs(runs, drone_range, args.coordinators, args.controllers)
    status_counts = count_by_status()

    print(f"Drones {drone_range.start}..{drone_range.stop - 1}, {len(runs)} runs, coordinators {args.coordinators}, controllers {args.controllers}")
    print(f"Inserted {inserted} new tasks into queue.")
    print(f"Queue status: {dict(status_counts)}")


if __name__ == "__main__":
    main()
