"""CLI: Write *expected* results for pending tasks whose outcome is already known.

Two cases:

  no-placement  Static configurations above a drone count where the start/target
                placement always fails (observed: every run from N = 10 on). Writes
                the same metrics row the worker writes on DronePlacementError.

  timeout       Configurations where every run at these drone counts is known to
                run into the 24 h timeout. Writes a 'timeout' metrics row with
                steps = 0 and NULL measurements, so time/distance plots (which only
                use finished runs) are not affected and the deadlock rate counts it.

Only 'pending' (or 'deferred', see queue_waves.py) rows are touched, never running or done ones. The scenario_runs row
is marked done with claimed_by = 'expected' and the reason in error_message, so the
synthetic rows stay distinguishable and can be reverted with `undo`.

Usage:
    python -m paper2_tools.distributed.expected_results no-placement [--min-n 10] [--coordinator X] [--dry-run]
    python -m paper2_tools.distributed.expected_results timeout --coordinator dmpc_admm --controller mpc_agent_adaptive --min-n 20 [--max-n 25] [--dry-run]
    python -m paper2_tools.distributed.expected_results list
    python -m paper2_tools.distributed.expected_results undo [--coordinator X] [--controller Y] [--min-n N] [--max-n M] [--dry-run]
"""

from __future__ import annotations

import argparse
from collections import Counter

import psycopg2.extras

from paper2_tools.distributed.config import HORIZON, COORDINATORS, CONTROLLERS
from paper2_tools.distributed.db import get_connection

EXPECTED_MARKER = "expected"
STATIC_CONTROLLER = "mpc_agent"

_METRIC_COLS = (
    "scenario_run_id, num_drones, horizon, coordinator, controller_type, status, steps, "
    "wall_time_s, min_step_time_s, max_step_time_s, mean_step_time_s, min_distance, "
    "max_distance, mean_distance_all_pairs, mean_distance_step_mean, jerk_3d_value"
)
_METRIC_PLACEHOLDERS = ", ".join(["%s"] * 16)


def _metrics_row(task: dict, status: str) -> tuple:
    """Build the metrics tuple for a synthetic result."""
    if status == "no_possible_placement":
        # Identical to what worker.py writes on DronePlacementError.
        measurements = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    elif status == "timeout":
        # Nothing was simulated: steps 0, measurements NULL.
        measurements = (0, None, None, None, None, None, None, None, None, None)
    else:
        raise ValueError(status)
    return (task["id"], task["n_drones"], HORIZON, task["coordinator"], task["controller"], status) + measurements


def _select_pending(cur, coordinator: str | None, controller: str | None, min_n: int, max_n: int | None) -> list[dict]:
    sql = """
        SELECT id, run_index, n_drones, coordinator, controller
        FROM scenario_runs r
        WHERE status IN ('pending', 'deferred')
          AND n_drones >= %s
          AND (%s::int IS NULL OR n_drones <= %s)
          AND (%s::text IS NULL OR coordinator = %s)
          AND (%s::text IS NULL OR controller = %s)
          AND NOT EXISTS (SELECT 1 FROM metrics m WHERE m.scenario_run_id = r.id)
        ORDER BY n_drones, coordinator, controller, run_index
        FOR UPDATE SKIP LOCKED
    """
    cur.execute(sql, (min_n, max_n, max_n, coordinator, coordinator, controller, controller))
    return [dict(r) for r in cur.fetchall()]


def _print_summary(tasks: list[dict], verb: str) -> None:
    if not tasks:
        print(f"{verb}: no matching rows.")
        return
    counts = Counter((t["n_drones"], t["coordinator"], t["controller"]) for t in tasks)
    print(f"{verb} {len(tasks)} rows:")
    for (n, coord, ctrl), c in sorted(counts.items()):
        print(f"  N={n:2d}  {coord:12s} {ctrl:20s} {c:3d}")


def _apply(status: str, reason: str, coordinator: str | None, controller: str | None,
           min_n: int, max_n: int | None, dry_run: bool) -> None:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            tasks = _select_pending(cur, coordinator, controller, min_n, max_n)
            _print_summary(tasks, "Would mark" if dry_run else "Marking")
            if dry_run or not tasks:
                conn.rollback()
                return
            psycopg2.extras.execute_batch(
                cur,
                f"INSERT INTO metrics ({_METRIC_COLS}) VALUES ({_METRIC_PLACEHOLDERS})",
                [_metrics_row(t, status) for t in tasks],
                page_size=500,
            )
            psycopg2.extras.execute_batch(
                cur,
                """
                UPDATE scenario_runs
                SET status = 'done', claimed_by = %s, claimed_at = now(), completed_at = now(),
                    error_message = %s
                WHERE id = %s AND status IN ('pending', 'deferred')
                """,
                [(EXPECTED_MARKER, reason, t["id"]) for t in tasks],
                page_size=500,
            )
        conn.commit()
    print(f"Done. {len(tasks)} metrics rows written with status='{status}', scenario_runs marked done.")


def cmd_no_placement(args: argparse.Namespace) -> None:
    reason = f"expected result, not simulated: static placement impossible for N >= {args.min_n}"
    _apply("no_possible_placement", reason, args.coordinator, STATIC_CONTROLLER, args.min_n, args.max_n, args.dry_run)


def cmd_timeout(args: argparse.Namespace) -> None:
    rng = f"N >= {args.min_n}" if args.max_n is None else f"{args.min_n} <= N <= {args.max_n}"
    reason = f"expected result, not simulated: {args.coordinator}/{args.controller} runs into 24 h timeout for {rng}"
    _apply("timeout", reason, args.coordinator, args.controller, args.min_n, args.max_n, args.dry_run)


def cmd_list(_args: argparse.Namespace) -> None:
    sql = """
        SELECT r.n_drones, r.coordinator, r.controller, m.status, count(*) AS n
        FROM scenario_runs r JOIN metrics m ON m.scenario_run_id = r.id
        WHERE r.claimed_by = %s
        GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (EXPECTED_MARKER,))
            rows = cur.fetchall()
    if not rows:
        print("No expected (synthetic) results in the table.")
        return
    total = 0
    for n, coord, ctrl, status, c in rows:
        print(f"  N={n:2d}  {coord:12s} {ctrl:20s} {status:22s} {c:3d}")
        total += c
    print(f"Total: {total} synthetic rows.")


def cmd_undo(args: argparse.Namespace) -> None:
    """Delete synthetic metrics rows and put their tasks back to pending."""
    sql = """
        SELECT id, run_index, n_drones, coordinator, controller
        FROM scenario_runs r
        WHERE status = 'done' AND claimed_by = %s
          AND n_drones >= %s
          AND (%s::int IS NULL OR n_drones <= %s)
          AND (%s::text IS NULL OR coordinator = %s)
          AND (%s::text IS NULL OR controller = %s)
        ORDER BY n_drones, coordinator, controller, run_index
        FOR UPDATE
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (EXPECTED_MARKER, args.min_n, args.max_n, args.max_n,
                              args.coordinator, args.coordinator, args.controller, args.controller))
            tasks = [dict(r) for r in cur.fetchall()]
            _print_summary(tasks, "Would revert" if args.dry_run else "Reverting")
            if args.dry_run or not tasks:
                conn.rollback()
                return
            ids = [t["id"] for t in tasks]
            cur.execute("DELETE FROM metrics WHERE scenario_run_id = ANY(%s)", (ids,))
            deleted = cur.rowcount
            cur.execute(
                """
                UPDATE scenario_runs
                SET status = 'pending', claimed_by = NULL, claimed_at = NULL, completed_at = NULL, error_message = NULL
                WHERE id = ANY(%s)
                """,
                (ids,),
            )
        conn.commit()
    print(f"Done. Deleted {deleted} metrics rows, {len(tasks)} tasks back to pending.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write expected results for pending tasks")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_filters(p: argparse.ArgumentParser, coordinator_required: bool = False, controller: bool = False) -> None:
        p.add_argument("--coordinator", choices=COORDINATORS, required=coordinator_required, default=None)
        if controller:
            p.add_argument("--controller", choices=CONTROLLERS, required=True)
        p.add_argument("--min-n", type=int, required=controller, default=10, help="lowest drone count to mark")
        p.add_argument("--max-n", type=int, default=None, help="highest drone count to mark (default: no limit)")
        p.add_argument("--dry-run", action="store_true", help="only show what would be marked")

    p = sub.add_parser("no-placement", help="static configs: placement impossible (default N >= 10)")
    add_filters(p)
    p.set_defaults(func=cmd_no_placement)

    p = sub.add_parser("timeout", help="mark one coordinator/controller as expected timeout from --min-n on")
    add_filters(p, coordinator_required=True, controller=True)
    p.set_defaults(func=cmd_timeout)

    p = sub.add_parser("list", help="show synthetic rows currently in the table")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("undo", help="revert synthetic rows back to pending")
    p.add_argument("--coordinator", choices=COORDINATORS, default=None)
    p.add_argument("--controller", choices=CONTROLLERS, default=None)
    p.add_argument("--min-n", type=int, default=0)
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_undo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
