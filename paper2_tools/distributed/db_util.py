"""DB utility commands for checking status, cleaning duplicates, etc.

Usage:
    python -m paper2_tools.distributed.db_util status
    python -m paper2_tools.distributed.db_util dedup
    python -m paper2_tools.distributed.db_util packages   # owner host and progress per (experiment, variant, N, coordinator, controller)
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg2


def _connect():
    return psycopg2.connect(os.environ["DRONESIM_DATABASE_URL"])


def cmd_status() -> None:
    conn = _connect()
    cur = conn.cursor()

    cur.execute("SELECT status, count(*) FROM scenario_runs GROUP BY status ORDER BY status")
    print("scenario_runs:")
    for r in cur.fetchall():
        print(f"  {r[0]}: {r[1]}")

    cur.execute("SELECT count(*) FROM metrics")
    print(f"\nmetrics: {cur.fetchone()[0]} rows")

    conn.close()


def cmd_packages() -> None:
    """Show every package (N, coordinator, controller): which host owns it and how far it is."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT experiment, variant, n_drones, coordinator, controller, "
        "  string_agg(DISTINCT claimed_by, ',') FILTER (WHERE claimed_by IS NOT NULL) AS owners, "
        "  sum((status = 'pending')::int), sum((status = 'running')::int), sum((status = 'done')::int), "
        "  sum((status = 'error')::int), count(*) "
        "FROM scenario_runs GROUP BY experiment, variant, n_drones, coordinator, controller "
        "ORDER BY experiment, variant, n_drones, coordinator, controller"
    )
    print(f"{'experiment':<10} {'var':<4} {'N':>3} {'coordinator':<12} {'controller':<20} {'owner':<24} {'pend':>4} {'run':>4} {'done':>4} {'err':>4} {'total':>5}")
    for experiment, variant, n, coord, ctrl, owners, pending, running, done, error, total in cur.fetchall():
        print(f"{experiment:<10} {variant:<4} {n:>3} {coord:<12} {ctrl:<20} {(owners or '-'):<24} {pending:>4} {running:>4} {done:>4} {error:>4} {total:>5}")
    conn.close()


def cmd_dedup() -> None:
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM metrics WHERE id NOT IN ("
        "  SELECT MIN(id) FROM metrics GROUP BY "
        "  num_drones, horizon, coordinator, controller_type, "
        "  status, steps, wall_time_s, jerk_3d_value"
        ")"
    )
    print(f"Deleted {cur.rowcount} duplicate metrics rows")
    conn.commit()

    cur.execute("SELECT count(*) FROM metrics")
    print(f"Remaining: {cur.fetchone()[0]} rows")

    conn.close()


def cmd_errors() -> None:
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        "SELECT id, n_drones, coordinator, controller, error_message "
        "FROM scenario_runs WHERE status = 'error' ORDER BY id LIMIT 20"
    )
    rows = cur.fetchall()
    if not rows:
        print("No errors found")
    else:
        for r in rows:
            print(f"  id={r[0]} n={r[1]} {r[2]}/{r[3]}: {r[4]}")

    conn.close()


def cmd_reset_stale() -> None:
    """Reset 'running' tasks back to 'pending' (e.g. after a worker was killed)."""
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        "UPDATE scenario_runs SET status = 'pending', claimed_by = NULL, claimed_at = NULL "
        "WHERE status = 'running'"
    )
    print(f"Reset {cur.rowcount} 'running' entries back to 'pending'")
    conn.commit()

    cur.execute("SELECT status, count(*) FROM scenario_runs GROUP BY status ORDER BY status")
    for r in cur.fetchall():
        print(f"  {r[0]}: {r[1]}")

    conn.close()


def cmd_reset_orphans() -> None:
    """Reset all imported 'done' back to 'pending', then re-mark only those
    with actual metrics as 'done'. Handles the NULL scenario_run_id case."""
    conn = _connect()
    cur = conn.cursor()

    # Step 1: reset all done to pending
    cur.execute("UPDATE scenario_runs SET status = 'pending' WHERE status = 'done'")
    reset = cur.rowcount
    print(f"Reset {reset} 'done' entries to 'pending'")

    # Step 2: for each metrics row, mark ONE matching pending scenario_run as done
    cur.execute(
        "SELECT num_drones, coordinator, controller_type FROM metrics ORDER BY id"
    )
    metrics_rows = cur.fetchall()
    marked = 0
    for n, coord, ctrl in metrics_rows:
        cur.execute(
            "UPDATE scenario_runs SET status = 'done' WHERE id = ("
            "  SELECT id FROM scenario_runs "
            "  WHERE n_drones = %s AND coordinator = %s AND controller = %s AND status = 'pending' "
            "  ORDER BY id LIMIT 1"
            ") RETURNING id",
            (n, coord, ctrl),
        )
        if cur.fetchone():
            marked += 1
    conn.commit()
    print(f"Re-marked {marked} entries as 'done' (matching metrics)")

    cur.execute("SELECT status, count(*) FROM scenario_runs GROUP BY status ORDER BY status")
    for r in cur.fetchall():
        print(f"  {r[0]}: {r[1]}")

    conn.close()


def cmd_runsql(sql_file: str) -> None:
    """Execute a SQL file against the database."""
    from pathlib import Path

    path = Path(sql_file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    sql = path.read_text()
    conn = _connect()
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    print(f"Executed {path} ({cur.statusmessage})")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="DB utility commands")
    parser.add_argument("command", choices=["status", "packages", "dedup", "errors", "reset-stale", "reset-orphans", "runsql"])
    parser.add_argument("sql_file", nargs="?", help="Path to SQL file (required for runsql)")
    args = parser.parse_args()

    if args.command == "runsql":
        if not args.sql_file:
            parser.error("runsql requires a SQL file path")
        cmd_runsql(args.sql_file)
    else:
        cmds = {"status": cmd_status, "packages": cmd_packages, "dedup": cmd_dedup, "errors": cmd_errors, "reset-stale": cmd_reset_stale, "reset-orphans": cmd_reset_orphans}
        cmds[args.command]()


if __name__ == "__main__":
    main()
