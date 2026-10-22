"""CLI: Process the queue breadth-first over drone counts, in waves of run indices.

The workers claim with ORDER BY n_drones, id, so by default every run of N is
finished before the first run of N+1 starts. To get at least one run per N
early, hide all pending rows except the lowest run indices from the workers by
setting their status to 'deferred' (claim_task only takes 'pending'). The
visible rows are then processed N = 2..25 for run 0, then run 1, ... Release the
next block(s) before the pending rows run dry: a worker whose queue is empty
exits with code 0 and systemd (Restart=on-failure) does NOT restart it.

No code change on the workers, no restart, running tasks are never touched.
Row ids are never changed (metrics reference them, workers hold them in memory).

Usage:
    python -m paper2_tools.distributed.queue_waves status
    python -m paper2_tools.distributed.queue_waves defer   --keep 2        [--dry-run]   # run 0,1 stay pending, rest -> deferred
    python -m paper2_tools.distributed.queue_waves release --through 4     [--dry-run]   # deferred rows with run < 4 -> pending
    python -m paper2_tools.distributed.queue_waves release --all           [--dry-run]
    python -m paper2_tools.distributed.queue_waves topup   --min-pending 40 [--dry-run]  # release next block(s) until >= 40 pending
"""

from __future__ import annotations

import argparse

from paper2_tools.distributed.db import get_connection

DEFERRED = "deferred"


def _counts_by_run(cur) -> list[tuple]:
    cur.execute(
        """
        SELECT run_index,
               sum((status = 'pending')::int)  AS pending,
               sum((status = %s)::int)         AS deferred,
               sum((status = 'running')::int)  AS running,
               sum((status = 'done')::int)     AS done,
               min(n_drones) FILTER (WHERE status = 'pending') AS next_n
        FROM scenario_runs
        GROUP BY run_index ORDER BY run_index
        """,
        (DEFERRED,),
    )
    return cur.fetchall()


def _print_counts(rows: list[tuple]) -> None:
    print("run  pending deferred running done  next_pending_N")
    for run, pend, defer, running, done, next_n in rows:
        print(f"{run:3d}  {pend:7d} {defer:8d} {running:7d} {done:4d}  {next_n if next_n is not None else '-'}")
    tot = [sum(r[i] for r in rows) for i in range(1, 5)]
    print(f"sum  {tot[0]:7d} {tot[1]:8d} {tot[2]:7d} {tot[3]:4d}")


def cmd_status(_args: argparse.Namespace) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            _print_counts(_counts_by_run(cur))


def _update(sql: str, params: tuple, dry_run: bool, verb: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql.replace("UPDATE scenario_runs SET status = %s", "SELECT count(*) FROM scenario_runs")
                        if dry_run else sql, params[1:] if dry_run else params)
            n = cur.fetchone()[0] if dry_run else cur.rowcount
            if dry_run:
                conn.rollback()
                print(f"Would {verb} {n} rows.")
                return
            conn.commit()
            print(f"{verb.capitalize()} {n} rows.")
        with conn.cursor() as cur:
            _print_counts(_counts_by_run(cur))


def cmd_defer(args: argparse.Namespace) -> None:
    sql = "UPDATE scenario_runs SET status = %s WHERE status = 'pending' AND run_index >= %s"
    _update(sql, (DEFERRED, args.keep), args.dry_run, f"defer (run >= {args.keep})")


def cmd_release(args: argparse.Namespace) -> None:
    if args.all:
        sql = "UPDATE scenario_runs SET status = %s WHERE status = %s AND run_index >= %s"
        _update(sql, ("pending", DEFERRED, 0), args.dry_run, "release (all)")
    else:
        sql = "UPDATE scenario_runs SET status = %s WHERE status = %s AND run_index < %s"
        _update(sql, ("pending", DEFERRED, args.through), args.dry_run, f"release (run < {args.through})")


def cmd_topup(args: argparse.Namespace) -> None:
    """Release deferred blocks, lowest run_index first, until >= min_pending rows are pending."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            released = []
            while True:
                cur.execute("SELECT count(*) FROM scenario_runs WHERE status = 'pending'")
                pending = cur.fetchone()[0]
                if pending >= args.min_pending:
                    break
                cur.execute("SELECT min(run_index) FROM scenario_runs WHERE status = %s", (DEFERRED,))
                nxt = cur.fetchone()[0]
                if nxt is None:
                    break
                cur.execute("UPDATE scenario_runs SET status = 'pending' WHERE status = %s AND run_index = %s", (DEFERRED, nxt))
                released.append((nxt, cur.rowcount))
            if args.dry_run:
                conn.rollback()
            else:
                conn.commit()
        verb = "Would release" if args.dry_run else "Released"
        if released:
            print(f"{verb}: " + ", ".join(f"run {r} ({n} rows)" for r, n in released))
        else:
            print(f"Nothing to release (pending = {pending}, min = {args.min_pending}).")
        if not args.dry_run:
            with conn.cursor() as cur:
                _print_counts(_counts_by_run(cur))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Breadth-first queue processing in run-index waves")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="pending/deferred/running/done per run index")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("defer", help="hide pending rows with run_index >= --keep from the workers")
    p.add_argument("--keep", type=int, required=True, help="number of run indices to keep pending (0..keep-1)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_defer)

    p = sub.add_parser("release", help="make deferred rows pending again")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--through", type=int, help="release run indices < this value")
    g.add_argument("--all", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("topup", help="release next block(s) until at least --min-pending rows are pending")
    p.add_argument("--min-pending", type=int, default=40)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_topup)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
