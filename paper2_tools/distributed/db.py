"""Database schema, task claiming, and metrics storage for distributed scenario runs."""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

from paper2_tools.distributed.config import get_database_url

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scenario_runs (
    id              SERIAL PRIMARY KEY,
    run_index       INTEGER NOT NULL,
    n_drones        INTEGER NOT NULL,
    coordinator     TEXT NOT NULL,
    controller      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    claimed_by      TEXT,
    claimed_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    UNIQUE(run_index, n_drones, coordinator, controller)
);

CREATE TABLE IF NOT EXISTS metrics (
    id                      SERIAL PRIMARY KEY,
    scenario_run_id         INTEGER REFERENCES scenario_runs(id),
    num_drones              INTEGER NOT NULL,
    horizon                 INTEGER NOT NULL,
    coordinator             TEXT NOT NULL,
    controller_type         TEXT NOT NULL,
    status                  TEXT NOT NULL,
    steps                   INTEGER NOT NULL,
    wall_time_s             DOUBLE PRECISION,
    min_step_time_s         DOUBLE PRECISION,
    max_step_time_s         DOUBLE PRECISION,
    mean_step_time_s        DOUBLE PRECISION,
    min_distance            DOUBLE PRECISION,
    max_distance            DOUBLE PRECISION,
    mean_distance_all_pairs DOUBLE PRECISION,
    mean_distance_step_mean DOUBLE PRECISION,
    jerk_3d_value           DOUBLE PRECISION,
    created_at              TIMESTAMPTZ DEFAULT now()
);
"""

# CPU-time columns of one run, filled from tools.utility.printer.cost_row (same names and order as COST_FIELDNAMES there).
# They are added to existing databases by ensure_schema(); rows written before that keep NULL.
_COST_COLUMNS: list[tuple[str, str]] = [
    ("cpu_total_s", "DOUBLE PRECISION"),
    *((f"cpu_{phase}_s", "DOUBLE PRECISION") for phase in
      ("setup", "cost", "constraints", "solver", "validate", "exchange", "admm_update", "physics", "metrics", "deadlock")),
    ("admm_iterations", "INTEGER"),
    ("cost_evals", "INTEGER"),
    ("constraint_evals", "INTEGER"),
]

_MIGRATION_SQL = "\n".join(f"ALTER TABLE metrics ADD COLUMN IF NOT EXISTS {name} {sql_type};" for name, sql_type in _COST_COLUMNS)

_BASE_METRIC_COLUMNS = [
    "num_drones", "horizon", "coordinator", "controller_type", "status", "steps",
    "wall_time_s", "min_step_time_s", "max_step_time_s", "mean_step_time_s",
    "min_distance", "max_distance", "mean_distance_all_pairs",
    "mean_distance_step_mean", "jerk_3d_value",
]
_METRIC_COLUMNS = _BASE_METRIC_COLUMNS + [name for name, _ in _COST_COLUMNS]


def get_connection():
    return psycopg2.connect(get_database_url())


def ensure_schema() -> None:
    """Create the tables and add the cost columns to a metrics table from before they existed. Idempotent."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
            cur.execute(_MIGRATION_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Queue population
# ---------------------------------------------------------------------------

def insert_runs(runs: range, drones: range, coordinators: list[str], controllers: list[str]) -> int:
    """Insert pending scenario runs into the queue. Returns number of rows inserted."""
    values = []
    for run_idx in runs:
        for n in drones:
            for coord in coordinators:
                for ctrl in controllers:
                    values.append((run_idx, n, coord, ctrl))

    sql = """
        INSERT INTO scenario_runs (run_index, n_drones, coordinator, controller)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (run_index, n_drones, coordinator, controller) DO NOTHING
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, values, page_size=500)
            inserted = cur.rowcount
        conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Task claiming in packages
# ---------------------------------------------------------------------------
#
# A package is one (n_drones, coordinator, controller) combination with all its runs. It belongs to the host that claimed its
# first row (claimed_by is the hostname, shared by all worker processes of a server). A process first takes a pending row of a
# package its host already owns, smallest N first, and only when none is left starts the lowest-N package nobody owns. The
# advisory lock serialises all claims, so two hosts can never start the same package at the same time.

_CLAIM_LOCK_ID = 7312001

_PACKAGE_OWNED_BY_HOST = """EXISTS (
                SELECT 1 FROM scenario_runs o
                WHERE o.n_drones = r.n_drones AND o.coordinator = r.coordinator AND o.controller = r.controller
                  AND o.claimed_by = %s
            )"""

_PACKAGE_UNOWNED = """NOT EXISTS (
                SELECT 1 FROM scenario_runs o
                WHERE o.n_drones = r.n_drones AND o.coordinator = r.coordinator AND o.controller = r.controller
                  AND o.claimed_by IS NOT NULL
            )"""


def _claim_sql(package_filter: str) -> str:
    return f"""
        UPDATE scenario_runs
        SET status = 'running', claimed_by = %s, claimed_at = now()
        WHERE id = (
            SELECT r.id FROM scenario_runs r
            WHERE r.status = 'pending' AND {package_filter}
            ORDER BY r.n_drones ASC, r.id ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, run_index, n_drones, coordinator, controller;
    """


def claim_task() -> dict[str, Any] | None:
    """Atomically claim the next pending task of this host's packages, or start a new package. Returns row dict or None."""
    hostname = socket.gethostname()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_CLAIM_LOCK_ID,))
            cur.execute(_claim_sql(_PACKAGE_OWNED_BY_HOST), (hostname, hostname))
            row = cur.fetchone()
            if row is None:
                cur.execute(_claim_sql(_PACKAGE_UNOWNED), (hostname,))
                row = cur.fetchone()
        conn.commit()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------

def insert_metrics(scenario_run_id: int, row: dict[str, Any]) -> None:
    """Insert a metrics row linked to the scenario run. Cost columns missing from ``row`` are stored as NULL."""
    columns = ["scenario_run_id", *_METRIC_COLUMNS]
    sql = f"INSERT INTO metrics ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
    params = (scenario_run_id, *[row[name] for name in _BASE_METRIC_COLUMNS], *[row.get(name) for name, _ in _COST_COLUMNS])
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def mark_done(scenario_run_id: int) -> None:
    sql = "UPDATE scenario_runs SET status = 'done', completed_at = now() WHERE id = %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (scenario_run_id,))
        conn.commit()


def mark_error(scenario_run_id: int, message: str) -> None:
    sql = "UPDATE scenario_runs SET status = 'error', completed_at = now(), error_message = %s WHERE id = %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (message, scenario_run_id))
        conn.commit()


# ---------------------------------------------------------------------------
# Status queries
# ---------------------------------------------------------------------------

def count_by_status() -> dict[str, int]:
    sql = "SELECT status, count(*) FROM scenario_runs GROUP BY status ORDER BY status"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return {row[0]: row[1] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_EXPORT_FIELDNAMES = list(_METRIC_COLUMNS)


def export_metrics_rows() -> tuple[list[str], list[dict[str, Any]]]:
    """Fetch all completed metrics rows. Returns (fieldnames, rows)."""
    cols = ", ".join(_EXPORT_FIELDNAMES)
    sql = f"SELECT {cols} FROM metrics ORDER BY num_drones, id"
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
    return _EXPORT_FIELDNAMES, rows
