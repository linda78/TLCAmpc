"""Database schema, task claiming, and metrics storage for distributed scenario runs.

A task row of ``scenario_runs`` names its campaign cell by (run_index, n_drones, coordinator, controller, experiment, variant) and
carries the per-run settings the worker needs beyond its command line: room_size, comm_radius, max_admm_iter and timeout_s.
NULL means "use the worker default". The volume campaigns (paper 2 revision, paper 3 costs) are experiment 'volume' with an
empty variant; the constant-density scaling experiment (paper2_tools/review_work/density_scaling) is experiment 'density'.
"""

from __future__ import annotations

import socket
from typing import Any

import psycopg2
import psycopg2.extras

from paper2_tools.distributed.config import VOLUME_EXPERIMENT, get_database_url

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TASK_KEY_CONSTRAINT = "scenario_runs_task_key"
_LEGACY_TASK_KEY_CONSTRAINT = "scenario_runs_run_index_n_drones_coordinator_controller_key"

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS scenario_runs (
    id              SERIAL PRIMARY KEY,
    run_index       INTEGER NOT NULL,
    n_drones        INTEGER NOT NULL,
    coordinator     TEXT NOT NULL,
    controller      TEXT NOT NULL,
    experiment      TEXT NOT NULL DEFAULT '{VOLUME_EXPERIMENT}',
    variant         TEXT NOT NULL DEFAULT '',
    room_size       DOUBLE PRECISION,
    comm_radius     DOUBLE PRECISION,
    max_admm_iter   INTEGER,
    timeout_s       INTEGER,
    status          TEXT NOT NULL DEFAULT 'pending',
    claimed_by      TEXT,
    claimed_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    CONSTRAINT {TASK_KEY_CONSTRAINT} UNIQUE (run_index, n_drones, coordinator, controller, experiment, variant)
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
    # per-drone DMPC columns, see CpuCostLedger.add_seconds
    ("local_solves", "INTEGER"),
    ("local_solve_cpu_s", "DOUBLE PRECISION"),
    ("parallel_step_cpu_s", "DOUBLE PRECISION"),
    ("max_admm_iter_step", "INTEGER"),
]

# Effective run settings the worker used, stored next to the metrics so an export needs no lookup of worker defaults.
_TASK_PARAM_COLUMNS: list[tuple[str, str]] = [
    ("room_size", "DOUBLE PRECISION"),
    ("comm_radius", "DOUBLE PRECISION"),
    ("max_admm_iter", "INTEGER"),
]

# Task columns of scenario_runs that came after the first campaigns.
_RUN_COLUMNS: list[tuple[str, str]] = [
    ("experiment", f"TEXT NOT NULL DEFAULT '{VOLUME_EXPERIMENT}'"),
    ("variant", "TEXT NOT NULL DEFAULT ''"),
    ("room_size", "DOUBLE PRECISION"),
    ("comm_radius", "DOUBLE PRECISION"),
    ("max_admm_iter", "INTEGER"),
    ("timeout_s", "INTEGER"),
]

_MIGRATION_SQL = "\n".join(f"ALTER TABLE metrics ADD COLUMN IF NOT EXISTS {name} {sql_type};" for name, sql_type in _COST_COLUMNS)
_TASK_PARAM_MIGRATION_SQL = "\n".join(f"ALTER TABLE metrics ADD COLUMN IF NOT EXISTS {name} {sql_type};" for name, sql_type in _TASK_PARAM_COLUMNS)
_RUNS_MIGRATION_SQL = "\n".join(f"ALTER TABLE scenario_runs ADD COLUMN IF NOT EXISTS {name} {sql_type};" for name, sql_type in _RUN_COLUMNS)

# The unique key grew by experiment and variant: swap the auto-named legacy constraint for the named one, once.
_TASK_KEY_MIGRATION_SQL = f"""
ALTER TABLE scenario_runs DROP CONSTRAINT IF EXISTS {_LEGACY_TASK_KEY_CONSTRAINT};
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '{TASK_KEY_CONSTRAINT}') THEN
        ALTER TABLE scenario_runs ADD CONSTRAINT {TASK_KEY_CONSTRAINT}
            UNIQUE (run_index, n_drones, coordinator, controller, experiment, variant);
    END IF;
END $$;
"""

_BASE_METRIC_COLUMNS = [
    "num_drones", "horizon", "coordinator", "controller_type", "status", "steps",
    "wall_time_s", "min_step_time_s", "max_step_time_s", "mean_step_time_s",
    "min_distance", "max_distance", "mean_distance_all_pairs",
    "mean_distance_step_mean", "jerk_3d_value",
]
_METRIC_COLUMNS = _BASE_METRIC_COLUMNS + [name for name, _ in _COST_COLUMNS]
_TASK_PARAM_NAMES = [name for name, _ in _TASK_PARAM_COLUMNS]
_TASK_COLUMNS = ["run_index", "n_drones", "coordinator", "controller", "experiment", "variant", "room_size", "comm_radius", "max_admm_iter", "timeout_s"]


def get_connection():
    return psycopg2.connect(get_database_url())


def ensure_schema() -> None:
    """Create the tables and bring a database from before the task settings and cost columns up to date. Idempotent."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
            cur.execute(_MIGRATION_SQL)
            cur.execute(_TASK_PARAM_MIGRATION_SQL)
            cur.execute(_RUNS_MIGRATION_SQL)
            cur.execute(_TASK_KEY_MIGRATION_SQL)
        conn.commit()


# ---------------------------------------------------------------------------
# Queue population
# ---------------------------------------------------------------------------

def make_task(run_index: int, n_drones: int, coordinator: str, controller: str, *, experiment: str = VOLUME_EXPERIMENT, variant: str = "",
              room_size: float | None = None, comm_radius: float | None = None, max_admm_iter: int | None = None,
              timeout_s: int | None = None) -> dict[str, Any]:
    """One queue row. A None setting means the worker default."""
    return {"run_index": run_index, "n_drones": n_drones, "coordinator": coordinator, "controller": controller, "experiment": experiment,
            "variant": variant, "room_size": room_size, "comm_radius": comm_radius, "max_admm_iter": max_admm_iter, "timeout_s": timeout_s}


def insert_tasks(tasks: list[dict[str, Any]]) -> int:
    """Insert pending tasks (dicts from make_task). Rows whose key already exists are left alone. Returns the number inserted."""
    if not tasks:
        return 0
    sql = f"""
        INSERT INTO scenario_runs ({', '.join(_TASK_COLUMNS)})
        VALUES %s
        ON CONFLICT ON CONSTRAINT {TASK_KEY_CONSTRAINT} DO NOTHING
        RETURNING id
    """
    template = "(" + ", ".join(f"%({name})s" for name in _TASK_COLUMNS) + ")"
    with get_connection() as conn:
        with conn.cursor() as cur:
            inserted = psycopg2.extras.execute_values(cur, sql, tasks, template=template, page_size=500, fetch=True)
        conn.commit()
    return len(inserted)


def insert_runs(runs: range, drones: range, coordinators: list[str], controllers: list[str]) -> int:
    """Insert the volume campaign runs for every (N, coordinator, controller) combination. Returns number of rows inserted."""
    tasks = [make_task(run_idx, n, coord, ctrl)
             for run_idx in runs for n in drones for coord in coordinators for ctrl in controllers]
    return insert_tasks(tasks)


# ---------------------------------------------------------------------------
# Task claiming in packages
# ---------------------------------------------------------------------------
#
# A package is one (experiment, variant, n_drones, coordinator, controller) combination with all its runs. It belongs to the host
# that claimed its first row (claimed_by is the hostname, shared by all worker processes of a server). A process first takes a
# pending row of a package its host already owns, smallest N first, and only when none is left starts the lowest-N package nobody
# owns. The advisory lock serialises all claims, so two hosts can never start the same package at the same time.

_CLAIM_LOCK_ID = 7312001

_PACKAGE_OWNED_BY_HOST = """EXISTS (
                SELECT 1 FROM scenario_runs o
                WHERE o.n_drones = r.n_drones AND o.coordinator = r.coordinator AND o.controller = r.controller
                  AND o.experiment = r.experiment AND o.variant = r.variant
                  AND o.claimed_by = %s
            )"""

_PACKAGE_UNOWNED = """NOT EXISTS (
                SELECT 1 FROM scenario_runs o
                WHERE o.n_drones = r.n_drones AND o.coordinator = r.coordinator AND o.controller = r.controller
                  AND o.experiment = r.experiment AND o.variant = r.variant
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
        RETURNING id, {', '.join(_TASK_COLUMNS)};
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

def insert_metrics(scenario_run_id: int, row: dict[str, Any], task_params: dict[str, Any] | None = None) -> None:
    """Insert a metrics row linked to the scenario run.

    Cost columns missing from ``row`` are stored as NULL. ``task_params`` holds the effective room_size, comm_radius and
    max_admm_iter of the run (see worker.resolve_task_settings); without it those columns stay NULL.
    """
    columns = ["scenario_run_id", *_METRIC_COLUMNS, *_TASK_PARAM_NAMES]
    sql = f"INSERT INTO metrics ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"
    params = (scenario_run_id,
              *[row[name] for name in _BASE_METRIC_COLUMNS],
              *[row.get(name) for name, _ in _COST_COLUMNS],
              *[(task_params or {}).get(name) for name in _TASK_PARAM_NAMES])
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

def count_by_status(experiment: str | None = None) -> dict[str, int]:
    where = "" if experiment is None else " WHERE experiment = %s"
    sql = f"SELECT status, count(*) FROM scenario_runs{where} GROUP BY status ORDER BY status"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, () if experiment is None else (experiment,))
            return {row[0]: row[1] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

# Metrics columns first (unchanged order for the plot scripts), then the effective run settings and the task identity.
_EXPORT_RUN_FIELDS = ["run_index", "experiment", "variant"]
_EXPORT_FIELDNAMES = [*_METRIC_COLUMNS, *_TASK_PARAM_NAMES, *_EXPORT_RUN_FIELDS]


def export_metrics_rows(experiment: str | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """Fetch all completed metrics rows, optionally of one experiment only. Returns (fieldnames, rows)."""
    cols = ", ".join([*(f"m.{c}" for c in _METRIC_COLUMNS), *(f"m.{c}" for c in _TASK_PARAM_NAMES), *(f"r.{c}" for c in _EXPORT_RUN_FIELDS)])
    where = "" if experiment is None else " WHERE r.experiment = %s"
    sql = f"SELECT {cols} FROM metrics m LEFT JOIN scenario_runs r ON r.id = m.scenario_run_id{where} ORDER BY m.num_drones, m.id"
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, () if experiment is None else (experiment,))
            rows = [dict(r) for r in cur.fetchall()]
    return _EXPORT_FIELDNAMES, rows
