"""Generate SQL to import existing results/ and results_n38/ into the distributed DB schema.

Usage:
    python -m paper2_tools.distributed.import_existing > import.sql
    psql "$DRONESIM_DATABASE_URL" -f import.sql
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "paper2_results"
COMBOS_PER_RUN = 35 * 2 * 2  # 35 n_drones (2-36) * 2 coordinators * 2 controllers = 140


def _escape(val: str) -> str:
    return val.replace("'", "''")


def _generate_scenario_runs(queue_path: Path, run_index_offset: int) -> list[str]:
    """Generate INSERT statements for scenario_runs from a queue CSV."""
    stmts = []
    with queue_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            queue_id = int(row["id"])
            run_index = run_index_offset + queue_id // COMBOS_PER_RUN
            n_drones = int(row["n_drones"])
            coordinator = _escape(row["coordinator"])
            controller = _escape(row["controller"])

            if row["status"] == "done":
                status = "done"
            elif row["status"] in ("toBeDone", "inProcess"):
                status = "pending"
            else:
                status = "pending"

            stmts.append(
                f"INSERT INTO scenario_runs (run_index, n_drones, coordinator, controller, status) "
                f"VALUES ({run_index}, {n_drones}, '{coordinator}', '{controller}', '{status}') "
                f"ON CONFLICT (run_index, n_drones, coordinator, controller) DO NOTHING;"
            )
    return stmts


def _generate_metrics(metrics_path: Path, source_label: str) -> list[str]:
    """Generate INSERT statements for metrics from a metrics CSV.

    These rows have no scenario_run_id link (not possible to reconstruct).
    """
    stmts = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            num_drones = int(row["num_drones"])
            horizon = int(row["horizon"])
            coordinator = _escape(row["coordinator"])
            controller_type = _escape(row["controller_type"])
            status = _escape(row["status"])
            steps = int(row["steps"])
            wall_time_s = float(row["wall_time_s"])
            min_step_time_s = float(row["min_step_time_s"])
            max_step_time_s = float(row["max_step_time_s"])
            mean_step_time_s = float(row["mean_step_time_s"])
            min_distance = float(row["min_distance"])
            max_distance = float(row["max_distance"])
            mean_distance_all_pairs = float(row["mean_distance_all_pairs"])
            mean_distance_step_mean = float(row["mean_distance_step_mean"])
            jerk_3d_value = float(row["jerk_3d_value"])

            stmts.append(
                f"INSERT INTO metrics ("
                f"scenario_run_id, num_drones, horizon, coordinator, controller_type, "
                f"status, steps, wall_time_s, min_step_time_s, max_step_time_s, "
                f"mean_step_time_s, min_distance, max_distance, mean_distance_all_pairs, "
                f"mean_distance_step_mean, jerk_3d_value"
                f") VALUES ("
                f"NULL, {num_drones}, {horizon}, '{coordinator}', '{controller_type}', "
                f"'{status}', {steps}, {wall_time_s}, {min_step_time_s}, {max_step_time_s}, "
                f"{mean_step_time_s}, {min_distance}, {max_distance}, {mean_distance_all_pairs}, "
                f"{mean_distance_step_mean}, {jerk_3d_value}"
                f"); -- source: {source_label}"
            )
    return stmts


def main() -> None:
    out = sys.stdout

    out.write("-- Auto-generated: Import existing CSV results into distributed DB schema\n")
    out.write("-- results/  -> run_index 0..19  (20 runs, drones 2-36)\n")
    out.write("-- results_n38/ -> run_index 20..39 (20 runs, drones 2-36)\n\n")

    out.write("BEGIN;\n\n")

    # --- scenario_runs from results/ (offset 0) ---
    queue1 = RESULTS_DIR / "results" / "scenarios_queue.csv"
    if queue1.exists():
        out.write(f"-- scenario_runs from {queue1.name} (run_index 0..19)\n")
        for stmt in _generate_scenario_runs(queue1, run_index_offset=0):
            out.write(stmt + "\n")
        out.write("\n")

    # # --- scenario_runs from results_n38/ (offset 20) ---
    # queue2 = RESULTS_DIR / "results_n38" / "scenarios_queue.csv"
    # if queue2.exists():
    #     out.write(f"-- scenario_runs from results_n38/{queue2.name} (run_index 20..39)\n")
    #     for stmt in _generate_scenario_runs(queue2, run_index_offset=20):
    #         out.write(stmt + "\n")
    #     out.write("\n")

    # --- metrics from results/ ---
    metrics1 = RESULTS_DIR / "results" / "metrics.csv"
    if metrics1.exists():
        out.write(f"-- metrics from results/ ({sum(1 for _ in open(metrics1)) - 1} rows)\n")
        for stmt in _generate_metrics(metrics1, "results/"):
            out.write(stmt + "\n")
        out.write("\n")

    # # --- metrics from results_n38/ ---
    # metrics2 = RESULTS_DIR / "results_n38" / "metrics.csv"
    # if metrics2.exists():
    #     out.write(f"-- metrics from results_n38/ ({sum(1 for _ in open(metrics2)) - 1} rows)\n")
    #     for stmt in _generate_metrics(metrics2, "results_n38/"):
    #         out.write(stmt + "\n")
    #     out.write("\n")

    out.write("COMMIT;\n")

    # Summary to stderr
    print(f"-- Summary:", file=sys.stderr)
    if queue1.exists():
        with queue1.open() as f:
            rows = list(csv.DictReader(f))
            done = sum(1 for r in rows if r["status"] == "done")
            pending = len(rows) - done
            print(f"  results/:     {done} done, {pending} pending -> run_index 0..19", file=sys.stderr)
    # if queue2.exists():
    #     with queue2.open() as f:
    #         rows = list(csv.DictReader(f))
    #         done = sum(1 for r in rows if r["status"] == "done")
    #         pending = len(rows) - done
    #         print(f"  results_n38/: {done} done, {pending} pending -> run_index 20..39", file=sys.stderr)
    if metrics1.exists():
        with metrics1.open() as f:
            n = sum(1 for _ in csv.DictReader(f))
            print(f"  results/ metrics:     {n} rows", file=sys.stderr)
    # if metrics2.exists():
    #     with metrics2.open() as f:
    #         n = sum(1 for _ in csv.DictReader(f))
    #         print(f"  results_n38/ metrics: {n} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
