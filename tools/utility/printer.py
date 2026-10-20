
from enum import StrEnum
from pathlib import Path
from PIL import Image
import numpy as np
import csv
from typing import Any

from drone_sim.simulation.utils.cpu_costs import CpuCostLedger

class Status(StrEnum):
   RUNNING = "running"
   FINISHED = "finished"
   MAX_STEPS = "max_steps"
   INFEASIBLE = "infeasible"
   NO_POSSIBLE_PLACEMENT = "no_possible_placement"
   ERROR = "error"
   TIMEOUT = "timeout"
   DEADLOCK = "deadlock"


# CPU-time phases of one run, see drone_sim.simulation.utils.cpu_costs. Each becomes a "cpu_<phase>_s" column.
COST_PHASES = ("setup", "cost", "constraints", "solver", "validate", "exchange", "admm_update", "physics", "metrics", "deadlock")
COST_FIELDNAMES = ["cpu_total_s", *(f"cpu_{phase}_s" for phase in COST_PHASES), "admm_iterations", "cost_evals", "constraint_evals"]


def cost_row(ledger: CpuCostLedger, cpu_total_s: float) -> dict[str, float | int]:
   """Flatten a ledger into the metric columns of COST_FIELDNAMES. cpu_total_s is the CPU time of the whole run."""
   row: dict[str, float | int] = {"cpu_total_s": cpu_total_s}
   for phase in COST_PHASES:
      row[f"cpu_{phase}_s"] = ledger.seconds.get(phase, 0.0)
   row["admm_iterations"] = ledger.counts.get("admm_iterations", 0)
   row["cost_evals"] = ledger.counts.get("cost", 0)
   row["constraint_evals"] = ledger.counts.get("constraints", 0)
   return row

def thread_unsafe_print_results(all_pair_dists: list[float], horizon: int, jerk_3d_value: float, num_drones: int, out_dir: Path, status: Status,
                                step_durations: list[float], step_mean_pair_dists: list[float], wall_time: float, coordinator_type: str, controller_type: str,
                                costs: dict[str, float | int] | None = None) -> None:
   fieldnames, row = print_results_prep(all_pair_dists=all_pair_dists, horizon=horizon, jerk_3d_value=jerk_3d_value, num_drones=num_drones, status=status,
                                        step_durations=step_durations, step_mean_pair_dists=step_mean_pair_dists, wall_time=wall_time, coordinator_type=coordinator_type, controller_type=controller_type,
                                        costs=costs)
   csv_path = out_dir / "metrics.csv"
   write_header = not csv_path.exists()
   with csv_path.open("a", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      if write_header:
         writer.writeheader()
      writer.writerow(row)

def print_gif(frames: list[Image.Image], gif_fps: float, gif_path: Path):
   if frames:
      duration_ms = int(round(1000.0 / max(0.1, float(gif_fps))))
      frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)

def print_results_prep(all_pair_dists: list[float], horizon: int, jerk_3d_value: float, num_drones: int, status: Status, step_durations: list[float],
                       step_mean_pair_dists: list[float], wall_time: float, coordinator_type: str, controller_type: str,
                       costs: dict[str, float | int] | None = None) -> tuple[list[str], dict[str, Any]]:
   """Build the metrics CSV/DB row of one run. ``costs`` is the cost_row() of the run; without it the cost columns stay empty."""
   # Print metrics summary for this scenario.
   num_steps = len(step_durations)

   if num_steps > 0:
      step_durations_arr = np.asarray(step_durations, dtype=float)
      min_step_time = float(step_durations_arr.min())
      max_step_time = float(step_durations_arr.max())
      mean_step_time = float(step_durations_arr.mean())
   else:
      min_step_time = max_step_time = mean_step_time = 0.0

   if all_pair_dists:
      all_pair_dists_arr = np.asarray(all_pair_dists, dtype=float)
      min_dist = float(all_pair_dists_arr.min())
      max_dist = float(all_pair_dists_arr.max())
      mean_dist = float(all_pair_dists_arr.mean())
   else:
      min_dist = max_dist = mean_dist = 0.0

   if step_mean_pair_dists:
      mean_step_mean_dist = float(np.asarray(step_mean_pair_dists, dtype=float).mean())
   else:
      mean_step_mean_dist = 0.0

   print(f"  distances: min={min_dist:.3f}, max={max_dist:.3f}, mean(all pairs)={mean_dist:.3f}, mean(step mean)={mean_step_mean_dist:.3f}")
   print(f"  jerk_3d_value (piecewise linear loss over 3D trajectories)={jerk_3d_value:.3f}")
   print(f"  timing: steps={num_steps}, wall_time={wall_time:.3f}s, min_step={min_step_time:.4f}s, max_step={max_step_time:.4f}s, mean_step={mean_step_time:.4f}s")
   if costs:
      phases = ", ".join(f"{phase}={costs.get(f'cpu_{phase}_s', 0.0):.2f}" for phase in COST_PHASES)
      print(f"  cpu: total={costs.get('cpu_total_s', 0.0):.2f}s [{phases}] admm_iterations={costs.get('admm_iterations', 0)}")

   fieldnames = ["num_drones", "horizon", "coordinator", "controller_type", "status", "steps", "wall_time_s", "min_step_time_s", "max_step_time_s",
                 "mean_step_time_s", "min_distance", "max_distance", "mean_distance_all_pairs", "mean_distance_step_mean", "jerk_3d_value", *COST_FIELDNAMES]
   row = {"num_drones": num_drones, "horizon": horizon, "coordinator": coordinator_type, "controller_type": controller_type, "status": status, "steps": num_steps,
          "wall_time_s": wall_time, "min_step_time_s": min_step_time, "max_step_time_s": max_step_time, "mean_step_time_s": mean_step_time, "min_distance": min_dist,
          "max_distance": max_dist, "mean_distance_all_pairs": mean_dist, "mean_distance_step_mean": mean_step_mean_dist, "jerk_3d_value": jerk_3d_value}
   row.update({name: (costs or {}).get(name) for name in COST_FIELDNAMES})

   return fieldnames, row