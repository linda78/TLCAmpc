from __future__ import annotations

import argparse
import csv
import io
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from drone_sim.api.render import render_png
from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator
from drone_sim.domain.utils.helper import all_drones_reached_destination
from tools.utility.constants import AZIM, DPI, ELEV, GIF_HEIGHT, GIF_WIDTH
from tools.utility.helper import DeadlockDetector, pairwise_distances, piecewise_linear_loss_3d
from tools.utility.printer import print_gif, print_results_prep, Status
from tools.utility.scenario_creator import create_scenario

# Lock for thread-safe CSV writing
_csv_lock = threading.Lock()


def _print_results(all_pair_dists: list[float], horizon: int, jerk_3d_value: float, num_drones: int, out_dir: Path, status: Status, step_durations: list[float],
                   step_mean_pair_dists: list[float], wall_time: float, coordinator_type: str, controller_type: str) -> None:
   fieldnames, row = print_results_prep(all_pair_dists=all_pair_dists, horizon=horizon, jerk_3d_value=jerk_3d_value, num_drones=num_drones, status=status,
                                        step_durations=step_durations, step_mean_pair_dists=step_mean_pair_dists, wall_time=wall_time,
                                        coordinator_type=coordinator_type, controller_type=controller_type)

   # Write metrics to CSV (one row per scenario) in the same output directory. Use lock for thread-safe writing.
   csv_path = out_dir / "metrics.csv"

   with _csv_lock:
      write_header = not csv_path.exists()
      with csv_path.open("a", newline="", encoding="utf-8") as f:
         writer = csv.DictWriter(f, fieldnames=fieldnames)
         if write_header:
            writer.writeheader()
         writer.writerow(row)


def _run_single_scenario_live_view(*, num_drones: int, horizon: int, out_dir: Path, controller_type: str = "mpc_agent",
                                   coordinator_type: str = "mpc_central", max_steps: int = 500, gif_fps: float = 20.0, trace_len: int = 50) -> None:
   # create and run scenario
   cfg = create_scenario(n_drones=num_drones, horizon=horizon, controller_type=controller_type, coordinator_type=coordinator_type)
   status, wall_time, jerk_3d_value, step_durations, step_mean_pair_dists, all_pair_dists, frames = run_single_scenario(cfg, max_steps, trace_len)

   # save gif and print results
   out_dir.mkdir(parents=True, exist_ok=True)
   coord_suffix = "dmpc" if coordinator_type == "dmpc_admm" else "central"
   gif_path = out_dir / f"N{num_drones}_H{horizon}_{coord_suffix}.gif"
   print(f"status={status}, frames={len(frames)}, wall_time={wall_time:.2f}s -> GIF: {gif_path}")
   print_gif(frames=frames, gif_path=gif_path, gif_fps=gif_fps)
   _print_results(all_pair_dists=all_pair_dists, horizon=horizon, jerk_3d_value=jerk_3d_value, num_drones=num_drones, out_dir=out_dir, status=status,
                  step_durations=step_durations, step_mean_pair_dists=step_mean_pair_dists, wall_time=wall_time, coordinator_type=coordinator_type,
                  controller_type=controller_type)


def _run_scenario_wrapper(num_drones: int, horizon: int, out_dir: Path, coordinator_type: str, max_steps: int, gif_fps: float, trace_len: int) -> \
tuple[int, int, str]:
   """Wrapper for thread pool execution. Returns (num_drones, horizon, status) for progress tracking."""
   try:
      _run_single_scenario_live_view(num_drones=num_drones, horizon=horizon, out_dir=out_dir, coordinator_type=coordinator_type, max_steps=max_steps,
                                     gif_fps=gif_fps, trace_len=trace_len, controller_type="mpc_agent")
      return (num_drones, horizon, "ok")
   except Exception as e:
      return (num_drones, horizon, f"exception: {e}")


def run_single_scenario(scenario: ScenarioConfig, max_steps: int, trace_len: int, timeout: float | None = None, store_gif: bool = True,
                        deadlock_steps: int | None = None, deadlock_min_movement: float = 0.0) -> tuple[
   Status, float, float, list[float], list[float], list[float], list[Image.Image]]:
   """
   Run one (N, H) pair using direct simulator execution and returns all relevant information to create csv and gif

   :param scenario: ScenarioConfig object.
   :param max_steps: Maximum number of steps to run.
   :param trace_len: Number of previous positions to store per drone for 3D jerk metric.
   :param timeout: Optional timeout in seconds. If exceeded, returns with Status.TIMEOUT.
   :param deadlock_steps: Optional number of consecutive steps in which every drone moved less than ``deadlock_min_movement``
                          before the run counts as a deadlock and returns with Status.DEADLOCK. None disables the check.
   :param deadlock_min_movement: Movement (metres per step) below which a drone counts as not moving for the deadlock check.
   :return: status, wall_time, jerk_3d_value, step_durations, step_mean_pair_dists, all_pair_dists, frames
   """
   status = Status.RUNNING

   t0 = time.perf_counter()

   # Metrics accumulators
   step_durations: list[float] = []
   step_mean_pair_dists: list[float] = []
   all_pair_dists: list[float] = []
   frames: list[Image.Image] = []

   # For 3D jerk metric: store full position history per drone.
   positions_by_drone: dict[str, list[np.ndarray]] = {}

   def _compute_jerk_3d_value() -> float:
      jerk_3d_total = 0.0
      for traj in positions_by_drone.values():
         if len(traj) < 2:
            continue
         pts = np.stack(traj, axis=0)
         loss, _, _ = piecewise_linear_loss_3d(pts)
         jerk_3d_total += float(loss)
      return jerk_3d_total

   def render_frame(sim: Simulator):
      # Render current frame
      traces = {d.drone_id: [] if trace_len == 0 else sim.traces.get(d.drone_id, [])[-trace_len:] for d in sim.drones}

      is_distributed = scenario.coordinator.type == "dmpc_admm"
      neighbor_links = None
      admm_iteration_count = sim.coordinator.get_last_iteration_count() if is_distributed else None
      admm_converged = sim.coordinator.get_last_converged() if is_distributed else None

      try:
         png_bytes = render_png(room_min=sim.room_min, room_max=sim.room_max, drones=sim.drones, drone_traces=traces,
                                obstacles=sim.obstacles, step_count=sim.step_count, compute_time_s=sim.compute_time_s, neighbor_links=neighbor_links,
                                admm_iteration_count=admm_iteration_count, admm_converged=admm_converged, width=GIF_WIDTH, height=GIF_HEIGHT, dpi=DPI,
                                elev=ELEV, azim=AZIM)
         img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
         frames.append(img)
      except Exception as e:
         print(f"  Render error at step {step_idx}: {e}")
         status = Status.ERROR

   def distance_calculations(sim: Simulator):
      # Collect metrics from current state
      drones_sorted = sorted(sim.drones, key=lambda d: d.drone_id)

      positions: list[np.ndarray] = []

      for d in drones_sorted:
         p = d.position()
         positions.append(p)

         # Store full position history per drone for 3D jerk metric.
         positions_by_drone.setdefault(d.drone_id, []).append(p)

      # Pairwise distance metrics.
      dists = pairwise_distances(positions)
      if dists.size > 0:
         all_pair_dists.extend(dists.tolist())
         step_mean_pair_dists.append(float(dists.mean()))

   deadlock_detector = DeadlockDetector(window=deadlock_steps, min_movement=deadlock_min_movement)

   try:
      sim = Simulator.from_config(scenario)
   except Exception as e:
      print(f"  Error creating simulator: {e}")
      return Status.ERROR, time.perf_counter() - t0, _compute_jerk_3d_value(), step_durations, step_mean_pair_dists, all_pair_dists, frames

   # Start positions are the baseline, so the very first step already counts for the deadlock check.
   deadlock_detector.update({d.drone_id: d.position() for d in sim.drones})

   for step_idx in range(max_steps):
      if store_gif:
         render_frame(sim)
      if status != Status.RUNNING:
         break

      t_step0 = time.perf_counter()
      # Advance simulation by one step
      try:
         sim.step()
      except Exception as e:
         print(f"  Step error at step {step_idx}: {e}")
         status = Status.ERROR
         break

      if sim.infeasible:
         print(f"  Infeasible at step {step_idx}: {sim.infeasible_reason or "MPC reported infeasible controls."}")
         status = Status.INFEASIBLE
         break
      t_step1 = time.perf_counter()
      step_durations.append(t_step1 - t_step0)

      distance_calculations(sim)

      # Check if all drones reached destination
      if all_drones_reached_destination(sim.drones):
         status = Status.FINISHED
         break

      # Deadlock: every drone has barely moved for deadlock_steps consecutive steps
      if deadlock_detector.update({d.drone_id: d.position() for d in sim.drones}):
         print(f"  Deadlock at step {step_idx}: all drones moved < {deadlock_min_movement} m per step for {deadlock_steps} steps")
         status = Status.DEADLOCK
         break

      if timeout is not None and (time.perf_counter() - t0) >= timeout:
         print(f"  Timeout reached after {timeout:.1f}s at step {step_idx}")
         status = Status.TIMEOUT
         break

   else:
      # Loop completed without break - hit max_steps
      status = Status.MAX_STEPS

   return status, time.perf_counter() - t0, _compute_jerk_3d_value(), step_durations, step_mean_pair_dists, all_pair_dists, frames


def main(argv: list[str] | None = None) -> None:
   p = argparse.ArgumentParser(description="Run the predefined horizon-feasibility scenarios and create live-view GIFs for N=2..7, H=1..20.")

   p.add_argument("--out-dir", type=Path, default=Path("live_view_horizon_grid/results"), help="Directory where GIFs will be written (one per N,H pair)")
   p.add_argument("--max-steps", type=int, default=500, help="Per-scenario maximum number of simulation steps")
   p.add_argument("--gif-fps", type=float, default=20.0, help="FPS for generated GIFs")
   p.add_argument("--trace-len", type=int, default=50, help="Number of trace points to render")
   p.add_argument("--coordinator", type=str, choices=["mpc_central", "dmpc_admm"], default="mpc_central",
                  help="Coordinator type: 'mpc_central' (centralized) or 'dmpc_admm' (distributed ADMM)")
   p.add_argument("--threads", type=int, default=1, help="Number of parallel threads for running simulations (default: 1, sequential)")

   args = p.parse_args(argv)

   drone_counts = list(range(2, 8))
   horizons = list(range(1, 21))

   coord_type: str = args.coordinator
   num_threads = max(1, args.threads)

   # Build list of all (N, H) scenarios to run
   scenarios = [(n, H) for n in drone_counts for H in horizons]
   total_scenarios = len(scenarios)

   print(f"Running live-view GIF grid for N in {drone_counts}, H in {horizons}")
   print(f"Coordinator: {coord_type}")
   print(f"Output directory: {args.out_dir}")
   print(f"Threads: {num_threads}")
   print(f"Total scenarios: {total_scenarios}")

   if num_threads == 1:
      # Sequential execution (original behavior)
      for idx, (n, H) in enumerate(scenarios, 1):
         print(f"=== [{idx}/{total_scenarios}] Scenario N={n}, H={H}, coordinator={coord_type} ===")
         _run_single_scenario_live_view(num_drones=n, horizon=H, out_dir=args.out_dir, coordinator_type=coord_type, max_steps=args.max_steps,
                                        gif_fps=args.gif_fps, trace_len=args.trace_len, controller_type="mpc_agent")
   else:
      # Parallel execution with thread pool
      print(f"Starting parallel execution with {num_threads} threads...")
      completed = 0

      with ThreadPoolExecutor(max_workers=num_threads) as executor:
         futures = {executor.submit(_run_scenario_wrapper, n, H, args.out_dir, coord_type, args.max_steps, args.gif_fps, args.trace_len): (n, H) for n, H in scenarios}

         for future in as_completed(futures):
            n, H = futures[future]
            completed += 1
            try:
               result_n, result_H, status = future.result()
               print(f"[{completed}/{total_scenarios}] Completed N={result_n}, H={result_H}: {status}")
            except Exception as e:
               print(f"[{completed}/{total_scenarios}] Failed N={n}, H={H}: {e}")

      print(f"All {total_scenarios} scenarios completed.")


if __name__ == "__main__":
   main()
