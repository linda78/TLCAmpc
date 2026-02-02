from __future__ import annotations

import argparse
import csv
import io
import json
import time
from enum import StrEnum
from pathlib import Path

import httpx
import numpy as np
from PIL import Image
from httpx import Client

from tools.utility.config_creater import create_default_config
from tools.utility.param_helper import _parse_kv_params
from tools.utility.metrics_fcts import compute_jerk_3d_value, pairwise_distances, all_drones_reached_destination


class Status(StrEnum):
   RUNNING = "running"
   FINISHED = "finished"
   STEP_TIMEOUT = "step_timeout"
   OVERALL_TIMEOUT = "overall_timeout"
   INFEASIBLE = "infeasible"
   ERROR = "error"

# check if we can use _all_drones_reached_destination from untility
def _all_routes_finished_from_state(state: dict, pos_tol: float = 0.2, vel_tol: float = 0.1) -> bool:
   drones = state.get("drones", [])
   for d in drones:
      x = np.asarray(d["x"], dtype=float)
      p = x[:3]
      v = x[3:]
      p_ref = np.asarray(d["p_ref"], dtype=float)

      if float(np.linalg.norm(p - p_ref)) > pos_tol:
         return False
      if float(np.linalg.norm(v)) > vel_tol:
         return False
   return True


def _current_state(client: Client, base_url: str, status: Status) -> tuple[dict | None, Status]:
   try:
      r_state = client.get(f"{base_url}/state")
      r_state.raise_for_status()
   except httpx.TimeoutException:
      return None, Status.STEP_TIMEOUT
   except httpx.HTTPError:
      return None, Status.ERROR

   return r_state.json(), status


def _next_step(client: Client, base_url: str, status: Status) -> tuple[dict | None, Status]:
   try:
      r_step = client.post(f"{base_url}/step", params={"n": 1})
      r_step.raise_for_status()
   except httpx.TimeoutException:
      return None, Status.STEP_TIMEOUT
   except httpx.HTTPError:
      return None, Status.ERROR

   step_payload = r_step.json()
   if step_payload.get("status") == "infeasible":
      return None, Status.INFEASIBLE

   return step_payload, status


def _print_results_if_not_running(all_pair_dists: list[float], frames: list[Image.Image], gif_fps: float,
                                  gif_path: Path, horizon: int, jerk_3d_value: float, num_drones: int, out_dir: Path,
                                  status: Status, step_durations: list[float], step_mean_pair_dists: list[float],
                                  wall_time: float) -> None:
   if frames:
      duration_ms = int(round(1000.0 / max(0.1, float(gif_fps))))
      frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)
   print(f"status={status}, frames={len(frames)}, wall_time={wall_time:.2f}s -> GIF: {gif_path}")

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

   print(f"  distances: min={min_dist:.3f}, max={max_dist:.3f}, "
         f"mean(all pairs)={mean_dist:.3f}, mean(step mean)={mean_step_mean_dist:.3f}")
   print(f"  jerk_3d_value (piecewise linear loss over 3D trajectories)={jerk_3d_value:.3f}")
   print(f"  timing: steps={num_steps}, wall_time={wall_time:.3f}s, "
         f"min_step={min_step_time:.4f}s, max_step={max_step_time:.4f}s, mean_step={mean_step_time:.4f}s")

   # Write metrics to CSV (one row per scenario) in the same output directory.
   csv_path = out_dir / "metrics.csv"
   fieldnames = ["num_drones", "horizon", "status", "frames", "steps", "wall_time_s", "min_step_time_s",
                 "max_step_time_s", "mean_step_time_s", "min_distance", "max_distance", "mean_distance_all_pairs",
                 "mean_distance_step_mean", "jerk_3d_value"]
   row = {"num_drones": num_drones, "horizon": horizon, "status": status, "frames": len(frames), "steps": num_steps,
          "wall_time_s": wall_time, "min_step_time_s": min_step_time, "max_step_time_s": max_step_time,
          "mean_step_time_s": mean_step_time, "min_distance": min_dist, "max_distance": max_dist,
          "mean_distance_all_pairs": mean_dist, "mean_distance_step_mean": mean_step_mean_dist,
          "jerk_3d_value": jerk_3d_value}
   # Append with header creation if needed.
   write_header = not csv_path.exists()
   with csv_path.open("a", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      if write_header:
         writer.writeheader()
      writer.writerow(row)


def _render_image(client: Client, base_url: str, width: int, height: int, dpi: int, elev: float, azim: float,
                  frames: list[Image.Image]) -> tuple[list[Image.Image], Status]:
   try:
      r_img = client.get(f"{base_url}/render",
                         params={"width": width, "height": height, "dpi": dpi, "elev": elev, "azim": azim,
                                 "trace_len": 50})
      r_img.raise_for_status()
   except httpx.TimeoutException:
      return frames, Status.STEP_TIMEOUT
   except httpx.HTTPError:
      return frames, Status.ERROR

   img = Image.open(io.BytesIO(r_img.content)).convert("RGBA")
   frames.append(img)

   return frames, Status.RUNNING


def _load_config(base_url: str, cfg_dict, client: Client, status: Status) -> Status:
   try:
      r = client.post(f"{base_url}/config", content=json.dumps(cfg_dict), headers={"Content-Type": "application/json"})
      r.raise_for_status()
   except httpx.TimeoutException:
      return Status.STEP_TIMEOUT
   except httpx.HTTPError:
      return Status.ERROR

   return status

def run_single_scenario_live_view(*, num_drones: int, horizon: int, base_url: str = "http://127.0.0.1:8000",
                                  out_dir: Path, max_steps: int = 500, per_request_timeout_s: float = 360.0,
                                  total_timeout_s: float = 600.0, gif_fps: float = 20.0, width: int = 900,
                                  height: int = 700, dpi: int = 120, elev: float = 20.0, azim: float = -60.0) -> None:
   """
   Run one (N, H) pair through the REST API and create a GIF.

   Returns (status, num_frames, wall_time_seconds), where status is one of:
     - "finished"           all drones reached their targets (heuristic)
     - "max_steps"          hit max_steps
     - "overall_timeout"    exceeded total_timeout_s
     - "step_timeout"       an HTTP request exceeded per_request_timeout_s
     - "infeasible"         central MPC reported infeasible
     - "error"              unexpected HTTP error
   """

   scenario = create_default_config(n_drones=num_drones, horizon=horizon)

   cfg_dict = scenario.model_dump(mode="json")

   out_dir.mkdir(parents=True, exist_ok=True)
   gif_path = out_dir / f"N{num_drones}_H{horizon}.gif"

   status = Status.RUNNING

   timeout = httpx.Timeout(per_request_timeout_s)
   t0 = time.perf_counter()

   # Metrics accumulators
   step_durations: list[float] = []
   step_mean_pair_dists: list[float] = []
   all_pair_dists: list[float] = []

   frames: list[Image.Image] = []

   # For 3D jerk metric: store full position history per drone.
   positions_by_drone: dict[str, list[np.ndarray]] = {}

   with httpx.Client(timeout=timeout) as client:
      wall_time = time.perf_counter() - t0
      # Load config
      status = _load_config(base_url, cfg_dict, client, status)

      for step_idx in range(max_steps):
         if status is Status.RUNNING:
            frames, status = _render_image(client, base_url, width, height, dpi, elev, azim, frames)
         if status is not Status.RUNNING:
            jerk_3d_value = compute_jerk_3d_value(positions_by_drone)
            _print_results_if_not_running(all_pair_dists, frames, gif_fps, gif_path, horizon, jerk_3d_value, num_drones,
                                          out_dir, status, step_durations, step_mean_pair_dists, wall_time)
            break

         now_global = time.perf_counter()
         if now_global - t0 > total_timeout_s:
            status = Status.OVERALL_TIMEOUT
            jerk_3d_value = compute_jerk_3d_value(positions_by_drone)
            _print_results_if_not_running(all_pair_dists, frames, gif_fps, gif_path, horizon, jerk_3d_value, num_drones,
                                          out_dir, status, step_durations, step_mean_pair_dists, wall_time)
            break

         t_step0 = time.perf_counter()

         # Advance simulation by one step.
         step_payload, status = _next_step(client, base_url, status)
         if status in [Status.STEP_TIMEOUT, Status.ERROR, Status.INFEASIBLE]:
            jerk_3d_value = compute_jerk_3d_value(positions_by_drone)
            _print_results_if_not_running(all_pair_dists, frames, gif_fps, gif_path, horizon, jerk_3d_value, num_drones,
                                          out_dir, status, step_durations, step_mean_pair_dists, wall_time)
            break

         # Check termination and collect metrics based on /state.
         state, status = _current_state(client, base_url, status)
         if status in [Status.STEP_TIMEOUT, Status.ERROR]:
            jerk_3d_value = compute_jerk_3d_value()
            _print_results_if_not_running(all_pair_dists, frames, gif_fps, gif_path, horizon, jerk_3d_value, num_drones,
                                          out_dir, status, step_durations, step_mean_pair_dists, wall_time)
            break

         drones_state = state.get("drones", [])
         drones_state_sorted = sorted(drones_state, key=lambda d: d["drone_id"])

         positions: list[np.ndarray] = []
         for d in drones_state_sorted:
            x = np.asarray(d["x"], dtype=float)
            p = x[:3]
            positions.append(p)

            did = d["drone_id"]

            # Store full position history per drone for 3D jerk metric.
            traj = positions_by_drone.setdefault(did, [])
            traj.append(p)

         # Pairwise distance metrics.
         dists = pairwise_distances(positions)
         if dists.size > 0:
            all_pair_dists.extend(dists.tolist())
            step_mean_pair_dists.append(float(dists.mean()))

         if _all_routes_finished_from_state(state):
            status = Status.FINISHED

         t_step1 = time.perf_counter()
         step_durations.append(t_step1 - t_step0)


def main(argv: list[str] | None = None) -> None:
   p = argparse.ArgumentParser(
      description="Run the predefined horizon-feasibility scenarios via the REST API and create live-view GIFs for N=2..7, H=1..10.")

   p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the running DroneSim REST API")
   p.add_argument("--param", action="append", default=[], help="Template parameter KEY=VALUE (may be repeated), see @overrite_param.py for details")

   p.add_argument("--out-dir", type=Path, default=Path("results/live_view_horizon_grid/results"),
                  help="Directory where GIFs will be written (one per N,H pair)")
   p.add_argument("--max-steps", type=int, default=500, help="Per-scenario maximum number of simulation steps")
   p.add_argument("--step-timeout", type=float, default=60.0,
                  help="Maximum wall time in seconds per HTTP request (config/step/state/render)")
   p.add_argument("--total-timeout", type=float, default=360.0,
                  help="Maximum total wall time in seconds per scenario before aborting")
   p.add_argument("--gif-fps", type=float, default=20.0, help="FPS for generated GIFs")

   args = p.parse_args(argv)

   params = _parse_kv_params(args.param)

   drone_counts = range(2, 8)
   horizons = range(1, 21)

   print(f"Running live-view GIF grid for N in {list(drone_counts)}, H in {list(horizons)}")
   print(f"Base URL: {args.base_url}")
   print(f"Output directory: {args.out_dir}")

   for n in drone_counts:
      for H in horizons:
         print(f"=== Scenario N={n}, H={H} ===")
         run_single_scenario_live_view(num_drones=n, horizon=H, base_url=args.base_url, out_dir=args.out_dir,
                                       max_steps=args.max_steps, per_request_timeout_s=args.step_timeout,
                                       total_timeout_s=args.total_timeout, gif_fps=args.gif_fps)


if __name__ == "__main__":
   main()
