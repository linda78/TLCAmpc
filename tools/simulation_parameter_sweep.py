import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from drone_sim.api.render import render_png
from drone_sim.simulation.simulator import Simulator
from matplotlib.image import AxesImage
from PIL import Image
from datetime import datetime
from drone_sim.domain.config import ScenarioConfig
from tools.utility.config_creater import create_config, create_default_config, predefined_patterns_for
from tools.utility.metrics_fcts import compute_jerk_3d_value, pairwise_distances, all_drones_reached_destination
from tools.metrics.analyze_sweep import combine_metrics, create_heatmaps


class Status(StrEnum):
   RUNNING = "running"
   FINISHED = "finished"
   MAX_STEPS = "max_steps"
   INFEASIBLE = "infeasible"
   TIMEOUT = "timeout"
   ERROR = "error"


# Timeout in seconds for a single simulation run
TIMEOUT_SECONDS = 60


# Default rendering parameters
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600
DEFAULT_DPI = 100
DEFAULT_ELEV = 30
DEFAULT_AZIM = 45
DEFAULT_TRACE_LEN = 500
DT = 0.1
# Rendering is expensive; disable live view for sweeps. GIF rendering can still run headlessly.
LIVE_VIEW = False
SAVE_GIF = True
# Default sweep parameters
DEFAULT_DRONE_COUNTS = range(2, 8)
DEFAULT_HORIZONS = range(1, 21)

# For 3D jerk metric: store full position history per drone.
positions_by_drone: dict[str, list[np.ndarray]] = {}


def _print_results_if_not_running(all_pair_dists: list[float], frames: list[Image.Image], gif_fps: float, gif_path: Path, horizon: int, jerk_3d_value: float,
                                  num_drones: int, out_dir: Path, status: Status, step_durations: list[float], step_mean_pair_dists: list[float],
                                  wall_time: float) -> None:
   if frames:
      duration_ms = int(round(1000.0 / max(0.1, float(gif_fps))))
      gif_path.parent.mkdir(parents=True, exist_ok=True)
      frames[0].save(f"{gif_path}.gif", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)
   if frames:
      print(f"status={status}, frames={len(frames)}, wall_time={wall_time:.2f}s -> GIF: {gif_path}")
   else:
      print(f"status={status}, frames={len(frames)}, wall_time={wall_time:.2f}s")

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
   fieldnames = ["num_drones", "horizon", "status", "frames", "steps", "wall_time_s", "min_step_time_s", "max_step_time_s", "mean_step_time_s", "min_distance",
                 "max_distance", "mean_distance_all_pairs", "mean_distance_step_mean", "jerk_3d_value"]
   row = {"num_drones": num_drones, "horizon": horizon, "status": status, "frames": len(frames), "steps": num_steps, "wall_time_s": wall_time,
          "min_step_time_s": min_step_time, "max_step_time_s": max_step_time, "mean_step_time_s": mean_step_time, "min_distance": min_dist,
          "max_distance": max_dist, "mean_distance_all_pairs": mean_dist, "mean_distance_step_mean": mean_step_mean_dist, "jerk_3d_value": jerk_3d_value}
   # Append with header creation if needed.
   write_header = not csv_path.exists()
   out_dir.mkdir(parents=True, exist_ok=True)
   with open(csv_path, "a", newline="") as file:
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      if write_header:
         writer.writeheader()
      writer.writerow(row)


def display_current_step(png_bytes: bytes, img_artist: AxesImage | None = None, ax: plt.Axes | None = None) -> AxesImage:
   img = mpimg.imread(io.BytesIO(png_bytes), format="png")
   if img_artist is None:
      img_artist = ax.imshow(img)
   else:
      img_artist.set_data(img)

   return img_artist


def record_current_setp(record_path: Path | None, gif_out: Path | None, frames: list[Image.Image], png_bytes: bytes, i: int) -> list[Image.Image]:
   if gif_out is not None:
      pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
      frames.append(pil_img)
   return frames


def render_step(sim: Simulator, record_path: Path | None, gif_out: Path | None, frames: list[Image.Image], i: int, img_artist: AxesImage | None = None,
                ax: plt.Axes | None = None) -> tuple[AxesImage, list[Image.Image]]:
   trace_len = DEFAULT_TRACE_LEN
   traces = [[] if trace_len == 0 else sim.traces.get(d.drone_id, [])[-trace_len:] for d in sim.drones]
   safety_zones = [float(d.safety_zone) for d in sim.drones]

   png_bytes = render_png(room_min=sim.room_min, room_max=sim.room_max, drone_positions=[d.position() for d in sim.drones],
                          drone_radii=[d.radius for d in sim.drones], drone_safety_zones=safety_zones, drone_colors=[d.color for d in sim.drones],
                          safety_colors=[d.safety_color for d in sim.drones], trace_colors=[d.trace_color for d in sim.drones], drone_traces=traces,
                          obstacles=sim.obstacles, step_count=sim.step_count, compute_time_s=sim.compute_time_s, room_radius=sim.room_radius,
                          width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, dpi=DEFAULT_DPI, elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)

   if LIVE_VIEW:
      img_artist = display_current_step(png_bytes, img_artist, ax)
   frames = record_current_setp(record_path, gif_out, frames, png_bytes, i)

   return img_artist, frames


def create_gif(frames: list[Image.Image], gif_out: Path | None, gif_fps: int):
   if gif_out is not None:
      if not frames:
         raise RuntimeError("No frames captured; cannot write GIF")
      duration_ms = int(round(1000.0 / max(0.1, float(gif_fps))))
      gif_out.parent.mkdir(parents=True, exist_ok=True)
      frames[0].save(gif_out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


def run_simulation(config: ScenarioConfig, record_dir: Path | None, gif_path: Path | None, out_dir: Path | None):
   max_steps = 500  # should be enough for small rooms

   step_durations: list[float] = []
   step_mean_pair_dists: list[float] = []
   all_pair_dists: list[float] = []
   positions_by_drone: dict[str, list[np.ndarray]] = {}
   img_artist = None
   frames: list[Image.Image] = []

   record_path = None
   gif_out = Path(gif_path) if (SAVE_GIF and gif_path is not None) else None

   fig, ax = None, None
   if LIVE_VIEW:
      plt.ion()
      fig, ax = plt.subplots()
      ax.set_axis_off()

   sim = Simulator.from_config(config)
   t_wall0 = time.perf_counter()
   for step in range(max_steps):
      if all_drones_reached_destination(sim.drones, thresh=0.3):
         jerk_3d_value = compute_jerk_3d_value(positions_by_drone)
         wall_time = time.perf_counter() - t_wall0
         _print_results_if_not_running(all_pair_dists, frames, 20.0, gif_path, config.controller.params["horizon"], jerk_3d_value, len(sim.drones), out_dir,
                                       Status.FINISHED, step_durations, step_mean_pair_dists, wall_time)
         break

      t_step0 = time.perf_counter()
      sim.step()

      if sim.infeasible:
         detail = sim.infeasible_reason or "Central MPC reported infeasible controls."
         jerk_3d_value = compute_jerk_3d_value(positions_by_drone)
         wall_time = time.perf_counter() - t_wall0
         _print_results_if_not_running(all_pair_dists, frames, 20.0, gif_path, config.controller.params["horizon"], jerk_3d_value, len(sim.drones), out_dir,
                                       Status.INFEASIBLE, step_durations, step_mean_pair_dists, wall_time)
         print(f"Unsolvable configuration (central MPC infeasible step {sim.step_count}). Detail: {detail}")
         break
      t_step1 = time.perf_counter()
      step_durations.append(t_step1 - t_step0)

      # Check timeout
      wall_time = time.perf_counter() - t_wall0
      if wall_time > TIMEOUT_SECONDS:
         jerk_3d_value = compute_jerk_3d_value(positions_by_drone)
         _print_results_if_not_running(all_pair_dists, frames, 20.0, gif_path, config.controller.params["horizon"], jerk_3d_value, len(sim.drones), out_dir,
                                       Status.TIMEOUT, step_durations, step_mean_pair_dists, wall_time)
         print(f"Timeout after {wall_time:.2f}s (limit: {TIMEOUT_SECONDS}s) at step {sim.step_count}")
         break

      for d in sim.drones:
         traj = positions_by_drone.setdefault(d.drone_id, [])
         traj.append(np.asarray(d.position(), dtype=float))

      positions = [drone.position() for drone in sim.drones]
      dists = pairwise_distances(positions)
      if dists.size > 0:
         all_pair_dists.extend(dists.tolist())
         step_mean_pair_dists.append(float(dists.mean()))

      # Render current state (headless if SAVE_GIF is enabled; interactive window only if LIVE_VIEW is enabled)
      if LIVE_VIEW or gif_out is not None:
         img_artist, frames = render_step(sim, record_path, gif_out, frames, step, img_artist, ax)

      if LIVE_VIEW:
         fig.canvas.draw_idle()
         plt.pause(0.001)

      # Check max steps
      if step == max_steps - 1:
         jerk_3d_value = compute_jerk_3d_value(positions_by_drone)
         wall_time = time.perf_counter() - t_wall0
         _print_results_if_not_running(all_pair_dists, frames, 20.0, gif_path, config.controller.params["horizon"], jerk_3d_value, len(sim.drones), out_dir,
                                       Status.MAX_STEPS, step_durations, step_mean_pair_dists, wall_time)
         break

   if LIVE_VIEW:
      plt.ioff()
      plt.close(fig)


def run_config(config: ScenarioConfig) -> bool:
   if config is None:
      return False

   H = config.controller.params["horizon"]
   n = len(config.drones)
   safety_zone = config.drones[0].safety_zone
   v_max = config.physics[0].params["v_max"]
   u = config.physics[0].params["u_max"]
   q_pos = config.controller.params["q_pos"]
   r_u = config.controller.params["r_u"]
   if config.room.r is None:
      room_size = config.room.max[0]*2
   else:
      room_size = config.room.r

   base = f's_{safety_zone}_u_{u}_v_{v_max}_r_{room_size}_q_{q_pos}_r_{r_u}'

   print(f"{datetime.now().strftime("%d-%H:%M:%S")}:\trun simulation with drones={n}, horizon={H}, safety={safety_zone}, v_max={v_max}, u={u}, room={room_size}, q={q_pos}, r={r_u}")
   run_simulation(config, record_dir=None, gif_path=Path(f"paper_test/param_swep_result/{base}/{n}_{H}"), out_dir=Path(f"paper_test/param_swep_result/{base}/out_dir/"))
   print(f"{datetime.now().strftime("%d-%H:%M:%S")}:\tdone, data saved in param_swep_result/{base}")

   return True


def create_config_and_run(safety_zone: float, v_max: float, u: float, room_size: float, r_room: float = 0.0,
                          n_drones: range = DEFAULT_DRONE_COUNTS, horizon: range = DEFAULT_HORIZONS, extra_space: float = 0.2,
                          use_predefined_patterns: bool = False):

   for n in n_drones:
      patterns = None
      if use_predefined_patterns:
         patterns = predefined_patterns_for(n, safety_zone, room_size if room_size > 0 else r_room, r_room > 0)
      for H in horizon:
         config = create_config(n_drones=n, horizon=H, safety_zone=safety_zone, v_max=v_max, u=u, room_size=room_size, r_room=r_room, extra_space=extra_space, positions=patterns)
         if run_config(config):
            print(f"{datetime.now().strftime("%d-%H:%M:%S")}:\trun simulation with drones={n}, horizon={H}, safety={safety_zone}, v_max={v_max}, u={u}, room={room_size}")
            continue


def run_some_interesting_configs(max_workers: int = 4):
   """
================================================================================================================================================================================================================================================================================================================================================================================================
side 5.0 - radius 1.0 -> N: 22|side 5.0 - radius 1.5 -> N: 6|side 5.0 - radius 2.0 -> N: 2|side 7.5 - radius 1.0 -> N:74|side 7.5 - radius 1.5 -> N:22|side 7.5 - radius 2.0 -> N: 9|side 8.0 - radius 1.0 -> N:90|side 8.0 - radius 1.5 -> N:26|side 8.0 - radius 2.0 -> N:11|side 9.0 - radius 1.0 ->N:128|side 9.0 - radius 1.5 -> N:38|side 9.0 - radius 2.0 -> N:16|
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 u_max v_max| rho	  N_j		   | u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		| u_max v_max| rho	  N_j		|
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 1.0	1.0	 | 1.250  11.3		| 1.0	1.0	 | 1.750   4.1		| 1.0	1.0	 | 2.250   1.9		| 1.0	1.0	 | 1.250  38.2		| 1.0	1.0	 | 1.750  13.9		| 1.0	1.0	 | 2.250   6.5		| 1.0	1.0	 | 1.250  46.3		| 1.0	1.0	 | 1.750  16.9		| 1.0	1.0	 | 2.250   7.9		| 1.0	1.0	 | 1.250  66.0		| 1.0	1.0	 | 1.750  24.0		| 1.0	1.0	 | 2.250  11.3		|
 1.0	2.5	 | 2.562   1.3		| 1.0	2.5	 | 3.062   0.8		| 1.0	2.5	 | 3.562   0.5		| 1.0	2.5	 | 2.562   4.4		| 1.0	2.5	 | 3.062   2.6		| 1.0	2.5	 | 3.562   1.6		| 1.0	2.5	 | 2.562   5.4		| 1.0	2.5	 | 3.062   3.2		| 1.0	2.5	 | 3.562   2.0		| 1.0	2.5	 | 2.562   7.7		| 1.0	2.5	 | 3.062   4.5		| 1.0	2.5	 | 3.562   2.9		|
 1.0	3.5	 | 4.062   0.3		| 1.0	3.5	 | 4.562   0.2		| 1.0	3.5	 | 5.062   0.2		| 1.0	3.5	 | 4.062   1.1		| 1.0	3.5	 | 4.562   0.8		| 1.0	3.5	 | 5.062   0.6		| 1.0	3.5	 | 4.062   1.3		| 1.0	3.5	 | 4.562   1.0		| 1.0	3.5	 | 5.062   0.7		| 1.0	3.5	 | 4.062   1.9		| 1.0	3.5	 | 4.562   1.4		| 1.0	3.5	 | 5.062   1.0		|
 1.0	5.0	 | 7.250   0.1		| 1.0	5.0	 | 7.750   0.0		| 1.0	5.0	 | 8.250   0.0		| 1.0	5.0	 | 7.250   0.2		| 1.0	5.0	 | 7.750   0.2		| 1.0	5.0	 | 8.250   0.1		| 1.0	5.0	 | 7.250   0.2		| 1.0	5.0	 | 7.750   0.2		| 1.0	5.0	 | 8.250   0.2		| 1.0	5.0	 | 7.250   0.3		| 1.0	5.0	 | 7.750   0.3		| 1.0	5.0	 | 8.250   0.2		|
 3.0	1.0	 | 1.083  17.4		| 3.0	1.0	 | 1.583   5.6		| 3.0	1.0	 | 2.083   2.4		| 3.0	1.0	 | 1.083  58.7		| 3.0	1.0	 | 1.583  18.8		| 3.0	1.0	 | 2.083   8.2		| 3.0	1.0	 | 1.083  71.2		| 3.0	1.0	 | 1.583  22.8		| 3.0	1.0	 | 2.083  10.0		| 3.0	1.0	 | 1.083 101.4		| 3.0	1.0	 | 1.583  32.5		| 3.0	1.0	 | 2.083  14.3		|
 3.0	2.5	 | 1.521   6.3		| 3.0	2.5	 | 2.021   2.7		| 3.0	2.5	 | 2.521   1.4		| 3.0	2.5	 | 1.521  21.2		| 3.0	2.5	 | 2.021   9.0		| 3.0	2.5	 | 2.521   4.7		| 3.0	2.5	 | 1.521  25.7		| 3.0	2.5	 | 2.021  11.0		| 3.0	2.5	 | 2.521   5.7		| 3.0	2.5	 | 1.521  36.6		| 3.0	2.5	 | 2.021  15.6		| 3.0	2.5	 | 2.521   8.0		|
 3.0	3.5	 | 2.021   2.7		| 3.0	3.5	 | 2.521   1.4		| 3.0	3.5	 | 3.021   0.8		| 3.0	3.5	 | 2.021   9.0		| 3.0	3.5	 | 2.521   4.7		| 3.0	3.5	 | 3.021   2.7		| 3.0	3.5	 | 2.021  11.0		| 3.0	3.5	 | 2.521   5.7		| 3.0	3.5	 | 3.021   3.3		| 3.0	3.5	 | 2.021  15.6		| 3.0	3.5	 | 2.521   8.0		| 3.0	3.5	 | 3.021   4.7		|
 3.0	5.0	 | 3.083   0.8		| 3.0	5.0	 | 3.583   0.5		| 3.0	5.0	 | 4.083   0.3		| 3.0	5.0	 | 3.083   2.5		| 3.0	5.0	 | 3.583   1.6		| 3.0	5.0	 | 4.083   1.1		| 3.0	5.0	 | 3.083   3.1		| 3.0	5.0	 | 3.583   2.0		| 3.0	5.0	 | 4.083   1.3		| 3.0	5.0	 | 3.083   4.4		| 3.0	5.0	 | 3.583   2.8		| 3.0	5.0	 | 4.083   1.9		|
 5.0	1.0	 | 1.050  19.1		| 5.0	1.0	 | 1.550   5.9		| 5.0	1.0	 | 2.050   2.6		| 5.0	1.0	 | 1.050  64.4		| 5.0	1.0	 | 1.550  20.0		| 5.0	1.0	 | 2.050   8.7		| 5.0	1.0	 | 1.050  78.2		| 5.0	1.0	 | 1.550  24.3		| 5.0	1.0	 | 2.050  10.5		| 5.0	1.0	 | 1.050 111.3		| 5.0	1.0	 | 1.550  34.6		| 5.0	1.0	 | 2.050  15.0		|
 5.0	2.5	 | 1.312   9.8		| 5.0	2.5	 | 1.812   3.7		| 5.0	2.5	 | 2.312   1.8		| 5.0	2.5	 | 1.312  33.0		| 5.0	2.5	 | 1.812  12.5		| 5.0	2.5	 | 2.312   6.0		| 5.0	2.5	 | 1.312  40.0		| 5.0	2.5	 | 1.812  15.2		| 5.0	2.5	 | 2.312   7.3		| 5.0	2.5	 | 1.312  57.0		| 5.0	2.5	 | 1.812  21.6		| 5.0	2.5	 | 2.312  10.4		|
 5.0	3.5	 | 1.613   5.3		| 5.0	3.5	 | 2.112   2.3		| 5.0	3.5	 | 2.612   1.2		| 5.0	3.5	 | 1.613  17.8		| 5.0	3.5	 | 2.112   7.9		| 5.0	3.5	 | 2.612   4.2		| 5.0	3.5	 | 1.613  21.6		| 5.0	3.5	 | 2.112   9.6		| 5.0	3.5	 | 2.612   5.1		| 5.0	3.5	 | 1.613  30.7		| 5.0	3.5	 | 2.112  13.7		| 5.0	3.5	 | 2.612   7.2		|
 5.0	5.0	 | 2.250   1.9		| 5.0	5.0	 | 2.750   1.1		| 5.0	5.0	 | 3.250   0.6		| 5.0	5.0	 | 2.250   6.5		| 5.0	5.0	 | 2.750   3.6		| 5.0	5.0	 | 3.250   2.2		| 5.0	5.0	 | 2.250   7.9		| 5.0	5.0	 | 2.750   4.4		| 5.0	5.0	 | 3.250   2.6		| 5.0	5.0	 | 2.250  11.3		| 5.0	5.0	 | 2.750   6.2		| 5.0	5.0	 | 3.250   3.8		|
================================================================================================================================================================================================================================================================================================================================================================================================
   """
   # Define all configurations as dictionaries
   configs = [
         # Feasibility correct? ... 5/1 -> 0, 3.5/3 -> 11, o3.5/5 -> 21, 5/5 -> 8, 3.5/3 -> 11
         {"safety_zone": 1.0, "v_max": 5.0, "u": 1.0, "room_size": 8.0, "r_room": 0.0, "n_drones": range(2, 30), "horizon": range(5, 6), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 1.0, "v_max": 3.5, "u": 3.0, "room_size": 8.0, "r_room": 0.0, "n_drones": range(2, 30), "horizon": range(5, 6), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 1.0, "v_max": 3.5, "u": 5.0, "room_size": 8.0, "r_room": 0.0, "n_drones": range(2, 30), "horizon": range(5, 6), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 1.0, "v_max": 5.0, "u": 5.0, "room_size": 8.0, "r_room": 0.0, "n_drones": range(2, 30), "horizon": range(5, 6), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 1.0, "v_max": 2.5, "u": 3.0, "room_size": 8.0, "r_room": 0.0, "n_drones": range(2, 30), "horizon": range(5, 6), "extra_space": 0.001, "use_predefined_patterns": True},
         # Feasibility correct? Room 9, Sphere different u=3.0 v_max = 5.0 ... 1 -> 7.7, 1.5 -> 4.5, 2 -> 2.9
         {"safety_zone": 1.0, "v_max": 2.5, "u": 1.0, "room_size": 9.0, "r_room": 0.0, "n_drones": range(2, 9), "horizon": range(4, 5), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 1.5, "v_max": 2.5, "u": 1.0, "room_size": 9.0, "r_room": 0.0, "n_drones": range(2, 9), "horizon": range(4, 5), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 2.0, "v_max": 2.5, "u": 1.0, "room_size": 9.0, "r_room": 0.0, "n_drones": range(2, 5), "horizon": range(4, 5), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 1.0, "v_max": 5.0, "u": 3.0, "room_size": 9.0, "r_room": 0.0, "n_drones": range(2, 9), "horizon": range(4, 5), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 1.5, "v_max": 5.0, "u": 3.0, "room_size": 9.0, "r_room": 0.0, "n_drones": range(2, 9), "horizon": range(4, 5), "extra_space": 0.001, "use_predefined_patterns": True},
         {"safety_zone": 2.0, "v_max": 5.0, "u": 3.0, "room_size": 9.0, "r_room": 0.0, "n_drones": range(2, 5), "horizon": range(4, 5), "extra_space": 0.001, "use_predefined_patterns": True},
         # some additional
         {"safety_zone": 1.0, "v_max": 1.0, "u": 5.0, "room_size": 5.0, "n_drones": range(2, 8), "horizon": range(1, 21), "extra_space": 0.0},
         {"safety_zone": 1.0, "v_max": 5.0, "u": 5.0, "room_size": 5.0, "n_drones": range(2, 8), "horizon": range(1, 21)},
         {"safety_zone": 1.5, "v_max": 2.5, "u": 5.0, "room_size": 8.0, "n_drones": range(2, 8), "horizon": range(1, 21), "extra_space": 0.01},
         {"safety_zone": 2.0, "v_max": 2.5, "u": 3.0, "room_size": 8.0, "n_drones": range(2, 8), "horizon": range(1, 21), "extra_space": 0.01},
         {"safety_zone": 2.0, "v_max": 2.5, "u": 3.0, "room_size": 7.5}, {"safety_zone": 2.0, "v_max": 2.5, "u": 5.0, "room_size": 7.5},
         {"safety_zone": 1.0, "v_max": 2.5, "u": 3.0, "room_size": 0.0, "r_room": 3.1, "n_drones": range(2, 8), "horizon": range(1, 21), "extra_space": 0.01},
         {"safety_zone": 1.0, "v_max": 2.5, "u": 5.0, "room_size": 0.0, "r_room": 3.1, "n_drones": range(2, 8), "horizon": range(1, 21), "extra_space": 0.01}, #
         {"safety_zone": 2.0, "v_max": 1.0, "u": 1.0, "room_size": 9.0, "n_drones": range(2, 13), "horizon": range(1, 21), "extra_space": 0.1},
         {"safety_zone": 1.0, "v_max": 2.5, "u": 3.0, "room_size": 8.0, "n_drones": range(2, 13), "horizon": range(1, 21), "extra_space": 0.01},
         {"safety_zone": 1.0, "v_max": 1.0, "u": 3.0, "room_size": 8.0, "n_drones": range(2, 13), "horizon": range(1, 21), "extra_space": 0.01},
         {"safety_zone": 1.0, "v_max": 2.5, "u": 5.0, "room_size": 8.0, "n_drones": range(2, 13), "horizon": range(1, 21), "extra_space": 0.01},
         # First example Calculated in paper:
         {"safety_zone": 1.0, "v_max": 3.0, "u": 2.5, "room_size": 5.0, "n_drones": range(2, 7), "horizon": range(4, 5), "extra_space": 0.01},
        ]

   print(f"Starting {len(configs)} configuration sweeps with {max_workers} workers...")

   with ThreadPoolExecutor(max_workers=max_workers) as executor:
      futures = {executor.submit(create_config_and_run, **cfg): i for i, cfg in enumerate(configs)}

      for future in as_completed(futures):
         config_idx = futures[future]
         try:
            future.result()
            print(f"Config {config_idx + 1}/{len(configs)} completed")
         except Exception as e:
            print(f"Config {config_idx + 1}/{len(configs)} failed: {e}")


def stupid_do_simply_all_sweep():
   counter = 0
   for room_size in [5.0, 7.5]:
      for n_drones in range(2, 8):
         for horizon in range(1, 21):
            for safety_zone in [0.5, 1.0, 1.5]:
               for v_max in [1.0, 2.0, 3.0]:
                  for u in [0.1, 0.2, 0.3]:
                     try:
                        config = create_config(n_drones=n_drones, horizon=horizon, safety_zone=safety_zone, v_max=v_max, u=u, room_size=5.0, extra_space=0.2)
                        counter += 1
                        print(f"{counter}: {n_drones}, {horizon}, {safety_zone}, {v_max}, {u}, {room_size}")
                     except Exception as e:
                        print(f"Failed to create config for:{counter}: {n_drones}, {horizon}, {safety_zone}, {v_max}, {u}, {room_size}")


def sweep_q_r(config: ScenarioConfig):
   for q in [1, 2, 3, 4, 5, 6, 7, 8]:
      for r in [0.01, 0.05, 0.1, 0.5, 0.8, 1, 3, 5, 8]:
         config.controller.params["q_pos"] = [q, q, q]
         config.controller.params["r_u"] = [r, r, r]
         run_config(config)


def move_sweep_outputs_to_results_dir(config_name: str):
   result_name = f'results/{config_name}'
   df = combine_metrics()
   basedir = Path(__file__).resolve().parent / "param_swep_result"

   # Save combined CSV
   output_csv = basedir / f'{result_name}.csv'
   output_csv.parent.mkdir(parents=True, exist_ok=True)
   df.to_csv(output_csv, index=False)
   print(f"Saved combined metrics to {output_csv}")
   print(f"Total runs: {len(df)}")

   # Create visualizations
   print("Creating visualizations...")
   create_heatmaps(df, result_name)

   target_dir = basedir / "results" / config_name
   target_dir.mkdir(parents=True, exist_ok=True)

   # BUG!: dir is created, but files are not moved
   for p in basedir.iterdir():
      if not p.is_file():
         continue
      if not p.name.startswith("s"):
         continue
      shutil.move(str(p), str(target_dir / p.name))

   print(f"Moved sweep output files starting with 's' to {target_dir}")


def run_qr_sweep():
   dic_config = {
         'd4_s1_u3_v2.5_r5': create_default_config(4, 4),
         'd6_s1_u3_v2.5_r5': create_default_config(6, 4),
         'd4_s1_u5_v3_r5': create_config(n_drones=4, horizon=4, safety_zone=1.0, v_max=3.0, u=5.0, room_size=5, extra_space=0.01,
                                         positions=predefined_patterns_for(4, 1.0, 5.0, False)),
         'd4_s1_u3_v2.5_r8': create_config(n_drones=4, horizon=4, safety_zone=1, v_max=2.5, u=3.0, room_size=8, extra_space=0.01),
         'd4_s1_u5_v3_r8': create_config(n_drones=4, horizon=4, safety_zone=1.0, v_max=3.0, u=5.0, room_size=8, extra_space=0.01,
                                         positions=predefined_patterns_for(4, 1.0, 8.0, False)),
         'd4_s1.5_u3_v2.5_r8': create_config(n_drones=4, horizon=4, safety_zone=1.5, v_max=2.5, u=3.0, room_size=8, extra_space=0.01),
         'd3_s1_u3_v2.5_r3.1': create_config(n_drones=3, horizon=4, safety_zone=1.0, v_max=2.5, u=3.0, room_size=0.0, r_room=3.1, extra_space=0.01,
                                             positions=predefined_patterns_for(3, 1.0, 3.1, True)),
         'd3_s1_u5_v3_r3.1': create_config(n_drones=3, horizon=4, safety_zone=1.0, v_max=3.0, u=5.0, room_size=0.0, r_room=3.1, extra_space=0.01,
                                           positions=predefined_patterns_for(3, 1.0, 3.1, True))
         }
   for config_name, config in dic_config.items():
      sweep_q_r(config)
      move_sweep_outputs_to_results_dir(config_name)


if __name__ == "__main__":
   run_some_interesting_configs(max_workers=10)
   # run_qr_sweep()
