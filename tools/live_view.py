from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from string import Template
from typing import Any, NoReturn

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

from drone_sim.api.render import render_png
from drone_sim.domain.config import ScenarioConfig
from drone_sim.domain.drone_model import resolve_drone_model
from drone_sim.simulation.simulator import Simulator
from drone_sim.simulation.distributed.distributed_coordinator import DistributedMPCCoordinator
from tools.utility import DronePlacementError
from tools.utility.scenario_creator import create_scenario
from drone_sim.domain.utils.helper import all_drones_reached_destination
import logging


def load_parametrized_json(path: str | Path, params: dict[str, str] | None = None) -> dict[str, Any]:
   """Load a JSON file and substitute `${var}` placeholders.

   This is intentionally minimal: the file contents are treated as a string.Template.

   Notes:
   - Substitution happens before JSON parsing.
   - If you want to substitute a *number*, pass the literal (e.g. "0.15").
   - If you want to substitute a *string*, include quotes in the template or pass a quoted value.

   Example:
       {"dt": ${dt}}
   with params {"dt": "0.05"}
   """

   text = Path(path).read_text(encoding="utf-8")
   if params:
      text = Template(text).safe_substitute(params)
   return json.loads(text)

def _create_scenario(config_path: str | Path | None, params: dict[str, str] | None) -> ScenarioConfig:
   if not config_path:
      # Get N and H from param and load default values
      num_str = params.get("num_drones")
      hor_str = params.get("horizon")
      if num_str is None or hor_str is None:
         _die("When --config has no path, you must provide both num_drones=<int> and horizon=<int> via --param.")

      try:
         scenario = create_scenario(n_drones=int(num_str), horizon=int(hor_str))
      except DronePlacementError as e:
         _die(f"Failed to create scenario: {e}")
   else:
      # Load raw JSON (with optional template substitution) and validate as ScenarioConfig.
      cfg_json = load_parametrized_json(config_path, params=params)
      if not isinstance(cfg_json, dict):
         raise TypeError(f"Config must decode to a JSON object/dict, got {type(cfg_json).__name__}")

      scenario = ScenarioConfig.model_validate(cfg_json)

   for param in params:
      print(f"{param}: {params[param]}")

   return scenario

def apply_obj_name(scenario: ScenarioConfig, obj_name: str | None) -> ScenarioConfig:
   """Point the scenario's default drone model at ``obj_name``, leaving per-drone overrides alone.

   The flag sets the *scenario* default, not the drones. For today's configs — none of which set a per-drone model — that behaves exactly as
   before: every drone gets the model named on the command line. In a heterogeneous scenario the deliberately configured variety survives instead
   of being flattened by the CLI.

   The name is resolved through the same function the config validator uses, so a typo fails here with a readable message rather than somewhere
   inside the first render.

   :param scenario: Parsed scenario; returned unchanged when no name is given.
   :param obj_name: File name under ``resources/assets/``, or an absolute path.
   :return: The scenario to run.
   """
   if not obj_name:
      return scenario

   try:
      resolve_drone_model("obj", obj_name)
   except ValueError as exc:
      _die(f"--obj-name: {exc}")

   return scenario.model_copy(update={"drone_model": "obj", "drone_model_path": obj_name})


def live_view_simulator(sim: Simulator, *, steps: int, step_n: int, sleep_s: float, trace_len: int,
                        width: int, height: int, dpi: int, elev: float, azim: float,
                        record_dir: str | Path | None, gif_path: str | Path | None, gif_fps: float,
                        obj_name: str | None = None, show_live: bool = True) -> None:
   """Step ``sim`` and render frames; optionally show a live matplotlib window and/or save PNGs/GIF."""
   record_path = Path(record_dir) if record_dir is not None else None
   if record_path is not None:
      record_path.mkdir(parents=True, exist_ok=True)

   gif_out = Path(gif_path) if gif_path is not None else None

   frames: list[Image.Image] = []

   fig = None
   ax = None
   img_artist = None
   if show_live:
      plt.ion()
      fig, ax = plt.subplots()
      ax.set_axis_off()

   all_reached = False
   for i in range(steps):
      if all_reached:
         break
      # Advance the simulation by ``step_n`` internal steps.
      if step_n > 0:
         for _ in range(step_n):
            sim.step()
            if sim.infeasible:
               # Mirror the REST API behaviour: treat an infeasible central MPC
               # configuration as a hard error so the caller can react.
               detail = sim.infeasible_reason or "Central MPC reported infeasible controls."
               raise RuntimeError(f"Unsolvable configuration (central MPC infeasible at frame {i}, step {sim.step_count}). Detail: {detail}")

      # Prepare the same inputs that the /render handler would use.
      traces = {d.drone_id: sim.traces.get(d.drone_id, [])[-trace_len:] for d in sim.drones}
      safety_zones: list[float] = []
      safety_alphas: list[float] = []
      for d in sim.drones:
          vel = d.velocity()
          if sim._lstm_provider is not None and d.safety_zone_mode == "lstm":
              radii = sim._lstm_provider.compute_neighbor_safety_radii([d.drone_id], {d.drone_id: d.safety_zone})
              safety_zones.append(float(np.mean(radii[d.drone_id])))
              safety_alphas.append(0.5)
          elif d.is_adaptive:
              safety_zones.append(float(d.compute_adaptive_radius(vel)))
              speed_ratio = min(float(np.linalg.norm(vel)) / d.v_max, 1.0) if d.v_max > 0 else 0.0
              safety_alphas.append(0.3 + 0.7 * speed_ratio)
          else:
              safety_zones.append(float(d.compute_adaptive_radius(vel)))
              safety_alphas.append(0.8)

      is_distributed = isinstance(sim.coordinator, DistributedMPCCoordinator) # TODO check of another field!!
      neighbor_links = None # sim.coordinator.get_neighbor_pairs() if is_distributed else None
      admm_iteration_count = sim.coordinator.get_last_iteration_count() if is_distributed else None
      admm_converged = sim.coordinator.get_last_converged() if is_distributed else None

      obj_path = str(Path(__file__).resolve().parent.parent / "src" / "drone_sim" / "resources" / "assets" / obj_name) if obj_name else None
      png_bytes = render_png(room_min=sim.room_min, room_max=sim.room_max, drones=sim.drones,
                             drone_traces=traces, obstacles=sim.obstacles,
                             step_count=sim.step_count, compute_time_s=sim.compute_time_s,
                             neighbor_links=neighbor_links, admm_iteration_count=admm_iteration_count, admm_converged=admm_converged,
                             safety_alphas=safety_alphas,
                             width=width, height=height, dpi=dpi, elev=elev, azim=azim,
                             obj_path=obj_path, obj_scale=0.5, draw_sphere_if_obj=False
                             )

      if show_live:
         img = mpimg.imread(io.BytesIO(png_bytes), format="png")
         if img_artist is None:
            img_artist = ax.imshow(img)
         else:
            img_artist.set_data(img)

      # For recording
      if record_path is not None or gif_out is not None:
         pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
         if record_path is not None:
            frame_path = record_path / f"frame_{i:05d}.png"
            frame_path.write_bytes(png_bytes)
         if gif_out is not None:
            frames.append(pil_img)

      if show_live:
         fig.canvas.draw_idle()
         plt.pause(0.001)

      if sleep_s > 0:
         time.sleep(sleep_s)

      all_reached = all_drones_reached_destination(sim.drones)
      print(f"All drones reached: {all_reached}")

   if show_live:
      plt.ioff()
      plt.show()

   if gif_out is not None:
      if not frames:
         raise RuntimeError("No frames captured; cannot write GIF")
      duration_ms = int(round(1000.0 / max(0.1, float(gif_fps))))
      gif_out.parent.mkdir(parents=True, exist_ok=True)
      frames[0].save(gif_out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


def run_live_view(*, config_path: str | Path | None, params: dict[str, str] | None, steps: int, step_n: int, sleep_s: float, trace_len: int,
                  width: int, height: int, dpi: int, elev: float, azim: float, record_dir: str | Path | None, gif_path: str | Path | None, gif_fps: float,
                  obj_name: str | None=None) -> None:
   scenario = _create_scenario(config_path=config_path, params=params)
   scenario = apply_obj_name(scenario, obj_name)
   sim = Simulator.from_config(scenario)
   live_view_simulator(sim, steps=steps, step_n=step_n, sleep_s=sleep_s, trace_len=trace_len,
                       width=width, height=height, dpi=dpi, elev=elev, azim=azim,
                       record_dir=record_dir, gif_path=gif_path, gif_fps=gif_fps, obj_name=obj_name,
                       show_live=True)


def _parse_kv_params(items: list[str]) -> dict[str, str]:
   out: dict[str, str] = {}
   for item in items:
      if "=" not in item:
         raise ValueError(f"Bad --param '{item}'. Expected KEY=VALUE")
      k, v = item.split("=", 1)
      out[k] = v
   return out


def main(argv: list[str] | None = None) -> None:
   p = argparse.ArgumentParser(description="Live-view DroneSim by stepping the simulator and rendering in-process")
   p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                  help="Logging level (default: WARNING; use DEBUG for full SLSQP + constraint breakdown)")
   p.add_argument("--config",
                  help="Either a JSON file path (e.g. configs/2DronesHorizon2.json) or param has to set at least 'num_drones' and 'horizon'")

   p.add_argument("--param", action="append", default=[], help="Template parameter KEY=VALUE (may be repeated)")
   p.add_argument("--steps", type=int, default=1000)
   p.add_argument("--step-n", type=int, default=1)
   p.add_argument("--sleep", type=float, default=0.05)
   p.add_argument("--trace-len", type=int, default=1000)

   p.add_argument("--record-dir", default=None, help="If set, write PNG frames into this directory")
   p.add_argument("--gif", dest="gif_path", default=None, help="If set, write an animated GIF to this path")
   p.add_argument("--gif-fps", type=float, default=20.0, help="FPS for the generated GIF")

   p.add_argument("--width", type=int, default=900)
   p.add_argument("--height", type=int, default=700)
   p.add_argument("--dpi", type=int, default=120)
   p.add_argument("--elev", type=float, default=20.0)
   p.add_argument("--azim", type=float, default=-60.0)

   p.add_argument("--obj-name", type=str)

   args = p.parse_args(argv)
   logging.basicConfig(level=getattr(logging, args.log_level), format="%(name)s %(levelname)s %(message)s")

   params = _parse_kv_params(args.param)
   config_str = args.config

   run_live_view(config_path=config_str, params=params, steps=args.steps, step_n=args.step_n, sleep_s=args.sleep, trace_len=args.trace_len,
                 width=args.width, height=args.height, dpi=args.dpi, elev=args.elev, azim=args.azim, record_dir=args.record_dir, gif_path=args.gif_path,
                 gif_fps=args.gif_fps, obj_name=args.obj_name)


def _die(msg: str, code: int = 1) -> "NoReturn":
   print(f" ==== {msg}", file=sys.stderr)
   raise SystemExit(code)


if __name__ == "__main__":
   logging.basicConfig(level=logging.DEBUG)
   try:
      main()
   except KeyboardInterrupt:
      _die("Interrupted by user", code=0)
   except RuntimeError as e:
      _die(str(e), code=1)
   except Exception:
      # Unexpected error: include traceback (much more helpful than just str(e))
      import traceback

      traceback.print_exc()
      raise SystemExit(1)
