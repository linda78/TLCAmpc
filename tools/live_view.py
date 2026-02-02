from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import NoReturn

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

from drone_sim.api.render import render_png
from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator
from tools.utility.config_creater import create_default_config
from tools.utility.param_helper import load_parametrized_json, _parse_kv_params
from tools.utility.metrics_fcts import all_drones_reached_destination


def create_scenario(config_path: str | Path | None, params: dict[str, str] | None) -> ScenarioConfig:
   if not config_path:
      # Get N and H from param and load default values
      num_str = params.get("num_drones")
      hor_str = params.get("horizon")
      if num_str is None or hor_str is None:
         _die("When --config has no path, you must provide both num_drones=<int> and horizon=<int> via --param.")

      scenario = create_default_config(n_drones=int(num_str), horizon=int(hor_str))
   else:
      # Load JSON (with optional template substitution) and validate as ScenarioConfig.
      cfg_json = load_parametrized_json(config_path, params=params)
      if not isinstance(cfg_json, dict):
         raise TypeError(f"Config must decode to a JSON object/dict, got {type(cfg_json).__name__}")

      scenario = ScenarioConfig.model_validate(cfg_json)
      print(scenario)

   return scenario

def run_live_view(*, config_path: str | Path | None, params: dict[str, str] | None, steps: int, step_n: int, sleep_s: float, trace_len: int,
                  width: int, height: int, dpi: int, elev: float, azim: float, record_dir: str | Path | None, gif_path: str | Path | None, gif_fps: float) -> None:
   scenario = create_scenario(config_path=config_path, params=params)

   sim = Simulator.from_config(scenario)

   record_path = Path(record_dir) if record_dir is not None else None
   if record_path is not None:
      record_path.mkdir(parents=True, exist_ok=True)

   gif_out = Path(gif_path) if gif_path is not None else None

   frames: list[Image.Image] = []

   plt.ion()
   fig, ax = plt.subplots()
   ax.set_axis_off()

   img_artist = None

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
      traces = [[] if trace_len == 0 else sim.traces.get(d.drone_id, [])[-trace_len:] for d in sim.drones]
      safety_zones = [float(d.safety_zone) for d in sim.drones]

      png_bytes = render_png(
            room_min=sim.room_min,
            room_max=sim.room_max,
            drone_positions=[d.position() for d in sim.drones],
            drone_radii=[d.radius for d in sim.drones],
            drone_safety_zones=safety_zones,
            drone_colors=[d.color for d in sim.drones],
            safety_colors=[d.safety_color for d in sim.drones],
            trace_colors=[d.trace_color for d in sim.drones],
            drone_traces=traces,
            obstacles=sim.obstacles,
            step_count=sim.step_count,
            compute_time_s=sim.compute_time_s,
            room_radius=sim.room_radius,
            width=width,
            height=height,
            dpi=dpi,
            elev=elev,
            azim=azim,
      )

      # For display
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

      fig.canvas.draw_idle()
      plt.pause(0.001)

      if sleep_s > 0:
         time.sleep(sleep_s)

      all_reached = all_drones_reached_destination(sim.drones)
      print(f"All drones reached: {all_reached}")

   plt.ioff()
   plt.show()

   if gif_out is not None:
      if not frames:
         raise RuntimeError("No frames captured; cannot write GIF")
      duration_ms = int(round(1000.0 / max(0.1, float(gif_fps))))
      gif_out.parent.mkdir(parents=True, exist_ok=True)
      frames[0].save(gif_out, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


def main(argv: list[str] | None = None) -> None:
   p = argparse.ArgumentParser(description="Live-view DroneSim by stepping the simulator and rendering in-process")
   p.add_argument("--config",
                  help="Either a JSON file path (e.g. configs/2DronesHorizon2.json) or param has to set at least 'num_drones' and 'horizon'")

   p.add_argument("--param", action="append", default=[], help="Template parameter KEY=VALUE (may be repeated), see @overrite_param.py for details")
   p.add_argument("--steps", type=int, default=500)
   p.add_argument("--step-n", type=int, default=1)
   p.add_argument("--sleep", type=float, default=0.05)
   p.add_argument("--trace-len", type=int, default=500)

   p.add_argument("--record-dir", default=None, help="If set, write PNG frames into this directory")
   p.add_argument("--gif", dest="gif_path", default=None, help="If set, write an animated GIF to this path")
   p.add_argument("--gif-fps", type=float, default=20.0, help="FPS for the generated GIF")

   p.add_argument("--width", type=int, default=900)
   p.add_argument("--height", type=int, default=700)
   p.add_argument("--dpi", type=int, default=120)
   p.add_argument("--elev", type=float, default=20.0)
   p.add_argument("--azim", type=float, default=-60.0)

   args = p.parse_args(argv)

   params = _parse_kv_params(args.param)
   config_str = args.config

   run_live_view(config_path=config_str, params=params, steps=args.steps, step_n=args.step_n, sleep_s=args.sleep, trace_len=args.trace_len,
                 width=args.width, height=args.height, dpi=args.dpi, elev=args.elev, azim=args.azim, record_dir=args.record_dir, gif_path=args.gif_path, gif_fps=args.gif_fps)


def _die(msg: str, code: int = 1) -> "NoReturn":
   print(f" ==== {msg}", file=sys.stderr)
   raise SystemExit(code)


if __name__ == "__main__":
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
