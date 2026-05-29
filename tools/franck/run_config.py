#!/usr/bin/env python3

from __future__ import annotations

import argparse
import numpy as np

from tools.franck.generate_points import drone_jam, generate_random_points

# [Red, Orange, Yellow-Green, Green, Cyan, Blue, Violet, Magenta]
DRONE_COLORS = ["#ff0000", "#ff7f00", "#d4ff00", "#00ff4f", "#00ffff", "#004fff", "#7f00ff", "#ff00d4"]



def random_pairs(*, n: int, room_extent: float, safety_zone: float, min_dist: float, seed: int, max_attempts: int, target_mode: str, ) -> list[
   tuple[list[float], list[float]]]:
   """Sample ``n`` (start, target) pairs via Franck's ``generate_random_points``."""
   cube_side = 2.0 * float(room_extent)
   starts = generate_random_points(float(safety_zone), cube_side, int(n), float(min_dist), int(seed), max_attempts=int(max_attempts))
   if target_mode == "antipodal":
      targets = [[-float(c) for c in s] for s in starts]
   elif target_mode == "independent":
      targets = generate_random_points(float(safety_zone), cube_side, int(n), float(min_dist), int(seed) + 1, max_attempts=int(max_attempts))
   else:
      raise SystemExit(f"unknown target_mode: {target_mode!r}")
   if len(starts) != n or len(targets) != n:
      raise SystemExit(f"generate_random_points produced fewer points than requested "
                       f"(starts={len(starts)}, targets={len(targets)}, requested={n}). "
                       f"Increase --room-extent, lower --min-dist, or raise --max-attempts.")
   return [([float(c) for c in s], [float(c) for c in t]) for s, t in zip(starts, targets)]


def build_config(pairs: list[tuple[list[float], list[float]]], *, room_extent: float, v_max: float, u_max: float, horizon: int, safety_zone: float,
      alpha: float, radius: float, ) -> dict:
   bound = float(room_extent)
   return {"dt": 0.1, "room": {"min": [-bound, -bound, -bound], "max": [bound, bound, bound]}, "physics": [
         {"id": "franck", "type": "linear_kinematics", "params": {"v_max": v_max, "u_min": [-u_max, -u_max, -u_max], "u_max": [u_max, u_max, u_max], }, }],
         "controller": {"type": "mpc_agent",
               "params": {"horizon": horizon, "q_pos": [2.0, 2.0, 2.0], "q_vel": [0.1, 0.1, 0.1], "r_u": [0.5, 0.5, 0.5], }, },
         "coordinator": {"type": "mpc_central", "params": {"horizon": horizon, "room_wall_tolerance": 0.5}, }, "drones": [
               {"drone_id": f"drone-{i + 1}", "physics": "franck", "start": list(start), "target": list(target), "radius": radius, "safety_zone": safety_zone,
                     "cons_stop": 0.0, "alpha": alpha, "drone_color": DRONE_COLORS[i % len(DRONE_COLORS)], } for i, (start, target) in enumerate(pairs)], }


def parse_args() -> argparse.Namespace:
   p = argparse.ArgumentParser(description="Build a Franck-style 8-drone scenario config in memory and run the Simulator.", )
   p.add_argument("--extent", type=float, default=2.0, help="Half-side of the corner cube (corners mode only; default: 2.0)")
   p.add_argument("--num-drones", type=int, default=8, help="Number of drones")
   p.add_argument("--seed", type=int, default=50, help="Random seed")
   p.add_argument("--min-dist", type=float, default=None, help="Minimum spacing between sampled points (default: 2 * safety_zone)")
   p.add_argument("--max-attempts", type=int, default=1000, help="Max sampling attempts per point")
   p.add_argument("--target-mode", choices=["antipodal", "independent"], default="antipodal",
                  help="In random mode, mirror starts through origin or sample independently with seed+1")
   p.add_argument("--room-extent", type=float, default=3.0)
   p.add_argument("--v-max", type=float, default=1.0)
   p.add_argument("--u-max", type=float, default=2.0)
   p.add_argument("--horizon", type=int, default=10)
   p.add_argument("--safety-zone", type=float, default=1.0)
   p.add_argument("--alpha", type=float, default=None)
   p.add_argument("--radius", type=float, default=0.5)
   p.add_argument("--start", nargs=4, action="append", default=[], metavar=("DRONE", "X", "Y", "Z"),
                  help="Override start of drone DRONE (1-indexed). Repeatable.")
   p.add_argument("--target", nargs=4, action="append", default=[], metavar=("DRONE", "X", "Y", "Z"),
                  help="Override target of drone DRONE (1-indexed). Repeatable.")
   p.add_argument("--steps", type=int, default=500, help="Number of Simulator.step() calls to perform (default: 500)")
   p.add_argument("--live-view", action="store_true", help="Show an interactive matplotlib live view while stepping")
   p.add_argument("--step-n", type=int, default=1, help="Sub-steps per rendered frame (live-view/save mode only; default: 1)")
   p.add_argument("--sleep", type=float, default=0.05, help="Seconds to sleep between frames (live-view/save mode only)")
   p.add_argument("--trace-len", type=int, default=1000, help="Number of past positions to draw as trace (live-view/save mode only)")
   p.add_argument("--record-dir", default=None, help="If set, write PNG frames into this directory")
   p.add_argument("--gif", dest="gif_path", default=None, help="If set, write an animated GIF to this path")
   p.add_argument("--gif-fps", type=float, default=20.0, help="FPS for the generated GIF")
   p.add_argument("--width", type=int, default=900, help="Render width (live-view/save mode only)")
   p.add_argument("--height", type=int, default=900, help="Render height (live-view/save mode only)")
   p.add_argument("--dpi", type=int, default=120, help="Render DPI (live-view/save mode only)")
   p.add_argument("--elev", type=float, default=20.0, help="Camera elevation (live-view/save mode only)")
   p.add_argument("--azim", type=float, default=-60.0, help="Camera azimuth (live-view/save mode only)")
   p.add_argument("--obj-name", type=str, default=None, help="Optional drone .obj asset name under src/drone_sim/resources/assets")

   return p.parse_args()


def apply_overrides(pairs: list[tuple[list[float], list[float]]], starts: list[list[str]], targets: list[list[str]], ) -> list[tuple[list[float], list[float]]]:
   pairs = [(list(s), list(t)) for s, t in pairs]
   for raw in starts:
      idx = int(raw[0]) - 1
      if not 0 <= idx < len(pairs):
         raise SystemExit(f"--start: drone index {raw[0]} out of range 1..{len(pairs)}")
      pairs[idx] = ([float(x) for x in raw[1:]], pairs[idx][1])
   for raw in targets:
      idx = int(raw[0]) - 1
      if not 0 <= idx < len(pairs):
         raise SystemExit(f"--target: drone index {raw[0]} out of range 1..{len(pairs)}")
      pairs[idx] = (pairs[idx][0], [float(x) for x in raw[1:]])
   return pairs


def main() -> None:
   args = parse_args()
   seeds = ['na', 14, 25, 27, 36, 40, 42, 45, 50, 55, 80]

   for seed in seeds:
      if seed == 'na':
         positions = drone_jam(cube_side=6.0, drone_radius=1.0, v_max=1.0, u_max=2.0)
         positions = positions[:args.num_drones]
         starts = [[float(c) for c in p] for p in positions]
         targets = [[-float(c) for c in s] for s in starts]
         pairs = list(zip(starts, targets))
           
      else: 
         min_dist = args.min_dist if args.min_dist is not None else 2.0 * args.safety_zone
         pairs = random_pairs(n=args.num_drones, room_extent=args.room_extent, safety_zone=args.safety_zone, min_dist=min_dist, seed=args.seed,
         max_attempts=args.max_attempts, target_mode=args.target_mode, )
      
      pairs = apply_overrides(pairs, args.start, args.target)
      config = build_config(pairs, room_extent=args.room_extent, v_max=args.v_max, u_max=args.u_max, horizon=args.horizon, safety_zone=args.safety_zone,
         alpha=args.alpha, radius=args.radius, )

      import json
      import jsbeautifier
      from datetime import datetime
       
      filename = datetime.now().strftime(f"configs/franck/scenario_seed_{seed}.json")  
      options = jsbeautifier.default_options()
      options.indent_size = 2
      formatted_json = jsbeautifier.beautify(json.dumps(config), options)   
      with open(filename, "w") as f:
           f.write(formatted_json)

   
   from drone_sim.domain.config import ScenarioConfig
   from drone_sim.simulation.simulator import Simulator

   scenario = ScenarioConfig.model_validate(config)
   sim = Simulator.from_config(scenario)

   print(f"drones={len(config['drones'])}, steps={args.steps}")
   for d in config["drones"]:
      print(f"  {d['drone_id']}: {d['start']} -> {d['target']}")

   if args.live_view or args.record_dir or args.gif_path:
      from tools.live_view import live_view_simulator
      live_view_simulator(sim, steps=args.steps, step_n=args.step_n, sleep_s=args.sleep, trace_len=args.trace_len,
            width=args.width, height=args.height, dpi=args.dpi, elev=args.elev, azim=args.azim,
            record_dir=args.record_dir, gif_path=args.gif_path, gif_fps=args.gif_fps,
            obj_name=args.obj_name, show_live=args.live_view, )
   else:
      for _ in range(args.steps):
         sim.step()

   print(f"\nt = {sim.t:.3f} (dt = {sim.dt})")
   for d in sim.drones:
      p = d.position()
      v = d.velocity()
      print(f"  {d.drone_id}: p={p.tolist()}, v={v.tolist()}, safety_zone={d.safety_zone}")
   if sim.last_collisions:
      print("collisions:")
      for ev in sim.last_collisions:
         print("  ", ev)
   else:
      print("collisions: none")


if __name__ == "__main__":
   main()
