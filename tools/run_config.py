from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator
from tools.utility.param_helper import load_parametrized_json


def run_scenario(config_path: str | Path, *, steps: int = 50) -> None:
   """Load a scenario JSON and run the Simulator for a fixed number of steps.

   This is a small helper for offline runs without the REST API.
   """

   cfg: Any = load_parametrized_json(config_path, params=None)
   if isinstance(cfg, str):
      cfg = json.loads(cfg)
   if not isinstance(cfg, dict):
      raise TypeError(f"Config must decode to a JSON object/dict, got {type(cfg).__name__}")

   scenario = ScenarioConfig.model_validate(cfg)
   sim = Simulator.from_config(scenario)

   for _ in range(steps):
      sim.step()

   print(f"t = {sim.t:.3f} (dt = {sim.dt})")
   print("drones:")
   for d in sim.drones:
      p = d.position()
      v = d.velocity()
      print(f"  {d.drone_id}: p={p.tolist()}, v={v.tolist()}, radius={d.radius}, safety_zone={d.safety_zone}")

   if sim.last_collisions:
      print("collisions:")
      for ev in sim.last_collisions:
         print("  ", ev)
   else:
      print("collisions: none")


def main(argv: list[str] | None = None) -> None:
   parser = argparse.ArgumentParser(description="Run a DroneSim scenario config directly with the Simulator")
   parser.add_argument("--config", required=True, help="Path to scenario JSON (supports ${var} placeholders)")
   parser.add_argument("--steps", type=int, default=50, help="Number of Simulator.step() calls to perform (default: 50)")

   args = parser.parse_args(argv)

   run_scenario(args.config, steps=args.steps)


if __name__ == "__main__":
   main()
