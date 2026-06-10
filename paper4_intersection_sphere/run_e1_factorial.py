"""E1 — factorial comparison (A1..A8 x scenarios x seeds).

Core result: per (mode, zone) cell, the sequencing-on vs sequencing-off delta in
outcome (deadlock removal) and control energy, across conflict densities.

Usage:
    python -m paper4_intersection_sphere.run_e1_factorial \
        --scenarios S-X3 S-X4 S-X6 --seeds 0 1 2 --max-steps 400
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paper4_intersection_sphere.sim_runner import (ARMS, SCENARIOS, results_to_csv, run_grid)

_OUT = Path(__file__).resolve().parent / "results" / "e1_factorial.csv"


def main() -> None:
   p = argparse.ArgumentParser(description="E1 factorial run")
   p.add_argument("--arms", nargs="+", default=list(ARMS))
   p.add_argument("--scenarios", nargs="+", default=["S-X3", "S-X4", "S-X6"])
   p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
   p.add_argument("--max-steps", type=int, default=400)
   p.add_argument("--out", type=Path, default=_OUT)
   p.add_argument("--quiet", action="store_true", help="silence drone_sim solver logs")
   args = p.parse_args()

   logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
   if args.quiet:
      logging.getLogger("drone_sim").setLevel(logging.CRITICAL)

   for sid in args.scenarios:
      if sid not in SCENARIOS:
         raise SystemExit(f"unknown scenario {sid!r}; choices: {list(SCENARIOS)}")

   results = run_grid(args.arms, args.scenarios, args.seeds, max_steps=args.max_steps)
   results_to_csv(results, args.out)


if __name__ == "__main__":
   main()
