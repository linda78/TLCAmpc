"""E3 — density sweep (the "under certain circumstances" crossover).

For each drone count N, run the four matched pairs (sequencing off/on) on the
shared-center crossing geometry, and record completion + energy. Shows the density
at which sequencing starts to pay off (crossover) and where the sequencing arms
themselves begin to fail (→ N_c^v / κ_a validation).

Usage:
    python -m paper4_intersection_sphere.run_e3_density --ns 3 4 6 8 --seeds 0 1 2 --quiet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paper4_intersection_sphere.sim_runner import (ARMS, Scenario, results_to_csv, run_trial)

_OUT = Path(__file__).resolve().parent / "results" / "e3_density.csv"


def main() -> None:
   p = argparse.ArgumentParser(description="E3 density sweep")
   p.add_argument("--ns", nargs="+", type=int, default=[3, 4, 6, 8])
   p.add_argument("--arms", nargs="+", default=list(ARMS))
   p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
   p.add_argument("--max-steps", type=int, default=400)
   p.add_argument("--kind", default="sphere", choices=["sphere", "corners"],
                  help="geometry: open 'sphere' ring or small-room 'corners'")
   p.add_argument("--room-half", type=float, default=None,
                  help="room half-size (defaults: sphere=5.0, corners=3.0)")
   p.add_argument("--out", type=Path, default=_OUT)
   p.add_argument("--quiet", action="store_true")
   args = p.parse_args()
   room_half = args.room_half if args.room_half is not None else (3.0 if args.kind == "corners" else 5.0)

   logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
   if args.quiet:
      logging.getLogger("drone_sim").setLevel(logging.CRITICAL)
   log = logging.getLogger(__name__)

   results = []
   total = len(args.ns) * len(args.arms) * len(args.seeds)
   i = 0
   for n in args.ns:
      scn = Scenario(f"N{n}", n, kind=args.kind, room_half=room_half)
      for aname in args.arms:
         arm = ARMS[aname]
         for seed in args.seeds:
            i += 1
            r = run_trial(arm, scn, seed, max_steps=args.max_steps)
            results.append(r)
            log.info("[%d/%d] N=%d %s seed=%d -> %s energy=%.2f viol=%d (%.1fs)",
                     i, total, n, aname, seed, r.outcome, r.energy,
                     r.violation_steps, r.wall_time_s)

   results_to_csv(results, args.out)


if __name__ == "__main__":
   main()
