"""E2 — horizon ablation (validates H_min / κ_a).

Fix one conflict-dense scenario, sweep the MPC horizon, and record feasibility /
completion / energy per horizon. Compare the empirical feasibility onset to the
theoretical H_min from §IV.

Usage:
    python -m paper4_intersection_sphere.run_e2_horizon \
        --scenario S-X4 --arms A3 A4 A7 A8 --horizons 2 3 4 5 6 8 10 --seeds 0 1 2 --quiet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paper4_intersection_sphere.sim_runner import (ARMS, SCENARIOS, results_to_csv, run_trial)

_OUT = Path(__file__).resolve().parent / "results" / "e2_horizon.csv"


def main() -> None:
   p = argparse.ArgumentParser(description="E2 horizon ablation")
   p.add_argument("--scenario", default="S-X4")
   p.add_argument("--arms", nargs="+", default=["A3", "A4", "A7", "A8"])
   p.add_argument("--horizons", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8, 10])
   p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
   p.add_argument("--max-steps", type=int, default=400)
   p.add_argument("--out", type=Path, default=_OUT)
   p.add_argument("--quiet", action="store_true")
   args = p.parse_args()

   logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
   if args.quiet:
      logging.getLogger("drone_sim").setLevel(logging.CRITICAL)
   log = logging.getLogger(__name__)

   scn = SCENARIOS[args.scenario]
   results = []
   total = len(args.horizons) * len(args.arms) * len(args.seeds)
   i = 0
   for H in args.horizons:
      for aname in args.arms:
         arm = ARMS[aname]
         for seed in args.seeds:
            i += 1
            r = run_trial(arm, scn, seed, max_steps=args.max_steps, horizon=H)
            results.append(r)
            log.info("[%d/%d] H=%d %s seed=%d -> %s energy=%.2f viol=%d (%.1fs)",
                     i, total, H, aname, seed, r.outcome, r.energy,
                     r.violation_steps, r.wall_time_s)

   results_to_csv(results, args.out)


if __name__ == "__main__":
   main()
