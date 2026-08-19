"""Trajectory comparison figure: sequencing off vs on, same scenario/seed.

Runs a matched pair (e.g. A1 vs A3, or A5 vs A7) on one scenario, captures the
real position traces, and draws top-down XY trajectories side by side. The
sequencing panel overlays the intersection sphere(s) that were placed. This is the
"deadlock / unsafe-graze vs sequenced-through" figure for §VI.

Usage:
    python -m paper4_intersection_sphere.plot_trajectories \
        --off A1 --on A3 --scenario S-X4 --seed 0
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drone_sim.domain.drone import GatedDrone
from drone_sim.simulation.simulator import Simulator
from drone_sim.domain.utils.helper import all_drones_reached_destination
from paper4_intersection_sphere.sim_runner import ARMS, SCENARIOS, build_arm_config

_FIG_DIR = Path(__file__).resolve().parent / "results" / "figures"


def _run_capture(arm_name: str, scenario: str, seed: int, max_steps: int):
   """Run one trial, return (traces, starts, targets, sphere_centers, outcome)."""
   arm = ARMS[arm_name]
   scn = SCENARIOS[scenario]
   cfg = build_arm_config(arm, scn, seed)
   sim = Simulator.from_config(cfg)
   coord = sim.coordinator
   has_spheres = hasattr(coord, "active_spheres")

   starts = [d.position().copy() for d in sim.drones]
   targets = [d.route.target.copy() for d in sim.drones]
   sphere_centers: list[np.ndarray] = []
   outcome = "TIMEOUT"

   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      for _ in range(max_steps):
         sim.step()
         if sim.infeasible:
            outcome = "INFEASIBLE"
            break
         if has_spheres:
            for rec in coord.active_spheres().values():
               sphere_centers.append(rec.sphere.center.copy())
         if all_drones_reached_destination(sim.drones):
            outcome = "COMPLETED"
            break
   traces = {did: np.asarray(t) for did, t in sim.traces.items()}
   return traces, starts, targets, sphere_centers, outcome, sim


def _panel(ax, traces, starts, targets, sphere_centers, title):
   colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(traces))))
   for i, (did, tr) in enumerate(sorted(traces.items())):
      if tr.ndim != 2 or len(tr) == 0:
         continue
      ax.plot(tr[:, 0], tr[:, 1], "-", color=colors[i], lw=1.5, alpha=0.9)
      ax.plot(starts[i][0], starts[i][1], "o", color=colors[i], ms=6)
      ax.plot(targets[i][0], targets[i][1], "*", color=colors[i], ms=11)
   # Deduplicate sphere centers (snap to grid) and draw them.
   seen = set()
   for c in sphere_centers:
      key = (round(float(c[0]), 1), round(float(c[1]), 1))
      if key in seen:
         continue
      seen.add(key)
      ax.add_patch(plt.Circle((c[0], c[1]), 0.25, color="k", alpha=0.15, zorder=0))
   ax.set_title(title, fontsize=10)
   ax.set_aspect("equal")
   ax.set_xlabel("x"); ax.set_ylabel("y")
   ax.grid(alpha=0.3)


def main() -> None:
   p = argparse.ArgumentParser(description="Trajectory off/on comparison")
   p.add_argument("--off", default="A1")
   p.add_argument("--on", default="A3")
   p.add_argument("--scenario", default="S-X4")
   p.add_argument("--seed", type=int, default=0)
   p.add_argument("--max-steps", type=int, default=400)
   p.add_argument("--quiet", action="store_true")
   args = p.parse_args()
   if args.quiet:
      logging.disable(logging.CRITICAL)

   off = _run_capture(args.off, args.scenario, args.seed, args.max_steps)
   on = _run_capture(args.on, args.scenario, args.seed, args.max_steps)

   fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
   _panel(axes[0], off[0], off[1], off[2], off[3],
          f"{args.off} (seq off) — {off[4]}")
   _panel(axes[1], on[0], on[1], on[2], on[3],
          f"{args.on} (seq on) — {on[4]}")
   fig.suptitle(f"Trajectories on {args.scenario} (seed {args.seed}): "
                f"○ start  ★ goal  ● sphere")
   fig.tight_layout()
   _FIG_DIR.mkdir(parents=True, exist_ok=True)
   out = _FIG_DIR / f"trajectories_{args.scenario}_{args.off}_vs_{args.on}.png"
   fig.savefig(out, dpi=150)
   fig.savefig(out.with_suffix(".pdf"))
   plt.close(fig)
   print("wrote", out)


if __name__ == "__main__":
   main()
