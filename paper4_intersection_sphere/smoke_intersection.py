"""C7 smoke run — confirm deadlock-free crossing under the intersection coordinators.

Runs a 4-drone shared-center crossing config to completion and reports:
  - outcome: COMPLETED / DEADLOCK / INFEASIBLE / TIMEOUT
  - min real pairwise separation vs. the safety threshold (safety check)
  - sequencing activity: peak active spheres, peak waiting drones, total parks

Not a pytest — a quick headless sanity check for the central (and DMPC) configs.

Usage:
    python -m paper4_intersection_sphere.smoke_intersection
    python -m paper4_intersection_sphere.smoke_intersection --config <path> --max-steps 800
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from drone_sim.domain.config import ScenarioConfig
from drone_sim.domain.drone import GatedDrone
from drone_sim.domain.utils.helper import all_drones_reached_destination
from drone_sim.simulation.simulator import Simulator

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT = _ROOT / "configs" / "franck_intersection" / "4drone_central_intersection.json"


def _min_separation(sim: Simulator) -> tuple[float, float]:
    """Smallest real pairwise (distance, distance - threshold) this step."""
    drones = sim.drones
    min_dist = float("inf")
    min_margin = float("inf")
    for i in range(len(drones)):
        for j in range(i + 1, len(drones)):
            di, dj = drones[i], drones[j]
            dist = float(np.linalg.norm(di.position() - dj.position()))
            thresh = float(di.safety_zone + dj.safety_zone)
            min_dist = min(min_dist, dist)
            min_margin = min(min_margin, dist - thresh)
    return min_dist, min_margin


def run(config: Path, max_steps: int = 800, eps_v: float = 1e-2, stall_window: int = 60) -> int:
    cfg = ScenarioConfig.model_validate(json.loads(config.read_text()))
    sim = Simulator.from_config(cfg)
    coord = sim.coordinator
    has_spheres = hasattr(coord, "active_spheres")

    print(f"config:      {config.name}")
    print(f"coordinator: {type(coord).__name__}  (horizon={getattr(coord, 'horizon', '?')})")
    print(f"drones:      {len(sim.drones)}  max_steps={max_steps}")

    min_margin_overall = float("inf")
    min_dist_overall = float("inf")
    peak_spheres = 0
    peak_waiting = 0
    ever_parked: set[str] = set()
    recent_speeds: list[float] = []

    outcome = "TIMEOUT"
    steps_used = max_steps

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for step in range(max_steps):
            sim.step()

            if sim.infeasible:
                outcome = "INFEASIBLE"
                steps_used = step + 1
                break

            d, m = _min_separation(sim)
            min_dist_overall = min(min_dist_overall, d)
            min_margin_overall = min(min_margin_overall, m)

            if has_spheres:
                spheres = coord.active_spheres()
                peak_spheres = max(peak_spheres, len(spheres))
            waiting = [dr.drone_id for dr in sim.drones
                       if isinstance(dr, GatedDrone) and dr.is_waiting]
            peak_waiting = max(peak_waiting, len(waiting))
            ever_parked.update(waiting)

            fleet_speed = sum(float(np.linalg.norm(dr.velocity())) for dr in sim.drones)
            recent_speeds.append(fleet_speed)
            if len(recent_speeds) > stall_window:
                recent_speeds.pop(0)

            if all_drones_reached_destination(sim.drones):
                outcome = "COMPLETED"
                steps_used = step + 1
                break

            # Stall detection: nobody reached the goal but the whole fleet is
            # near-stationary for a sustained window -> deadlock.
            if (len(recent_speeds) == stall_window
                    and max(recent_speeds) < eps_v):
                outcome = "DEADLOCK"
                steps_used = step + 1
                break

    reached = sum(
        dr.route.target_reached(position=dr.position(), thresh=0.1)
        for dr in sim.drones
    )
    safe = min_margin_overall >= 0.0

    print("-" * 56)
    print(f"outcome:        {outcome}  (after {steps_used} steps, t={sim.t:.1f}s)")
    print(f"reached target: {reached}/{len(sim.drones)}")
    print(f"min separation: dist={min_dist_overall:.3f}  margin={min_margin_overall:+.3f}  "
          f"-> {'SAFE' if safe else 'VIOLATION'}")
    print(f"sequencing:     peak_spheres={peak_spheres}  peak_waiting={peak_waiting}  "
          f"drones_parked={sorted(ever_parked)}")
    print("-" * 56)

    ok = outcome == "COMPLETED" and safe and peak_spheres > 0
    print("RESULT:", "PASS" if ok else "CHECK", "\n")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description="C7 intersection smoke run")
    p.add_argument("--config", type=Path, default=_DEFAULT)
    p.add_argument("--max-steps", type=int, default=800)
    args = p.parse_args()
    raise SystemExit(run(args.config, args.max_steps))


if __name__ == "__main__":
    main()