"""Plot the E2 (horizon → H_min) and E3 (density → capacity) ablations.

E2: completion rate vs MPC horizon per arm, with the theoretical H_min line.
E3: completion rate vs drone count per arm (baseline vs sequencing).

H_min = ceil( (1/dt) * sqrt(2 r_min alpha / U_max) ).
With dt=0.1, r_min=0.6, alpha=0.33, U_max=3  ->  H_min = 4.

Usage:
    python -m paper4_intersection_sphere.plot_ablations
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_RES = _HERE / "results"
_FIG = _RES / "figures"

DT, R_MIN, ALPHA, U_MAX = 0.1, 0.6, 0.33, 3.0
H_MIN = math.ceil((1.0 / DT) * math.sqrt(2.0 * R_MIN * ALPHA / U_MAX))

_ARM_LABEL = {
   "A1": "A1 cen/base/fixed", "A2": "A2 cen/base/adapt",
   "A3": "A3 cen/seq/fixed", "A4": "A4 cen/seq/adapt",
   "A5": "A5 dis/base/fixed", "A6": "A6 dis/base/adapt",
   "A7": "A7 dis/seq/fixed", "A8": "A8 dis/seq/adapt",
}


def _load(path: Path) -> list[dict]:
   with path.open(newline="", encoding="utf-8") as f:
      return list(csv.DictReader(f))


def _completion_by(rows: list[dict], xkey: str) -> dict[str, dict[int, float]]:
   """arm -> {x -> completion rate}."""
   buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
   for r in rows:
      buckets[(r["arm"], int(r[xkey]))].append(int(r["completed"]))
   out: dict[str, dict[int, float]] = defaultdict(dict)
   for (arm, x), cs in buckets.items():
      out[arm][x] = sum(cs) / len(cs)
   return out


def plot_e2(path: Path) -> Path | None:
   if not path.exists():
      print(f"(skip E2: no {path})")
      return None
   rows = _load(path)
   comp = _completion_by(rows, "horizon")
   fig, ax = plt.subplots(figsize=(5.2, 3.6))
   for arm in sorted(comp, key=lambda a: int(a[1:])):
      xs = sorted(comp[arm])
      ax.plot(xs, [comp[arm][x] * 100 for x in xs], "o-", label=_ARM_LABEL.get(arm, arm))
   ax.axvline(H_MIN, color="k", ls="--", lw=1)
   ax.text(H_MIN + 0.1, 5, f"$H_{{\\min}}={H_MIN}$", fontsize=9)
   ax.set_xlabel("MPC horizon $H$")
   ax.set_ylabel("completion rate [%]")
   ax.set_ylim(-5, 105)
   ax.set_title("E2: feasibility vs horizon (S-C6)")
   ax.grid(alpha=0.3)
   ax.legend(fontsize=7)
   fig.tight_layout()
   out = _FIG / "e2_horizon.png"
   fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf"))
   plt.close(fig)
   return out


def plot_e3(path: Path) -> Path | None:
   if not path.exists():
      print(f"(skip E3: no {path})")
      return None
   rows = _load(path)
   comp = _completion_by(rows, "n_drones")
   fig, ax = plt.subplots(figsize=(5.2, 3.6))
   for arm in sorted(comp, key=lambda a: int(a[1:])):
      xs = sorted(comp[arm])
      ax.plot(xs, [comp[arm][x] * 100 for x in xs], "s-", label=_ARM_LABEL.get(arm, arm))
   ax.set_xlabel("number of drones $N$")
   ax.set_ylabel("completion rate [%]")
   ax.set_ylim(-5, 105)
   ax.set_title("E3: feasibility vs density (corner room)")
   ax.grid(alpha=0.3)
   ax.legend(fontsize=8)
   fig.tight_layout()
   out = _FIG / "e3_density.png"
   fig.savefig(out, dpi=150); fig.savefig(out.with_suffix(".pdf"))
   plt.close(fig)
   return out


def main() -> None:
   p = argparse.ArgumentParser(description="Plot E2/E3 ablations")
   p.add_argument("--e2", type=Path, default=_RES / "e2_horizon.csv")
   p.add_argument("--e3", type=Path, default=_RES / "e3_density.csv")
   args = p.parse_args()
   print(f"H_min (theory) = {H_MIN}")
   f2 = plot_e2(args.e2)
   f3 = plot_e3(args.e3)
   for f in (f2, f3):
      if f:
         print("wrote", f)


if __name__ == "__main__":
   main()
