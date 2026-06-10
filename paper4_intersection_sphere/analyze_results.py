"""Aggregate E1 factorial CSV → summary tables + figures.

Reads results/e1_factorial.csv and produces:
  - console summary: per (scenario, arm) outcome + mean energy / makespan / safety
  - matched-pair deltas: sequencing-on minus sequencing-off, per (mode, zone, scenario)
  - figures/e1_energy_factorial.png : grouped energy bars (on/off per mode,zone)
  - figures/e1_outcomes.png         : completion/deadlock stacked bars per arm

Pure stdlib + matplotlib (no pandas) to stay dependency-light.

Usage:
    python -m paper4_intersection_sphere.analyze_results
    python -m paper4_intersection_sphere.analyze_results --csv results/e1_factorial.csv
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
_DEFAULT_CSV = _HERE / "results" / "e1_factorial.csv"
_FIG_DIR = _HERE / "results" / "figures"

_MODE_ORDER = ["central", "distributed"]
_ZONE_ORDER = ["fixed", "adaptive"]


def _load(path: Path) -> list[dict]:
   with path.open(newline="", encoding="utf-8") as f:
      rows = list(csv.DictReader(f))
   for r in rows:
      r["sequencing"] = bool(int(r["sequencing"]))
      r["completed"] = bool(int(r["completed"]))
      r["energy"] = float(r["energy"])
      r["makespan_s"] = float(r["makespan_s"])
      r["min_margin"] = math.inf if r["min_margin"] == "inf" else float(r["min_margin"])
      r["violation_steps"] = int(r["violation_steps"])
      # New (post-metric-fix) columns; default gracefully if reading an old CSV.
      r["min_body_margin"] = (math.inf if r.get("min_body_margin", "inf") == "inf"
                              else float(r["min_body_margin"]))
      r["collision_steps"] = int(r.get("collision_steps", 0) or 0)
      r["n_drones"] = int(r["n_drones"])
   return rows


def _mean(xs: list[float]) -> float:
   return sum(xs) / len(xs) if xs else float("nan")


def _agg(rows: list[dict]) -> dict[tuple, dict]:
   """Aggregate over seeds, keyed by (scenario, arm)."""
   groups: dict[tuple, list[dict]] = defaultdict(list)
   for r in rows:
      groups[(r["scenario"], r["arm"])].append(r)
   out = {}
   for key, g in groups.items():
      n = len(g)
      completed = [x for x in g if x["completed"]]
      out[key] = {
         "mode": g[0]["mode"], "zone": g[0]["zone"], "sequencing": g[0]["sequencing"],
         "n_drones": g[0]["n_drones"], "n": n,
         "completion_rate": len(completed) / n,
         "n_deadlock": sum(x["outcome"] == "DEADLOCK" for x in g),
         "n_infeasible": sum(x["outcome"] == "INFEASIBLE" for x in g),
         "n_timeout": sum(x["outcome"] == "TIMEOUT" for x in g),
         # Energy/makespan averaged over completed runs only (fair comparison).
         "energy": _mean([x["energy"] for x in completed]),
         "makespan_s": _mean([x["makespan_s"] for x in completed]),
         "min_margin": min((x["min_margin"] for x in g), default=float("nan")),
         "violation_runs": sum(x["violation_steps"] > 0 for x in g),
         "min_body_margin": min((x["min_body_margin"] for x in g), default=float("nan")),
         "collision_runs": sum(x["collision_steps"] > 0 for x in g),
      }
   return out


def _print_summary(agg: dict[tuple, dict]) -> None:
   scenarios = sorted({k[0] for k in agg})
   arms = sorted({k[1] for k in agg}, key=lambda a: int(a[1:]))
   print("\n=== E1 summary (mean over seeds; energy/makespan over completed runs) ===")
   hdr = f"{'scn':6s} {'arm':3s} {'mode':11s} {'seq':3s} {'zone':8s} " \
         f"{'compl':6s} {'dead':4s} {'inf':3s} {'energy':>8s} {'mksp_s':>7s} " \
         f"{'coll':>4s} {'zviol':>5s} {'bodymrg':>8s}"
   print(hdr)
   print("-" * len(hdr))
   print("(coll = runs with a real body collision; zviol = runs breaching the "
         "arm's own zone constraint; bodymrg = min physical margin)")
   for scn in scenarios:
      for arm in arms:
         a = agg.get((scn, arm))
         if not a:
            continue
         print(f"{scn:6s} {arm:3s} {a['mode']:11s} {'on' if a['sequencing'] else 'off':3s} "
               f"{a['zone']:8s} {a['completion_rate']*100:5.0f}% {a['n_deadlock']:4d} "
               f"{a['n_infeasible']:3d} {a['energy']:8.2f} {a['makespan_s']:7.1f} "
               f"{a['collision_runs']:4d} {a['violation_runs']:5d} {a['min_body_margin']:+8.3f}")


def _matched_pairs(agg: dict[tuple, dict]) -> None:
   """sequencing-on minus sequencing-off, holding (scenario, mode, zone) fixed."""
   print("\n=== Matched-pair sequencing effect (ON − OFF) ===")
   hdr = f"{'scn':6s} {'mode':11s} {'zone':8s} {'dEnergy':>9s} {'dCompl':>7s} " \
         f"{'off_out':>16s} {'on_out':>16s}"
   print(hdr)
   print("-" * len(hdr))
   # index by (scenario, mode, zone, sequencing)
   idx = {}
   for (scn, arm), a in agg.items():
      idx[(scn, a["mode"], a["zone"], a["sequencing"])] = a
   scns = sorted({k[0] for k in agg})
   for scn in scns:
      for mode in _MODE_ORDER:
         for zone in _ZONE_ORDER:
            off = idx.get((scn, mode, zone, False))
            on = idx.get((scn, mode, zone, True))
            if not off or not on:
               continue
            d_e = on["energy"] - off["energy"]
            d_c = on["completion_rate"] - off["completion_rate"]
            off_out = f"{off['completion_rate']*100:.0f}%c/{off['collision_runs']}coll"
            on_out = f"{on['completion_rate']*100:.0f}%c/{on['collision_runs']}coll"
            print(f"{scn:6s} {mode:11s} {zone:8s} {d_e:+9.2f} {d_c*100:+6.0f}% "
                  f"{off_out:>16s} {on_out:>16s}")


def _fig_energy(agg: dict[tuple, dict], scenarios: list[str], tag: str = "") -> Path:
   """Grouped energy bars: per (mode, zone) cell, sequencing off vs on, one panel per scenario."""
   _FIG_DIR.mkdir(parents=True, exist_ok=True)
   cells = [(m, z) for m in _MODE_ORDER for z in _ZONE_ORDER]
   labels = [f"{m[:4]}\n{z}" for (m, z) in cells]
   idx = {}
   for (scn, arm), a in agg.items():
      idx[(scn, a["mode"], a["zone"], a["sequencing"])] = a

   n = len(scenarios)
   fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), sharey=True)
   if n == 1:
      axes = [axes]
   x = range(len(cells))
   width = 0.38
   for ax, scn in zip(axes, scenarios):
      off_e = [idx.get((scn, m, z, False), {}).get("energy", float("nan")) for (m, z) in cells]
      on_e = [idx.get((scn, m, z, True), {}).get("energy", float("nan")) for (m, z) in cells]
      ax.bar([i - width / 2 for i in x], off_e, width, label="seq off", color="#bbbbbb")
      ax.bar([i + width / 2 for i in x], on_e, width, label="seq on", color="#1f77b4")
      ax.set_title(scn)
      ax.set_xticks(list(x))
      ax.set_xticklabels(labels, fontsize=8)
      ax.grid(axis="y", alpha=0.3)
   axes[0].set_ylabel(r"control energy  $\sum\|u\|^2\,\Delta t$")
   axes[-1].legend(fontsize=8)
   fig.suptitle("E1: control energy — sequencing off vs on, per (mode, zone)")
   fig.tight_layout()
   out = _FIG_DIR / f"e1{tag}_energy_factorial.png"
   fig.savefig(out, dpi=150)
   fig.savefig(out.with_suffix(".pdf"))
   plt.close(fig)
   return out


def _fig_outcomes(agg: dict[tuple, dict], scenarios: list[str], tag: str = "") -> Path:
   """Stacked outcome bars per arm, one panel per scenario."""
   _FIG_DIR.mkdir(parents=True, exist_ok=True)
   arms = sorted({k[1] for k in agg}, key=lambda a: int(a[1:]))
   n = len(scenarios)
   fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), sharey=True)
   if n == 1:
      axes = [axes]
   for ax, scn in zip(axes, scenarios):
      compl, dead, inf, to = [], [], [], []
      for arm in arms:
         a = agg.get((scn, arm))
         if not a:
            compl.append(0); dead.append(0); inf.append(0); to.append(0); continue
         tot = a["n"]
         c = a["completion_rate"] * tot
         compl.append(c)
         dead.append(a["n_deadlock"])
         inf.append(a["n_infeasible"])
         to.append(a["n_timeout"])
      x = range(len(arms))
      ax.bar(x, compl, label="completed", color="#2ca02c")
      ax.bar(x, dead, bottom=compl, label="deadlock", color="#d62728")
      ax.bar(x, inf, bottom=[c + d for c, d in zip(compl, dead)], label="infeasible", color="#9467bd")
      ax.bar(x, to, bottom=[c + d + i for c, d, i in zip(compl, dead, inf)], label="timeout", color="#ff7f0e")
      ax.set_title(scn)
      ax.set_xticks(list(x))
      ax.set_xticklabels(arms, fontsize=8)
      ax.grid(axis="y", alpha=0.3)
   axes[0].set_ylabel("runs")
   axes[-1].legend(fontsize=7)
   fig.suptitle("E1: outcomes per arm (A1–A8)")
   fig.tight_layout()
   out = _FIG_DIR / f"e1{tag}_outcomes.png"
   fig.savefig(out, dpi=150)
   fig.savefig(out.with_suffix(".pdf"))
   plt.close(fig)
   return out


def main() -> None:
   p = argparse.ArgumentParser(description="Analyze E1 factorial results")
   p.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
   p.add_argument("--tag", default="", help="suffix for figure filenames, e.g. '_corners'")
   args = p.parse_args()
   if not args.csv.exists():
      raise SystemExit(f"no CSV at {args.csv} — run run_e1_factorial first")

   rows = _load(args.csv)
   agg = _agg(rows)
   scenarios = sorted({k[0] for k in agg})
   _print_summary(agg)
   _matched_pairs(agg)
   f1 = _fig_energy(agg, scenarios, args.tag)
   f2 = _fig_outcomes(agg, scenarios, args.tag)
   print(f"\nfigures: {f1}\n         {f2}")


if __name__ == "__main__":
   main()
