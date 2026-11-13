"""Figures of the constant-density scaling experiment (todo.md, block B).

Reads ``results/metrics.csv`` next to this file (export with ``export_csv --experiment density``) and writes to ``results/plots``:

- ``time_per_drone``: CPU seconds one drone spends per step, over N, log scale. Central MPC: step CPU / N. DMPC: mean local
  solve * mean rounds per step, one line per k.
- ``fleet_step_time``: what a fleet with one computer per drone waits per step. Central: step CPU. DMPC: sum over the rounds of
  the slowest local solve (parallel_step_cpu_s / steps).
- ``rounds_per_step``: mean and largest Gauss-Seidel rounds per step of the DMPC cells.
- ``outcome_rates``: finished / deadlock / other share per cell.
- ``scaling_density``: time per drone and rounds per step side by side (paper figure).
- ``scaling_table.csv``: the complete result, one row per cell (series, N): outcomes, run time, step time, per-drone quantities.
- ``scaling_factors.csv``: how each time grows with N at constant density: factor against the smallest N of the series, and the
  exponent b of a log-log fit time ~ N^b with the factor 2^b per doubling of the fleet.

CPU seconds come from time.process_time of the worker process, so the overbooked Hetzner wall clock does not enter; ``--wall``
switches the two time figures to the wall-clock step time instead. By default every run with at least one step counts, because
the per-step cost does not depend on how the run ended; ``--finished-only`` restricts to finished runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
for entry in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
   if entry not in sys.path:
      sys.path.insert(0, entry)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paper2_tools.plot_scenario_1_2 import IEEE_COL_WIDTH, IEEE_FULL_WIDTH, set_font_scale  # noqa: E402
from tools.utility.constants import COLOR_BY_DRONE_INDEX  # noqa: E402

METRICS_CSV = _HERE / "results" / "metrics.csv"
PLOTS_DIR = _HERE / "results" / "plots"

SERIES_ORDER = ["MPC", "DMPC k=1", "DMPC k=3"]
SERIES_COLORS = {"MPC": COLOR_BY_DRONE_INDEX[2], "DMPC k=1": COLOR_BY_DRONE_INDEX[20], "DMPC k=3": COLOR_BY_DRONE_INDEX[8]}
SERIES_MARKERS = {"MPC": "o", "DMPC k=1": "s", "DMPC k=3": "^"}
_FALLBACK_COLORS = [COLOR_BY_DRONE_INDEX[i] for i in (6, 7, 11, 32)]

ROUNDS_LABEL = "Gauss\u2013Seidel rounds per step"

NO_STEP_STATUSES = ("no_possible_placement", "error")

# column of cell_table -> what it is, for the scaling factors. run_time_s only counts finished runs: a timeout or a deadlock says
# nothing about how long the run would have taken. The per-step times take every run with steps, see select_runs.
SCALING_QUANTITIES = {"run_time_cpu_s": "CPU time of a finished run", "run_time_wall_s": "wall time of a finished run", "step_time_s": "CPU time per step",
                      "time_per_drone_s": "CPU time per drone and step", "fleet_step_s": "step time, fleet view", "local_solve_s": "CPU time per local solve"}


def load(path: Path = METRICS_CSV) -> pd.DataFrame:
   df = pd.read_csv(path)
   df.columns = df.columns.str.strip()
   return df


def series_name(row: pd.Series) -> str:
   if row["coordinator"] == "mpc_central":
      return "MPC"
   k = row.get("max_admm_iter")
   if pd.isna(k):
      return "DMPC"
   return f"DMPC k={int(k)}"


def series_label(series: str) -> str:
   """Name of a series in the figures, in the words of the paper. The CSV tables keep the short series name."""
   if not series.startswith("DMPC k="):
      return series
   rounds = int(series.removeprefix("DMPC k="))
   return f"DMPC, {rounds} Gauss\u2013Seidel round{'' if rounds == 1 else 's'}"


def add_derived(df: pd.DataFrame, wall: bool = False) -> pd.DataFrame:
   """Per-run quantities the figures show. Rows without steps get NaN."""
   out = df.copy()
   out["series"] = out.apply(series_name, axis=1)
   steps = out["steps"].astype(float).where(out["steps"] > 0)
   n = out["num_drones"].astype(float)
   step_time = out["mean_step_time_s"] if wall else out["cpu_total_s"] / steps
   out["step_time_s"] = step_time
   out["rounds_per_step"] = out["admm_iterations"] / steps
   local = out["local_solves"].astype(float).where(out["local_solves"] > 0)
   out["local_solve_s"] = out["local_solve_cpu_s"] / local
   is_dmpc = out["coordinator"] == "dmpc_admm"
   # one drone per step: its local solve times the rounds; central: the whole step shared by all drones
   out["time_per_drone_s"] = np.where(is_dmpc, out["local_solve_s"] * out["rounds_per_step"], step_time / n)
   # one step as a fleet sees it: slowest drone per round, summed; central: the whole step
   out["fleet_step_s"] = np.where(is_dmpc, out["parallel_step_cpu_s"] / steps, step_time)
   if wall:
      # wall clock has no per-solve split: scale the CPU split by wall / cpu per step
      cpu_step = out["cpu_total_s"] / steps
      ratio = (out["mean_step_time_s"] / cpu_step).replace([np.inf, -np.inf], np.nan)
      out.loc[is_dmpc, "time_per_drone_s"] = (out["local_solve_s"] * out["rounds_per_step"] * ratio)[is_dmpc]
      out.loc[is_dmpc, "fleet_step_s"] = (out["parallel_step_cpu_s"] / steps * ratio)[is_dmpc]
   return out


def select_runs(df: pd.DataFrame, finished_only: bool = False) -> pd.DataFrame:
   if finished_only:
      return df[df["status"] == "finished"]
   return df[(~df["status"].isin(NO_STEP_STATUSES)) & (df["steps"] > 0)]


def aggregate(df: pd.DataFrame, column: str) -> pd.DataFrame:
   """mean, std and count of ``column`` per (series, N), NaN-free rows only."""
   rows = df.dropna(subset=[column])
   agg = rows.groupby(["series", "num_drones"])[column].agg(["mean", "std", "count"]).reset_index()
   agg["std"] = agg["std"].fillna(0.0)
   return agg


def outcome_rates(df: pd.DataFrame) -> pd.DataFrame:
   """Share of finished, deadlock and other outcomes per (series, N), in percent."""
   grouped = df.groupby(["series", "num_drones"])["status"]
   total = grouped.count()
   finished = grouped.apply(lambda s: (s == "finished").sum())
   deadlock = grouped.apply(lambda s: (s == "deadlock").sum())
   out = pd.DataFrame({"total": total, "finished": finished / total * 100, "deadlock": deadlock / total * 100})
   out["other"] = 100 - out["finished"] - out["deadlock"]
   return out.reset_index()


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
   """The complete result, one row per (series, N). ``df`` holds every run of add_derived, whatever its status."""
   rows = []
   for (series, n), cell in df.groupby(["series", "num_drones"]):
      with_steps = select_runs(cell)
      finished = cell[cell["status"] == "finished"]
      status = cell["status"]
      rows.append({"series": series, "num_drones": n, "room_size": cell["room_size"].mean(), "runs": len(cell), "finished": int((status == "finished").sum()),
                   "deadlock": int((status == "deadlock").sum()), "timeout": int((status == "timeout").sum()),
                   "other": int((~status.isin(["finished", "deadlock", "timeout"])).sum()),
                   "steps_finished": finished["steps"].mean(), "run_time_cpu_s": finished["cpu_total_s"].mean(), "run_time_wall_s": finished["wall_time_s"].mean(),
                   **{column: with_steps[column].mean() for column in ("step_time_s", "time_per_drone_s", "fleet_step_s", "local_solve_s", "rounds_per_step")}})
   table = pd.DataFrame(rows)
   order = {name: i for i, name in enumerate(_series_in(df))}
   return table.sort_values(["series", "num_drones"], key=lambda c: c.map(order) if c.name == "series" else c).reset_index(drop=True)


def scaling_factors(table: pd.DataFrame) -> pd.DataFrame:
   """Per series and quantity: exponent b of time ~ N^b, factor 2^b per doubling, and time(N) / time(smallest N) as columns x<N>."""
   rows = []
   for series, cells in table.groupby("series", sort=False):
      for column, label in SCALING_QUANTITIES.items():
         points = cells.dropna(subset=[column])
         points = points[points[column] > 0]
         if len(points) < 2:
            continue
         n = points["num_drones"].to_numpy(dtype=float)
         time = points[column].to_numpy(dtype=float)
         exponent = float(np.polyfit(np.log(n), np.log(time), 1)[0])
         rows.append({"series": series, "quantity": label, "exponent": exponent, "per_doubling": 2.0 ** exponent, "cells": len(points),
                      **{f"x{int(ni)}": ti / time[0] for ni, ti in zip(n, time)}})
   out = pd.DataFrame(rows)
   factor_columns = sorted((c for c in out.columns if c.startswith("x")), key=lambda c: int(c[1:]))
   return out[[c for c in out.columns if not c.startswith("x")] + factor_columns]


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
   """Write and print scaling_table.csv and scaling_factors.csv. ``df`` holds every run of add_derived."""
   out_dir.mkdir(parents=True, exist_ok=True)
   table = cell_table(df)
   factors = scaling_factors(table)
   table.to_csv(out_dir / "scaling_table.csv", index=False)
   factors.to_csv(out_dir / "scaling_factors.csv", index=False)
   print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
   print(factors.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
   print(f"Saved {out_dir / 'scaling_table.csv'} and scaling_factors.csv")


def mirrored_rows(df: pd.DataFrame) -> int:
   """Rows from runs with targets mirrored at the room centre: an old campaign, or a worker that predates travel_distance."""
   return len(df) if "travel_distance" not in df else int(df["travel_distance"].isna().sum())


def _series_in(df: pd.DataFrame) -> list[str]:
   present = list(df["series"].unique())
   ordered = [s for s in SERIES_ORDER if s in present] + sorted(s for s in present if s not in SERIES_ORDER)
   return ordered


def _color(series: str, index: int) -> str:
   return SERIES_COLORS.get(series, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


def _draw_lines(ax, agg: pd.DataFrame, series_list: list[str], log_y: bool, annotate_counts: bool) -> None:
   for i, series in enumerate(series_list):
      part = agg[agg["series"] == series].sort_values("num_drones")
      if part.empty:
         continue
      x = part["num_drones"].to_numpy()
      mean = part["mean"].to_numpy()
      std = part["std"].to_numpy()
      color = _color(series, i)
      ax.plot(x, mean, marker=SERIES_MARKERS.get(series, "o"), linestyle="-", color=color, linewidth=1.2, markersize=3.5, label=series_label(series))
      lower = np.maximum(mean - std, mean * 0.05) if log_y else np.maximum(mean - std, 0.0)
      ax.fill_between(x, lower, mean + std, color=color, alpha=0.2)
      if annotate_counts:
         for xi, yi, c in zip(x, mean, part["count"]):
            ax.annotate(f"n={int(c)}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=5, color="gray")
   ax.set_xscale("log", base=2)
   ax.set_xticks(sorted(agg["num_drones"].unique()))
   ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
   ax.minorticks_off()
   if log_y:
      ax.set_yscale("log")
   ax.set_xlabel("Number of Drones $N$")
   ax.grid(alpha=0.3, which="major")


def _save(fig, out_dir: Path, name: str) -> None:
   out_dir.mkdir(parents=True, exist_ok=True)
   fig.savefig(out_dir / f"{name}.png", dpi=300)
   fig.savefig(out_dir / f"{name}.eps", format="eps")
   print(f"Saved {out_dir / name}.png  (+eps)")
   plt.close(fig)


def plot_line_metric(df: pd.DataFrame, column: str, ylabel: str, name: str, out_dir: Path, log_y: bool = True, figsize=(IEEE_COL_WIDTH * 1.5, 3.0),
                     annotate_counts: bool = True) -> None:
   agg = aggregate(df, column)
   if agg.empty:
      print(f"  no data for {column}, skipping {name}")
      return
   fig, ax = plt.subplots(figsize=figsize)
   _draw_lines(ax, agg, _series_in(df), log_y, annotate_counts)
   ax.set_ylabel(ylabel)
   ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)  # above the axes, the curves fill the plot
   fig.tight_layout()
   _save(fig, out_dir, name)


def plot_rounds(df: pd.DataFrame, out_dir: Path, figsize=(IEEE_COL_WIDTH * 1.5, 3.0)) -> None:
   dmpc = df[df["coordinator"] == "dmpc_admm"]
   if dmpc.empty:
      print("  no DMPC rows, skipping rounds_per_step")
      return
   mean_rounds = aggregate(dmpc, "rounds_per_step")
   max_rounds = aggregate(dmpc, "max_admm_iter_step")
   fig, ax = plt.subplots(figsize=figsize)
   series_list = _series_in(dmpc)
   _draw_lines(ax, mean_rounds, series_list, log_y=False, annotate_counts=False)
   for i, series in enumerate(series_list):
      part = max_rounds[max_rounds["series"] == series].sort_values("num_drones")
      ax.plot(part["num_drones"], part["mean"], linestyle="--", marker="x", color=_color(series, i), linewidth=0.9, markersize=3, label=f"{series_label(series)}, max per step")
   ax.set_ylabel(ROUNDS_LABEL)
   ax.set_ylim(0, float(max_rounds["mean"].max()) * 1.45)  # headroom for the legend above the capped round counts
   ax.legend(loc="upper left", ncol=2)
   fig.tight_layout()
   _save(fig, out_dir, "rounds_per_step")


def plot_outcomes(df: pd.DataFrame, out_dir: Path, figsize=(IEEE_COL_WIDTH * 1.5, 3.0)) -> None:
   rates = outcome_rates(df)
   if rates.empty:
      return
   series_list = _series_in(df)
   drone_counts = sorted(rates["num_drones"].unique())
   x = np.arange(len(drone_counts))
   width = 0.8 / max(len(series_list), 1)
   fig, ax = plt.subplots(figsize=figsize)
   for i, series in enumerate(series_list):
      part = rates[rates["series"] == series].set_index("num_drones").reindex(drone_counts)
      offset = (i - (len(series_list) - 1) / 2) * width
      color = _color(series, i)
      ax.bar(x + offset, part["finished"].fillna(0), width, color=color, edgecolor="white", label=f"{series_label(series)} finished")
      ax.bar(x + offset, part["deadlock"].fillna(0), width, bottom=part["finished"].fillna(0), color=color, alpha=0.45, edgecolor="white", hatch="//",
             label=f"{series_label(series)} deadlock")
   ax.set_xticks(x)
   ax.set_xticklabels(drone_counts)
   ax.set_xlabel("Number of Drones $N$")
   ax.set_ylabel("Share of runs [%]")
   ax.set_ylim(0, 105)
   ax.legend(fontsize=5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)  # below the axes, the bars fill the plot
   ax.grid(axis="y", alpha=0.3)
   fig.tight_layout()
   _save(fig, out_dir, "outcome_rates")


def plot_combined(df: pd.DataFrame, out_dir: Path, figsize=(IEEE_FULL_WIDTH, 2.8)) -> None:
   """Paper figure: time per drone (left, log) and Gauss-Seidel rounds per step (right)."""
   per_drone = aggregate(df, "time_per_drone_s")
   dmpc = df[df["coordinator"] == "dmpc_admm"]
   if per_drone.empty:
      return
   fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
   _draw_lines(ax_l, per_drone, _series_in(df), log_y=True, annotate_counts=False)
   ax_l.set_ylabel("CPU time per drone and step [s]")
   if not dmpc.empty:
      _draw_lines(ax_r, aggregate(dmpc, "rounds_per_step"), _series_in(dmpc), log_y=False, annotate_counts=False)
      ax_r.set_ylabel(ROUNDS_LABEL)
      ax_r.set_ylim(0, float(dmpc["max_admm_iter_step"].max()) * 1.15)
   # one legend for both panels, above them: every series of the right panel is also in the left one
   fig.legend(*ax_l.get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
   fig.tight_layout(rect=(0, 0, 1, 0.92))
   _save(fig, out_dir, "scaling_density")


def main(argv: list[str] | None = None) -> None:
   parser = argparse.ArgumentParser(description="Figures of the constant-density scaling experiment")
   parser.add_argument("--metrics", type=Path, default=METRICS_CSV)
   parser.add_argument("--out", type=Path, default=None, help="default: <metrics dir>/plots")
   parser.add_argument("--wall", action="store_true", help="wall-clock step times instead of CPU seconds")
   parser.add_argument("--finished-only", action="store_true", help="only runs that reached their targets")
   parser.add_argument("--font-scale", type=float, default=1.0)
   args = parser.parse_args(argv)

   set_font_scale(args.font_scale)
   out_dir = args.out if args.out is not None else args.metrics.parent / "plots"
   df = add_derived(load(args.metrics), wall=args.wall)
   runs = select_runs(df, finished_only=args.finished_only)
   unit = "wall" if args.wall else "CPU"
   print(f"Loaded {len(df)} rows, {len(runs)} with steps; series {sorted(runs['series'].unique())}; N {sorted(runs['num_drones'].unique())}")
   print(runs.groupby(["series", "num_drones"])["status"].value_counts().to_string())
   if mirrored_rows(df):
      print(f"WARNING: {mirrored_rows(df)} of {len(df)} rows have no travel_distance, so their targets were mirrored and the density did not stay constant")

   plot_line_metric(runs, "time_per_drone_s", f"{unit} time per drone and step [s]", "time_per_drone", out_dir)
   plot_line_metric(runs, "fleet_step_s", f"{unit} time per step, fleet view [s]", "fleet_step_time", out_dir)
   plot_line_metric(runs, "local_solve_s", f"{unit} time per local solve [s]", "local_solve_time", out_dir)
   plot_rounds(runs, out_dir)
   plot_outcomes(df[~df["status"].isin(NO_STEP_STATUSES)], out_dir)
   plot_combined(runs, out_dir)
   write_tables(df, out_dir)
   print("Done.")


if __name__ == "__main__":
   main()
