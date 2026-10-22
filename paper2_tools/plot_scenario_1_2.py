"""Scenario 1.2 – Radius-Effekt auf Deadlockrate.

Generates two diagrams from paper2_results/results/metrics.csv:
1. Deadlock rate (bar chart) per number of drones, adaptive vs fixed.
2. Arrival time (box-whisker plot) per number of drones, adaptive vs fixed.
3. Mean step time (box-whisker plot) per number of drones, adaptive vs fixed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ── IEEE two-column paper style ──────────────────────────────────────
IEEE_COL_WIDTH = 3.5        # single column width in inches
IEEE_FULL_WIDTH = 7.16      # full text width in inches
_BASE_FONT_SIZES = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
}

def set_font_scale(scale: float = 1.0) -> None:
   """Apply font-size scaling factor on top of IEEE base sizes."""
   matplotlib.rcParams.update({k: v * scale for k, v in _BASE_FONT_SIZES.items()})

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",        # Times-compatible math
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
set_font_scale(1.0)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.utility.constants import COLOR_BY_DRONE_INDEX

# METRICS_CSV = Path(__file__).resolve().parent / "paper2_results" / "result_8_3" / "metrics.csv"
# OUTPUT_DIR = Path(__file__).resolve().parent / "paper2_results" / "plots"

N_DRONES_FIELD = "num_drones"
SPHERE_TYPE_FIELD = "controller_type"
STATUS_FIELD = "status"
FINISHED_STATUS = "finished"
COORDINATOR_TYPE_FIELD = "coordinator"

CSV_FIELDS = {"mean_step_time": "mean_step_time_s", "arrival": "wall_time_s", "steps": "steps"}

COORDINATOR_TYPES = ["mpc_central", "dmpc_admm"]
SPHERE_TYPES = ["mpc_agent_adaptive", "mpc_agent"]

LABELS = {
      SPHERE_TYPES[0]: "$r_{a}(v)$", SPHERE_TYPES[1]: "$r_{s}$",
      COORDINATOR_TYPES[0]: "MPC", COORDINATOR_TYPES[1]: "DMPC",
      CSV_FIELDS["mean_step_time"]: "Mean Step Time [s]", "mean_step_time_without": "Mean Step Time",
      CSV_FIELDS["arrival"]: "Mean Arrival Time [s]", "arrival_without": "Mean Arrival Time",
      CSV_FIELDS["steps"]: "Mean Steps"
}


def load_data(path: Path) -> pd.DataFrame:
   df = pd.read_csv(path)
   # Strip whitespace from column names (Windows line-endings in CSV)
   df.columns = df.columns.str.strip()
   return df


SINGLE_COLOR_INDEX = {
   COORDINATOR_TYPES[0]: 10,
   COORDINATOR_TYPES[1]: 20,
   SPHERE_TYPES[0]: 11,
   SPHERE_TYPES[1]: 32,
}
COMBO_COLOR_INDEX = {
   (COORDINATOR_TYPES[0], SPHERE_TYPES[0]): 2,
   (COORDINATOR_TYPES[0], SPHERE_TYPES[1]): 7,
   (COORDINATOR_TYPES[1], SPHERE_TYPES[0]): 8,
   (COORDINATOR_TYPES[1], SPHERE_TYPES[1]): 6,
}


def _build_combos(coordinator_list: list[str], controller_list: list[str]) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str], dict[tuple[str, str], str]]:
   """Build coordinator/controller combos with their labels and colors."""
   combos: list[tuple[str, str]] = []
   for coord in coordinator_list:
      for ctrl in controller_list:
         combos.append((coord, ctrl))

   combo_labels = {(co, ct): f"{LABELS.get(co, co)} / {LABELS.get(ct, ct)}" for co, ct in combos}
   
   # Determine color: use combo color if both vary, otherwise use single type color
   combo_colors = {}
   for co, ct in combos:
      if (co, ct) in COMBO_COLOR_INDEX and len(coordinator_list) > 1 and len(controller_list) > 1:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[COMBO_COLOR_INDEX[(co, ct)]]
      elif len(controller_list) > 1:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[SINGLE_COLOR_INDEX.get(ct, 1)]
      else:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[SINGLE_COLOR_INDEX.get(co, 1)]
   
   return combos, combo_labels, combo_colors


def _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df: pd.DataFrame, output_dir: Path, coordinator_list: list[str], controller_list: list[str],
                                                                                          column: str, ylabel: str, title: str, filename_base: str, figsize: tuple[float, float] = (IEEE_COL_WIDTH, 2.4),
                                                                                          n_drones_range: range | None = None, log_scale: bool = False, show_title: bool = True) -> None:
   """Box-whisker plot for every coordinator+controller combo, filtered by n_drones_range."""
   finished = df[df[STATUS_FIELD] == FINISHED_STATUS].copy()
   n_drones_range = n_drones_range if n_drones_range is not None else sorted(df[N_DRONES_FIELD].unique())
   drone_counts = sorted(n for n in df[N_DRONES_FIELD].unique() if n in n_drones_range)

   combos, combo_labels, combo_colors = _build_combos(coordinator_list, controller_list)

   data: dict[tuple[str, str], list[list[float]]] = {combo: [] for combo in combos}
   for n in drone_counts:
      for combo in combos:
         coord, ctrl = combo
         subset = finished[(finished[N_DRONES_FIELD] == n) & (finished[COORDINATOR_TYPE_FIELD] == coord) & (finished[SPHERE_TYPE_FIELD] == ctrl)]
         data[combo].append(subset[column].tolist())

   fig, ax = plt.subplots(figsize=figsize)
   num_combos = len(combos)
   width = 0.8 / max(num_combos, 1)
   gap = 0.8
   has_empty = False

   for idx, n in enumerate(drone_counts):
      center = idx * (1 + gap)
      for j, combo in enumerate(combos):
         pos = center + (j - (num_combos - 1) / 2) * width
         vals = data[combo][idx]
         if not vals:
            ax.text(pos, 0.5, r"$\emptyset$", ha="center", va="center", fontsize=10, color=combo_colors[combo], alpha=0.8, transform=ax.get_xaxis_transform(),
                    clip_on=False)
            has_empty = True
            continue
         ax.boxplot(vals, positions=[pos], widths=width * 0.85, patch_artist=True, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=3), medianprops=dict(color="black", linewidth=1.2),
                    boxprops=dict(facecolor=combo_colors[combo], alpha=0.75), whiskerprops=dict(color="gray"), capprops=dict(color="gray"),
                    flierprops=dict(marker="o", markersize=2, alpha=0.5))

   centers = [idx * (1 + gap) for idx in range(len(drone_counts))]
   ax.set_xticks(centers)
   ax.set_xticklabels(drone_counts)
   ax.set_xlabel("Number of Drones $N$")
   ax.set_ylabel(ylabel)
   if show_title:
      ax.set_title(title)

   if log_scale:
      ax.set_yscale("log")

   legend_elements = [Patch(facecolor=combo_colors[c], alpha=0.75, label=combo_labels[c]) for c in combos]
   ax.legend(handles=legend_elements)
   ax.grid(axis="y", alpha=0.3)

   if has_empty:
      x_min, x_max = ax.get_xlim()
      ax.set_xlim(x_min, x_max + 0.5)

   fig.tight_layout()
   out_png = output_dir / f"{filename_base}.png"
   out_eps = output_dir / f"{filename_base}.eps"
   fig.savefig(out_png, dpi=300)
   fig.savefig(out_eps, format="eps")
   print(f"Saved {out_png}  (+eps)")
   plt.close(fig)


def _deadlock_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df: pd.DataFrame, output_dir: Path,  coordinator_list: list[str], controller_list: list[str],
                                                                                           n_drones_range: list[int], title: str, filename_base: str, figsize: tuple[float, float] = (IEEE_COL_WIDTH, 3.0), show_title: bool = True) -> None:
   """Bar chart: deadlock rate (%) for every coordinator+controller combo, filtered by n_drones_range."""
   drone_counts = sorted(n for n in df[N_DRONES_FIELD].unique() if n in n_drones_range)

   combos, combo_labels, combo_colors = _build_combos(coordinator_list, controller_list)

   rates: dict[tuple[str, str], list[float]] = {combo: [] for combo in combos}
   for n in drone_counts:
      for combo in combos:
         coord, ctrl = combo
         subset = df[(df[N_DRONES_FIELD] == n) & (df[COORDINATOR_TYPE_FIELD] == coord) & (df[SPHERE_TYPE_FIELD] == ctrl)]
         total = len(subset)
         deadlocks = (subset[STATUS_FIELD] != FINISHED_STATUS).sum()
         rate = (deadlocks / total * 100) if total > 0 else 0.0
         rates[combo].append(rate)

   x = np.arange(len(drone_counts))
   num_combos = len(combos)
   width = 0.8 / max(num_combos, 1)

   fig, ax = plt.subplots(figsize=figsize)
   for i, combo in enumerate(combos):
      offset = (i - (num_combos - 1) / 2) * width
      ax.bar(x + offset, rates[combo], width, label=combo_labels[combo], color=combo_colors[combo], edgecolor="white")

   ax.set_xlabel("Number of Drones $N$")
   ax.set_ylabel("Deadlock Rate [\\%]")
   if show_title:
      ax.set_title(title)
   ax.set_xticks(x)
   ax.set_xticklabels(drone_counts)
   ax.set_ylim(0, 110)
   ax.legend()
   ax.grid(axis="y", alpha=0.3)

   fig.tight_layout()
   out_png = output_dir / f"{filename_base}.png"
   out_eps = output_dir / f"{filename_base}.eps"
   fig.savefig(out_png, dpi=300)
   fig.savefig(out_eps, format="eps")
   print(f"Saved {out_png}  (+eps)")
   plt.close(fig)


def plot_coord_and_sphere_combination(df: pd.DataFrame, output_dir: Path, field: str, n_drones_range: range | list[int] | None = None, show_title: bool = True):
   # Mean step time:
   for coordinator in COORDINATOR_TYPES:
      for log_scale in [True, False]:
         _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df=df, output_dir=output_dir, filename_base=f"{CSV_FIELDS[field]}_{coordinator}_{('log' if log_scale else '')}_{n_drones_range}",
                                                                                               title=f"{LABELS[f"{field}_without"]}: {LABELS[SPHERE_TYPES[0]]} vs. {LABELS[SPHERE_TYPES[1]]} Spheres ({LABELS[coordinator]})",
                                                                                               ylabel=LABELS[CSV_FIELDS[field]], column=CSV_FIELDS[field], coordinator_list=[coordinator], controller_list=SPHERE_TYPES,
                                                                                               n_drones_range=n_drones_range, figsize=(IEEE_COL_WIDTH, 2.4), log_scale=log_scale, show_title=show_title)
   for sphere in SPHERE_TYPES:
      for log_scale in [True, False]:
         _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df=df, output_dir=output_dir, filename_base=f"{CSV_FIELDS[field]}_{sphere}_{('log' if log_scale else '')}_{n_drones_range}",
                                                                                               title=f"{LABELS[f"{field}_without"]}: {LABELS[COORDINATOR_TYPES[0]]} vs. {LABELS[COORDINATOR_TYPES[1]]} ({LABELS[sphere]})",
                                                                                               ylabel=LABELS[CSV_FIELDS[field]], column=CSV_FIELDS[field], coordinator_list=COORDINATOR_TYPES, controller_list=[sphere],
                                                                                               n_drones_range=n_drones_range, figsize=(IEEE_COL_WIDTH, 2.4), log_scale=log_scale, show_title=show_title)


def plot_all_four_comparison(df: pd.DataFrame, output_dir: Path, column: str, ylabel: str, title_prefix: str, filename_base: str, n_drones_range: range | list[int] | None = None, figsize: tuple[float, float] = (IEEE_FULL_WIDTH, 2.8), show_title: bool = True) -> None:
   """Boxplot with all 4 combos (2 coordinators x 2 controllers) side by side, for a given column."""
   for log_scale in [False, True]:
      _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df=df, output_dir=output_dir, coordinator_list=COORDINATOR_TYPES,
            controller_list=SPHERE_TYPES, n_drones_range=n_drones_range, column=column, ylabel=ylabel, title=f"{title_prefix}: All Coordinators & Controllers",
            filename_base=f"{filename_base}_all4{('_log' if log_scale else '')}", figsize=figsize, log_scale=log_scale, show_title=show_title)


SCALING_COLUMNS_ALL = [("mean_step_time_s", "Mean Step Time [s]"),
                       ("wall_time_s", "Wall Time [s]")]


def plot_scaling_curves(df: pd.DataFrame, output_dir: Path,
                        coordinator: str, controller: str,
                        n_drones_range: range | list[int] | None = None,
                        figsize: tuple[float, float] | None = None,
                        log_scale: bool = False, show_title: bool = True,
                        columns: list[tuple[str, str]] | None = None) -> None:
   """Line plot showing how mean_step_time and/or wall_time scale with N.

   Shows mean ± std as shaded band for finished runs only.
   *columns* selects which metrics to plot (default: both).
   Single-column → single plot; two columns → side-by-side subplots.
   """
   finished = df[(df[STATUS_FIELD] == FINISHED_STATUS) &
                 (df[COORDINATOR_TYPE_FIELD] == coordinator) &
                 (df[SPHERE_TYPE_FIELD] == controller)].copy()
   n_range = n_drones_range if n_drones_range is not None else sorted(df[N_DRONES_FIELD].unique())
   drone_counts = sorted(n for n in finished[N_DRONES_FIELD].unique() if n in n_range)

   if not drone_counts:
      return

   if columns is None:
      columns = SCALING_COLUMNS_ALL
   ncols = len(columns)
   if figsize is None:
      figsize = (IEEE_FULL_WIDTH, 2.8) if ncols > 1 else (IEEE_COL_WIDTH, 2.8)
   color = COLOR_BY_DRONE_INDEX[COMBO_COLOR_INDEX.get(
      (coordinator, controller), SINGLE_COLOR_INDEX.get(coordinator, 1))]

   fig, axes = plt.subplots(1, ncols, figsize=figsize)
   if ncols == 1:
      axes = [axes]
   for ax, (col, ylabel) in zip(axes, columns):
      means, stds, counts = [], [], []
      for n in drone_counts:
         vals = finished[finished[N_DRONES_FIELD] == n][col]
         means.append(vals.mean())
         stds.append(vals.std())
         counts.append(len(vals))

      means_arr = np.array(means)
      stds_arr = np.array(stds)

      ax.plot(drone_counts, means_arr, "o-", color=color, linewidth=1.2, markersize=3)
      lower = means_arr - stds_arr
      if log_scale:
         lower = np.maximum(lower, means_arr * 0.05)  # clamp to 5% of mean for log scale
      else:
         lower = np.maximum(lower, 0)
      ax.fill_between(drone_counts, lower, means_arr + stds_arr,
                       color=color, alpha=0.2)

      # Annotate sample counts
      for x, y, c in zip(drone_counts, means_arr, counts):
         # f"n={c}"
         ax.annotate("", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=6, color="gray")

      ax.set_xlabel("Number of Drones $N$")
      ax.set_ylabel(ylabel)
      if log_scale:
         ax.set_yscale("log")
      ax.grid(alpha=0.3)

   coord_label = LABELS.get(coordinator, coordinator)
   ctrl_label = LABELS.get(controller, controller)
   if show_title:
      fig.suptitle(f"Scaling: {coord_label} / {ctrl_label}")
   fig.tight_layout()

   suffix = "_log" if log_scale else ""
   col_suffix = "_" + "_".join(c[0] for c in columns) if len(columns) != len(SCALING_COLUMNS_ALL) else ""
   base = f"scaling_{coordinator}_{controller}_{n_range}{col_suffix}{suffix}"
   for ext, kwargs in [("png", dict(dpi=300)), ("eps", dict(format="eps"))]:
      fig.savefig(output_dir / f"{base}.{ext}", **kwargs)
   print(f"Saved {output_dir / base}.png  (+eps)")
   plt.close(fig)


def plot_all_four_deadlock(df: pd.DataFrame, output_dir: Path, n_drones_range: range | list[int] | None = None, figsize: tuple[float, float] = (IEEE_FULL_WIDTH, 2.8), show_title: bool = True) -> None:
   """Deadlock bar chart with all 4 combos (2 coordinators x 2 controllers) side by side."""
   n_range = n_drones_range if n_drones_range is not None else sorted(df[N_DRONES_FIELD].unique())
   _deadlock_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df=df, output_dir=output_dir,
         coordinator_list=COORDINATOR_TYPES, controller_list=SPHERE_TYPES, n_drones_range=n_range, title="Deadlock Rate: All Coordinators & Controllers",
         filename_base="deadlock_rate_all4", figsize=figsize, show_title=show_title)


def main(metrics_csv: Path | None = None, show_title: bool = True, font_scale: float = 1.0,
         metrics_path: Path | None = None, output_dir: Path | None = None, n_max: int = 24) -> None:
   """Generate all figures.

   *metrics_csv* selects a result folder below ``paper2_results`` (legacy).
   *metrics_path* / *output_dir* give explicit paths instead (e.g. review_work).
   *n_max* is the largest N in the full-range plots (legacy default 24).
   """
   set_font_scale(font_scale)
   if metrics_path is not None:
      METRICS_CSV = Path(metrics_path).resolve()
      OUTPUT_DIR = Path(output_dir).resolve() if output_dir is not None else METRICS_CSV.parent / "plots"
   else:
      METRICS_CSV = Path(__file__).resolve().parent / "paper2_results" / "results_hetzner" / "metrics.csv" if metrics_csv is None else Path(
            __file__).resolve().parent / "paper2_results" / metrics_csv / "metrics.csv"
      OUTPUT_DIR = Path(__file__).resolve().parent / "paper2_results" / "results_hetzner" / "plots" if metrics_csv is None else Path(
            __file__).resolve().parent / "paper2_results" / metrics_csv / "plots"

   OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

   df = load_data(METRICS_CSV)
   print(f"Loaded {len(df)} rows from {METRICS_CSV}")
   print(f"  num_drones: {sorted(df['num_drones'].unique())}")
   print(f"  sphere_type: {sorted(df['controller_type'].unique())}")
   print(f"  status counts:\n{df['status'].value_counts().to_string()}")
   print()

   n_range = range(2, 12)
   n_range_all = range(2, n_max + 1)

   plot_coord_and_sphere_combination(df, OUTPUT_DIR, n_drones_range=n_range, field="mean_step_time", show_title=show_title)
   plot_coord_and_sphere_combination(df, OUTPUT_DIR, n_drones_range=n_range, field="arrival", show_title=show_title)
   plot_coord_and_sphere_combination(df, OUTPUT_DIR, n_drones_range=n_range_all, field="mean_step_time", show_title=show_title)
   plot_coord_and_sphere_combination(df, OUTPUT_DIR, n_drones_range=n_range_all, field="arrival", show_title=show_title)

   # All 4 combos (2 coordinators x 2 controllers) direct comparison
   plot_all_four_comparison(df, OUTPUT_DIR, column="mean_step_time_s", ylabel="Mean Step Time [s]", title_prefix="Mean Step Time", filename_base=f"mean_step_time{n_range}", n_drones_range=n_range, show_title=show_title, figsize=(IEEE_COL_WIDTH*2.5, 4.8))
   plot_all_four_comparison(df, OUTPUT_DIR, column="wall_time_s", ylabel="Wall Time [s]", title_prefix="Wall Time", filename_base=f"wall_time{n_range}", n_drones_range=n_range, show_title=show_title)
   plot_all_four_comparison(df, OUTPUT_DIR, column="steps", ylabel="Arrival Time [steps]", title_prefix="Arrival Time", filename_base=f"arrival_time{n_range}", n_drones_range=n_range, show_title=show_title)
   plot_all_four_comparison(df, OUTPUT_DIR, column="mean_step_time_s", ylabel="Mean Step Time [s]", title_prefix="Mean Step Time", filename_base=f"mean_step_time{n_range_all}", n_drones_range=n_range_all, show_title=show_title, figsize=(IEEE_COL_WIDTH*2.2, 3.0))
   plot_all_four_comparison(df, OUTPUT_DIR, column="wall_time_s", ylabel="Wall Time [s]", title_prefix="Wall Time", filename_base=f"wall_time{n_range_all}", n_drones_range=n_range_all, show_title=show_title)
   plot_all_four_comparison(df, OUTPUT_DIR, column="steps", ylabel="Arrival Time [steps]", title_prefix="Arrival Time", filename_base=f"arrival_time{n_range_all}", n_drones_range=n_range_all, show_title=show_title)

   plot_all_four_deadlock(df, OUTPUT_DIR, n_drones_range=n_range, show_title=show_title, figsize=(IEEE_COL_WIDTH*2.2, 3.0))
   plot_all_four_deadlock(df, OUTPUT_DIR, n_drones_range=n_range_all, show_title=show_title, figsize=(IEEE_COL_WIDTH*2.2, 3.0))

   # Scaling curves: mean_step_time & wall_time vs N
   for coord, ctrl in [("mpc_central", "mpc_agent_adaptive"), ("dmpc_admm", "mpc_agent_adaptive")]:
      for log in [False, True]:
         plot_scaling_curves(df, OUTPUT_DIR, columns=[("wall_time_s", "Wall Time [s]")], coordinator=coord, controller=ctrl, n_drones_range=n_range,
                             log_scale=log, figsize=(IEEE_COL_WIDTH*1.5, 3.5), show_title=show_title)
         plot_scaling_curves(df, OUTPUT_DIR, columns=[("wall_time_s", "Wall Time [s]")], coordinator=coord, controller=ctrl, n_drones_range=n_range_all,
                             log_scale=log, figsize=(IEEE_COL_WIDTH*1.5, 3.5), show_title=show_title)
         plot_scaling_curves(df, OUTPUT_DIR, columns=[("mean_step_time_s", "Mean Step Time [s]")], coordinator=coord, controller=ctrl, n_drones_range=n_range,
                             log_scale=log, figsize=(IEEE_COL_WIDTH*1.5, 3.5), show_title=show_title)
         plot_scaling_curves(df, OUTPUT_DIR, columns=[("mean_step_time_s", "Mean Step Time [s]")], coordinator=coord, controller=ctrl, n_drones_range=n_range_all,
                             log_scale=log, figsize=(IEEE_COL_WIDTH*1.5, 3.5), show_title=show_title)


   print("\nDone.")


if __name__ == "__main__":
   import argparse

   parser = argparse.ArgumentParser(description="Scenario 1.2 figures (deadlock rate, arrival time, step time, scaling)")
   parser.add_argument("--metrics", type=Path, default=None, help="explicit metrics.csv (default: paper2_results/results_hetzner/metrics.csv)")
   parser.add_argument("--out", type=Path, default=None, help="output directory (default: <metrics dir>/plots)")
   parser.add_argument("--n-max", type=int, default=24, help="largest N in the full-range plots (default 24)")
   parser.add_argument("--font-scale", type=float, default=2.5)
   parser.add_argument("--show-title", action="store_true")
   args = parser.parse_args()
   if args.metrics is not None:
      main(metrics_path=args.metrics, output_dir=args.out, show_title=args.show_title, font_scale=args.font_scale, n_max=args.n_max)
   else:
      main(metrics_csv="results_hetzner", show_title=args.show_title, font_scale=args.font_scale, n_max=args.n_max)
