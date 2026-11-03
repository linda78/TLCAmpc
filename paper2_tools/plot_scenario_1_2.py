"""Block A figures for the revision of paper 2: deadlock rate, compute time per computing unit, scaling of Config. A and B.

Reads the Block-A export (default ``review_work/results_hetzner/volume/metrics.csv``, 28 Oct 2026) and writes four figures in
their final print size to ``paper2_results/block_a_terminal/plots/``. The file names follow the content, not the figure number:

   fig_deadlock_rate   deadlock rate over N for Config. A to D (submitted as Fig. 5)
   fig_time_per_unit   mean compute time per step and computing unit over N for Config. A to D (submitted as Fig. 6)
   fig_scaling_A       step time of Config. A over N, mean and one standard deviation (submitted as Fig. 7)
   fig_scaling_B       compute time per drone and step of Config. B over N (submitted as Fig. 8)

Configurations as in Table II of the paper: A ``mpc_central`` with ``mpc_agent_adaptive``, B ``dmpc_admm`` with
``mpc_agent_adaptive``, C ``mpc_central`` with ``mpc_agent``, D ``dmpc_admm`` with ``mpc_agent``. The time plots use finished and
deadlock runs; ``no_possible_placement`` is not a simulated run and is left out everywhere. The box plots of the first
submission are still available with ``--legacy``.

   .venv/bin/python paper2_tools/plot_scenario_1_2.py                          # the four paper figures
   .venv/bin/python paper2_tools/plot_scenario_1_2.py --metrics x.csv --out d   # another export
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from tools.utility.constants import COLOR_BY_DRONE_INDEX  # noqa: E402

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
    # TrueType (Type 42) instead of Type 3 fonts in EPS and PDF; IEEE PDF eXpress rejects Type 3. Also applies to
    # review_work/narrow_gap/narrow_gap.py, which imports set_font_scale from here.
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})
set_font_scale(1.0)

BLOCK_A_METRICS = _HERE / "review_work" / "results_hetzner" / "volume" / "metrics.csv"
BLOCK_A_OUTPUT = _HERE / "paper2_results" / "block_a_terminal" / "plots"

N_DRONES_FIELD = "num_drones"
SPHERE_TYPE_FIELD = "controller_type"
STATUS_FIELD = "status"
COORDINATOR_TYPE_FIELD = "coordinator"
FINISHED_STATUS = "finished"
DEADLOCK_STATUS = "deadlock"
NO_PLACEMENT_STATUS = "no_possible_placement"
VALID_STATUSES = (FINISHED_STATUS, DEADLOCK_STATUS)   # simulated runs: the time plots use both outcomes
UNIT_TIME_FIELD = "unit_time_s"                        # added by load_data, see there

CSV_FIELDS = {"mean_step_time": "mean_step_time_s", "arrival": "wall_time_s", "steps": "steps"}

COORDINATOR_TYPES = ["mpc_central", "dmpc_admm"]
SPHERE_TYPES = ["mpc_agent_adaptive", "mpc_agent"]

# configurations of Table II, (coordinator, controller); the order A, B, C, D is the legend order
CONFIG_A = ("mpc_central", "mpc_agent_adaptive")
CONFIG_B = ("dmpc_admm", "mpc_agent_adaptive")
CONFIG_C = ("mpc_central", "mpc_agent")
CONFIG_D = ("dmpc_admm", "mpc_agent")
CONFIG_ORDER = [CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D]
CONFIG_LABELS = {CONFIG_A: "Config. A", CONFIG_B: "Config. B", CONFIG_C: "Config. C", CONFIG_D: "Config. D"}
CONFIG_MARKERS = {CONFIG_A: "o", CONFIG_B: "s", CONFIG_C: "^", CONFIG_D: "D"}
CONFIG_OPEN_MARKER = {CONFIG_C}   # C lies on 0 % everywhere, open marker so that A underneath stays visible

LABELS = {   # legacy titles of the box plots
      SPHERE_TYPES[0]: "$r_{a}(v)$", SPHERE_TYPES[1]: "$r_{s}$",
      COORDINATOR_TYPES[0]: "MPC", COORDINATOR_TYPES[1]: "DMPC",
      CSV_FIELDS["mean_step_time"]: "step time in s", "mean_step_time_without": "Mean Step Time",
      CSV_FIELDS["arrival"]: "wall time in s", "arrival_without": "Mean Arrival Time",
      CSV_FIELDS["steps"]: "steps"
}

# axis titles in one style with narrow_gap.py ("$r$ in m", "$t$ in s")
XLABEL_N = "number of drones $N$"
YLABEL_DEADLOCK = "deadlock rate in %"
YLABEL_STEP_TIME = "step time in s"
YLABEL_UNIT_TIME = "compute time per step\nand computing unit in s"
YLABEL_UNIT_TIME_B = "compute time per drone and step in s"
REFERENCE_TIME_S = 1.0                  # ten sampling times, dt = 0.1 s
REFERENCE_TIME_LABEL = r"$10\,\Delta t$"
CUBIC_REFERENCE = (0.002048, 3.10, 8)   # a, b, n_min of the fit a N^b to Config. A (results section), guide line for N >= n_min
POWER_REFERENCE_OFFSET = 0.4            # the guide line is drawn below the data by this factor, on top of it the markers would hide it
BAND_ALPHA = 0.2                        # shade of the std band, blended with white and drawn opaque (EPS has no transparency)
GRID_COLOR = "0.82"                     # opaque light grey, an alpha grid comes out mid-grey in EPS


def load_data(path: Path) -> pd.DataFrame:
   df = pd.read_csv(path)
   # Strip whitespace from column names (Windows line-endings in CSV)
   df.columns = df.columns.str.strip()
   # Compute time per step and computing unit (decision of 1 Nov 2026). Config. A and C run on one computer, the step time is
   # the measure. Config. B and D run one computer per drone and nobody waits for the slowest, so the measure is the mean
   # local solve CPU time per drone and step from the CPU ledger. Two B rows (ids 1119 and 1121) have no ledger; their step
   # time divided by N agrees with the ledger of the neighbouring runs to 1 %. parallel_step_cpu_s is not drawn.
   per_drone = df["local_solve_cpu_s"] / (df[N_DRONES_FIELD] * df["steps"])
   fallback = df["mean_step_time_s"] / df[N_DRONES_FIELD]
   df[UNIT_TIME_FIELD] = np.where(df[COORDINATOR_TYPE_FIELD] == "dmpc_admm", per_drone.fillna(fallback), df["mean_step_time_s"])
   return df


SINGLE_COLOR_INDEX = {
   COORDINATOR_TYPES[0]: 10,
   COORDINATOR_TYPES[1]: 20,
   SPHERE_TYPES[0]: 11,
   SPHERE_TYPES[1]: 32,
}
COMBO_COLOR_INDEX = {
   CONFIG_A: 2,
   CONFIG_C: 7,
   CONFIG_B: 8,
   CONFIG_D: 6,
}


def _build_combos(coordinator_list: list[str], controller_list: list[str]) -> tuple[list[tuple[str, str]], dict[tuple[str, str], str], dict[tuple[str, str], str]]:
   """Coordinator/controller combos in the order A, B, C, D of Table II with their labels and colors."""
   combos = [c for c in CONFIG_ORDER if c[0] in coordinator_list and c[1] in controller_list]
   combo_labels = {c: CONFIG_LABELS[c] for c in combos}

   # Determine color: use combo color if both vary, otherwise use single type color
   combo_colors = {}
   for co, ct in combos:
      if len(coordinator_list) > 1 and len(controller_list) > 1:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[COMBO_COLOR_INDEX[(co, ct)]]
      elif len(controller_list) > 1:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[SINGLE_COLOR_INDEX.get(ct, 1)]
      else:
         combo_colors[(co, ct)] = COLOR_BY_DRONE_INDEX[SINGLE_COLOR_INDEX.get(co, 1)]

   return combos, combo_labels, combo_colors


def _marker_kwargs(combo: tuple[str, str], color: str, open_marker: bool = False, markersize: float = 3.2) -> dict:
   """Marker per configuration (A circle, B square, C triangle, D diamond); open markers show what lies underneath."""
   face = "none" if open_marker or combo in CONFIG_OPEN_MARKER else color
   return dict(marker=CONFIG_MARKERS.get(combo, "o"), markersize=markersize, markeredgewidth=0.8, markeredgecolor=color, markerfacecolor=face)


def _band_color(color: str, alpha: float = BAND_ALPHA) -> tuple[float, float, float]:
   """*color* blended with white as an opaque colour: what alpha blending on white paper gives, but EPS-safe."""
   return tuple(1.0 - alpha * (1.0 - c) for c in to_rgb(color))


def _select_runs(df: pd.DataFrame, n: int, combo: tuple[str, str]) -> pd.DataFrame:
   coord, ctrl = combo
   return df[(df[N_DRONES_FIELD] == n) & (df[COORDINATOR_TYPE_FIELD] == coord) & (df[SPHERE_TYPE_FIELD] == ctrl)]


def _format_n_axis(ax, drone_counts: list[int]) -> None:
   """Major ticks at the smallest N and every fifth drone, minor ticks at every N."""
   lo, hi = int(min(drone_counts)), int(max(drone_counts))
   ax.set_xlim(lo - 0.7, hi + 0.7)
   ax.set_xticks(sorted({lo, *[n for n in range(lo, hi + 1) if n % 5 == 0]}))
   ax.set_xticks(range(lo, hi + 1), minor=True)
   ax.set_xlabel(XLABEL_N)


def _reference_time_line(ax, x_left: float, reference_s: float) -> None:
   """Dashed line at *reference_s* with its label just above the left end."""
   ax.axhline(reference_s, ls="--", color="k", lw=0.8)
   ax.text(x_left, reference_s * 1.12, REFERENCE_TIME_LABEL, ha="left", va="bottom", fontsize=matplotlib.rcParams["legend.fontsize"])


def _save(fig, output_dir: Path, filename_base: str) -> None:
   out_png = output_dir / f"{filename_base}.png"
   fig.savefig(out_png, dpi=300)
   fig.savefig(output_dir / f"{filename_base}.eps", format="eps")
   print(f"Saved {out_png}  (+eps)")
   plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------------
# Paper figures
# ---------------------------------------------------------------------------------------------------------------------------

def plot_deadlock_rate(df: pd.DataFrame, output_dir: Path, coordinator_list: list[str], controller_list: list[str], n_drones_range: range | list[int],
                       filename_base: str = "fig_deadlock_rate", title: str = "Deadlock rate", figsize: tuple[float, float] = (IEEE_COL_WIDTH, 2.4),
                       show_title: bool = False) -> None:
   """Deadlock rate in % over N, one line with markers per configuration.

   Numerator: runs with status ``deadlock``. Denominator: the placed runs (status other than ``no_possible_placement``); a failed
   placement is no deadlock. A configuration without a placed run at some N ends there. The first N at which the static spheres
   could not be placed at all is marked with a dotted vertical line. There is only one failure mode: no run reached the step limit
   or the compute budget, so there is no timeout curve."""
   placed = df[df[STATUS_FIELD] != NO_PLACEMENT_STATUS]
   drone_counts = sorted(int(n) for n in df[N_DRONES_FIELD].unique() if n in n_drones_range)
   combos, combo_labels, combo_colors = _build_combos(coordinator_list, controller_list)

   fig, ax = plt.subplots(figsize=figsize)
   for combo in combos:
      ns, rates = [], []
      for n in drone_counts:
         subset = _select_runs(placed, n, combo)
         if len(subset) == 0:
            continue
         ns.append(n)
         rates.append((subset[STATUS_FIELD] == DEADLOCK_STATUS).sum() / len(subset) * 100.0)
      if ns:
         ax.plot(ns, rates, "-", color=combo_colors[combo], linewidth=1.0, label=combo_labels[combo], **_marker_kwargs(combo, combo_colors[combo]))

   # placement limit of the static spheres: first N without a placed static run after at least one placed one
   if "mpc_agent" in controller_list:
      static_placed = [n for n in drone_counts if len(placed[(placed[N_DRONES_FIELD] == n) & (placed[SPHERE_TYPE_FIELD] == "mpc_agent")]) > 0]
      missing = [n for n in drone_counts if static_placed and n > static_placed[0] and n not in static_placed]
      if missing:
         n_limit = min(missing)
         ax.axvline(n_limit - 0.5, color="k", ls=":", lw=0.8)
         ax.text(n_limit - 0.75, 27, f"static spheres cannot be\nplaced for $N \\geq {n_limit}$", rotation=90, ha="right", va="bottom",
                 fontsize=matplotlib.rcParams["legend.fontsize"], linespacing=1.1)

   _format_n_axis(ax, drone_counts)
   ax.set_ylabel(YLABEL_DEADLOCK)
   ax.set_ylim(-4, 104)
   ax.set_yticks(range(0, 101, 20))
   if show_title:
      ax.set_title(title)
   ax.legend(loc="center right", frameon=False, handlelength=2.2)
   ax.grid(color=GRID_COLOR, lw=0.3)

   fig.tight_layout()
   _save(fig, output_dir, filename_base)


def plot_time_per_unit(df: pd.DataFrame, output_dir: Path, coordinator_list: list[str], controller_list: list[str], n_drones_range: range | list[int],
                       filename_base: str = "fig_time_per_unit", column: str = UNIT_TIME_FIELD, ylabel: str = YLABEL_UNIT_TIME,
                       title: str = "Compute time per step and computing unit", figsize: tuple[float, float] = (IEEE_COL_WIDTH, 2.4),
                       show_title: bool = False, band: bool = False, reference_s: float | None = REFERENCE_TIME_S) -> None:
   """Mean of *column* over N per configuration as a line with markers on a log axis, finished and deadlock runs.

   Replaces the box plots of the first submission (four boxes per N are unreadable at 24 values of N, the spread is in the
   scaling figures). *band* adds the range from minimum to maximum as a shaded band, no outlier points, drawn opaque in a
   light shade behind the lines (EPS has no transparency). *reference_s* draws the dashed line at ten sampling times."""
   runs = df[df[STATUS_FIELD].isin(VALID_STATUSES)]
   drone_counts = sorted(int(n) for n in df[N_DRONES_FIELD].unique() if n in n_drones_range)
   combos, combo_labels, combo_colors = _build_combos(coordinator_list, controller_list)

   fig, ax = plt.subplots(figsize=figsize)
   for combo in combos:
      ns, means, mins, maxs = [], [], [], []
      for n in drone_counts:
         vals = _select_runs(runs, n, combo)[column].dropna()
         if vals.empty:
            continue
         ns.append(n)
         means.append(vals.mean())
         mins.append(vals.min())
         maxs.append(vals.max())
      if not ns:
         continue
      if band:
         ax.fill_between(ns, mins, maxs, color=_band_color(combo_colors[combo]), lw=0, zorder=1)
      ax.plot(ns, means, "-", color=combo_colors[combo], linewidth=1.0, label=combo_labels[combo], zorder=3, **_marker_kwargs(combo, combo_colors[combo]))

   _format_n_axis(ax, drone_counts)
   ax.set_yscale("log")
   ax.set_ylabel(ylabel)
   if reference_s is not None:
      _reference_time_line(ax, drone_counts[0] - 0.4, reference_s)
   if show_title:
      ax.set_title(title)
   ax.legend(loc="upper left", frameon=False, handlelength=2.2)
   ax.grid(color=GRID_COLOR, lw=0.3)

   fig.tight_layout()
   _save(fig, output_dir, filename_base)


SCALING_COLUMNS_ALL = [("mean_step_time_s", YLABEL_STEP_TIME),
                       ("wall_time_s", "wall time in s")]


def plot_scaling_curves(df: pd.DataFrame, output_dir: Path,
                        coordinator: str, controller: str,
                        n_drones_range: range | list[int] | None = None,
                        figsize: tuple[float, float] | None = None,
                        log_scale: bool = False, show_title: bool = True,
                        columns: list[tuple[str, str]] | None = None,
                        filename_base: str | None = None,
                        reference_s: float | None = None,
                        power_reference: tuple[float, float, int] | None = None,
                        power_reference_offset: float = POWER_REFERENCE_OFFSET,
                        markersize: float = 3.2) -> None:
   """Line plot showing how a metric of one configuration scales with N.

   Mean as a line with markers, one standard deviation over the runs as a shaded band (clamped to 5 % of the mean below on a
   log axis). Finished and deadlock runs count; an N without a finished run gets an open marker. *columns* selects which
   metrics to plot (default: step time and wall time side by side). *reference_s* draws the dashed line at ten sampling times.
   *power_reference* = (a, b, n_min) draws a dotted guide line a N^b for N >= n_min, shifted below the data by
   *power_reference_offset* so that the markers do not hide it; its legend entry names the exponent only."""
   runs = df[(df[STATUS_FIELD].isin(VALID_STATUSES)) &
             (df[COORDINATOR_TYPE_FIELD] == coordinator) &
             (df[SPHERE_TYPE_FIELD] == controller)].copy()
   n_range = n_drones_range if n_drones_range is not None else sorted(df[N_DRONES_FIELD].unique())
   drone_counts = sorted(int(n) for n in runs[N_DRONES_FIELD].unique() if n in n_range)

   if not drone_counts:
      return

   if columns is None:
      columns = SCALING_COLUMNS_ALL
   ncols = len(columns)
   if figsize is None:
      figsize = (IEEE_FULL_WIDTH, 2.8) if ncols > 1 else (IEEE_COL_WIDTH, 2.8)
   combo = (coordinator, controller)
   color = COLOR_BY_DRONE_INDEX[COMBO_COLOR_INDEX.get(combo, SINGLE_COLOR_INDEX.get(coordinator, 1))]
   has_finished = np.array([(runs[runs[N_DRONES_FIELD] == n][STATUS_FIELD] == FINISHED_STATUS).any() for n in drone_counts])

   fig, axes = plt.subplots(1, ncols, figsize=figsize)
   if ncols == 1:
      axes = [axes]
   for ax, (col, ylabel) in zip(axes, columns):
      means, stds = [], []
      for n in drone_counts:
         vals = runs[runs[N_DRONES_FIELD] == n][col].dropna()
         means.append(vals.mean())
         stds.append(vals.std() if len(vals) > 1 else 0.0)
      means_arr = np.array(means)
      stds_arr = np.array(stds)
      ns_arr = np.array(drone_counts)

      lower = means_arr - stds_arr
      if log_scale:
         lower = np.maximum(lower, means_arr * 0.05)  # clamp to 5% of mean for log scale
      else:
         lower = np.maximum(lower, 0)
      ax.fill_between(ns_arr, lower, means_arr + stds_arr, color=_band_color(color), lw=0, zorder=1)
      ax.plot(ns_arr, means_arr, "-", color=color, linewidth=1.2, zorder=3)
      for mask, open_marker in ((has_finished, False), (~has_finished, True)):
         if mask.any():
            ax.plot(ns_arr[mask], means_arr[mask], ls="none", color=color, zorder=4,
                    **_marker_kwargs(combo, color, open_marker=open_marker, markersize=markersize))

      if power_reference is not None:
         a, b, n_min = power_reference
         ref_n = ns_arr[ns_arr >= n_min]
         if ref_n.size:
            ax.plot(ref_n, power_reference_offset * a * ref_n.astype(float) ** b, ":", color="k", lw=0.9, zorder=2, label=rf"$\propto N^{{{b:g}}}$")
            ax.legend(loc="upper left", frameon=False, handlelength=2.2)

      _format_n_axis(ax, drone_counts)
      ax.set_ylabel(ylabel)
      if log_scale:
         ax.set_yscale("log")
      if reference_s is not None:
         _reference_time_line(ax, drone_counts[0] - 0.4, reference_s)
      ax.grid(color=GRID_COLOR, lw=0.3)

   if show_title:
      fig.suptitle(f"Scaling: {CONFIG_LABELS.get(combo, f'{coordinator} / {controller}')}")
   fig.tight_layout()

   if filename_base is None:
      suffix = "_log" if log_scale else ""
      col_suffix = "_" + "_".join(c[0] for c in columns) if len(columns) != len(SCALING_COLUMNS_ALL) else ""
      filename_base = f"scaling_{coordinator}_{controller}_{n_range}{col_suffix}{suffix}"
   _save(fig, output_dir, filename_base)


def plot_paper_figures(df: pd.DataFrame, output_dir: Path, n_drones_range: range | list[int], show_title: bool = False,
                       cubic_reference: tuple[float, float, int] | None = CUBIC_REFERENCE) -> None:
   """The four Block-A figures of the revision in print size (PLAN_ABBILDUNGEN.md, step 2a to 2d)."""
   column_size = (IEEE_COL_WIDTH, 2.4)
   half_size = (IEEE_COL_WIDTH * 0.48, 2.2)
   plot_deadlock_rate(df, output_dir, COORDINATOR_TYPES, SPHERE_TYPES, n_drones_range, filename_base="fig_deadlock_rate", figsize=column_size,
                      show_title=show_title)
   plot_time_per_unit(df, output_dir, COORDINATOR_TYPES, SPHERE_TYPES, n_drones_range, filename_base="fig_time_per_unit", figsize=column_size,
                      show_title=show_title)
   plot_scaling_curves(df, output_dir, *CONFIG_A, n_drones_range=n_drones_range, figsize=half_size, log_scale=True, show_title=show_title,
                       columns=[("mean_step_time_s", YLABEL_STEP_TIME)], filename_base="fig_scaling_A", reference_s=REFERENCE_TIME_S,
                       power_reference=cubic_reference, markersize=2.4)
   plot_scaling_curves(df, output_dir, *CONFIG_B, n_drones_range=n_drones_range, figsize=half_size, log_scale=True, show_title=show_title,
                       columns=[(UNIT_TIME_FIELD, YLABEL_UNIT_TIME_B)], filename_base="fig_scaling_B", reference_s=REFERENCE_TIME_S, markersize=2.4)


# ---------------------------------------------------------------------------------------------------------------------------
# Legacy figures of the first submission (box plots), kept for comparison: --legacy
# ---------------------------------------------------------------------------------------------------------------------------

def _boxplot_by_specific_coordinator_list_and_specific_controller_list_for_n_drones_range(df: pd.DataFrame, output_dir: Path, coordinator_list: list[str], controller_list: list[str],
                                                                                          column: str, ylabel: str, title: str, filename_base: str, figsize: tuple[float, float] = (IEEE_COL_WIDTH, 2.4),
                                                                                          n_drones_range: range | None = None, log_scale: bool = False, show_title: bool = True) -> None:
   """Box-whisker plot for every coordinator+controller combo, filtered by n_drones_range (finished and deadlock runs)."""
   runs = df[df[STATUS_FIELD].isin(VALID_STATUSES)].copy()
   n_drones_range = n_drones_range if n_drones_range is not None else sorted(df[N_DRONES_FIELD].unique())
   drone_counts = sorted(n for n in df[N_DRONES_FIELD].unique() if n in n_drones_range)

   combos, combo_labels, combo_colors = _build_combos(coordinator_list, controller_list)

   data: dict[tuple[str, str], list[list[float]]] = {combo: [] for combo in combos}
   for n in drone_counts:
      for combo in combos:
         data[combo].append(_select_runs(runs, n, combo)[column].dropna().tolist())

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
   ax.set_xlabel(XLABEL_N)
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
   _save(fig, output_dir, filename_base)


def plot_coord_and_sphere_combination(df: pd.DataFrame, output_dir: Path, field: str, n_drones_range: range | list[int] | None = None, show_title: bool = True):
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


def plot_all_four_deadlock(df: pd.DataFrame, output_dir: Path, n_drones_range: range | list[int] | None = None, figsize: tuple[float, float] = (IEEE_FULL_WIDTH, 2.8), show_title: bool = True) -> None:
   """Deadlock rate with all 4 combos (2 coordinators x 2 controllers), legacy file name."""
   n_range = n_drones_range if n_drones_range is not None else sorted(df[N_DRONES_FIELD].unique())
   plot_deadlock_rate(df, output_dir, COORDINATOR_TYPES, SPHERE_TYPES, n_range, filename_base="deadlock_rate_all4", title="Deadlock Rate: All Coordinators & Controllers",
                      figsize=figsize, show_title=show_title)


def plot_legacy_figures(df: pd.DataFrame, output_dir: Path, n_range: range, n_range_all: range, show_title: bool) -> None:
   plot_coord_and_sphere_combination(df, output_dir, n_drones_range=n_range, field="mean_step_time", show_title=show_title)
   plot_coord_and_sphere_combination(df, output_dir, n_drones_range=n_range, field="arrival", show_title=show_title)
   plot_coord_and_sphere_combination(df, output_dir, n_drones_range=n_range_all, field="mean_step_time", show_title=show_title)
   plot_coord_and_sphere_combination(df, output_dir, n_drones_range=n_range_all, field="arrival", show_title=show_title)

   plot_all_four_comparison(df, output_dir, column="mean_step_time_s", ylabel=YLABEL_STEP_TIME, title_prefix="Mean Step Time", filename_base=f"mean_step_time{n_range}", n_drones_range=n_range, show_title=show_title, figsize=(IEEE_COL_WIDTH*2.5, 4.8))
   plot_all_four_comparison(df, output_dir, column="wall_time_s", ylabel="wall time in s", title_prefix="Wall Time", filename_base=f"wall_time{n_range}", n_drones_range=n_range, show_title=show_title)
   plot_all_four_comparison(df, output_dir, column="steps", ylabel="arrival time in steps", title_prefix="Arrival Time", filename_base=f"arrival_time{n_range}", n_drones_range=n_range, show_title=show_title)
   plot_all_four_comparison(df, output_dir, column="mean_step_time_s", ylabel=YLABEL_STEP_TIME, title_prefix="Mean Step Time", filename_base=f"mean_step_time{n_range_all}", n_drones_range=n_range_all, show_title=show_title, figsize=(IEEE_COL_WIDTH*2.2, 3.0))
   plot_all_four_comparison(df, output_dir, column="wall_time_s", ylabel="wall time in s", title_prefix="Wall Time", filename_base=f"wall_time{n_range_all}", n_drones_range=n_range_all, show_title=show_title)
   plot_all_four_comparison(df, output_dir, column="steps", ylabel="arrival time in steps", title_prefix="Arrival Time", filename_base=f"arrival_time{n_range_all}", n_drones_range=n_range_all, show_title=show_title)

   plot_all_four_deadlock(df, output_dir, n_drones_range=n_range, show_title=show_title, figsize=(IEEE_COL_WIDTH*2.2, 3.0))
   plot_all_four_deadlock(df, output_dir, n_drones_range=n_range_all, show_title=show_title, figsize=(IEEE_COL_WIDTH*2.2, 3.0))

   for coord, ctrl in [CONFIG_A, CONFIG_B]:
      for log in [False, True]:
         for rng in (n_range, n_range_all):
            plot_scaling_curves(df, output_dir, columns=[("wall_time_s", "wall time in s")], coordinator=coord, controller=ctrl, n_drones_range=rng,
                                log_scale=log, figsize=(IEEE_COL_WIDTH*1.5, 3.5), show_title=show_title)
            plot_scaling_curves(df, output_dir, columns=[("mean_step_time_s", YLABEL_STEP_TIME)], coordinator=coord, controller=ctrl, n_drones_range=rng,
                                log_scale=log, figsize=(IEEE_COL_WIDTH*1.5, 3.5), show_title=show_title)


def main(metrics_csv: str | None = None, show_title: bool = False, font_scale: float = 1.0,
         metrics_path: Path | None = None, output_dir: Path | None = None, n_max: int = 25, legacy: bool = False) -> None:
   """Generate the figures.

   Without arguments: Block-A export ``BLOCK_A_METRICS`` -> ``BLOCK_A_OUTPUT``. *metrics_csv* selects a result folder below
   ``paper2_results`` (legacy layout with ``metrics.csv`` and ``plots/`` next to it). *metrics_path* / *output_dir* give
   explicit paths. *n_max* is the largest N in the full-range plots. *legacy* adds the box plots of the first submission.
   """
   set_font_scale(font_scale)
   if metrics_path is not None:
      METRICS_CSV = Path(metrics_path).resolve()
      OUTPUT_DIR = Path(output_dir).resolve() if output_dir is not None else METRICS_CSV.parent / "plots"
   elif metrics_csv is not None:
      METRICS_CSV = _HERE / "paper2_results" / metrics_csv / "metrics.csv"
      OUTPUT_DIR = _HERE / "paper2_results" / metrics_csv / "plots"
   else:
      METRICS_CSV = BLOCK_A_METRICS
      OUTPUT_DIR = Path(output_dir).resolve() if output_dir is not None else BLOCK_A_OUTPUT

   OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

   df = load_data(METRICS_CSV)
   print(f"Loaded {len(df)} rows from {METRICS_CSV}")
   print(f"  num_drones: {sorted(df['num_drones'].unique())}")
   print(f"  sphere_type: {sorted(df['controller_type'].unique())}")
   print(f"  status counts:\n{df['status'].value_counts().to_string()}")
   print()

   n_range_all = range(2, n_max + 1)
   plot_paper_figures(df, OUTPUT_DIR, n_range_all, show_title=show_title)

   if legacy:
      plot_legacy_figures(df, OUTPUT_DIR, range(2, 12), n_range_all, show_title)

   print("\nDone.")


if __name__ == "__main__":
   import argparse

   parser = argparse.ArgumentParser(description="Block A figures of the revision (deadlock rate, compute time per unit, scaling A and B)")
   parser.add_argument("--metrics", type=Path, default=None, help=f"explicit metrics.csv (default: {BLOCK_A_METRICS.relative_to(_HERE.parent)})")
   parser.add_argument("--out", type=Path, default=None, help=f"output directory (default: {BLOCK_A_OUTPUT.relative_to(_HERE.parent)}, or <metrics dir>/plots with --metrics)")
   parser.add_argument("--n-max", type=int, default=25, help="largest N in the plots (default 25)")
   parser.add_argument("--font-scale", type=float, default=1.0, help="factor on the IEEE base font sizes (default 1.0, the figures are drawn in print size)")
   parser.add_argument("--show-title", action="store_true")
   parser.add_argument("--legacy", action="store_true", help="also draw the box plots of the first submission")
   args = parser.parse_args()
   main(metrics_path=args.metrics, output_dir=args.out, show_title=args.show_title, font_scale=args.font_scale, n_max=args.n_max, legacy=args.legacy)
