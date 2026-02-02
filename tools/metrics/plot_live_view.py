from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Use a slightly smaller default font size to fit multi-row figures better.
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,    "legend.fontsize": 7})

# Default location of the live_view_horizon_grid folder, resolved relative to this file so it does not depend on the current
# working directory when the script is executed.
_DEFAULT_ROOT = (Path(__file__).resolve().parents[1] / "live_view_horizon_grid").resolve()
FAILED = ["step_timeout", "overall_timeout", "infeasible", "error", "timeout", "max_steps", ]


def find_result_dirs(root: Path) -> list[Path]:
   """ Find all result folders under `root` that look like:
     - results
     - results_<tolerance>
   """
   result_dirs: list[Path] = []
   for entry in root.iterdir():
      if not entry.is_dir():
         continue
      if not entry.name.startswith("results"):
         continue
      if not (entry / "metrics.csv").exists():
         continue
      result_dirs.append(entry)
   return sorted(result_dirs)


def tolerance_from_dirname(dirname: str) -> tuple[str, float | None]:
   """ Map directory name to a human-readable label and a numeric tolerance (if possible).
   """
   if dirname == "results":
      return "baseline", None
   suffix = dirname.split("_", 1)[1] if "_" in dirname else dirname
   try:
      tol_value = float(suffix)
   except ValueError:
      tol_value = None
   return suffix, tol_value


def load_all_metrics(root: Path) -> pd.DataFrame:
   """ Load and concatenate all metrics.csv files, adding tolerance info.
   """
   frames: list[pd.DataFrame] = []
   for d in find_result_dirs(root):
      label, tol = tolerance_from_dirname(d.name)
      csv_path = d / "metrics.csv"
      df = pd.read_csv(csv_path)
      df["tolerance_label"] = label
      df["tolerance"] = tol
      frames.append(df)

   if not frames:
      raise RuntimeError(f"No metrics.csv files found under {root}")

   # TODO: solve future warning about empty of inconsistent frames
   df_all = pd.concat(frames, ignore_index=True)

   # Keep all rows; some infeasible runs have steps == 0. We handle those explicitly when computing jerk_total and wall_time_s.
   return df_all.copy()


def load_metrics_from_csv_paths(csv_paths: list[Path]) -> pd.DataFrame:
   frames: list[pd.DataFrame] = []
   for csv_path in csv_paths:
      if not csv_path.exists():
         continue
      label = csv_path.parent.parent.name
      df = pd.read_csv(csv_path)
      df["tolerance_label"] = label
      df["tolerance"] = None
      frames.append(df)

   if not frames:
      raise RuntimeError("No metrics.csv files found")

   df_all = pd.concat(frames, ignore_index=True)
   return df_all.copy()


def find_metrics_csv_under_out_dir(root: Path) -> list[Path]:
   return sorted(root.glob("*/out_dir/metrics.csv"))


def cut_after_first_gap(df: pd.DataFrame, col: str = "horizon") -> pd.DataFrame | None:
   """ Removes all lines, after first gap, from `df`.
   """
   if df.empty or col not in df.columns:
      return None

   df_sorted = df.sort_values(col)
   h = df_sorted[col].to_numpy(dtype=float)

   if len(h) < 2:
      return df.copy()

   diffs = np.diff(h)
   gap_idx = np.where(diffs > 1)[0]

   if gap_idx.size == 0:
      # no gab -> hold all
      return df.copy()

   last_ok_value = h[gap_idx[0]]

   mask = df[col].astype(float) <= last_ok_value
   return df[mask].copy()


def score_best_horizon(df_N: pd.DataFrame) -> float | None:
   """Return the best horizon H using 70% time / 30% jerk weighting.

   Only finished runs are considered. Returns None if no finished simulations are available for this N.
   """
   df_finished = cut_after_first_gap(df_N[df_N["status"].astype(str).str.lower() == "finished"].copy())
   if df_finished is None or df_finished.empty:
      return None

   if "jerk_3d_value" in df_finished.columns:
      jerk = df_finished["jerk_3d_value"].astype(float)
   elif "jerk_mean" in df_finished.columns:
      jerk = df_finished["jerk_mean"].astype(float)
   else:
      return None

   if np.isnan(jerk).all():
      return None
   time = df_finished["wall_time_s"].astype(float)

   j_min, j_max = float(jerk.min()), float(jerk.max())
   t_min, t_max = float(time.min()), float(time.max())

   if t_max > t_min:
      t_norm = (time - t_min) / (t_max - t_min)
   else:
      t_norm = time * 0.0

   if j_max > j_min:
      j_norm = (jerk - j_min) / (j_max - j_min)
   else:
      j_norm = jerk * 0.0

   score = 0.3 * j_norm + 0.7 * t_norm
   best_idx = score.idxmin()
   return float(df_finished.loc[best_idx, "horizon"])


def _find_tail_start(horizons: np.ndarray, infeasible_mask: np.ndarray) -> float | None:
   """Find the first infeasible horizon that occurs after at least one valid horizon."""
   seen_valid = False
   for h, is_inf in zip(horizons, infeasible_mask):
      if not is_inf:
         seen_valid = True
      elif seen_valid:
         return float(h)
   return None


def _shade_infeasible_regions(ax: plt.Axes, horizons: np.ndarray, infeasible_horizons: np.ndarray, max_gap: int = 3) -> None:
   """Shade grey regions for infeasible horizons, merging nearby ones into blocks."""
   if infeasible_horizons.size == 0:
      return

   # Find where the "tail" of continuous infeasibility starts
   infeasible_set = set(infeasible_horizons)
   infeasible_mask = np.array([h in infeasible_set for h in horizons])
   tail_start = _find_tail_start(horizons, infeasible_mask)
   max_h = float(horizons.max())

   # Shade from tail_start to end if it exists
   if tail_start is not None:
      ax.axvspan(tail_start - 0.5, max_h + 0.5, color="grey", alpha=0.3, zorder=0)
      early_infeasible = infeasible_horizons[infeasible_horizons < tail_start]
   else:
      early_infeasible = infeasible_horizons

   # Shade early infeasible horizons as merged blocks
   if early_infeasible.size == 0:
      return

   block_start = early_infeasible[0] - 0.5
   prev_h = early_infeasible[0]
   for h in early_infeasible[1:]:
      if h - prev_h > max_gap:
         ax.axvspan(block_start, prev_h + 0.5, color="grey", alpha=0.3, zorder=0)
         block_start = h - 0.5
      prev_h = h
   ax.axvspan(block_start, prev_h + 0.5, color="grey", alpha=0.3, zorder=0)


def plot_subplot_for_n(ax: plt.Axes, df_N: pd.DataFrame) -> None:
   """Render one row (fixed num_drones N) of the multi-diagram figure.

   Left y-axis shows normalised jerk metrics right y-axis shows overall simulation time.
   """

   if df_N.empty:
      ax.set_visible(False)
      return

   df_N = df_N.sort_values("horizon").copy()

   infeasible_mask = df_N["status"].astype(str).str.lower().isin(FAILED)
   if infeasible_mask.any():
      horizons = df_N["horizon"].astype(float).to_numpy()
      infeasible_horizons = df_N.loc[infeasible_mask, "horizon"].astype(float).sort_values().to_numpy()
      _shade_infeasible_regions(ax, horizons, infeasible_horizons)

   # Normalise jerk_3d_value if present.
   j3_plot = None
   if "jerk_3d_value" in df_N.columns:
      j3_all = df_N["jerk_3d_value"].astype(float)
      j3_min, j3_max = float(j3_all.min()), float(j3_all.max())
      if j3_max > j3_min:
         j3_plot = (j3_all - j3_min) / (j3_max - j3_min)
      else:
         j3_plot = j3_all * 0.0

   # Left axis: normalised jerk metrics.
   ax.plot(df_N["horizon"], j3_plot, marker="o", color="tab:cyan", linewidth=1, markersize=4, label="normalised jerk")

   N_val = int(df_N["num_drones"].iloc[0])
   ax.set_ylabel(f"{N_val} drones\nnormalised jerk", color="black")
   ax.tick_params(axis="y", labelcolor="black")
   ax.grid(True, which="both", linestyle="--", alpha=0.3)

   # Right axis: overall simulation time (paper-style purple color)
   ax2 = ax.twinx()
   ax2.plot(df_N["horizon"], df_N["wall_time_s"], marker="o", color="tab:purple", linewidth=1, markersize=4,
            label="overall time [s]")
   ax2.set_ylabel("time [s]", color="black")
   ax2.tick_params(axis="y", labelcolor="black")

   # Draw a vertical dashed line at the best horizon, similar to the "best H" indicator in the paper figure.
   best_H = score_best_horizon(df_N)
   if best_H is not None:
      ax.axvline(best_H, color="red", linestyle=":", alpha=0.6)
      ax.text(best_H, ax.get_ylim()[1], f"best H={int(best_H)}", color="red", rotation=90, va="top", ha="right",
              fontsize=6)


def plot_multidiagrams_per_tolerance(df: pd.DataFrame, out_dir: Path) -> None:
   """Create multi-diagram figures per tolerance for jerk and overall time."""
   out_dir.mkdir(parents=True, exist_ok=True)

   # Sort N so that 2 is at the top and 7 at the bottom (assuming 2..7 exist).
   all_N = sorted(df["num_drones"].unique())

   for tol_label, df_tol_raw in df.groupby("tolerance_label"):
      df_tol = df_tol_raw.sort_values(["num_drones", "horizon"]).copy()

      fig, axes = plt.subplots(nrows=len(all_N), ncols=1, sharex=True, figsize=(8, 2.0 * len(all_N)))
      if len(all_N) == 1:
         axes = [axes]

      for ax, N in zip(axes, all_N):
         df_N = df_tol[df_tol["num_drones"] == N]
         plot_subplot_for_n(ax, df_N)

      # Integer ticks on the shared MPC horizon axis.
      horizons = sorted(df_tol["horizon"].unique())
      if horizons:
         axes[-1].set_xticks(horizons)
         axes[-1].set_xlim(min(horizons) - 0.5, max(horizons) + 0.5)

         axes[-1].set_xlabel("MPC horizon")
         n_min, n_max = int(min(all_N)), int(max(all_N))
         fig.suptitle(f"Horizon sweep (H={min(horizons)}..{max(horizons)}) for {n_min}..{n_max} drones")

      legend_elements = [
            Line2D([0], [0], color="tab:cyan", marker="o", linewidth=1, markersize=4, label="normalised jerk"),
            Line2D([0], [0], color="tab:purple", marker="o", linewidth=1, markersize=4, label="overall time [s]"),
            Patch(facecolor="grey", edgecolor="none", alpha=0.3, label="simulation timing or solving error")]
      fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements), bbox_to_anchor=(0.5, 0.01),
                 frameon=False)
      fig.tight_layout(rect=[0, 0.03, 1, 0.95])

      summary_path = out_dir / f"multidiag_jerk_time_tol-{tol_label}.png"
      fig.savefig(summary_path, dpi=200)
      plt.close(fig)


def main() -> None:
   parser = argparse.ArgumentParser(description=("Plot live_view_horizon_grid results as curves over horizon H."))
   parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT,
                       help="Root folder that contains result folders (default: live_view_horizon_grid next to repo root)")
   parser.add_argument("--metrics-glob", action="append", default=None,
                       help="Glob to one or more metrics.csv files (repeatable), e.g. 'param_swep_result/*/out_dir/metrics.csv'")
   parser.add_argument("--debug", action="store_true", help="Print debug information (e.g. which CSV files were loaded)")
   parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots (default: <root>/plots)")
   args = parser.parse_args()

   root: Path = args.root
   if not root.exists():
      raise FileNotFoundError(
            f"Root folder does not exist: {root}. "
            f"If you want to plot param sweeps, use --metrics-glob 'param_swep_result/*/out_dir/metrics.csv' "
            f"(or point --root at an existing folder)."
      )
   if args.metrics_glob:
      csv_paths: list[Path] = []
      for pattern in args.metrics_glob:
         p = Path(pattern)
         if p.is_absolute():
            csv_paths.extend([Path(x) for x in sorted(p.parent.glob(p.name))])
         else:
            csv_paths.extend([Path(x) for x in sorted(Path().glob(pattern))])
      csv_paths = [p for p in csv_paths if p.exists()]
      if args.debug:
         print("CSV files (from --metrics-glob):")
         for p in csv_paths:
            print(f"  - {p}")
      print(f"Plotting {len(csv_paths)} metrics.csv files")
      out_dir: Path = args.out_dir or Path("plots")
      df = load_metrics_from_csv_paths(csv_paths)
   else:
      sweep_csvs = find_metrics_csv_under_out_dir(root)
      if sweep_csvs:
         if args.debug:
            print("CSV files (auto-detected under --root):")
            for p in sweep_csvs:
               print(f"  - {p}")
         print(f"Plotting {len(sweep_csvs)} metrics.csv files under {root}")
         out_dir = args.out_dir or (root / "plots")
         df = load_metrics_from_csv_paths(sweep_csvs)
      else:
         print(f"Plotting {root}")
         out_dir = args.out_dir or (root / "plots")
         df = load_all_metrics(root)

   # Create one multi-diagram figure per tolerance with
   #   - total jerk (sum over direction changes)
   #   - overall simulation time (approx. frames * mean_step_time_s)
   plot_multidiagrams_per_tolerance(df, out_dir)

   print(f"Multi-diagram jerk/time plots written to: {out_dir}")


if __name__ == "__main__":
   main()
