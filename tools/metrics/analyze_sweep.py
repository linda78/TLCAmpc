#!/usr/bin/env python3
"""Combine metrics from parameter sweep and create visualization."""

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def extract_q_r_from_folder(folder_name):
    """Extract q and r values from folder name."""
    # Pattern: s_1.0_u_[3.0, 3.0, 3.0]_v_2.5_r_5.0_q_[1, 1, 1]_r_[0.01, 0.01, 0.01]
    q_match = re.search(r'_q_\[(\d+(?:\.\d+)?),\s*\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\]', folder_name)
    r_match = re.search(r'_r_\[(\d+(?:\.\d+)?),\s*\d+(?:\.\d+)?,\s*\d+(?:\.\d+)?\]$', folder_name)

    q_val = float(q_match.group(1)) if q_match else None
    r_val = float(r_match.group(1)) if r_match else None
    return q_val, r_val


def combine_metrics(base_dir):
    """Combine all metrics.csv files into one DataFrame."""
    all_data = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path) or not folder.startswith('s_'):
            continue

        metrics_path = os.path.join(folder_path, 'out_dir', 'metrics.csv')
        if not os.path.exists(metrics_path):
            continue

        q_val, r_val = extract_q_r_from_folder(folder)
        if q_val is None or r_val is None:
            continue

        df = pd.read_csv(metrics_path)
        df['q'] = q_val
        df['r'] = r_val
        df['folder'] = folder
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['q', 'r']).reset_index(drop=True)
    return combined


def create_heatmaps(df, output_path):
    """Create heatmap visualizations for different metrics."""
    q_vals = sorted(df['q'].unique())
    r_vals = sorted(df['r'].unique())

    metrics_to_plot = [
        ('status', 'Finished (1) / Not Finished (0)', lambda x: 1 if x == 'finished' else 0),
        ('wall_time_s', 'Wall Time (s)', None),
        ('max_step_time_s', 'Max Step Time (s)', None),
        ('mean_distance_step_mean', 'Mean Distance', None),
        ('jerk_3d_value', 'Jerk', None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (col, title, transform) in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break

        pivot_data = np.zeros((len(q_vals), len(r_vals)))

        for i, q in enumerate(q_vals):
            for j, r in enumerate(r_vals):
                row = df[(df['q'] == q) & (df['r'] == r)]
                if len(row) > 0:
                    val = row[col].values[0]
                    if transform:
                        val = transform(val)
                    pivot_data[i, j] = val
                else:
                    pivot_data[i, j] = np.nan

        ax = axes[idx]
        im = ax.imshow(pivot_data, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(r_vals)))
        ax.set_xticklabels([f'{r:.2f}' for r in r_vals], rotation=45, ha='right')
        ax.set_yticks(range(len(q_vals)))
        ax.set_yticklabels([f'{q:.0f}' for q in q_vals])
        ax.set_xlabel('r value')
        ax.set_ylabel('q value')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        # Add text annotations
        for i in range(len(q_vals)):
            for j in range(len(r_vals)):
                val = pivot_data[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val < (pivot_data.max() + pivot_data.min()) / 2 else 'black'
                    if col == 'status':
                        ax.text(j, i, 'Y' if val == 1 else 'N', ha='center', va='center', color=text_color, fontsize=8)
                    else:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=6)

    # Remove unused subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Analyze parameter sweep results and create heatmaps.')
    parser.add_argument('--base_path', nargs='?', default='.',
                        help='Base path containing s_* folders (default: current directory)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output name prefix (default: param_sweep_analysis)')
    args = parser.parse_args(argv)

    # Resolve base path (can be relative or absolute)
    base_dir = os.path.abspath(args.base_path)

    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return 1

    output_name = args.output or 'param_sweep_analysis'

    print(f"Scanning for metrics in: {base_dir}")
    print("Combining metrics from all runs...")
    df = combine_metrics(base_dir)

    if df.empty:
        print("No metrics found!")
        return 1

    # Save combined CSV
    output_csv = os.path.join(base_dir, f'{output_name}.csv')
    df.to_csv(output_csv, index=False)
    print(f"Saved combined metrics to {output_csv}")
    print(f"Total runs: {len(df)}")

    # Create visualizations
    print("Creating visualizations...")
    output_png = os.path.join(base_dir, f'{output_name}.png')
    create_heatmaps(df, output_png)

    # Print summary
    print("\n=== Summary ===")
    print(f"Finished runs: {(df['status'] == 'finished').sum()} / {len(df)}")
    print(f"\nMetrics ranges:")
    for col in ['wall_time_s', 'max_step_time_s', 'mean_distance_step_mean', 'jerk_3d_value']:
        print(f"  {col}: {df[col].min():.3f} - {df[col].max():.3f}")

    return 0


if __name__ == '__main__':
    exit(main())