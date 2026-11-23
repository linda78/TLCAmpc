"""Record drone positions and velocities of a scenario run, one CSV row per drone per step.

The columns drone_id, t, x, y, z, vx, vy, vz are the ones the ReplayBackend reads, so the CSV can also be replayed.

Two subcommands (run from the repo root):

   record   run a scenario and write the CSV
      .venv/bin/python -m tools.regression_positions record --config configs/narrow_gap/NarrowGapBaseline.json --steps 50 --seed 0 --out run_a.csv

   compare  compare two CSVs exactly (text equality, which is bit equality for floats)
      .venv/bin/python -m tools.regression_positions compare run_a.csv run_b.csv

Exit codes of compare (same idea as `cmp`): 0 identical, 1 different, 2 a file is missing or unusable.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator
from tools.live_view import load_parametrized_json

FIELDS = ["step", "t", "drone_id", "x", "y", "z", "vx", "vy", "vz"]


def state_rows(sim: Simulator, step: int) -> list[dict[str, Any]]:
   """Return one row (a dict) per drone for the current state of the simulator.

   .tolist() turns numpy numbers into plain Python floats. They are written with full
   precision (never rounded), so two runs can be compared exactly.
   """
   rows: list[dict[str, Any]] = []
   for d in sim.drones:
      x, y, z = d.position().tolist()
      vx, vy, vz = d.velocity().tolist()
      rows.append({
         "step": step, "t": float(sim.t), "drone_id": d.drone_id,
         "x": x, "y": y, "z": z,
         "vx": vx, "vy": vy, "vz": vz,
      })
   return rows


def record_run(config_path: str | Path, *, steps: int, out_csv: str | Path, seed: int = 0) -> int:
   """Run the scenario for `steps` steps and write the state after every step to `out_csv`.

   Step 0 is the starting state, before the first sim.step().
   The random generators are seeded first, so the same seed gives the same CSV (the solver
   uses random numbers for its symmetry break, see global_mpc.py).
   :return: number of data rows written (without the header).
   """
   # Seed before anything is built: everything after these two lines is reproducible.
   random.seed(seed)
   np.random.seed(seed)

   cfg = load_parametrized_json(config_path, params=None)
   if not isinstance(cfg, dict):
      raise TypeError(f"Config must decode to a JSON object/dict, got {type(cfg).__name__}")

   scenario = ScenarioConfig.model_validate(cfg)
   sim = Simulator.from_config(scenario)

   out_path = Path(out_csv)
   out_path.parent.mkdir(parents=True, exist_ok=True)

   n_rows = 0
   with out_path.open("w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=FIELDS)
      writer.writeheader()

      for row in state_rows(sim, 0):
         writer.writerow(row)
         n_rows += 1

      for step in range(1, steps + 1):
         sim.step()
         for row in state_rows(sim, step):
            writer.writerow(row)
            n_rows += 1

   return n_rows


def read_csv_rows(path: str | Path) -> tuple[list[str], list[list[str]]]:
   """Read a CSV file as plain text. :return: (header, data rows); every cell stays a string."""
   with Path(path).open("r", newline="", encoding="utf-8") as f:
      reader = csv.reader(f)
      try:
         header = next(reader)
      except StopIteration:
         raise ValueError(f"{path} is empty (no header line)") from None
      return header, list(reader)


def compare_runs(path_a: str | Path, path_b: str | Path) -> int:
   """Compare two CSVs exactly and print the first difference.

   Cells are compared as text. Python writes floats as the shortest text that reads back as the
   same float, so equal text means equal bits. It stops at the first difference: later rows
   differ as a consequence, because every step starts from the previous one.
   :return: 0 identical, 1 different, 2 a file is missing or unusable.
   """
   try:
      header_a, rows_a = read_csv_rows(path_a)
      header_b, rows_b = read_csv_rows(path_b)
   except (OSError, ValueError) as err:
      print(f"ERROR: {err}", file=sys.stderr)
      return 2

   if header_a != header_b:
      print("ERROR: the header lines differ, these are not the same kind of file")
      print(f"  {path_a}: {header_a}")
      print(f"  {path_b}: {header_b}")
      return 2

   for i, (row_a, row_b) in enumerate(zip(rows_a, rows_b), start=1):
      if row_a == row_b:
         continue
      named_a = dict(zip(header_a, row_a))
      named_b = dict(zip(header_b, row_b))
      print(f"DIFFERENT: first difference in data row {i} (file line {i + 1})")
      print(f"  step {named_a.get('step')}, drone {named_a.get('drone_id')}")
      for column in header_a:
         if named_a.get(column) != named_b.get(column):
            print(f"  column {column}: {named_a.get(column)} (A) vs {named_b.get(column)} (B)")
      print(f"  A = {path_a}")
      print(f"  B = {path_b}")
      return 1

   if len(rows_a) != len(rows_b):
      print(f"DIFFERENT: the common rows are identical, but A has {len(rows_a)} data rows and B has {len(rows_b)}")
      return 1

   print(f"IDENTICAL: {len(rows_a)} data rows, {len(header_a)} columns")
   return 0


def main(argv: list[str] | None = None) -> int:
   parser = argparse.ArgumentParser(description="Record drone positions of a scenario run, or compare two recordings")
   sub = parser.add_subparsers(dest="command", required=True)

   rec = sub.add_parser("record", help="Run a scenario and write its positions to a CSV file")
   rec.add_argument("--config", required=True, help="Path to scenario JSON")
   rec.add_argument("--steps", type=int, default=50, help="Number of Simulator.step() calls (default: 50)")
   rec.add_argument("--out", required=True, help="Output CSV path")
   rec.add_argument("--seed", type=int, default=0, help="Seed for numpy and random (default: 0)")

   cmp_ = sub.add_parser("compare", help="Compare two recorded CSV files exactly")
   cmp_.add_argument("file_a", help="First CSV (for example the reference from before the change)")
   cmp_.add_argument("file_b", help="Second CSV (for example the run after the change)")

   args = parser.parse_args(argv)

   if args.command == "record":
      n_rows = record_run(args.config, steps=args.steps, out_csv=args.out, seed=args.seed)
      print(f"Wrote {n_rows} rows to {args.out} (seed {args.seed})")
      return 0

   return compare_runs(args.file_a, args.file_b)


if __name__ == "__main__":
   raise SystemExit(main())