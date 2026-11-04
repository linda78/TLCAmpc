"""ReplayBackend: SimulationBackend that replays a recorded run from a per-step CSV instead of simulating."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from drone_sim.domain.config import ScenarioConfig
from drone_sim.domain.utils.helper import all_drones_reached_destination
from drone_sim.gui.backend import SimulationBackend, SimState, StepResult
from drone_sim.gui.direct_backend import make_drone_states
from drone_sim.simulation.simulator import Simulator

REQUIRED_COLUMNS = ("drone_id", "t", "x", "y", "z", "vx", "vy", "vz")

Frame = tuple[float, dict[str, tuple[np.ndarray, np.ndarray]]]  # (t, drone_id -> (position, velocity))


class ReplayBackend(SimulationBackend):
   """Replays a recorded run. The scenario JSON gives room, obstacles, colours and radii; the CSV gives one row per drone
   and time step with the columns ``drone_id, t, x, y, z, vx, vy, vz`` (further columns are ignored, so the per-step CSV
   of ``paper2_tools/review_work/narrow_gap/narrow_gap.py`` works as it is).

   Rows are grouped by ``t`` into frames. ``step()`` shows the next frame: ``step_count`` is the frame index starting at
   0, ``t`` the frame's time. Safety radii are computed from the recorded velocities with the drone parameters of the
   scenario, collisions with the simulator's own check. On the last frame ``step()`` sets ``end_of_data`` and keeps
   returning that frame. ``reset()`` goes back to the first frame, ``seek()`` jumps to any frame and ``step()`` continues
   from there; ``positions_up_to()`` gives the recorded path up to a frame for the traces. Nothing is simulated.
   """

   def __init__(self) -> None:
      self._sim: Simulator | None = None
      self._cfg: ScenarioConfig | None = None
      self._config_path: Path | None = None
      self._csv_path: Path | None = None
      self._frames: list[Frame] = []
      self._index: int = -1

   # ------------------------------------------------------------------ #
   # Public API                                                           #
   # ------------------------------------------------------------------ #

   def load_config(self, path: Path) -> SimState:
      cfg = ScenarioConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
      self._cfg = cfg
      self._config_path = Path(path)
      self._sim = Simulator.from_config(cfg)
      self._frames = []
      self._csv_path = None
      self._index = -1
      return self._make_sim_state()

   def load_csv(self, path: Path) -> SimState:
      """Read the recorded run; raises ValueError when columns are missing or the drones do not match the scenario."""
      if self._sim is None:
         raise RuntimeError("Call load_config() before load_csv()")
      path = Path(path)
      with path.open(newline="", encoding="utf-8") as fh:
         reader = csv.DictReader(fh)
         missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
         if missing:
            raise ValueError(f"{path.name}: missing columns {', '.join(missing)} (needed: {', '.join(REQUIRED_COLUMNS)})")
         rows = list(reader)
      if not rows:
         raise ValueError(f"{path.name}: no rows")
      known = {d.drone_id for d in self._sim.drones}
      unknown = sorted({r["drone_id"] for r in rows} - known)
      if unknown:
         raise ValueError(f"{path.name}: drone ids not in {self._config_path.name}: {', '.join(unknown)}")
      by_t: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
      for r in rows:
         t = round(float(r["t"]), 6)
         pos = np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=float)
         vel = np.array([float(r["vx"]), float(r["vy"]), float(r["vz"])], dtype=float)
         by_t.setdefault(t, {})[r["drone_id"]] = (pos, vel)
      frames = sorted(by_t.items())
      for t, states in frames:
         absent = sorted(known - states.keys())
         if absent:
            raise ValueError(f"{path.name}: frame t={t} has no row for {', '.join(absent)}")
      self._frames = frames
      self._csv_path = path
      self._index = -1
      self._apply_frame(0)
      return self._make_sim_state()

   def step(self) -> StepResult:
      if self._sim is None:
         raise RuntimeError("Call load_config() before step()")
      if not self._frames:
         raise RuntimeError("Call load_csv() before step()")
      if self._index < len(self._frames) - 1:
         self._index += 1
      self._apply_frame(self._index)
      return self._make_step_result(end_of_data=self._index == len(self._frames) - 1)

   def get_state(self) -> SimState:
      if self._sim is None:
         raise RuntimeError("Call load_config() before get_state()")
      return self._make_sim_state()

   def reset(self) -> None:
      if self._sim is None:
         raise RuntimeError("Call load_config() before reset()")
      self._index = -1
      if self._frames:
         self._apply_frame(0)

   def seek(self, index: int) -> StepResult:
      """Jump to frame ``index`` (clamped to the recorded range) and return it; ``step()`` continues from there."""
      if self._sim is None:
         raise RuntimeError("Call load_config() before seek()")
      if not self._frames:
         raise RuntimeError("Call load_csv() before seek()")
      self._index = self._clamp(index)
      self._apply_frame(self._index)
      return self._make_step_result(end_of_data=self._index == len(self._frames) - 1)

   def positions_up_to(self, index: int | None = None) -> dict[str, list[list[float]]]:
      """Recorded positions per drone from the first frame up to and including ``index`` (default: the current frame)."""
      last = self.frame_index if index is None else self._clamp(index)
      out: dict[str, list[list[float]]] = {}
      for _, states in self._frames[: last + 1]:
         for drone_id, (pos, _) in states.items():
            out.setdefault(drone_id, []).append(pos.tolist())
      return out

   @property
   def frame_count(self) -> int:
      return len(self._frames)

   @property
   def frame_index(self) -> int:
      """Index of the frame currently applied (0 before the first ``step()``)."""
      return max(self._index, 0)

   # ------------------------------------------------------------------ #
   # Private helpers                                                      #
   # ------------------------------------------------------------------ #

   def _clamp(self, index: int) -> int:
      return min(max(int(index), 0), len(self._frames) - 1)

   def _apply_frame(self, index: int) -> None:
      t, states = self._frames[index]
      sim = self._sim
      for d in sim.drones:
         pos, vel = states[d.drone_id]
         d.x = np.concatenate([pos, vel])  # a fresh array: positions handed out earlier are views and must not change
         d.route.advance_if_reached(d.position())
      sim.t = t
      sim.step_count = max(index, 0)
      sim.last_collisions = sim._compute_collisions()  # same check as a simulated step, on the recorded positions

   def _make_step_result(self, end_of_data: bool) -> StepResult:
      sim = self._sim
      drone_states, safety_radii = make_drone_states(sim.drones)
      return StepResult(drones=drone_states, safety_radii=safety_radii, last_collisions=list(sim.last_collisions), infeasible=False, infeasible_reason=None,
            step_count=max(self._index, 0), t=float(sim.t), all_reached=all_drones_reached_destination(sim.drones), admm_iteration_count=None,
            end_of_data=end_of_data)

   def _make_sim_state(self) -> SimState:
      sim = self._sim
      kind = f"replay: {self._csv_path.name}" if self._csv_path is not None else "replay (no CSV loaded)"
      return SimState(drone_count=len(sim.drones), obstacle_count=len(sim.obstacles), obstacles=sim.obstacles, coordinator_type=kind, dt=sim.dt,
                      step_count=max(self._index, 0), room_min=sim.room_min, room_max=sim.room_max,
                      config_path=str(self._config_path) if self._config_path is not None else None)
