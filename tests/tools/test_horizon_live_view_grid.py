"""Tests for tools.horizon_live_view_grid.

Only the deadlock exit of run_single_scenario is covered: the rest of the loop drives a real simulator and rendering. The simulator
is replaced by a stub whose drones move by a fixed amount per step, so the test controls exactly how much movement each step shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import tools.horizon_live_view_grid as hlvg
from drone_sim.simulation.utils.cpu_costs import CpuCostLedger
from drone_sim.domain.config import ControllerSpec, DroneConfig, PhysicsSpec, ScenarioConfig
from tools.utility.printer import COST_FIELDNAMES, Status


def _scenario() -> ScenarioConfig:
   return ScenarioConfig(
      physics=PhysicsSpec(type="linear_kinematics"),
      controller=ControllerSpec(type="mpc_agent"),
      drones=[DroneConfig(drone_id="d1", start=[0.0, 0.0, 0.0], target=[5.0, 5.0, 5.0])],
   )


class _Route:
   def __init__(self, target):
      self.target = np.asarray(target, dtype=float)

   def target_reached(self, position: np.ndarray, thresh: float = 1e-3) -> bool:
      return bool(np.linalg.norm(position - self.target) < thresh)


@dataclass
class _Drone:
   drone_id: str
   x: np.ndarray
   route: _Route
   step_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))

   def position(self) -> np.ndarray:
      return self.x


class _Simulator:
   infeasible = False
   infeasible_reason = None

   def __init__(self, drones: list[_Drone]):
      self.drones = drones
      self.step_count = 0
      self.costs = CpuCostLedger()

   def step(self) -> None:
      for d in self.drones:
         d.x = d.x + d.step_delta
      self.step_count += 1


def _drone(drone_id: str, start, target=(10.0, 10.0, 10.0), step_delta=(0.0, 0.0, 0.0)) -> _Drone:
   return _Drone(drone_id, np.asarray(start, dtype=float), _Route(target), np.asarray(step_delta, dtype=float))


def _run(monkeypatch, drones: list[_Drone], **kwargs):
   sim = _Simulator(drones)
   monkeypatch.setattr(hlvg, "Simulator", type("SimStub", (), {"from_config": staticmethod(lambda cfg: sim)}))
   status, _wall, _jerk, step_durations, _mean, _all, _frames, costs = hlvg.run_single_scenario(_scenario(), max_steps=50, trace_len=0, store_gif=False,
                                                                                       **kwargs)
   return status, len(step_durations), costs


class TestRunSingleScenarioDeadlock:
   """Tests for the deadlock exit of run_single_scenario."""

   def test_stationary_drones_end_as_deadlock_after_the_window(self, monkeypatch):
      drones = [_drone("a", (0, 0, 0)), _drone("b", (3, 0, 0))]

      status, steps, _costs = _run(monkeypatch, drones, deadlock_steps=5, deadlock_min_movement=0.01)

      assert status == Status.DEADLOCK
      assert steps == 5

   def test_crawling_drones_below_the_threshold_count_as_deadlock(self, monkeypatch):
      drones = [_drone("a", (0, 0, 0), step_delta=(0.001, 0, 0)), _drone("b", (3, 0, 0), step_delta=(0, 0.001, 0))]

      status, steps, _costs = _run(monkeypatch, drones, deadlock_steps=5, deadlock_min_movement=0.01)

      assert status == Status.DEADLOCK
      assert steps == 5

   def test_without_a_window_stationary_drones_run_to_max_steps(self, monkeypatch):
      drones = [_drone("a", (0, 0, 0)), _drone("b", (3, 0, 0))]

      status, steps, _costs = _run(monkeypatch, drones)

      assert status == Status.MAX_STEPS
      assert steps == 50

   def test_one_progressing_drone_prevents_the_deadlock(self, monkeypatch):
      drones = [_drone("a", (0, 0, 0)), _drone("b", (3, 0, 0), step_delta=(0.05, 0, 0))]

      status, steps, _costs = _run(monkeypatch, drones, deadlock_steps=5, deadlock_min_movement=0.01)

      assert status == Status.MAX_STEPS
      assert steps == 50

   def test_reaching_the_targets_wins_over_the_deadlock(self, monkeypatch):
      """Test drones that sit on their targets are finished, not deadlocked, even with the tightest window."""
      drones = [_drone("a", (0, 0, 0), target=(0, 0, 0)), _drone("b", (3, 0, 0), target=(3, 0, 0))]

      status, steps, _costs = _run(monkeypatch, drones, deadlock_steps=1, deadlock_min_movement=0.01)

      assert status == Status.FINISHED
      assert steps == 1


class TestRunSingleScenarioCosts:
   """Tests for the cost row that run_single_scenario returns."""

   def test_cost_row_has_every_column_and_books_the_runner_phases(self, monkeypatch):
      drones = [_drone("a", (0, 0, 0)), _drone("b", (3, 0, 0))]

      _status, steps, costs = _run(monkeypatch, drones, deadlock_steps=5, deadlock_min_movement=0.01)

      assert list(costs) == COST_FIELDNAMES
      assert all(costs[name] >= 0 for name in COST_FIELDNAMES)
      assert costs["cpu_total_s"] >= costs["cpu_metrics_s"] + costs["cpu_deadlock_s"]
      assert costs["admm_iterations"] == 0
