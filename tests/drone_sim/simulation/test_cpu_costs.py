"""Tests for drone_sim.simulation.utils.cpu_costs and its wiring into the simulator.

The ledger decides what the campaign reports as solver, constraint or setup cost, so the exclusive bookkeeping has to be exact.
The wiring test runs the real simulator for a few steps with both coordinators and checks that every phase is reached.
"""

from __future__ import annotations

import time

import pytest

from drone_sim.domain.config import ControllerSpec, DroneConfig, PhysicsSpec, RoomConfig, ScenarioConfig
from drone_sim.simulation.simulator import Simulator
from drone_sim.simulation.utils.cpu_costs import CpuCostLedger


def _burn(seconds: float) -> None:
   """Consume CPU time, not wall time, so process_time moves."""
   t0 = time.process_time()
   while time.process_time() - t0 < seconds:
      sum(range(1000))


class TestCpuCostLedger:
   """Tests for CpuCostLedger."""

   def test_books_cpu_time_and_counts_entries(self):
      ledger = CpuCostLedger()

      with ledger.phase("a"):
         _burn(0.02)
      with ledger.phase("a"):
         _burn(0.01)

      assert ledger.seconds["a"] >= 0.03
      assert ledger.counts["a"] == 2
      assert ledger.total() == pytest.approx(ledger.seconds["a"])

   def test_nested_phase_time_is_booked_only_once(self):
      """Test the outer phase keeps only its own time, so the phases add up to the measured total."""
      ledger = CpuCostLedger()
      t0 = time.process_time()

      with ledger.phase("outer"):
         _burn(0.02)
         with ledger.phase("inner"):
            _burn(0.03)
         _burn(0.01)

      elapsed = time.process_time() - t0
      assert ledger.seconds["inner"] >= 0.03
      assert ledger.seconds["outer"] >= 0.03
      assert ledger.seconds["outer"] < elapsed - 0.025
      assert ledger.total() == pytest.approx(elapsed, abs=0.005)

   def test_exception_inside_a_phase_still_books_it(self):
      ledger = CpuCostLedger()

      with pytest.raises(RuntimeError):
         with ledger.phase("a"):
            _burn(0.01)
            raise RuntimeError("boom")

      assert ledger.seconds["a"] >= 0.01
      assert ledger.counts["a"] == 1

   def test_count_adds_without_booking_time(self):
      ledger = CpuCostLedger()

      ledger.count("admm_iterations", 3)
      ledger.count("admm_iterations")

      assert ledger.counts["admm_iterations"] == 4
      assert ledger.total() == 0.0

   def test_phase_objects_are_reused(self):
      ledger = CpuCostLedger()

      assert ledger.phase("a") is ledger.phase("a")
      assert ledger.phase("a") is not ledger.phase("b")

   def test_inclusive_timings_stay_apart_from_the_phases(self):
      ledger = CpuCostLedger()

      ledger.add_seconds("local_solve", 0.5)
      ledger.add_seconds("local_solve", 0.25)

      assert ledger.timings["local_solve"] == pytest.approx(0.75)
      assert ledger.total() == 0.0
      assert "local_solve" not in ledger.seconds

   def test_record_max_keeps_the_largest_value(self):
      ledger = CpuCostLedger()

      ledger.record_max("max_admm_iter_step", 2)
      ledger.record_max("max_admm_iter_step", 5)
      ledger.record_max("max_admm_iter_step", 3)

      assert ledger.maxima["max_admm_iter_step"] == 5


def _scenario(coordinator: str, horizon: int = 3) -> ScenarioConfig:
   return ScenarioConfig(
      dt=0.1,
      physics=PhysicsSpec(type="linear_kinematics", params={"v_max": 2.0, "u_min": [-3.0] * 3, "u_max": [3.0] * 3}),
      controller=ControllerSpec(type="mpc_agent", params={"horizon": horizon}),
      coordinator=ControllerSpec(type=coordinator, params={"horizon": horizon, **({"max_admm_iter": 5} if coordinator == "dmpc_admm" else {})}),
      room=RoomConfig(min=[0.0, 0.0, 0.0], max=[8.0, 8.0, 8.0]),
      drones=[
         DroneConfig(drone_id="d1", start=[1.0, 1.0, 2.0], target=[6.0, 6.0, 2.0], radius=0.3, safety_zone=0.8),
         DroneConfig(drone_id="d2", start=[6.0, 1.0, 2.0], target=[1.0, 6.0, 2.0], radius=0.3, safety_zone=0.8),
      ],
   )


class TestSimulatorCostWiring:
   """Tests for the shared ledger between simulator, coordinator and solvers."""

   def test_simulator_and_coordinator_share_one_ledger(self):
      sim = Simulator.from_config(_scenario("mpc_central"))

      assert sim.costs is sim.coordinator.costs
      assert sim.costs is sim.coordinator._solver.costs

   @pytest.mark.parametrize("coordinator, extra", [("mpc_central", set()), ("dmpc_admm", {"exchange", "admm_update"})])
   def test_every_phase_is_reached_after_a_few_steps(self, coordinator, extra):
      sim = Simulator.from_config(_scenario(coordinator))

      for _ in range(3):
         sim.step()

      booked = {name for name, seconds in sim.costs.seconds.items() if seconds >= 0.0}
      assert {"setup", "cost", "constraints", "solver", "validate", "physics"} | extra <= booked
      assert sim.costs.counts["cost"] > 0
      assert sim.costs.counts["constraints"] > 0
      # Simulator.step enters the physics phase three times: before, between and after the coordinator call.
      assert sim.costs.counts["physics"] == 3 * 3
      if coordinator == "dmpc_admm":
         assert sim.costs.counts["admm_iterations"] >= 3
      assert sim.costs._stack == []

   @pytest.mark.parametrize("gauss_seidel", [True, False])
   def test_dmpc_books_every_local_solve_and_the_slowest_per_round(self, gauss_seidel):
      """Test the per-drone columns: one local solve per drone and round, and the fleet step time never exceeds the summed solves."""
      sim = Simulator.from_config(_scenario("dmpc_admm"))
      sim.coordinator.gauss_seidel = gauss_seidel

      for _ in range(3):
         sim.step()

      costs = sim.costs
      rounds = costs.counts["admm_iterations"]
      assert costs.counts["local_solves"] == 2 * rounds
      assert costs.timings["local_solve"] > 0.0
      assert 0.0 < costs.timings["parallel_step"] <= costs.timings["local_solve"]
      assert 1 <= costs.maxima["max_admm_iter_step"] <= 5
      assert costs.maxima["max_admm_iter_step"] <= rounds

   def test_central_mpc_has_no_local_solves(self):
      sim = Simulator.from_config(_scenario("mpc_central"))

      for _ in range(2):
         sim.step()

      assert "local_solves" not in sim.costs.counts
      assert sim.costs.timings == {}
      assert sim.costs.maxima == {}
