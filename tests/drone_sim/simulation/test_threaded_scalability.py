"""Scalability measurement tests for threaded DMPC coordinator.

Tests for:
- Solve time vs drone count (2, 4, 8 drones)
- Threaded vs synchronous ADMM coordinator performance comparison
- All drones produce movement at scale
- Scalability trend analysis

All tests marked @pytest.mark.slow for CI separation.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

from drone_sim.domain.config import (
    ControllerSpec,
    DroneConfig,
    PhysicsSpec,
    RoomConfig,
    ScenarioConfig,
)
from drone_sim.simulation.simulator import Simulator


def make_grid_drones(n_drones: int, spacing: float = 2.0) -> list[DroneConfig]:
    """Generate drones in a grid pattern to avoid head-on scenarios.

    Each drone starts at (col*spacing, row*spacing, 0) targeting
    (col*spacing, row*spacing + 5, 0). Grid: row = i // 4, col = i % 4.

    :param n_drones: Number of drones to create
    :param spacing: Grid spacing in meters
    :return: List of DroneConfig objects
    """
    drones = []
    for i in range(n_drones):
        row = i // 4
        col = i % 4
        start = [col * spacing, row * spacing, 0.0]
        target = [col * spacing, row * spacing + 5.0, 0.0]

        drones.append(
            DroneConfig(
                drone_id=f"d{i+1}",
                start=start,
                target=target,
                safety_zone=0.5,
            )
        )

    return drones


def make_scalability_config(n_drones: int, coordinator_type: str) -> ScenarioConfig:
    """Create scenario config for scalability testing.

    :param n_drones: Number of drones (2, 4, 8)
    :param coordinator_type: "dmpc_admm" or "dmpc_threaded"
    :return: ScenarioConfig for testing
    """
    drones = make_grid_drones(n_drones)

    # Coordinator-specific params
    if coordinator_type == "dmpc_admm":
        coord_params = {"horizon": 4, "max_admm_iter": 30}
    elif coordinator_type == "dmpc_threaded":
        coord_params = {
            "horizon": 4,
            "convergence_timeout_sec": 15.0,
            "convergence_threshold": 1e-3,
        }
    else:
        raise ValueError(f"Unknown coordinator type: {coordinator_type}")

    return ScenarioConfig(
        dt=0.1,
        physics=PhysicsSpec(type="linear_kinematics"),
        controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
        coordinator=ControllerSpec(type=coordinator_type, params=coord_params),
        drones=drones,
        room=RoomConfig(min=[-10.0, -10.0, -10.0], max=[10.0, 10.0, 10.0]),
    )


@pytest.mark.slow
class TestScalabilitySolveTime:
    """Scalability tests measuring solve time vs drone count."""

    @pytest.mark.parametrize("n_drones", [2, 4, 8])
    @pytest.mark.parametrize("coordinator_type", ["dmpc_admm", "dmpc_threaded"])
    def test_scalability_solve_time(self, n_drones: int, coordinator_type: str):
        """Measure solve time vs drone count for different coordinators.

        Pattern: Warm-up step, then measure 5 steps with time.perf_counter().
        Reports mean and std of solve times for analysis.
        """
        cfg = make_scalability_config(n_drones, coordinator_type)
        sim = Simulator.from_config(cfg)

        # Warm-up step (first solve often slower due to initialization)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sim.step()

        # Measure 5 steps
        solve_times = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(5):
                start = time.perf_counter()
                sim.step()
                elapsed = time.perf_counter() - start
                solve_times.append(elapsed)

        mean_time = float(np.mean(solve_times))
        std_time = float(np.std(solve_times))

        # Print results for analysis
        print(f"{coordinator_type} {n_drones} drones: {mean_time:.4f}s +/- {std_time:.4f}s")

        # Sanity checks
        assert mean_time < 30.0, f"Solve time {mean_time:.2f}s exceeds timeout"
        assert mean_time > 0, "Solve time should be positive"

    @pytest.mark.parametrize("n_drones", [2, 4, 8])
    def test_all_drones_move_at_scale(self, n_drones: int):
        """Validate threaded coordinator produces movement for all drones at scale.

        Pattern: Run 10 steps, assert all drones moved from initial position.
        """
        cfg = make_scalability_config(n_drones, "dmpc_threaded")
        sim = Simulator.from_config(cfg)

        # Record initial positions
        initial_positions = {d.drone_id: d.position().copy() for d in sim.drones}

        # Run simulation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(10):
                sim.step()

        # Verify all drones moved
        for drone in sim.drones:
            final_pos = drone.position()
            initial_pos = initial_positions[drone.drone_id]
            dist = np.linalg.norm(final_pos - initial_pos)
            assert dist > 0.01, f"Drone {drone.drone_id} should have moved (dist={dist:.4f})"

    @pytest.mark.flaky
    def test_threaded_vs_admm_scalability_trend(self):
        """Compare threaded vs ADMM solve time scalability trends.

        Pattern: Measure mean solve time for both coordinators at 2, 4, 8 drones.
        Assert threaded is within 10x of ADMM (sanity check—threaded has overhead
        but shouldn't be dramatically worse).
        """
        drone_counts = [2, 4, 8]
        results = {}

        for coordinator_type in ["dmpc_admm", "dmpc_threaded"]:
            results[coordinator_type] = {}

            for n_drones in drone_counts:
                cfg = make_scalability_config(n_drones, coordinator_type)
                sim = Simulator.from_config(cfg)

                # Warm-up
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sim.step()

                # Measure 5 steps
                solve_times = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for _ in range(5):
                        start = time.perf_counter()
                        sim.step()
                        elapsed = time.perf_counter() - start
                        solve_times.append(elapsed)

                mean_time = float(np.mean(solve_times))
                results[coordinator_type][n_drones] = mean_time

        # Print comparison table
        print("\nScalability Comparison:")
        print("Drones | ADMM (s) | Threaded (s) | Ratio")
        print("-------|----------|--------------|-------")
        for n_drones in drone_counts:
            admm_time = results["dmpc_admm"][n_drones]
            threaded_time = results["dmpc_threaded"][n_drones]
            ratio = threaded_time / admm_time if admm_time > 0 else float('inf')
            print(f"{n_drones:6d} | {admm_time:8.4f} | {threaded_time:12.4f} | {ratio:6.2f}x")

        # Assert threaded is within 10x of ADMM (sanity check)
        for n_drones in drone_counts:
            admm_time = results["dmpc_admm"][n_drones]
            threaded_time = results["dmpc_threaded"][n_drones]
            ratio = threaded_time / admm_time if admm_time > 0 else float('inf')
            assert ratio < 10.0, (
                f"{n_drones} drones: threaded {ratio:.2f}x slower than ADMM (should be <10x)"
            )
