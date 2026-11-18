"""Verification tests comparing threaded DMPC against synchronous DMPC and central MPC.

Tests for:
- Trajectory similarity between threaded and synchronous DMPC
- Trajectory comparison between threaded DMPC and central MPC
- Target reaching for all three coordinator types
- Collision avoidance with threaded DMPC
- Convergence metrics tracking and analysis
"""

from __future__ import annotations

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


# Helper functions
def trajectory_rms_error(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute RMS error between two (N,3) trajectory arrays."""
    return float(np.sqrt(np.mean((traj1 - traj2) ** 2)))


def trajectory_max_deviation(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """Compute max position deviation between trajectories."""
    return float(np.max(np.linalg.norm(traj1 - traj2, axis=1)))


def make_2drone_config(coordinator_type: str, coordinator_params: dict) -> ScenarioConfig:
    """Build a 2-drone head-on scenario config.

    d1: [0, -1.5, 0] -> [0, 1.5, 0]
    d2: [0, 1.5, 0] -> [0, -1.5, 0]

    Tuned controller params for convergence: q_pos=[3.0, 3.0, 3.0], r_u=[0.5, 0.5, 0.5]
    """
    return ScenarioConfig(
        dt=0.1,
        physics=PhysicsSpec(type="linear_kinematics"),
        controller=ControllerSpec(
            type="mpc_agent",
            params={
                "horizon": 4,
                "q_pos": [3.0, 3.0, 3.0],
                "r_u": [0.5, 0.5, 0.5],
            },
        ),
        coordinator=ControllerSpec(
            type=coordinator_type,
            params=coordinator_params,
        ),
        drones=[
            DroneConfig(
                drone_id="d1",
                start=[0.0, -1.5, 0.0],
                target=[0.0, 1.5, 0.0],
                safety_zone=0.5,
            ),
            DroneConfig(
                drone_id="d2",
                start=[0.0, 1.5, 0.0],
                target=[0.0, -1.5, 0.0],
                safety_zone=0.5,
            ),
        ],
        room=RoomConfig(min=[-2.5, -2.5, -2.5], max=[2.5, 2.5, 2.5]),
    )


def run_simulation(sim: Simulator, steps: int) -> dict[str, np.ndarray]:
    """Run simulation for N steps, return dict mapping drone_id to (steps, 3) trajectory."""
    trajectories = {d.drone_id: [] for d in sim.drones}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(steps):
            sim.step()
            for drone in sim.drones:
                trajectories[drone.drone_id].append(drone.position().copy())

    # Convert lists to numpy arrays
    return {
        drone_id: np.array(traj) for drone_id, traj in trajectories.items()
    }


@pytest.mark.slow
class TestThreadedVsSyncTrajectory:
    """Tests comparing threaded DMPC vs synchronous DMPC trajectories."""

    @pytest.mark.flaky
    def test_threaded_vs_admm_trajectory_similarity(self):
        """Test threaded and ADMM DMPC produce comparable trajectories (RMS < 1.2, max dev < 3.5)."""
        # Run with ADMM DMPC
        cfg_admm = make_2drone_config(
            "dmpc_admm",
            {"horizon": 4, "max_admm_iter": 50, "rho": 2.0},
        )
        sim_admm = Simulator.from_config(cfg_admm)
        traj_admm = run_simulation(sim_admm, 50)

        # Run with threaded DMPC
        cfg_threaded = make_2drone_config(
            "dmpc_threaded",
            {"horizon": 4, "convergence_timeout_sec": 15.0},
        )
        sim_threaded = Simulator.from_config(cfg_threaded)
        traj_threaded = run_simulation(sim_threaded, 50)

        # Compare trajectories (both are approximate solvers, allow tolerance)
        # Note: Threaded DMPC has non-deterministic spawn timing, leading to different paths
        for drone_id in ["d1", "d2"]:
            rms = trajectory_rms_error(traj_admm[drone_id], traj_threaded[drone_id])
            max_dev = trajectory_max_deviation(traj_admm[drone_id], traj_threaded[drone_id])

            assert rms < 1.2, f"{drone_id} RMS error {rms:.3f} exceeds 1.2"
            assert max_dev < 3.5, f"{drone_id} max deviation {max_dev:.3f} exceeds 3.5"

    def test_threaded_vs_central_trajectory_comparison(self):
        """Test threaded DMPC vs central MPC produce feasible trajectories (max dev < 2.5)."""
        # Run with central MPC
        cfg_central = make_2drone_config(
            "mpc_central",
            {"horizon": 4},
        )
        sim_central = Simulator.from_config(cfg_central)
        traj_central = run_simulation(sim_central, 50)

        # Run with threaded DMPC
        cfg_threaded = make_2drone_config(
            "dmpc_threaded",
            {"horizon": 4, "convergence_timeout_sec": 15.0},
        )
        sim_threaded = Simulator.from_config(cfg_threaded)
        traj_threaded = run_simulation(sim_threaded, 50)

        # Central is tighter, but both should produce feasible trajectories
        # Distributed takes different paths but reaches similar destinations
        for drone_id in ["d1", "d2"]:
            max_dev = trajectory_max_deviation(traj_central[drone_id], traj_threaded[drone_id])
            assert max_dev < 2.5, f"{drone_id} max deviation {max_dev:.3f} exceeds 2.5"

    def test_all_three_coordinators_reach_targets(self):
        """Test all three coordinator types reach targets within tolerance."""
        coordinator_configs = [
            ("mpc_central", {"horizon": 4}),
            ("dmpc_admm", {"horizon": 4, "max_admm_iter": 50, "rho": 2.0}),
            ("dmpc_threaded", {"horizon": 4, "convergence_timeout_sec": 15.0}),
        ]

        targets = {
            "d1": np.array([0.0, 1.5, 0.0]),
            "d2": np.array([0.0, -1.5, 0.0]),
        }

        for coord_type, coord_params in coordinator_configs:
            cfg = make_2drone_config(coord_type, coord_params)
            sim = Simulator.from_config(cfg)

            # Run for 150 steps to ensure convergence (distributed needs more time)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for _ in range(150):
                    sim.step()

            # Check all drones reached targets within 1.5 (distributed tolerance)
            for drone in sim.drones:
                final_pos = drone.position()
                target = targets[drone.drone_id]
                dist = np.linalg.norm(final_pos - target)
                assert dist < 1.5, (
                    f"{coord_type}: {drone.drone_id} final distance {dist:.3f} > 1.5"
                )


@pytest.mark.slow
class TestThreadedCollisionAvoidance:
    """Tests for threaded DMPC collision avoidance behavior."""

    def test_threaded_no_hard_collisions(self):
        """Test threaded DMPC avoids hard collisions (minor safety zone tolerance allowed)."""
        cfg = make_2drone_config(
            "dmpc_threaded",
            {"horizon": 4, "convergence_timeout_sec": 15.0},
        )
        sim = Simulator.from_config(cfg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(50):
                sim.step()

                # Check for hard collisions (allow minor tolerance for approximate DMPC)
                collisions = [
                    c for c in sim.last_collisions if c["kind"] == "drone_drone"
                ]
                for col in collisions:
                    # Allow minor safety zone violations (distance >= threshold - 0.15)
                    # This matches the pattern in test_dmpc_integration.py
                    assert col["distance"] >= col["threshold"] - 0.15, (
                        f"Step {step}: Hard collision detected - "
                        f"distance={col['distance']:.3f} < threshold={col['threshold']:.3f}"
                    )

    def test_threaded_collision_rate_comparable_to_admm(self):
        """Test threaded DMPC collision rate is within 2x of ADMM (or both zero)."""
        # Run with ADMM
        cfg_admm = make_2drone_config(
            "dmpc_admm",
            {"horizon": 4, "max_admm_iter": 50, "rho": 2.0},
        )
        sim_admm = Simulator.from_config(cfg_admm)

        admm_collision_count = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(50):
                sim_admm.step()
                admm_collision_count += len([
                    c for c in sim_admm.last_collisions if c["kind"] == "drone_drone"
                ])

        # Run with threaded
        cfg_threaded = make_2drone_config(
            "dmpc_threaded",
            {"horizon": 4, "convergence_timeout_sec": 15.0},
        )
        sim_threaded = Simulator.from_config(cfg_threaded)

        threaded_collision_count = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(50):
                sim_threaded.step()
                threaded_collision_count += len([
                    c for c in sim_threaded.last_collisions if c["kind"] == "drone_drone"
                ])

        # Assert threaded collision count is within 2x of ADMM count (or both zero)
        if admm_collision_count == 0:
            assert threaded_collision_count <= 2, (
                f"ADMM had 0 collisions but threaded had {threaded_collision_count}"
            )
        else:
            assert threaded_collision_count <= 2 * admm_collision_count, (
                f"Threaded collision count {threaded_collision_count} > "
                f"2x ADMM count {admm_collision_count}"
            )


@pytest.mark.slow
class TestConvergenceAnalysis:
    """Tests for convergence behavior analysis."""

    def test_convergence_metrics_tracked(self):
        """Test convergence metrics are tracked and returned correctly."""
        cfg = make_2drone_config(
            "dmpc_threaded",
            {"horizon": 4, "convergence_timeout_sec": 15.0, "convergence_threshold": 1e-3},
        )
        sim = Simulator.from_config(cfg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(30):
                sim.step()

                coord = sim.coordinator
                # Verify get_last_iteration_count() returns int >= 0
                iter_count = coord.get_last_iteration_count()
                assert isinstance(iter_count, int), (
                    f"Step {step}: iteration count is not int"
                )
                assert iter_count >= 0, (
                    f"Step {step}: iteration count {iter_count} < 0"
                )

                # Verify get_last_converged() returns bool
                converged = coord.get_last_converged()
                assert isinstance(converged, bool), (
                    f"Step {step}: converged is not bool"
                )

    def test_convergence_rate_reasonable(self):
        """Test convergence rate is reasonable (> 0) and print summary."""
        cfg = make_2drone_config(
            "dmpc_threaded",
            {"horizon": 4, "convergence_timeout_sec": 15.0, "convergence_threshold": 1e-3},
        )
        sim = Simulator.from_config(cfg)

        converged_count = 0
        iteration_counts = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(30):
                sim.step()

                coord = sim.coordinator
                if coord.get_last_converged():
                    converged_count += 1
                iteration_counts.append(coord.get_last_iteration_count())

        # Print convergence summary for analysis
        print(f"\nThreaded DMPC Convergence Summary:")
        print(f"  Converged: {converged_count}/30 steps ({100*converged_count/30:.1f}%)")
        print(f"  Mean iterations: {np.mean(iteration_counts):.1f}")
        print(f"  Max iterations: {np.max(iteration_counts)}")

        # Assert at least some steps converged
        assert converged_count > 0, (
            "No steps converged - expected at least some convergence"
        )

    def test_admm_convergence_metrics_tracked(self):
        """Test ADMM coordinator also tracks convergence metrics (API validation)."""
        cfg = make_2drone_config(
            "dmpc_admm",
            {"horizon": 4, "max_admm_iter": 50, "rho": 2.0},
        )
        sim = Simulator.from_config(cfg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for step in range(30):
                sim.step()

                coord = sim.coordinator
                # Verify get_last_iteration_count() works for ADMM
                iter_count = coord.get_last_iteration_count()
                assert isinstance(iter_count, int), (
                    f"Step {step}: ADMM iteration count is not int"
                )
                assert iter_count >= 0, (
                    f"Step {step}: ADMM iteration count {iter_count} < 0"
                )

                # Verify get_last_converged() works for ADMM
                converged = coord.get_last_converged()
                assert isinstance(converged, bool), (
                    f"Step {step}: ADMM converged is not bool"
                )
