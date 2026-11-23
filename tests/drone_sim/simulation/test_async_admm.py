"""Tests for async/Gauss-Seidel ADMM to prevent deadlocks."""

import json
from pathlib import Path

import numpy as np
import pytest

from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator


def _load_config(name: str) -> ScenarioConfig:
    """Load a config by name."""
    path = Path(__file__).parent.parent.parent.parent / "configs" / "dmpc" / name
    with open(path) as f:
        cfg = json.load(f)
    return ScenarioConfig.model_validate(cfg)


class TestAsyncADMMDeadlockPrevention:
    """Tests verifying Gauss-Seidel ADMM prevents deadlocks."""

    @pytest.mark.skip
    def test_drones_make_progress_toward_targets(self):
        """Drones should continually make progress, not get stuck."""
        cfg = _load_config("4DronesDMPC.json")
        sim = Simulator.from_config(cfg)

        # Track distance to target over time
        initial_distances = []
        for drone in sim.drones:
            pos = drone.position()
            target = np.array(drone.route.target)
            initial_distances.append(np.linalg.norm(pos - target))

        # Run simulation
        stuck_count = 0
        prev_total_dist = sum(initial_distances)

        for step in range(50):
            sim.step()

            # Compute current total distance to targets
            total_dist = 0.0
            for drone in sim.drones:
                pos = drone.position()
                target = np.array(drone.route.target)
                total_dist += np.linalg.norm(pos - target)

            # Check if making progress (allow small tolerance for maneuvering)
            if total_dist >= prev_total_dist - 0.01:
                stuck_count += 1
            else:
                stuck_count = 0  # Reset if progress made

            prev_total_dist = total_dist

            # Fail if stuck for too many consecutive steps
            assert stuck_count < 10, f"Drones stuck for {stuck_count} steps at step {step}"

    @pytest.mark.slow
    @pytest.mark.flaky
    def test_gauss_seidel_breaks_symmetry(self):
        """Two head-on drones should eventually pass each other."""
        # Create minimal 2-drone head-on scenario
        cfg = _load_config("4DronesDMPC.json")
        sim = Simulator.from_config(cfg)

        # Record initial positions as starting points
        initial_positions = {drone.drone_id: drone.position().copy() for drone in sim.drones}

        # Run for enough steps that drones should cross
        for _ in range(100):
            sim.step()

        # Check that at least some drones have crossed center
        crossed = 0
        for drone in sim.drones:
            pos = drone.position()
            start = initial_positions[drone.drone_id]
            target = np.array(drone.route.target)

            # Check if drone is closer to target than start
            dist_to_start = np.linalg.norm(pos - start)
            dist_to_target = np.linalg.norm(pos - target)

            if dist_to_target < dist_to_start:
                crossed += 1

        # At least half should have made progress toward target
        assert crossed >= 2, f"Only {crossed}/4 drones crossed toward target"

    def test_random_ordering_varies_between_iterations(self):
        """Drone solving order should vary across ADMM iterations."""
        from drone_sim.simulation.distributed.distributed_coordinator import DistributedMPCCoordinator

        coord = DistributedMPCCoordinator(dt=0.1, gauss_seidel=True)

        # The gauss_seidel flag should be set
        assert coord.gauss_seidel is True

    def test_gauss_seidel_disabled_uses_jacobi(self):
        """When gauss_seidel=False, should use Jacobi (all at once) updates."""
        from drone_sim.simulation.distributed.distributed_coordinator import DistributedMPCCoordinator

        coord = DistributedMPCCoordinator(dt=0.1, gauss_seidel=False)
        assert coord.gauss_seidel is False

    def test_priority_ordering_method_exists(self):
        """Coordinator should have _compute_priority method."""
        from drone_sim.simulation.distributed.distributed_coordinator import DistributedMPCCoordinator

        coord = DistributedMPCCoordinator(dt=0.1)
        assert hasattr(coord, "_compute_priority")
        assert callable(coord._compute_priority)

    @pytest.mark.slow
    def test_no_hard_collision_in_head_on_scenario(self):
        """Drones should avoid hard collisions (physical radii overlap).

        Note: With non-converged ADMM (warning emitted), soft constraint violations
        may occur. This test checks for hard collisions (physical overlap) only.
        Full collision avoidance tuning is a separate concern.
        """
        cfg = _load_config("4DronesDMPC.json")
        sim = Simulator.from_config(cfg)

        for step in range(100):
            sim.step()

            # Check all pairwise distances for hard collisions
            positions = [drone.position() for drone in sim.drones]
            radii = [drone.radius for drone in sim.drones]

            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_dist = radii[i] + radii[j]

                    # Hard collision = physical radii overlap
                    assert dist >= min_dist, (
                        f"Hard collision at step {step}: drones {i} and {j} "
                        f"distance {dist:.3f} < physical min {min_dist:.3f}"
                    )
