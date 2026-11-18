"""Integration tests for DMPC simulator integration.

Tests for:
- Loading DMPC config via Simulator.from_config
- Running simulation with DMPC coordinator
- Comparing DMPC vs central MPC behavior
- Backward compatibility with central MPC
"""

from __future__ import annotations

import json
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
from drone_sim.simulation.distributed.distributed_coordinator import DistributedMPCCoordinator
from drone_sim.simulation.simulator import Simulator


class TestSimulatorLoadsDMPCConfig:
    """Tests for loading DMPC configs via Simulator.from_config."""

    def test_simulator_loads_dmpc_config_from_file(self):
        """Test Simulator.from_config loads 4DronesDMPC.json correctly."""
        with open("configs/dmpc/4DronesDMPC.json") as f:
            cfg_dict = json.load(f)

        cfg = ScenarioConfig(**cfg_dict)
        sim = Simulator.from_config(cfg)

        assert isinstance(sim, Simulator)
        assert isinstance(sim.coordinator, DistributedMPCCoordinator)
        assert len(sim.drones) == 4

    def test_simulator_dmpc_coordinator_params_set(self):
        """Test DMPC coordinator receives correct parameters."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={
                    "horizon": 4,
                    "rho": 2.0,
                    "max_admm_iter": 30,
                    "primal_tol": 1e-4,
                    "dual_tol": 1e-4,
                },
            ),
            drones=[
                DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 0, 5]),
                DroneConfig(drone_id="d2", start=[5, 0, 5], target=[0, 0, 5]),
            ],
            room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10]),
        )

        sim = Simulator.from_config(cfg)
        coord = sim.coordinator

        assert isinstance(coord, DistributedMPCCoordinator)
        assert coord.dt == 0.1
        assert coord.horizon == 4
        assert coord.rho == 2.0
        assert coord.max_admm_iter == 30
        assert coord.primal_tol == 1e-4
        assert coord.dual_tol == 1e-4

    def test_simulator_passes_comm_radius_to_dmpc(self):
        """Test comm_radius is passed from config to DMPC coordinator."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(type="dmpc_admm", params={"horizon": 4}),
            comm_radius=5.0,
            drones=[
                DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 0, 5]),
                DroneConfig(drone_id="d2", start=[10, 0, 5], target=[15, 0, 5]),
            ],
            room=RoomConfig(min=[-20, -20, 0], max=[20, 20, 20]),
        )

        sim = Simulator.from_config(cfg)
        coord = sim.coordinator

        assert isinstance(coord, DistributedMPCCoordinator)
        assert coord.comm_radius == 5.0

@pytest.mark.slow
class TestSimulatorStepWithDMPC:
    """Tests for running simulation steps with DMPC coordinator."""

    def test_simulator_step_with_dmpc_no_error(self):
        """Test simulation step with DMPC runs without errors."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={"horizon": 4, "max_admm_iter": 20},
            ),
            drones=[
                DroneConfig(
                    drone_id="d1",
                    start=[-2.0, 0.0, 5.0],
                    target=[2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
                DroneConfig(
                    drone_id="d2",
                    start=[2.0, 0.0, 5.0],
                    target=[-2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
            ],
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim = Simulator.from_config(cfg)

        # Run 10 steps - should not raise RuntimeError
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(10):
                sim.step()

        # Verify simulation progressed
        assert sim.step_count == 10
        assert sim.t == pytest.approx(1.0)

    def test_simulator_step_with_dmpc_drones_move(self):
        """Test drones move during DMPC simulation."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={"horizon": 4, "max_admm_iter": 30},
            ),
            drones=[
                DroneConfig(
                    drone_id="d1",
                    start=[-2.0, 0.0, 5.0],
                    target=[2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
                DroneConfig(
                    drone_id="d2",
                    start=[2.0, 0.0, 5.0],
                    target=[-2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
            ],
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim = Simulator.from_config(cfg)
        initial_positions = [d.position().copy() for d in sim.drones]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(10):
                sim.step()

        # Verify drones moved toward targets
        final_positions = [d.position().copy() for d in sim.drones]

        for i, (init, final) in enumerate(zip(initial_positions, final_positions)):
            dist = np.linalg.norm(final - init)
            assert dist > 0.1, f"Drone {i} should have moved"

    def test_simulator_step_with_dmpc_no_collisions(self):
        """Test DMPC maintains collision avoidance."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={"horizon": 4, "max_admm_iter": 50, "rho": 2.0},
            ),
            drones=[
                DroneConfig(
                    drone_id="d1",
                    start=[-2.0, 0.0, 5.0],
                    target=[2.0, 0.0, 5.0],
                    radius=0.2,
                    safety_zone=0.8,
                ),
                DroneConfig(
                    drone_id="d2",
                    start=[2.0, 0.0, 5.0],
                    target=[-2.0, 0.0, 5.0],
                    radius=0.2,
                    safety_zone=0.8,
                ),
            ],
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim = Simulator.from_config(cfg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(30):
                sim.step()

                # Check no safety zone violations
                collisions = [
                    c for c in sim.last_collisions if c["kind"] == "drone_drone"
                ]
                # Minor tolerance for numerical precision
                for col in collisions:
                    assert col["distance"] >= col["threshold"] - 0.1, (
                        f"Safety zone violation: distance={col['distance']:.3f} < "
                        f"threshold={col['threshold']:.3f}"
                    )

@pytest.mark.slow
class TestDMPCVsCentralComparison:
    """Tests comparing DMPC and central MPC behavior."""

    def test_both_produce_feasible_trajectories(self):
        """Test both coordinators produce trajectories without collisions."""
        base_drones = [
            DroneConfig(
                drone_id="d1",
                start=[-1.5, 0.0, 5.0],
                target=[1.5, 0.0, 5.0],
                radius=0.2,
                safety_zone=0.6,
            ),
            DroneConfig(
                drone_id="d2",
                start=[1.5, 0.0, 5.0],
                target=[-1.5, 0.0, 5.0],
                radius=0.2,
                safety_zone=0.6,
            ),
        ]

        # Central MPC config
        cfg_central = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(type="mpc_central", params={"horizon": 4}),
            drones=base_drones,
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        # DMPC config
        cfg_dmpc = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={"horizon": 4, "max_admm_iter": 50},
            ),
            drones=base_drones,
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim_central = Simulator.from_config(cfg_central)
        sim_dmpc = Simulator.from_config(cfg_dmpc)

        steps = 50
        central_collisions = 0
        dmpc_collisions = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(steps):
                sim_central.step()
                sim_dmpc.step()

                central_collisions += len(
                    [c for c in sim_central.last_collisions if c["kind"] == "drone_drone"]
                )
                dmpc_collisions += len(
                    [c for c in sim_dmpc.last_collisions if c["kind"] == "drone_drone"]
                )

        # Both should have minimal or no collisions
        # (allow some tolerance for DMPC as it's approximate)
        assert central_collisions <= 5, f"Central MPC had {central_collisions} collisions"
        assert dmpc_collisions <= 10, f"DMPC had {dmpc_collisions} collisions"

    def test_both_reach_targets(self):
        """Test both coordinators get drones to their targets."""
        base_drones = [
            DroneConfig(
                drone_id="d1",
                start=[-1.5, 0.0, 5.0],
                target=[1.5, 0.0, 5.0],
                radius=0.2,
                safety_zone=0.6,
            ),
            DroneConfig(
                drone_id="d2",
                start=[1.5, 0.0, 5.0],
                target=[-1.5, 0.0, 5.0],
                radius=0.2,
                safety_zone=0.6,
            ),
        ]

        cfg_central = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(type="mpc_central", params={"horizon": 4}),
            drones=base_drones,
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        cfg_dmpc = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={"horizon": 4, "max_admm_iter": 50},
            ),
            drones=base_drones,
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim_central = Simulator.from_config(cfg_central)
        sim_dmpc = Simulator.from_config(cfg_dmpc)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(120): # distributed needs some more steps
                sim_central.step()
                sim_dmpc.step()

        # Check final positions are near targets (within 1.0 unit)
        targets = [np.array([1.5, 0.0, 5.0]), np.array([-1.5, 0.0, 5.0])]
        tolerance = 1.0

        for i, target in enumerate(targets):
            central_dist = np.linalg.norm(sim_central.drones[i].position() - target)
            dmpc_dist = np.linalg.norm(sim_dmpc.drones[i].position() - target)

            assert central_dist < tolerance, (
                f"Central MPC drone {i} final distance {central_dist:.2f} > {tolerance}"
            )
            assert dmpc_dist < tolerance, (
                f"DMPC drone {i} final distance {dmpc_dist:.2f} > {tolerance}"
            )

@pytest.mark.slow
class TestDMPCWithCommRadius:
    """Tests for DMPC with limited communication radius."""

    def test_dmpc_with_comm_radius_runs(self):
        """Test DMPC runs with limited communication radius."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={"horizon": 4, "max_admm_iter": 30},
            ),
            comm_radius=3.0,
            drones=[
                DroneConfig(
                    drone_id="d1",
                    start=[-2.0, 0.0, 5.0],
                    target=[2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
                DroneConfig(
                    drone_id="d2",
                    start=[2.0, 0.0, 5.0],
                    target=[-2.0, 0.0, 5.0],
                    safety_zone=0.5,
                ),
            ],
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim = Simulator.from_config(cfg)

        assert sim.coordinator.comm_radius == 3.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(20):
                sim.step()

        assert sim.step_count == 20

    @pytest.mark.flaky
    def test_dmpc_with_large_comm_radius_same_as_none(self):
        """Test large comm_radius behaves like comm_radius=None (all neighbors)."""
        base_drones = [
            DroneConfig(
                drone_id="d1",
                start=[-1.0, 0.0, 5.0],
                target=[1.0, 0.0, 5.0],
                safety_zone=0.5,
            ),
            DroneConfig(
                drone_id="d2",
                start=[1.0, 0.0, 5.0],
                target=[-1.0, 0.0, 5.0],
                safety_zone=0.5,
            ),
        ]

        cfg_none = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={"horizon": 4, "max_admm_iter": 30},
            ),
            comm_radius=None,
            drones=base_drones,
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        cfg_large = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(
                type="dmpc_admm",
                params={"horizon": 4, "max_admm_iter": 30},
            ),
            comm_radius=100.0,  # Much larger than any drone separation
            drones=base_drones,
            room=RoomConfig(min=[-5, -5, 0], max=[5, 5, 10]),
        )

        sim_none = Simulator.from_config(cfg_none)
        sim_large = Simulator.from_config(cfg_large)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(20):
                sim_none.step()
                sim_large.step()

        # Final positions should be similar (not identical due to solver variations)
        for i in range(2):
            pos_none = sim_none.drones[i].position()
            pos_large = sim_large.drones[i].position()
            dist = np.linalg.norm(pos_none - pos_large)
            assert dist < 1.5, f"Drone {i} positions differ by {dist:.3f}"


class TestBackwardCompatibility:
    """Tests ensuring central MPC still works after DMPC integration."""

    def test_central_mpc_still_loads(self):
        """Test 4DronesHorizon4.json (central MPC) still loads correctly."""
        with open("configs/drones10020139/4DronesHorizon4.json") as f:
            cfg_dict = json.load(f)

        cfg = ScenarioConfig(**cfg_dict)
        sim = Simulator.from_config(cfg)

        assert isinstance(sim, Simulator)
        assert sim.coordinator is not None
        # Should NOT be DistributedMPCCoordinator
        assert not isinstance(sim.coordinator, DistributedMPCCoordinator)
        assert len(sim.drones) == 4

    def test_central_mpc_still_runs(self):
        """Test simulation with central MPC still runs correctly."""
        with open("configs/drones10020139/4DronesHorizon4.json") as f:
            cfg_dict = json.load(f)

        cfg = ScenarioConfig(**cfg_dict)
        sim = Simulator.from_config(cfg)

        for _ in range(10):
            sim.step()

        assert sim.step_count == 10

    def test_central_mpc_config_programmatic(self):
        """Test central MPC config created programmatically still works."""
        cfg = ScenarioConfig(
            dt=0.1,
            physics=PhysicsSpec(type="linear_kinematics"),
            controller=ControllerSpec(type="mpc_agent", params={"horizon": 4}),
            coordinator=ControllerSpec(type="mpc_central", params={"horizon": 4}),
            drones=[
                DroneConfig(drone_id="d1", start=[0, 0, 5], target=[5, 0, 5]),
                DroneConfig(drone_id="d2", start=[5, 0, 5], target=[0, 0, 5]),
            ],
            room=RoomConfig(min=[-10, -10, 0], max=[10, 10, 10]),
        )

        sim = Simulator.from_config(cfg)

        for _ in range(10):
            sim.step()

        assert sim.step_count == 10
        assert not isinstance(sim.coordinator, DistributedMPCCoordinator)
