"""Unit tests for ConflictDetector — TTC, closing velocity gate, hysteresis, evasion waypoints.

Test classes:
- TestTTCDetection: validates TTC-based conflict detection and closing velocity gate
- TestHysteresis: validates HYSTERESIS_STEPS entry/exit logic
- TestEvasionWaypoint: validates z-reflection logic and offset distance
- TestMultiNeighbor: validates multi-conflict handling

Sign convention note: ``closing = dot(pos_j - pos_i, vel_j - vel_i) / |pos_j - pos_i|``.
A *negative* value means the drones are approaching each other (gap is shrinking); the
detector returns conflict when ``closing < 0``.
"""
from __future__ import annotations

import numpy as np
import pytest

from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.domain.drone import Drone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics
from drone_sim.simulation.utils.conflict_detection import (
    EVASION_OFFSET_DISTANCE,
    HYSTERESIS_STEPS,
    TTC_THRESHOLD_FACTOR,
    VZ_ZERO_THRESHOLD,
    ConflictDetector,
)
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_drone(
    drone_id: str,
    x: np.ndarray,
    target: np.ndarray | None = None,
    radius: float = 0.1,
    safety_zone: float = 0.5,
    cons_stop: float = 0.0,
    v_max: float = 5.0,
    dt: float = 0.1,
    alpha: float | None = 0.3,  # adaptive by default for conflict detection tests
) -> Drone:
    """Create a minimal Drone for testing. Alpha=0.3 → is_adaptive=True by default."""
    if target is None:
        target = np.zeros(3, dtype=float)
    physics = LinearKinematicsPhysics(dt=dt, v_max=v_max)
    controller = CentralMPCAgent(dt=dt, horizon=5)
    return Drone(
        drone_id=drone_id,
        radius=radius,
        safety_zone=safety_zone,
        cons_stop=cons_stop,
        color="tab:blue",
        safety_color="tab:cyan",
        trace_color="tab:blue",
        controller=controller,
        physics=physics,
        x=np.asarray(x, dtype=float).reshape(6),
        route=Route(start=np.asarray(x, dtype=float).reshape(6)[:3], waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
        alpha=alpha,
    )


def _make_head_on_trajs(
    pos_i: float,
    vel_i: float,
    pos_j: float,
    vel_j: float,
    horizon: int = 10,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build linear predicted trajectories for two drones moving along the x-axis.

    Returns:
        traj_i: shape (H, 3)
        traj_j: shape (H, 3)
    """
    steps = np.arange(1, horizon + 1) * dt
    traj_i = np.stack(
        [pos_i + vel_i * steps, np.zeros(horizon), np.zeros(horizon)], axis=1
    )
    traj_j = np.stack(
        [pos_j + vel_j * steps, np.zeros(horizon), np.zeros(horizon)], axis=1
    )
    return traj_i, traj_j


def _all_neighbors_graph(drone_ids: list[str]) -> NeighborGraph:
    """Build a NeighborGraph where all drones are neighbours (comm_radius=None)."""
    graph = NeighborGraph(comm_radius=None)
    positions = {did: np.zeros(3) for did in drone_ids}
    graph.update(positions)
    return graph


def _run_detector_to_evading(
    drones: list[Drone],
    trajs: dict[str, np.ndarray],
    graph: NeighborGraph,
) -> ConflictDetector:
    """Run detector for HYSTERESIS_STEPS steps and return it."""
    detector = ConflictDetector()
    for _ in range(HYSTERESIS_STEPS):
        detector.update(drones, trajs, graph)
    return detector


# ---------------------------------------------------------------------------
# TestTTCDetection
# ---------------------------------------------------------------------------


class TestTTCDetection:
    """Validate TTC-based conflict detection and closing velocity gate."""

    def test_head_on_pair_flagged_as_conflict(self):
        """Two drones heading directly at each other should be flagged as conflict.

        Sign convention: rel_pos = pos_j - pos_i; rel_vel = vel_j - vel_i.
        When approaching, rel_vel is anti-parallel to rel_pos → dot < 0.
        The detector flags conflict when closing < 0.
        """
        # drone_i at x=-2 moving right (+1), drone_j at x=+2 moving left (-1)
        drone_i = _make_drone("d1", x=[-2, 0, 0, 1.0, 0, 0])
        drone_j = _make_drone("d2", x=[2, 0, 0, -1.0, 0, 0])

        # Verify trajectories converge within TTC threshold
        traj_i, traj_j = _make_head_on_trajs(
            pos_i=-2.0, vel_i=1.0, pos_j=2.0, vel_j=-1.0, horizon=20
        )
        assert traj_i.shape == (20, 3)
        assert traj_j.shape == (20, 3)

        dists = np.linalg.norm(traj_i - traj_j, axis=1)
        threshold = TTC_THRESHOLD_FACTOR * (drone_i.safety_zone + drone_j.safety_zone)
        assert dists.min() < threshold, "Trajectories must converge within TTC threshold"

        # Verify closing velocity is negative (approaching)
        rel_pos = drone_j.position() - drone_i.position()  # [4, 0, 0]
        rel_vel = drone_j.velocity() - drone_i.velocity()  # [-2, 0, 0]
        closing = float(np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) + 1e-9))
        assert closing < 0, (
            "Head-on approach: dot(rel_pos, rel_vel) < 0 — rel_vel anti-parallel to rel_pos"
        )

        # Run detector — at HYSTERESIS_STEPS consecutive conflict steps, both enter evading
        graph = _all_neighbors_graph(["d1", "d2"])
        trajs = {"d1": traj_i, "d2": traj_j}
        detector = _run_detector_to_evading([drone_i, drone_j], trajs, graph)

        assert detector.is_evading("d1") or detector.is_evading("d2"), (
            "At least one drone should be evading after consecutive conflicts"
        )

    def test_diverging_pair_not_flagged(self):
        """Two drones moving away from each other must NOT be flagged as conflict.

        The closing velocity gate blocks conflict even if predicted trajectories
        passed through close range in the past.

        Sign convention: when diverging, rel_vel is parallel to rel_pos → dot > 0 → no conflict.
        """
        # drone_i at x=0 moving left (-1), drone_j at x=0.5 moving right (+1) — moving apart
        drone_i = _make_drone("d1", x=[0.0, 0, 0, -1.0, 0, 0])
        drone_j = _make_drone("d2", x=[0.5, 0, 0, 1.0, 0, 0])

        traj_i, traj_j = _make_head_on_trajs(
            pos_i=0.0, vel_i=-1.0, pos_j=0.5, vel_j=1.0, horizon=20
        )

        # Verify closing velocity is positive (diverging)
        rel_pos = drone_j.position() - drone_i.position()  # [0.5, 0, 0]
        rel_vel = drone_j.velocity() - drone_i.velocity()  # [2, 0, 0]
        closing = float(np.dot(rel_pos, rel_vel) / (np.linalg.norm(rel_pos) + 1e-9))
        assert closing > 0, (
            "Diverging pair: dot(rel_pos, rel_vel) > 0 — rel_vel parallel to rel_pos"
        )

        graph = _all_neighbors_graph(["d1", "d2"])
        trajs = {"d1": traj_i, "d2": traj_j}

        # Many steps — closing velocity gate should always block conflict
        detector = ConflictDetector()
        for _ in range(HYSTERESIS_STEPS + 5):
            detector.update([drone_i, drone_j], trajs, graph)

        assert not detector.is_evading("d1"), "Diverging drone d1 must NOT be evading"
        assert not detector.is_evading("d2"), "Diverging drone d2 must NOT be evading"

    def test_safe_distance_no_conflict(self):
        """Two drones at rest, far apart — no conflict detected."""
        drone_i = _make_drone("d1", x=[-10, 0, 0, 0, 0, 0])
        drone_j = _make_drone("d2", x=[10, 0, 0, 0, 0, 0])

        # Stationary trajectories, well beyond TTC threshold
        traj_i = np.tile([-10.0, 0, 0], (10, 1))
        traj_j = np.tile([10.0, 0, 0], (10, 1))

        graph = _all_neighbors_graph(["d1", "d2"])
        trajs = {"d1": traj_i, "d2": traj_j}

        detector = ConflictDetector()
        for _ in range(HYSTERESIS_STEPS + 2):
            detector.update([drone_i, drone_j], trajs, graph)

        assert not detector.is_evading("d1"), "Far-apart resting drone d1 must NOT be evading"
        assert not detector.is_evading("d2"), "Far-apart resting drone d2 must NOT be evading"

    def test_non_adaptive_drones_ignored(self):
        """Drones with alpha=None (not adaptive) must never be detected as conflicting."""
        drone_i = _make_drone("d1", x=[-2, 0, 0, 1.0, 0, 0], alpha=None)
        drone_j = _make_drone("d2", x=[2, 0, 0, -1.0, 0, 0], alpha=None)

        traj_i, traj_j = _make_head_on_trajs(
            pos_i=-2.0, vel_i=1.0, pos_j=2.0, vel_j=-1.0, horizon=20
        )

        graph = _all_neighbors_graph(["d1", "d2"])
        trajs = {"d1": traj_i, "d2": traj_j}

        detector = ConflictDetector()
        for _ in range(HYSTERESIS_STEPS + 2):
            detector.update([drone_i, drone_j], trajs, graph)

        assert not detector.is_evading("d1"), "Non-adaptive drone d1 must NOT be evading"
        assert not detector.is_evading("d2"), "Non-adaptive drone d2 must NOT be evading"

    def test_outside_comm_radius_not_detected(self):
        """Drones outside comm_radius should not be detected even on a collision course."""
        drone_i = _make_drone("d1", x=[-2, 0, 0, 1.0, 0, 0])
        drone_j = _make_drone("d2", x=[2, 0, 0, -1.0, 0, 0])

        traj_i, traj_j = _make_head_on_trajs(
            pos_i=-2.0, vel_i=1.0, pos_j=2.0, vel_j=-1.0, horizon=20
        )

        # comm_radius=1.0 — drones are 4.0 m apart, outside radius
        graph = NeighborGraph(comm_radius=1.0)
        graph.update({"d1": drone_i.position(), "d2": drone_j.position()})

        detector = ConflictDetector()
        trajs = {"d1": traj_i, "d2": traj_j}

        for _ in range(HYSTERESIS_STEPS + 2):
            detector.update([drone_i, drone_j], trajs, graph)

        assert not detector.is_evading("d1"), "Out-of-comm-range drone must NOT be evading"
        assert not detector.is_evading("d2"), "Out-of-comm-range drone must NOT be evading"


# ---------------------------------------------------------------------------
# TestHysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:
    """Validate HYSTERESIS_STEPS entry/exit logic."""

    def _make_head_on_pair(self) -> tuple[Drone, Drone, dict[str, np.ndarray]]:
        """Return two drones on a head-on course + trajectories dict."""
        drone_i = _make_drone("d1", x=[-2, 0, 0, 1.0, 0, 0])
        drone_j = _make_drone("d2", x=[2, 0, 0, -1.0, 0, 0])
        traj_i, traj_j = _make_head_on_trajs(-2.0, 1.0, 2.0, -1.0, horizon=20)
        trajs = {"d1": traj_i, "d2": traj_j}
        return drone_i, drone_j, trajs

    def _make_safe_pair(self) -> tuple[Drone, Drone, dict[str, np.ndarray]]:
        """Return drones far apart at rest — no conflict possible."""
        drone_i = _make_drone("d1", x=[-10, 0, 0, 0, 0, 0])
        drone_j = _make_drone("d2", x=[10, 0, 0, 0, 0, 0])
        trajs = {"d1": np.tile([-10.0, 0, 0], (20, 1)), "d2": np.tile([10.0, 0, 0], (20, 1))}
        return drone_i, drone_j, trajs

    def test_one_step_below_hysteresis_does_not_evade(self):
        """1 consecutive conflict step with HYSTERESIS_STEPS=2 must NOT trigger evading."""
        assert HYSTERESIS_STEPS == 2, "Test assumes HYSTERESIS_STEPS == 2"

        drone_i, drone_j, conflict_trajs = self._make_head_on_pair()
        graph = _all_neighbors_graph(["d1", "d2"])

        detector = ConflictDetector()
        # Only one conflict step — below hysteresis threshold
        detector.update([drone_i, drone_j], conflict_trajs, graph)

        assert not detector.is_evading("d1"), "After 1 step, must not yet be evading"
        assert not detector.is_evading("d2"), "After 1 step, must not yet be evading"

    def test_two_consecutive_steps_triggers_evading(self):
        """2 consecutive conflict steps with HYSTERESIS_STEPS=2 MUST trigger evading."""
        assert HYSTERESIS_STEPS == 2, "Test assumes HYSTERESIS_STEPS == 2"

        drone_i, drone_j, conflict_trajs = self._make_head_on_pair()
        graph = _all_neighbors_graph(["d1", "d2"])

        detector = ConflictDetector()
        detector.update([drone_i, drone_j], conflict_trajs, graph)
        detector.update([drone_i, drone_j], conflict_trajs, graph)

        assert detector.is_evading("d1") or detector.is_evading("d2"), (
            "After 2 consecutive steps at HYSTERESIS_STEPS=2, at least one must be evading"
        )

    def test_conflict_clears_immediately_exits_evading(self):
        """When conflict clears, evading must be set to False immediately (no hold time)."""
        drone_i, drone_j, conflict_trajs = self._make_head_on_pair()
        graph = _all_neighbors_graph(["d1", "d2"])

        detector = ConflictDetector()
        # Drive into evading state
        for _ in range(HYSTERESIS_STEPS):
            detector.update([drone_i, drone_j], conflict_trajs, graph)

        # Confirm evading
        assert detector.is_evading("d1") or detector.is_evading("d2"), (
            "Must be evading before testing exit"
        )

        # Send safe drones and trajectories — conflict cleared
        drone_i_safe, drone_j_safe, safe_trajs = self._make_safe_pair()
        detector.update([drone_i_safe, drone_j_safe], safe_trajs, graph)

        assert not detector.is_evading("d1"), "After conflict clears, d1 must exit evading"
        assert not detector.is_evading("d2"), "After conflict clears, d2 must exit evading"

    def test_evasion_waypoint_cleared_on_conflict_exit(self):
        """evasion_waypoint must be set to None when conflict clears."""
        drone_i, drone_j, conflict_trajs = self._make_head_on_pair()
        graph = _all_neighbors_graph(["d1", "d2"])

        detector = ConflictDetector()
        for _ in range(HYSTERESIS_STEPS):
            detector.update([drone_i, drone_j], conflict_trajs, graph)

        # At least one has a waypoint
        wp_d1 = detector.get_evasion_waypoint("d1")
        wp_d2 = detector.get_evasion_waypoint("d2")
        assert wp_d1 is not None or wp_d2 is not None, (
            "At least one evading drone should have a waypoint"
        )

        # Clear conflict with safe state
        drone_i_safe, drone_j_safe, safe_trajs = self._make_safe_pair()
        detector.update([drone_i_safe, drone_j_safe], safe_trajs, graph)

        assert detector.get_evasion_waypoint("d1") is None, (
            "d1 waypoint must be cleared after conflict exit"
        )
        assert detector.get_evasion_waypoint("d2") is None, (
            "d2 waypoint must be cleared after conflict exit"
        )

    def test_reset_then_count_restarts(self):
        """After a conflict gap (one safe step), conflict counter resets from zero."""
        drone_i, drone_j, conflict_trajs = self._make_head_on_pair()
        drone_i_safe, drone_j_safe, safe_trajs = self._make_safe_pair()
        graph = _all_neighbors_graph(["d1", "d2"])

        detector = ConflictDetector()
        # One conflict step
        detector.update([drone_i, drone_j], conflict_trajs, graph)
        # One safe step — resets counter
        detector.update([drone_i_safe, drone_j_safe], safe_trajs, graph)
        # Only one more conflict step — not enough to trigger evading
        detector.update([drone_i, drone_j], conflict_trajs, graph)

        # Both counters must be exactly 1 (not 2), so not evading
        state_d1 = detector.get_state("d1")
        state_d2 = detector.get_state("d2")
        assert state_d1.consecutive_conflict_steps == 1 or state_d2.consecutive_conflict_steps == 1, (
            "Counter must restart after safe step — should be 1, not 2"
        )
        assert not detector.is_evading("d1"), "Not enough consecutive steps to evade"
        assert not detector.is_evading("d2"), "Not enough consecutive steps to evade"


# ---------------------------------------------------------------------------
# TestEvasionWaypoint
# ---------------------------------------------------------------------------


class TestEvasionWaypoint:
    """Validate z-reflection evasion waypoint logic.

    All tests use a head-on scenario where BOTH drones enter evading.
    We check the waypoint of whichever drone has the velocity we want to test
    by explicitly controlling both drones' positions and velocities such that
    the closing velocity gate (dot < 0) fires.

    Key: for closing < 0, we need dot(pos_j - pos_i, vel_j - vel_i) < 0.
    This holds when drone_i moves toward drone_j while drone_j is to the right.
    Classic setup: i at -2 vel=+1, j at +2 vel=-1.
    """

    def _head_on_conflict(
        self,
        pos_i: float,
        vel_ix: float,
        vel_iy: float,
        vel_iz: float,
        pos_j: float,
        vel_jx: float,
    ) -> tuple[Drone, Drone, dict[str, np.ndarray], NeighborGraph]:
        """Build a head-on conflict where both drones approach each other on the x-axis.

        Returns drones, trajs dict, and graph.
        """
        did_i, did_j = "d1", "d2"
        drone_i = _make_drone(did_i, x=[pos_i, 0, 0, vel_ix, vel_iy, vel_iz])
        drone_j = _make_drone(did_j, x=[pos_j, 0, 0, vel_jx, 0, 0])

        # Trajectories: linear motion along x from initial positions toward each other
        traj_i, traj_j = _make_head_on_trajs(
            pos_i=pos_i, vel_i=vel_ix, pos_j=pos_j, vel_j=vel_jx, horizon=20
        )
        trajs = {did_i: traj_i, did_j: traj_j}
        graph = _all_neighbors_graph([did_i, did_j])
        return drone_i, drone_j, trajs, graph

    def test_positive_vz_reflects_to_negative(self):
        """Velocity (1.0, 0.5, 2.0) → evasion waypoint has z < current_z (vz reflected)."""
        # d1 at x=-2 moving right with vz=+2.0; d2 at x=+2 moving left — both approaching
        drone_i, drone_j, trajs, graph = self._head_on_conflict(
            pos_i=-2.0, vel_ix=1.0, vel_iy=0.5, vel_iz=2.0,
            pos_j=2.0, vel_jx=-1.0,
        )
        detector = _run_detector_to_evading([drone_i, drone_j], trajs, graph)

        # Both should enter evading; verify d1's waypoint has z < 0 (reflected from vz=+2.0)
        assert detector.is_evading("d1"), "d1 must be evading in head-on scenario"
        wp = detector.get_evasion_waypoint("d1")
        assert wp is not None, "Evading drone must have a waypoint"

        current_z = drone_i.position()[2]  # = 0.0
        assert wp[2] < current_z, (
            f"With vz=+2.0, evasion direction reflects to -vz → waypoint z < {current_z}, "
            f"got wp[2]={wp[2]:.4f}"
        )

    def test_negative_vz_reflects_to_positive(self):
        """Velocity (-1.0, -0.5, -1.5) → evasion waypoint has z > current_z (vz reflected).

        Setup: d1 at x=+2 moving left (vel=-1.0) with vz=-1.5; d2 at x=-2 moving right.
        dot(rel_pos, rel_vel) = dot([-4,0,0], [1-(-1),0,0]) = dot([-4,0,0],[2,0,0]) = -8 < 0 → conflict.
        """
        drone_i, drone_j, trajs, graph = self._head_on_conflict(
            pos_i=2.0, vel_ix=-1.0, vel_iy=-0.5, vel_iz=-1.5,
            pos_j=-2.0, vel_jx=1.0,
        )
        detector = _run_detector_to_evading([drone_i, drone_j], trajs, graph)

        assert detector.is_evading("d1"), "d1 must be evading in head-on scenario"
        wp = detector.get_evasion_waypoint("d1")
        assert wp is not None

        current_z = drone_i.position()[2]  # = 0.0
        assert wp[2] > current_z, (
            f"With vz=-1.5, evasion direction reflects to +vz → waypoint z > {current_z}, "
            f"got wp[2]={wp[2]:.4f}"
        )

    def test_horizontal_flight_positive_net_deflects_up(self):
        """vz≈0, positive net horizontal (vx+vy > 0) → waypoint has z > current_z.

        When |vz| < VZ_ZERO_THRESHOLD, a synthetic z-component is added: +1 if vx+vy >= 0.
        """
        assert VZ_ZERO_THRESHOLD > 0, "VZ_ZERO_THRESHOLD must be positive"
        # vx=1.0, vy=0.5, vz=0.0 → vx+vy=1.5 > 0 → deflect_z = +1.0
        drone_i, drone_j, trajs, graph = self._head_on_conflict(
            pos_i=-2.0, vel_ix=1.0, vel_iy=0.5, vel_iz=0.0,
            pos_j=2.0, vel_jx=-1.0,
        )
        detector = _run_detector_to_evading([drone_i, drone_j], trajs, graph)

        assert detector.is_evading("d1"), "d1 must be evading in head-on scenario"
        wp = detector.get_evasion_waypoint("d1")
        assert wp is not None

        current_z = drone_i.position()[2]  # = 0.0
        assert wp[2] > current_z, (
            f"vz=0 with positive net horizontal → synthetic +z deflection → wp[2] > {current_z}, "
            f"got {wp[2]:.4f}"
        )

    def test_horizontal_flight_negative_net_deflects_down(self):
        """vz≈0, negative net horizontal (vx+vy < 0) → waypoint has z < current_z.

        When |vz| < VZ_ZERO_THRESHOLD, a synthetic z-component is added: -1 if vx+vy < 0.
        """
        # vx=-1.0, vy=-0.5, vz=0.0 → vx+vy=-1.5 < 0 → deflect_z = -1.0
        # Place d1 at x=+2 moving left, d2 at x=-2 moving right → head-on
        drone_i, drone_j, trajs, graph = self._head_on_conflict(
            pos_i=2.0, vel_ix=-1.0, vel_iy=-0.5, vel_iz=0.0,
            pos_j=-2.0, vel_jx=1.0,
        )
        detector = _run_detector_to_evading([drone_i, drone_j], trajs, graph)

        assert detector.is_evading("d1"), "d1 must be evading in head-on scenario"
        wp = detector.get_evasion_waypoint("d1")
        assert wp is not None

        current_z = drone_i.position()[2]  # = 0.0
        assert wp[2] < current_z, (
            f"vz=0 with negative net horizontal → synthetic -z deflection → wp[2] < {current_z}, "
            f"got {wp[2]:.4f}"
        )

    def test_waypoint_is_offset_distance_away(self):
        """Evasion waypoint must be placed exactly EVASION_OFFSET_DISTANCE from drone position."""
        drone_i = _make_drone("d1", x=[-2, 0, 0, 1.0, 0, 2.0])
        drone_j = _make_drone("d2", x=[2, 0, 0, -1.0, 0, 0])
        traj_i, traj_j = _make_head_on_trajs(-2.0, 1.0, 2.0, -1.0, horizon=20)
        trajs = {"d1": traj_i, "d2": traj_j}
        graph = _all_neighbors_graph(["d1", "d2"])

        detector = _run_detector_to_evading([drone_i, drone_j], trajs, graph)

        for did, drone in [("d1", drone_i), ("d2", drone_j)]:
            if detector.is_evading(did):
                wp = detector.get_evasion_waypoint(did)
                assert wp is not None
                dist = float(np.linalg.norm(wp - drone.position()))
                assert abs(dist - EVASION_OFFSET_DISTANCE) < 1e-6, (
                    f"Waypoint distance {dist:.6f} must equal "
                    f"EVASION_OFFSET_DISTANCE={EVASION_OFFSET_DISTANCE}"
                )
                return

        pytest.fail("No drone entered evading state")

    def test_waypoint_held_after_entry(self):
        """Once set on Evading entry, the evasion waypoint is held on subsequent conflict steps."""
        drone_i = _make_drone("d1", x=[-2, 0, 0, 1.0, 0, 2.0])
        drone_j = _make_drone("d2", x=[2, 0, 0, -1.0, 0, 0])
        traj_i, traj_j = _make_head_on_trajs(-2.0, 1.0, 2.0, -1.0, horizon=20)
        trajs = {"d1": traj_i, "d2": traj_j}
        graph = _all_neighbors_graph(["d1", "d2"])

        detector = ConflictDetector()
        # Enter evading
        for _ in range(HYSTERESIS_STEPS):
            detector.update([drone_i, drone_j], trajs, graph)

        # Find which drone is evading and record waypoint
        evading_id = next(
            (did for did in ["d1", "d2"] if detector.is_evading(did)), None
        )
        assert evading_id is not None, "At least one drone must be evading"

        wp_first = detector.get_evasion_waypoint(evading_id).copy()

        # One more step with same conflict trajs
        detector.update([drone_i, drone_j], trajs, graph)
        wp_second = detector.get_evasion_waypoint(evading_id)

        np.testing.assert_array_equal(
            wp_first, wp_second, err_msg="Waypoint must be held constant, not recomputed"
        )


# ---------------------------------------------------------------------------
# TestMultiNeighbor
# ---------------------------------------------------------------------------


class TestMultiNeighbor:
    """Validate multi-conflict neighbor handling."""

    def test_drone_conflicts_with_two_neighbors_enters_evading(self):
        """Drone B conflicts with both A and C simultaneously.

        Setup: A at x=-2 moving right, B at x=0 moving right slower, C at x=2 moving left.
        Pair A-B: rel_pos=[2,0,0], rel_vel=[vB-vA]. If vA > vB, rel_vel < 0 → conflict.
        We use A and B both approaching each other on x; B also approaches C from C's side.

        Simplest case: A and B head-on, B and C head-on. Use three distinct drones.
        """
        # Pair (A, B): A at -3 moving right fast, B at -1 moving right slowly
        # → A approaching B from behind? No, A is to the left of B and moving faster right.
        # For conflict we need closing < 0: dot(pos_B - pos_A, vel_B - vel_A) < 0.
        # pos_B - pos_A = [2, 0, 0]; vel_B - vel_A = [1-2, 0, 0] = [-1, 0, 0]
        # dot = 2 * (-1) = -2 < 0 → CONFLICT (A chasing B from behind).
        # Pair (B, C): B at -1 moving right, C at +1 moving left → head-on.
        # pos_C - pos_B = [2, 0, 0]; vel_C - vel_B = [-1-1, 0, 0] = [-2, 0, 0]
        # dot = -4 < 0 → CONFLICT.

        drone_a = _make_drone("a", x=[-3, 0, 0, 2.0, 0, 0])  # fast right
        drone_b = _make_drone("b", x=[-1, 0, 0, 1.0, 0, 0])  # slow right
        drone_c = _make_drone("c", x=[1, 0, 0, -1.0, 0, 0])  # moving left

        steps = np.arange(1, 21) * 0.1
        traj_a = np.stack([-3.0 + 2.0 * steps, np.zeros(20), np.zeros(20)], axis=1)
        traj_b = np.stack([-1.0 + 1.0 * steps, np.zeros(20), np.zeros(20)], axis=1)
        traj_c = np.stack([1.0 - 1.0 * steps, np.zeros(20), np.zeros(20)], axis=1)

        trajs = {"a": traj_a, "b": traj_b, "c": traj_c}
        graph = _all_neighbors_graph(["a", "b", "c"])

        detector = _run_detector_to_evading([drone_a, drone_b, drone_c], trajs, graph)

        # B conflicts with both A (approaching from behind) and C (head-on)
        state_b = detector.get_state("b")
        assert state_b is not None

        # The state machine will have updated all drones — at least B should be evading
        # (it's in conflict with two neighbors)
        assert detector.is_evading("b"), (
            "Drone B is in conflict with both A and C — must enter evading"
        )

    def test_single_evasion_waypoint_per_drone(self):
        """A drone in conflict with multiple neighbors gets exactly one evasion waypoint."""
        drone_a = _make_drone("a", x=[-2, 0, 0, 1.0, 0, 0])
        drone_b = _make_drone("b", x=[2, 0, 0, -1.0, 0, 0])

        traj_a, traj_b = _make_head_on_trajs(-2.0, 1.0, 2.0, -1.0, horizon=20)
        trajs = {"a": traj_a, "b": traj_b}
        graph = _all_neighbors_graph(["a", "b"])

        detector = _run_detector_to_evading([drone_a, drone_b], trajs, graph)

        # Find the evading drone — verify it has exactly one waypoint of correct shape
        for did in ["a", "b"]:
            if detector.is_evading(did):
                wp = detector.get_evasion_waypoint(did)
                assert wp is not None, "Evading drone must have exactly one waypoint"
                assert wp.shape == (3,), f"Waypoint must be shape (3,), got {wp.shape}"
                return

        pytest.fail("No drone entered evading state")

    def test_get_evasion_waypoint_none_when_not_evading(self):
        """get_evasion_waypoint returns None for drones not in evading state."""
        detector = ConflictDetector()
        assert detector.get_evasion_waypoint("nonexistent") is None

        drone_i = _make_drone("d1", x=[0, 0, 0, 0, 0, 0])
        traj = np.tile([0.0, 0, 0], (10, 1))
        graph = _all_neighbors_graph(["d1"])
        detector.update([drone_i], {"d1": traj}, graph)

        assert detector.get_evasion_waypoint("d1") is None

    def test_is_evading_false_for_unknown_drone(self):
        """is_evading returns False for unknown drone IDs."""
        detector = ConflictDetector()
        assert not detector.is_evading("unknown-drone-xyz")

    def test_reset_clears_state(self):
        """reset() removes the per-drone state entry so drone is no longer tracked as evading."""
        drone_i = _make_drone("d1", x=[-2, 0, 0, 1.0, 0, 0])
        drone_j = _make_drone("d2", x=[2, 0, 0, -1.0, 0, 0])
        traj_i, traj_j = _make_head_on_trajs(-2.0, 1.0, 2.0, -1.0, horizon=20)
        trajs = {"d1": traj_i, "d2": traj_j}
        graph = _all_neighbors_graph(["d1", "d2"])

        detector = _run_detector_to_evading([drone_i, drone_j], trajs, graph)

        # Reset d1
        detector.reset("d1")
        assert not detector.is_evading("d1"), "After reset, d1 must not be evading"
        assert detector.get_evasion_waypoint("d1") is None, "After reset, waypoint must be None"
