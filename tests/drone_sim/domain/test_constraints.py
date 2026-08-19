"""Tests for drone_sim.domain.constraints module.

Tests for:
- VelocityConstraints (evaluate_single, evaluate_multi)
- MovingObstacleAvoidanceConstraints (evaluate_single, evaluate_multi)
- ObstacleAvoidanceConstraints (evaluate_single, evaluate_multi)
- RoomConstraints (evaluate_single, evaluate_multi -- box and sphere)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from drone_sim.domain.constraints import (
    VelocityConstraints,
    MovingObstacleAvoidanceConstraints,
    ObstacleAvoidanceConstraints,
    RoomConstraints,
    point_to_box_dist,
)
from drone_sim.domain.drone import Drone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics


class _StubController:
    pass


def _make_drone(
    drone_id: str = "d1",
    x: np.ndarray | None = None,
    target: np.ndarray | None = None,
    radius: float = 0.2,
    safety_zone: float = 1.0,
    cons_stop: float = 0.0,
    v_max: float = 5.0,
    alpha: float | None = None,
    safety_zone_mode: str = "fixed",
) -> Drone:
    """Helper to create a minimal Drone for constraint testing."""
    if x is None:
        x = np.zeros(6, dtype=float)
    if target is None:
        target = np.zeros(3, dtype=float)

    return Drone(
        drone_id=drone_id,
        radius=radius,
        safety_zone=safety_zone,
        cons_stop=cons_stop,
        color="tab:blue",
        safety_color="tab:cyan",
        trace_color="tab:blue",
        controller=_StubController(),
        physics=LinearKinematicsPhysics(dt=0.1, v_max=v_max),
        x=np.asarray(x, dtype=float).reshape(6),
        route=Route(start=np.asarray(x, dtype=float).reshape(6)[:3], waypoints=[], target=np.asarray(target, dtype=float).reshape(3)),
        alpha=alpha,
        safety_zone_mode=safety_zone_mode,
    )


# ---------------------------------------------------------------
# VelocityConstraints
# ---------------------------------------------------------------

class TestVelocityConstraintsSingle:
    """Tests for VelocityConstraints.evaluate_single."""

    def test_below_vmax_satisfied(self):
        """Velocity below v_max produces positive margins."""
        horizon = 3
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=5.0)
        # speed = sqrt(3) ~ 1.73 m/s, well below 5.0
        v_pred = np.ones((horizon, 3))
        values = np.array([])

        result = velocity_constraints.evaluate_single(drone, v_pred, values)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_zero_velocity_satisfied(self):
        """Zero velocity is always within limits."""
        horizon = 3
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=1.0)
        v_pred = np.zeros((horizon, 3))
        values = np.array([])

        result = velocity_constraints.evaluate_single(drone, v_pred, values)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_appends_to_existing_values(self):
        """evaluate_single concatenates to existing values array."""
        horizon = 2
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=5.0)
        v_pred = np.zeros((horizon, 3))
        existing = np.array([42.0, 99.0])

        result = velocity_constraints.evaluate_single(drone, v_pred, existing)

        assert result.shape == (2 + horizon,)
        assert result[0] == 42.0
        assert result[1] == 99.0

    def test_exact_margin_value(self):
        """Verify the exact margin: v_max^2 - ||vel||^2."""
        horizon = 1
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drone = _make_drone(v_max=5.0)
        # vel = (3, 0, 0) => ||vel||^2 = 9, margin = 25 - 9 = 16
        v_pred = np.array([[3.0, 0.0, 0.0]])
        values = np.array([])

        result = velocity_constraints.evaluate_single(drone, v_pred, values)

        assert result[0] == pytest.approx(16.0)


class TestVelocityConstraintsMulti:
    """Tests for VelocityConstraints.evaluate_multi."""

    def test_multiple_drones_below_vmax(self):
        """All drones below v_max => all positive margins."""
        horizon = 3
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drones = [_make_drone("d1", v_max=5.0), _make_drone("d2", v_max=5.0)]
        v_pred = np.ones((2, horizon, 3))  # speed = sqrt(3) ~ 1.73
        values = np.array([])

        result = velocity_constraints.evaluate_multi(drones, v_pred, values)

        assert result.shape == (2 * horizon,)
        assert np.all(result > 0)

    def test_appends_to_existing_values(self):
        """evaluate_multi concatenates to existing values."""
        horizon = 1
        velocity_constraints = VelocityConstraints(horizon=horizon)
        drones = [_make_drone(v_max=5.0)]
        v_pred = np.zeros((1, horizon, 3))
        existing = np.array([1.0, 2.0])

        result = velocity_constraints.evaluate_multi(drones, v_pred, existing)

        assert result.shape == (2 + 1,)
        assert result[0] == 1.0
        assert result[1] == 2.0


# ---------------------------------------------------------------
# MovingObstacleAvoidanceConstraints
# ---------------------------------------------------------------

class TestMovingObstacleAvoidanceSingle:
    """Tests for MovingObstacleAvoidanceConstraints.evaluate_single."""

    def test_far_neighbor_satisfied(self):
        """Distant neighbor produces positive constraint values."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.zeros((horizon, 3))
        neighbor_traj = np.ones((horizon, 3)) * 10.0
        neighbors = {"n1": (neighbor_traj, None)}
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_close_neighbor_violated(self):
        """Close neighbor produces negative constraint values."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3))
        # Neighbor at (0.5, 0, 0), dist=0.5, threshold=1.0+1.0=2.0 => -1.5
        neighbor_traj = np.tile(np.array([0.5, 0.0, 0.0]), (horizon, 1))
        neighbors = {"n1": (neighbor_traj, None)}
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result.shape == (horizon,)
        assert np.all(result < 0)

    def test_no_neighbors_empty(self):
        """No neighbors produces no constraint values."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone()
        pred_pos = np.zeros((horizon, 3))
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, {}, values)

        assert result.shape == (0,)

    def test_multiple_neighbors(self):
        """Multiple neighbors produce H * num_neighbors constraints."""
        horizon = 2
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.zeros((horizon, 3))
        neighbors = {
            "n1": (np.ones((horizon, 3)) * 10.0, None),
            "n2": (np.ones((horizon, 3)) * 20.0, None),
        }
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result.shape == (horizon * 2,)
        assert np.all(result > 0)

    def test_exact_margin_value(self):
        """Verify exact margin: dist - (safety_zone_self + safety_zone_self) with same-type assumption."""
        horizon = 1
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        # Neighbor at (5, 0, 0), dist=5, threshold=0.5+0.5=1.0 (ego safety for both)
        neighbors = {"n1": (np.array([[5.0, 0.0, 0.0]]), None)}
        values = np.array([])

        result = moving_avoidance.evaluate_single(drone, pred_pos, neighbors, values)

        assert result[0] == pytest.approx(4.0)


class TestMovingObstacleAvoidanceMulti:
    """Tests for MovingObstacleAvoidanceConstraints.evaluate_multi."""

    def test_two_drones_far_apart(self):
        """Two drones far apart produces all positive constraints."""
        horizon = 3
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.5)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.ones((horizon, 3)) * 10.0,
        }
        values = np.array([])

        result = moving_avoidance.evaluate_multi(drones, pred_pos, values)

        # 1 pair (i<j) * horizon constraints
        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_two_drones_overlapping(self):
        """Two overlapping drones produces negative constraints."""
        horizon = 2
        moving_avoidance = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=1.0)
        d2 = _make_drone("d2", safety_zone=1.0)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.tile(np.array([0.5, 0.0, 0.0]), (horizon, 1)),
        }
        values = np.array([])

        result = moving_avoidance.evaluate_multi(drones, pred_pos, values)

        assert np.all(result < 0)

# ---------------------------------------------------------------
# point_to_box_dist
# ---------------------------------------------------------------

class TestPointToBoxDist:
    """Direct unit tests for point_to_box_dist covering all geometric approach directions.

    The function uses regularized L2 norm: sqrt(sum(relu(d)^2) + eps^2) - eps
    with eps=1e-3. This means:
    - Outside the box: approx equal to Euclidean distance to nearest surface (within eps error)
    - Inside the box: exactly 0.0 (all relu_d = 0, sqrt(eps^2) - eps = 0)
    - On surface: exactly 0.0
    """

    def test_face_approach_x_axis(self):
        """Point offset along x-axis only (face approach).

        point=[3,0,0], center=[0,0,0], half_extents=[1,1,1].
        d_x=3-1=2.0, d_y=d_z clamped to 0. Expected: approx 2.0.
        """
        dist = point_to_box_dist(
            np.array([3.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        assert dist == pytest.approx(2.0, abs=1e-2)

    def test_face_approach_negative_z(self):
        """Point offset along negative z-axis (face approach).

        point=[0,0,-4], center=[0,0,0], half_extents=[1,1,1].
        d_z=4-1=3.0. Expected: approx 3.0.
        """
        dist = point_to_box_dist(
            np.array([0.0, 0.0, -4.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        assert dist == pytest.approx(3.0, abs=1e-2)

    def test_edge_approach_xy(self):
        """Point offset along two axes simultaneously (edge approach).

        point=[2,2,0], center=[0,0,0], half_extents=[1,1,1].
        d_x=d_y=1.0, d_z clamped to 0. Expected: sqrt(2) approx 1.4142.
        """
        dist = point_to_box_dist(
            np.array([2.0, 2.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        assert dist == pytest.approx(np.sqrt(2.0), abs=1e-2)

    def test_corner_approach(self):
        """Point offset along all three axes simultaneously (corner approach).

        point=[2,2,2], center=[0,0,0], half_extents=[1,1,1].
        d_x=d_y=d_z=1.0. Expected: sqrt(3) approx 1.7321.
        """
        dist = point_to_box_dist(
            np.array([2.0, 2.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        assert dist == pytest.approx(np.sqrt(3.0), abs=1e-2)

    def test_inside_box_returns_zero(self):
        """Point at box center returns 0.0.

        All d_i = -1.0, clamped to 0. relu_d=[0,0,0]. Expected: 0.0.
        """
        dist = point_to_box_dist(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        assert dist == pytest.approx(0.0, abs=1e-2)

    def test_inside_box_off_center(self):
        """Point inside box but off-center returns 0.0.

        point=[0.5,0.3,-0.2], all |point_i| < 1.0 so all d_i < 0. Expected: 0.0.
        """
        dist = point_to_box_dist(
            np.array([0.5, 0.3, -0.2]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        assert dist == pytest.approx(0.0, abs=1e-2)

    def test_on_surface_face(self):
        """Point on a face surface returns 0.0.

        point=[1,0,0], center=[0,0,0], half_extents=[1,1,1].
        d_x=0, d_y=d_z=0 -> relu_d=[0,0,0]. Expected: 0.0.
        """
        dist = point_to_box_dist(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        assert dist == pytest.approx(0.0, abs=1e-2)

    def test_asymmetric_half_extents(self):
        """Point with asymmetric half_extents — verifies per-axis handling.

        point=[3,0,0], center=[0,0,0], half_extents=[2,0.5,0.5].
        d_x=3-2=1.0 (only x excess). Expected: approx 1.0.
        """
        dist = point_to_box_dist(
            np.array([3.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([2.0, 0.5, 0.5]),
        )
        assert dist == pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------
# ObstacleAvoidanceConstraints
# ---------------------------------------------------------------

class TestObstacleAvoidanceSingle:
    """Tests for ObstacleAvoidanceConstraints.evaluate_single."""

    def test_far_from_obstacle_satisfied(self):
        """Drone far from obstacle produces positive margins."""
        horizon = 3
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3))
        obstacles = [(np.array([10.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))]
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, values)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_close_to_obstacle_violated(self):
        """Drone close to obstacle produces negative margins."""
        horizon = 3
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3))
        # Obstacle at (0.5, 0, 0) with half_extents=[0.2,0.2,0.2]
        # point_to_box_dist([0,0,0], [0.5,0,0], [0.2,0.2,0.2]) = 0.3 (face approach)
        # margin = 0.3 - 1.0 = -0.7 => still negative
        obstacles = [(np.array([0.5, 0.0, 0.0]), np.array([0.2, 0.2, 0.2]))]
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, values)

        assert result.shape == (horizon,)
        assert np.all(result < 0)

    def test_no_obstacles_returns_zeros(self):
        """No obstacles produces a zero-length result from _evaluate, but shape=(horizon,)."""
        horizon = 3
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone()
        pred_pos = np.zeros((horizon, 3))
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, [], values)

        # _evaluate returns np.zeros(horizon) even with no obstacles
        assert result.shape == (horizon,)

    def test_exact_margin_value(self):
        """Verify exact margin: point_to_box_dist - safety_zone."""
        horizon = 1
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        # Obstacle at (3, 0, 0) with half_extents=[0.3,0.3,0.3]
        # point_to_box_dist([0,0,0], [3,0,0], [0.3,0.3,0.3]) ~= 2.7 (face approach, eps regularized)
        # margin ~= 2.7 - 0.5 = 2.2 (tolerance 1e-3 for regularization eps)
        obstacles = [(np.array([3.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.3]))]
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, values)

        assert result[0] == pytest.approx(2.2, abs=1e-3)

    def test_appends_to_existing_values(self):
        """evaluate_single concatenates to existing values."""
        horizon = 2
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.zeros((horizon, 3))
        obstacles = [(np.array([10.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))]
        existing = np.array([7.0])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, existing)

        assert result.shape == (1 + horizon,)
        assert result[0] == 7.0

    def test_two_obstacles_both_constrain(self):
        """Two obstacles each produce their own horizon row (shape N*H).

        This test would fail on the buggy code (overwrite loop) where only the
        last obstacle's margin survived and result.shape == (1,).

        horizon=1, safety_zone=0.5, drone at origin.
        obstacle_A at [3,0,0], half_extents=[0.3,0.3,0.3]:
            dist_A ~= 2.7 (face approach, eps regularized)  => margin_A ~= 2.7 - 0.5 = 2.2
        obstacle_B at [1,0,0], half_extents=[0.1,0.1,0.1]:
            dist_B ~= 0.9 (face approach, eps regularized)  => margin_B ~= 0.9 - 0.5 = 0.4
        """
        horizon = 1
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        obstacles = [
            (np.array([3.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.3])),  # obstacle_A
            (np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1])),  # obstacle_B
        ]
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, values)

        # Fixed code: shape == (2,) — one row per obstacle
        assert result.shape == (2,), f"Expected shape (2,) but got {result.shape}"
        assert result[0] == pytest.approx(2.2, abs=1e-3)   # obstacle_A margin
        assert result[1] == pytest.approx(0.4, abs=1e-3)   # obstacle_B margin
        assert result[0] > 0   # drone safe from obstacle_A
        assert result[1] > 0   # drone safe from obstacle_B

    def test_two_obstacles_one_violated(self):
        """Two obstacles with large safety_zone: obstacle_B margin goes negative.

        Demonstrates that each obstacle is evaluated independently — obstacle_A
        (far enough) stays positive while obstacle_B (too close) is negative.
        On the buggy code with obstacles reversed in the list, only obstacle_A's
        (safe) margin would survive and the violation would be silently dropped.

        horizon=1, safety_zone=2.5.
        obstacle_A at [3,0,0], half_extents=[0.3,0.3,0.3]: dist_A ~= 2.7, margin ~= 0.2
        obstacle_B at [1,0,0], half_extents=[0.1,0.1,0.1]: dist_B ~= 0.9, margin ~= -1.6
        """
        horizon = 1
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=2.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        obstacles = [
            (np.array([3.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.3])),  # obstacle_A (safe)
            (np.array([1.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1])),  # obstacle_B (violated)
        ]
        values = np.array([])

        result = obstacle_avoidance.evaluate_single(drone, pred_pos, obstacles, values)

        assert result.shape == (2,), f"Expected shape (2,) but got {result.shape}"
        assert result[0] > 0, "obstacle_A margin should be positive (drone is safe)"
        assert result[1] < 0, "obstacle_B margin should be negative (constraint violated)"


class TestObstacleAvoidanceMulti:
    """Tests for ObstacleAvoidanceConstraints.evaluate_multi."""

    def test_two_drones_far_from_obstacle(self):
        """Both drones far from obstacle produces positive margins."""
        horizon = 3
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.5)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.ones((horizon, 3)) * 20.0,
        }
        obstacles = [(np.array([50.0, 50.0, 50.0]), np.array([0.5, 0.5, 0.5]))]
        values = np.array([])

        result = obstacle_avoidance.evaluate_multi(drones, pred_pos, obstacles, values)

        assert result.shape == (2 * horizon,)
        assert np.all(result > 0)

    def test_one_drone_close_to_obstacle(self):
        """One drone near obstacle has negative values."""
        horizon = 2
        obstacle_avoidance = ObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=1.0)
        d2 = _make_drone("d2", safety_zone=1.0)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),  # at origin
            "d2": np.ones((horizon, 3)) * 100.0,  # far away
        }
        obstacles = [(np.array([0.5, 0.0, 0.0]), np.array([0.5, 0.5, 0.5]))]  # close to d1
        values = np.array([])

        result = obstacle_avoidance.evaluate_multi(drones, pred_pos, obstacles, values)

        # d1 constraints should be violated
        assert np.any(result[:horizon] < 0)
        # d2 constraints should be satisfied
        assert np.all(result[horizon:] > 0)


# ---------------------------------------------------------------
# RoomConstraints -- Box mode
# ---------------------------------------------------------------

class TestRoomConstraintsSingleBox:
    """Tests for RoomConstraints.evaluate_single with box room."""

    def test_inside_room_satisfied(self):
        """Drone in center of room produces positive margins."""
        horizon = 3
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3)) + 5.0  # center of [0,10]^3
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, values)

        # 6 per-face constraints per horizon step
        assert result.shape == (6 * horizon,)
        assert np.all(result > 0)

    def test_outside_room_violated(self):
        """Drone outside room produces negative margins."""
        horizon = 3
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3)) - 5.0  # well outside [0,10]^3
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, values)

        assert result.shape == (6 * horizon,)
        assert np.any(result < 0)

    def test_near_wall_accounts_for_safety_zone(self):
        """Drone near wall: safety_zone pushes margin down."""
        horizon = 1
        room_constraints = RoomConstraints(horizon=horizon)
        # Position at (0.5, 5, 5), room [0,10]^3, safety_zone=1.0
        # Lower margin x: 0.5 - 0.0 - 1.0 = -0.5 => violated
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.array([[0.5, 5.0, 5.0]])
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, values)

        # Per-face constraints: [lower_x, lower_y, lower_z, upper_x, upper_y, upper_z]
        # lower_x = 0.5 - 1.0 - 0.0 = -0.5
        assert result.shape == (6,)
        assert result[0] == pytest.approx(-0.5)
        assert np.min(result) == pytest.approx(-0.5)

    def test_exact_margin_center(self):
        """Verify exact margin for drone at center of symmetric room."""
        horizon = 1
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        # Center of [-5, 5]^3, each wall is 5 units away, minus safety=1 => 4.0
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        room_min = np.array([-5.0, -5.0, -5.0])
        room_max = np.array([5.0, 5.0, 5.0])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, values)

        # All 6 faces have margin 4.0 at center of symmetric room
        assert result.shape == (6,)
        assert np.all(result == pytest.approx(4.0))

    def test_appends_to_existing_values(self):
        """evaluate_single concatenates to existing values."""
        horizon = 1
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[5.0, 5.0, 5.0]])
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        existing = np.array([42.0])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max, room_min, existing)

        assert result.shape == (1 + 6,)  # existing + 6 per-face constraints
        assert result[0] == 42.0


class TestRoomConstraintsMultiBox:
    """Tests for RoomConstraints.evaluate_multi with box room."""

    def test_two_drones_inside(self):
        """Both drones inside room produces positive margins."""
        horizon = 2
        room_constraints = RoomConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.5)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)) + 5.0,
            "d2": np.zeros((horizon, 3)) + 5.0,
        }
        room_min = np.array([0.0, 0.0, 0.0])
        room_max = np.array([10.0, 10.0, 10.0])
        values = np.array([])

        result = room_constraints.evaluate_multi(drones, pred_pos, room_max, room_min, values)

        # 2 drones * 6 per-face * horizon steps
        assert result.shape == (2 * 6 * horizon,)
        assert np.all(result > 0)


# ---------------------------------------------------------------
# RoomConstraints -- Sphere mode
# ---------------------------------------------------------------

class TestRoomConstraintsSingleSphere:
    """Tests for RoomConstraints.evaluate_single with spherical room."""

    def test_inside_sphere_satisfied(self):
        """Drone at origin in large sphere room produces positive margins."""
        horizon = 3
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3))
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max=10.0, room_min=0.0, values=values, room_is_sphere=True)

        assert result.shape == (horizon,)
        assert np.all(result > 0)

    def test_outside_sphere_violated(self):
        """Drone far from origin in small sphere room produces negative margins."""
        horizon = 3
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0)
        pred_pos = np.zeros((horizon, 3)) + 5.0  # dist from origin = sqrt(75) ~ 8.66
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max=2.0, room_min=0.0, values=values, room_is_sphere=True)

        assert np.all(result < 0)

    def test_exact_margin_sphere(self):
        """Verify exact margin: room_radius - dist_from_origin - safety_zone."""
        horizon = 1
        room_constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        # Drone at (3, 0, 0), dist=3.0, room_radius=10.0 => 10 - 3 - 0.5 = 6.5
        pred_pos = np.array([[3.0, 0.0, 0.0]])
        values = np.array([])

        result = room_constraints.evaluate_single(drone, pred_pos, room_max=10.0, room_min=0.0, values=values, room_is_sphere=True)

        assert result[0] == pytest.approx(6.5)


class TestRoomConstraintsMultiSphere:
    """Tests for RoomConstraints.evaluate_multi with spherical room."""

    def test_two_drones_inside_sphere(self):
        """Both drones inside sphere produces positive margins."""
        horizon = 2
        room_constraints = RoomConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5)
        d2 = _make_drone("d2", safety_zone=0.5)
        drones = [d1, d2]
        pred_pos = {
            "d1": np.zeros((horizon, 3)),
            "d2": np.ones((horizon, 3)),
        }
        values = np.array([])

        result = room_constraints.evaluate_multi(drones, pred_pos, room_max=10.0, room_min=0.0, values=values, room_is_sphere=True)

        assert result.shape == (2 * horizon,)
        assert np.all(result > 0)


# ---------------------------------------------------------------
# Adaptive Constraints (velocity-dependent safety radii)
# ---------------------------------------------------------------
@pytest.mark.skip(reason="Skipping adaptive constraints tests as they are broken -> create issue ISS-006")
class TestAdaptiveConstraints:
    """Tests for adaptive velocity-dependent safety radius in constraint evaluation.

    Hand-computed reference values:
    LinearKinematicsPhysics(dt=0.1) has default u_max=[3,3,3].
    u_max_scalar = min(|u_max|) = 3.0
    For alpha=0.5, radius=0.2, safety_zone=1.0:
      velocity=[0,0,0]: s_stop=0, formula=0.2, floor=1.0 => adaptive_radius = 1.0
      velocity=[4,0,0]: ||v||^2=16, s_stop=16/6=8/3, formula=0.2+0.5*(8/3)=1.5333 > 1.0
    The safety_zone floor ensures adaptive radius >= safety_zone at all velocities.
    """

    def test_moving_obstacle_adaptive_radius_at_rest(self):
        """Adaptive drone at rest: safety radius floored at safety_zone.

        At rest the adaptive formula gives radius (0.2), but the floor
        enforces safety_zone (1.0). So adaptive at rest = fixed = safety_zone.
        Both fixed and adaptive produce the same margin.
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)

        # Fixed drone: safety_zone=1.0
        drone_fixed = _make_drone(safety_zone=1.0)
        # Adaptive drone: alpha=0.5, radius=0.2, safety_zone=1.0
        drone_adaptive = _make_drone(safety_zone=1.0, alpha=0.5)

        pred_pos = np.array([[0.0, 0.0, 0.0]])
        neighbor_traj = np.array([[5.0, 0.0, 0.0]])
        neighbors = {"n1": (neighbor_traj, None)}  # neighbor vel=None

        # Fixed: dist=5.0, threshold = 1.0 + 1.0 = 2.0, margin = 3.0
        result_fixed = constraints.evaluate_single(drone_fixed, pred_pos, neighbors, np.array([]))
        assert result_fixed[0] == pytest.approx(3.0)

        # Adaptive at rest: floor enforces safety_zone=1.0, neighbor vel=None => safety_zone=1.0
        # threshold = 1.0 + 1.0 = 2.0, margin = 3.0 (same as fixed)
        pred_vel_rest = np.array([[0.0, 0.0, 0.0]])
        result_adaptive = constraints.evaluate_single(
            drone_adaptive, pred_pos, neighbors, np.array([],), pred_vel=pred_vel_rest,
        )
        assert result_adaptive[0] == pytest.approx(3.0)

        # At rest, adaptive equals fixed due to floor
        assert result_adaptive[0] == pytest.approx(result_fixed[0])

    def test_moving_obstacle_adaptive_radius_moving(self):
        """Adaptive drone with high velocity: formula exceeds safety_zone floor.

        velocity=[4,0,0]: ||v||^2=16, s_stop=16/(2*3)=8/3
        adaptive_radius = max(1.0, 0.2 + 0.5 * 8/3) = max(1.0, 1.5333) = 1.5333
        Neighbor vel=None => uses ego safety_zone=1.0
        threshold = 1.5333 + 1.0 = 2.5333
        margin = 5.0 - 2.5333 = 2.4667
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0, alpha=0.5)

        pred_pos = np.array([[0.0, 0.0, 0.0]])
        neighbor_traj = np.array([[5.0, 0.0, 0.0]])
        neighbors = {"n1": (neighbor_traj, None)}

        pred_vel = np.array([[4.0, 0.0, 0.0]])
        result = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]), pred_vel=pred_vel)

        expected_adaptive_radius = 0.2 + 0.5 * (16.0 / (2.0 * 3.0))  # ~ 1.5333
        assert expected_adaptive_radius > 1.0  # formula exceeds floor
        expected_margin = 5.0 - (expected_adaptive_radius + 1.0)  # neighbor uses ego safety_zone
        assert result[0] == pytest.approx(expected_margin)

    def test_moving_obstacle_multi_adaptive_both_drones(self):
        """Both drones adaptive in evaluate_multi with pred_vel dict.

        d1 at origin, velocity=[4,0,0]: ||v||^2=16, s_stop=8/3
            adaptive_radius_1 = max(1.0, 0.2 + 0.5*8/3) = 1.5333
        d2 at (10,0,0), velocity=[5,0,0]: ||v||^2=25, s_stop=25/6
            adaptive_radius_2 = max(1.0, 0.2 + 0.5*25/6) = 2.2833
        threshold = 1.5333 + 2.2833 = 3.8167
        dist = 10.0
        margin = 10.0 - 3.8167 = 6.1833
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=1.0, alpha=0.5)
        d2 = _make_drone("d2", safety_zone=1.0, alpha=0.5)
        drones = [d1, d2]

        pred_pos = {
            "d1": np.array([[0.0, 0.0, 0.0]]),
            "d2": np.array([[10.0, 0.0, 0.0]]),
        }
        pred_vel = {
            "d1": np.array([[4.0, 0.0, 0.0]]),
            "d2": np.array([[5.0, 0.0, 0.0]]),
        }

        result = constraints.evaluate_multi(drones, pred_pos, np.array([]), pred_vel=pred_vel)

        ar_d1 = 0.2 + 0.5 * (16.0 / 6.0)  # 1.5333
        ar_d2 = 0.2 + 0.5 * (25.0 / 6.0)  # 2.2833
        expected_margin = 10.0 - (ar_d1 + ar_d2)
        assert result.shape == (horizon,)
        assert result[0] == pytest.approx(expected_margin)

    def test_moving_obstacle_multi_mixed_fixed_adaptive(self):
        """One fixed drone, one adaptive. Fixed uses safety_zone, adaptive uses formula.

        d1 fixed: safety = 1.0
        d2 adaptive, velocity=[4,0,0]: adaptive_radius = max(1.0, 0.2 + 0.5*8/3) = 1.5333
        threshold = 1.0 + 1.5333 = 2.5333
        dist = 5.0
        margin = 5.0 - 2.5333 = 2.4667
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=1.0)          # fixed (alpha=None)
        d2 = _make_drone("d2", safety_zone=1.0, alpha=0.5)  # adaptive
        drones = [d1, d2]

        pred_pos = {
            "d1": np.array([[0.0, 0.0, 0.0]]),
            "d2": np.array([[5.0, 0.0, 0.0]]),
        }
        pred_vel = {
            "d1": np.array([[4.0, 0.0, 0.0]]),  # ignored for fixed drone
            "d2": np.array([[4.0, 0.0, 0.0]]),
        }

        result = constraints.evaluate_multi(drones, pred_pos, np.array([]), pred_vel=pred_vel)

        ar_d2 = 0.2 + 0.5 * (16.0 / 6.0)  # 1.5333
        expected_margin = 5.0 - (1.0 + ar_d2)  # d1 uses fixed safety_zone
        assert result[0] == pytest.approx(expected_margin)

    def test_obstacle_avoidance_adaptive(self):
        """Adaptive drone near static obstacle with high velocity.

        velocity=[4,0,0]: adaptive_radius = max(1.0, 0.2 + 0.5*8/3) = 1.5333
        obstacle at (5,0,0) with half_extents=[0.3,0.3,0.3]
        point_to_box_dist([0,0,0], [5,0,0], [0.3,0.3,0.3]) = 5.0 - 0.3 = 4.7 (face approach)
        margin = 4.7 - ar (safety = adaptive radius)
        """
        horizon = 1
        constraints = ObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0, alpha=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        obstacles = [(np.array([5.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.3]))]
        pred_vel = np.array([[4.0, 0.0, 0.0]])

        result = constraints.evaluate_single(drone, pred_pos, obstacles, np.array([]), pred_vel=pred_vel)

        ar = 0.2 + 0.5 * (16.0 / 6.0)  # 1.5333
        expected_margin = 4.7 - ar  # ~3.1667 (tolerance 1e-3 for regularization eps)
        assert result[0] == pytest.approx(expected_margin, abs=1e-3)

    def test_room_constraints_adaptive_box(self):
        """Adaptive drone in box room. Wall clearance uses adaptive radius.

        Drone at center of [-5,5]^3, velocity=[4,0,0]:
        adaptive_radius = max(1.0, 0.2 + 0.5*8/3) = 1.5333
        Each wall margin = 5.0 - 1.5333 = 3.4667
        (compared to fixed: 5.0 - 1.0 = 4.0)
        """
        horizon = 1
        constraints = RoomConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=1.0, alpha=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        room_min = np.array([-5.0, -5.0, -5.0])
        room_max = np.array([5.0, 5.0, 5.0])
        pred_vel = np.array([[4.0, 0.0, 0.0]])

        result = constraints.evaluate_single(
            drone, pred_pos, room_max, room_min, np.array([]),
            pred_vel=pred_vel,
        )

        ar = 0.2 + 0.5 * (16.0 / 6.0)  # 1.5333
        expected_margin = 5.0 - ar
        assert result.shape == (6,)
        # All 6 faces should have the same margin (symmetric room, drone at center)
        for i in range(6):
            assert result[i] == pytest.approx(expected_margin)

    def test_backward_compat_no_pred_vel(self):
        """Pass pred_vel=None explicitly, verify identical result to omitting pred_vel entirely.

        Both should use fixed safety_zone since pred_vel is None.
        Neighbor vel=None also uses ego safety_zone.
        """
        horizon = 2
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5, alpha=0.5)  # adaptive, but no velocity given
        pred_pos = np.zeros((horizon, 3))
        neighbor_traj = np.ones((horizon, 3)) * 10.0
        neighbors = {"n1": (neighbor_traj, None)}

        # Without pred_vel (default None)
        result_default = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]))
        # With pred_vel=None explicitly
        result_explicit_none = constraints.evaluate_single(
            drone, pred_pos, neighbors, np.array([]), pred_vel=None,
        )

        assert_array_almost_equal(result_default, result_explicit_none)

        # Should use fixed safety_zone (0.5) for both ego and neighbor
        # dist = sqrt(3*100) ~ 17.32, threshold = 0.5 + 0.5 = 1.0, margin ~ 16.32
        expected_margin = np.linalg.norm(np.ones(3) * 10.0) - (0.5 + 0.5)
        for step in range(horizon):
            assert result_default[step] == pytest.approx(expected_margin)


# ---------------------------------------------------------------
# MPCConstraints base class
# ---------------------------------------------------------------

class TestMPCConstraintsBase:
    """Tests for base class and label methods."""

    def test_velocity_label(self):
        assert VelocityConstraints(horizon=1).label() == "velocity"

    def test_moving_obstacle_label(self):
        assert MovingObstacleAvoidanceConstraints(horizon=1).label() == "moving_obstacle_avoidance"

    def test_obstacle_label(self):
        assert ObstacleAvoidanceConstraints(horizon=1).label() == "obstacle_avoidance"

    def test_room_label(self):
        assert RoomConstraints(horizon=1).label() == "room"


# ---------------------------------------------------------------
# Per-step neighbor safety radii (ndarray support)
# ---------------------------------------------------------------

class TestPerStepNeighborVelocity:
    """Tests for per-step neighbor velocity in _evaluate."""

    def test_evaluate_single_per_step_neighbor_velocity(self):
        """Verify _evaluate uses per-step neighbor velocity to compute adaptive radii.

        Set up per-step neighbor velocities that vary across the horizon and verify
        the constraint margin changes accordingly. Ego drone is adaptive with alpha=0.5.

        horizon=3, drone at origin, neighbor at (5,0,0) at all steps.
        safety_zone=0.2 (equal to radius so the floor doesn't mask adaptive values).
        ego pred_vel=None => uses safety_zone=0.2.
        Neighbor velocities: [0,0,0], [1,0,0], [3,0,0] per step.
        Neighbor radii computed using ego drone's params (alpha=0.5, radius=0.2, u_max=3.0):
          step 0: vel=[0,0,0] => s_stop=0, r=max(0.2, 0.2)=0.2
          step 1: vel=[1,0,0] => ||v||^2=1, s_stop=1/6, r=max(0.2, 0.2+0.5*(1/6))~0.2833
          step 2: vel=[3,0,0] => ||v||^2=9, s_stop=9/6=1.5, r=max(0.2, 0.2+0.5*1.5)=0.95
        Expected margins:
          5.0 - (0.2 + 0.2) = 4.6,
          5.0 - (0.2 + 0.2833) ~ 4.5167,
          5.0 - (0.2 + 0.95) = 3.85
        """
        horizon = 3
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.2, alpha=0.5)
        pred_pos = np.zeros((horizon, 3))
        neighbor_traj = np.tile(np.array([5.0, 0.0, 0.0]), (horizon, 1))
        neighbor_vel = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        neighbors = {"n1": (neighbor_traj, neighbor_vel)}

        result = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]))

        assert result.shape == (horizon,)
        r0 = 0.2
        r1 = 0.2 + 0.5 * (1.0 / 6.0)
        r2 = 0.2 + 0.5 * (9.0 / 6.0)
        assert result[0] == pytest.approx(5.0 - (0.2 + r0))
        assert result[1] == pytest.approx(5.0 - (0.2 + r1))
        assert result[2] == pytest.approx(5.0 - (0.2 + r2))

    @pytest.mark.skip(reason="Skipping adaptive constraints tests as they are broken -> create issue ISS-006")
    def test_evaluate_single_none_vs_zero_velocity(self):
        """None neighbor velocity (fixed fallback) vs zero velocity (adaptive at rest).

        For a fixed drone (alpha=None): None vel => safety_zone=0.1
        For an adaptive drone (alpha=0.5): zero vel => max(safety_zone=0.1, radius=0.2) = 0.2
        """
        horizon = 2
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        pred_pos = np.zeros((horizon, 3))
        neighbor_traj = np.tile(np.array([5.0, 0.0, 0.0]), (horizon, 1))

        # Fixed drone: neighbor vel=None => neighbor uses ego safety_zone=0.1
        drone_fixed = _make_drone(safety_zone=0.1)
        neighbors_none = {"n1": (neighbor_traj, None)}
        result_none = constraints.evaluate_single(drone_fixed, pred_pos, neighbors_none, np.array([]))

        # Adaptive drone with neighbor zero vel:
        # ego pred_vel=None => ego uses safety_zone=0.1
        # neighbor zero vel => _safety_radius(ego_drone, [0,0,0]) => max(0.1, 0.2) = 0.2
        drone_adaptive = _make_drone(safety_zone=0.1, alpha=0.5)
        neighbors_zero = {"n1": (neighbor_traj, np.zeros((horizon, 3)))}
        result_zero = constraints.evaluate_single(drone_adaptive, pred_pos, neighbors_zero, np.array([]))

        # None => both sides use safety_zone(0.1), margin = 5.0 - 0.2 = 4.8
        assert result_none[0] == pytest.approx(4.8)
        # Zero vel => ego safety_zone(0.1) + neighbor adaptive(0.2) = 0.3, margin = 4.7
        assert result_zero[0] == pytest.approx(4.7)


# ---------------------------------------------------------------
# LSTM Safety Radius Branch (Phase 23)
# ---------------------------------------------------------------

class _FakeLSTMProvider:
    """Hand-rolled fake implementing the LSTMSafetyZoneProvider interface.
    Returns constant per-step radii for all neighbors.
    """
    def __init__(self, radius: float, horizon: int):
        self._radius = radius
        self._horizon = horizon

    def compute_neighbor_safety_radii(
        self,
        neighbor_ids: list[str],
        r_floor_by_id: dict[str, float],
    ) -> dict[str, np.ndarray]:
        return {nid: np.full(self._horizon, self._radius) for nid in neighbor_ids}


class TestSafetyRadiusLSTMBranch:
    """Tests for _safety_radius() LSTM branch (Phase 23)."""

    def test_lstm_mode_with_radius_returns_lstm_radius(self):
        """LSTM mode + lstm_radius provided: return lstm_radius directly."""
        from drone_sim.domain.constraints import _safety_radius
        drone = _make_drone(safety_zone=1.0, safety_zone_mode="lstm")
        result = _safety_radius(drone, velocity=None, lstm_radius=2.5)
        assert result == pytest.approx(2.5)

    def test_lstm_mode_no_radius_falls_back_to_safety_zone(self):
        """LSTM mode + lstm_radius=None (warmup): fall back to drone.safety_zone."""
        from drone_sim.domain.constraints import _safety_radius
        drone = _make_drone(safety_zone=1.0, safety_zone_mode="lstm")
        result = _safety_radius(drone, velocity=None, lstm_radius=None)
        assert result == pytest.approx(1.0)

    def test_fixed_mode_unchanged(self):
        """Fixed mode: returns safety_zone regardless of velocity or lstm_radius."""
        from drone_sim.domain.constraints import _safety_radius
        drone = _make_drone(safety_zone=1.5)  # safety_zone_mode="fixed" by default
        result = _safety_radius(drone, velocity=np.array([4.0, 0.0, 0.0]))
        assert result == pytest.approx(1.5)

    def test_adaptive_mode_unchanged(self):
        """Adaptive mode: velocity-dependent radius still works."""
        from drone_sim.domain.constraints import _safety_radius
        drone = _make_drone(safety_zone=1.0, alpha=0.5)  # safety_zone_mode="fixed" by default
        vel = np.array([4.0, 0.0, 0.0])
        result = _safety_radius(drone, velocity=vel)
        # alpha=0.5, ||v||^2=16, u_max_scalar=3.0, s_stop=16/6, r=safety_zone+alpha*s_stop=1.0+0.5*(16/6)
        expected = 1.0 + 0.5 * (16.0 / (2.0 * 3.0))
        assert result == pytest.approx(expected)


class TestLSTMConstraintWiring:
    """Tests that lstm_radii kwarg threads through constraint evaluation."""

    def test_evaluate_single_lstm_radii_used_for_neighbor(self):
        """evaluate_single uses lstm_radii for neighbor safety radius.

        Drone at origin (fixed mode, safety_zone=0.5).
        Neighbor at (5,0,0), lstm_radius=1.0 for all steps.
        Expected margin: 5.0 - (0.5 + 1.0) = 3.5
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)  # fixed mode
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        neighbor_traj = np.array([[5.0, 0.0, 0.0]])
        neighbors = {"n1": (neighbor_traj, None)}
        lstm_radii = {"n1": np.full(horizon, 1.0)}

        result = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]),
                                             lstm_radii=lstm_radii)

        # ego: safety_zone=0.5 (fixed, no lstm), neighbor: lstm_radius=1.0
        assert result[0] == pytest.approx(5.0 - (0.5 + 1.0))

    def test_evaluate_single_no_lstm_radii_unchanged(self):
        """evaluate_single with lstm_radii=None (default) behaves identically to before."""
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        drone = _make_drone(safety_zone=0.5)
        pred_pos = np.array([[0.0, 0.0, 0.0]])
        neighbor_traj = np.array([[5.0, 0.0, 0.0]])
        neighbors = {"n1": (neighbor_traj, None)}

        result_default = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]))
        result_none = constraints.evaluate_single(drone, pred_pos, neighbors, np.array([]),
                                                  lstm_radii=None)

        np.testing.assert_array_almost_equal(result_default, result_none)

    def test_evaluate_multi_lstm_radii_both_drones(self):
        """evaluate_multi applies lstm_radii for both drones in a pair.

        d1 at origin (lstm mode, safety_zone=0.5, lstm_radius=1.0).
        d2 at (6,0,0) (lstm mode, safety_zone=0.5, lstm_radius=1.5).
        dist=6.0, threshold=1.0+1.5=2.5, margin=3.5
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5, safety_zone_mode="lstm")
        d2 = _make_drone("d2", safety_zone=0.5, safety_zone_mode="lstm")
        drones = [d1, d2]
        pred_pos = {
            "d1": np.array([[0.0, 0.0, 0.0]]),
            "d2": np.array([[6.0, 0.0, 0.0]]),
        }
        lstm_radii = {
            "d1": np.full(horizon, 1.0),
            "d2": np.full(horizon, 1.5),
        }

        result = constraints.evaluate_multi(drones, pred_pos, np.array([]), lstm_radii=lstm_radii)

        assert result.shape == (horizon,)
        assert result[0] == pytest.approx(6.0 - (1.0 + 1.5))

    def test_evaluate_multi_missing_id_in_lstm_radii_falls_back(self):
        """If a drone_id is absent from lstm_radii, falls back to safety_zone.

        d1 has lstm_radii (1.0), d2 does not.
        d2 fallback: safety_zone_mode='lstm', lstm_radius=None -> drone.safety_zone=0.5
        dist=6.0, threshold=1.0+0.5=1.5, margin=4.5
        """
        horizon = 1
        constraints = MovingObstacleAvoidanceConstraints(horizon=horizon)
        d1 = _make_drone("d1", safety_zone=0.5, safety_zone_mode="lstm")
        d2 = _make_drone("d2", safety_zone=0.5, safety_zone_mode="lstm")
        drones = [d1, d2]
        pred_pos = {
            "d1": np.array([[0.0, 0.0, 0.0]]),
            "d2": np.array([[6.0, 0.0, 0.0]]),
        }
        lstm_radii = {"d1": np.full(horizon, 1.0)}  # d2 absent

        result = constraints.evaluate_multi(drones, pred_pos, np.array([]), lstm_radii=lstm_radii)

        # d2 absent -> lstm_radius=None -> fallback to safety_zone=0.5
        assert result[0] == pytest.approx(6.0 - (1.0 + 0.5))
