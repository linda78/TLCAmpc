"""Tests for the stateless conflict-detection primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from drone_sim.simulation.utils.stateless_conflict_detection import (
   closest_approach,
   compute_threshold,
   constant_velocity_trajectory,
   detect_intersections,
   is_closing,
)


@dataclass
class _StubDrone:
   """Light-weight stand-in. Carries a state vector so ``is_closing`` works."""

   drone_id: str
   safety_zone: float
   x: np.ndarray = field(default_factory=lambda: np.zeros(6))


def _stub(drone_id: str, safety_zone: float, pos=(0, 0, 0), vel=(0, 0, 0)) -> _StubDrone:
   x = np.zeros(6)
   x[:3] = pos
   x[3:6] = vel
   return _StubDrone(drone_id=drone_id, safety_zone=safety_zone, x=x)


# ---------- threshold -------------------------------------------------------


def test_threshold_uses_minimum_safety_zone():
   drones = [_stub("a", 0.5), _stub("b", 1.2), _stub("c", 0.8)]
   assert compute_threshold(drones) == pytest.approx(1.0)


def test_threshold_handles_empty_list():
   assert compute_threshold([]) == 0.0


def test_threshold_heterogeneous_safety_zones():
   drones = [_stub("a", 0.3), _stub("b", 1.0), _stub("c", 0.7)]
   assert compute_threshold(drones) == pytest.approx(0.6)


# ---------- constant_velocity_trajectory ------------------------------------


def test_constant_velocity_extrapolation():
   pos = np.array([0.0, 0.0, 0.0])
   vel = np.array([1.0, 0.0, 0.0])
   traj = constant_velocity_trajectory(pos, vel, horizon=3, dt=0.5)
   expected = np.array([[0.5, 0, 0], [1.0, 0, 0], [1.5, 0, 0]])
   np.testing.assert_allclose(traj, expected)


# ---------- closest_approach ------------------------------------------------


def test_closest_approach_returns_midpoint_index_for_head_on():
   traj_i = np.stack([np.array([t, 0, 0]) for t in [0.0, 1.0, 2.0, 3.0]])
   traj_j = np.stack([np.array([3.0 - t, 0, 0]) for t in [0.0, 1.0, 2.0, 3.0]])
   min_dist, k_min, p_i, p_j = closest_approach(traj_i, traj_j)
   assert k_min in (1, 2)
   assert min_dist == pytest.approx(1.0)


def test_closest_approach_rejects_shape_mismatch():
   ti = np.zeros((4, 3))
   tj = np.zeros((3, 3))
   with pytest.raises(ValueError):
      closest_approach(ti, tj)


# ---------- is_closing ------------------------------------------------------


def test_is_closing_true_for_head_on_pair():
   a = _stub("a", 0.5, pos=(0, 0, 0), vel=(1, 0, 0))
   b = _stub("b", 0.5, pos=(4, 0, 0), vel=(-1, 0, 0))
   assert is_closing(a, b) is True
   # Symmetric.
   assert is_closing(b, a) is True


def test_is_closing_false_for_drones_moving_apart():
   a = _stub("a", 0.5, pos=(0, 0, 0), vel=(-1, 0, 0))
   b = _stub("b", 0.5, pos=(4, 0, 0), vel=(1, 0, 0))
   assert is_closing(a, b) is False


def test_is_closing_false_for_stationary_pair():
   a = _stub("a", 0.5, pos=(0, 0, 0), vel=(0, 0, 0))
   b = _stub("b", 0.5, pos=(4, 0, 0), vel=(0, 0, 0))
   assert is_closing(a, b) is False


def test_is_closing_false_for_parallel_pair():
   # Same direction, offset on y — dot(rel_pos, rel_vel) = 0.
   a = _stub("a", 0.5, pos=(0, 0, 0), vel=(1, 0, 0))
   b = _stub("b", 0.5, pos=(0, 3, 0), vel=(1, 0, 0))
   assert is_closing(a, b) is False


# ---------- detect_intersections --------------------------------------------


def test_detect_intersection_head_on_pair():
   horizon = 5
   dt = 1.0
   drone_a = _stub("a", 0.5, pos=(0, 0, 0), vel=(1, 0, 0))
   drone_b = _stub("b", 0.5, pos=(4, 0, 0), vel=(-1, 0, 0))
   traj_a = constant_velocity_trajectory(drone_a.x[:3], drone_a.x[3:6], horizon, dt)
   traj_b = constant_velocity_trajectory(drone_b.x[:3], drone_b.x[3:6], horizon, dt)
   events = detect_intersections(
      drones=[drone_a, drone_b],
      predicted_trajs={"a": traj_a, "b": traj_b},
      threshold=compute_threshold([drone_a, drone_b]),
      dt=dt,
   )
   assert len(events) == 1
   event = events[0]
   assert event.pair == frozenset({"a", "b"})
   np.testing.assert_allclose(event.detection_point[1:], [0.0, 0.0])
   assert event.detection_point[0] == pytest.approx(2.0, abs=0.6)
   assert event.ttc["a"] == event.ttc["b"]


def test_detect_intersection_parallel_pair_skipped():
   horizon = 5
   dt = 1.0
   drone_a = _stub("a", 0.5, pos=(0, 0, 0), vel=(1, 0, 0))
   drone_b = _stub("b", 0.5, pos=(0, 3, 0), vel=(1, 0, 0))
   traj_a = constant_velocity_trajectory(drone_a.x[:3], drone_a.x[3:6], horizon, dt)
   traj_b = constant_velocity_trajectory(drone_b.x[:3], drone_b.x[3:6], horizon, dt)
   events = detect_intersections(
      drones=[drone_a, drone_b],
      predicted_trajs={"a": traj_a, "b": traj_b},
      threshold=compute_threshold([drone_a, drone_b]),
      dt=dt,
   )
   assert events == []


def test_detect_intersection_diverging_pair_skipped_by_closing_gate():
   """Trajectories happen to dip below threshold within the horizon but the
   drones are *already past each other* — closing-velocity gate filters it.
   """
   horizon = 5
   dt = 1.0
   # Just past each other: a at +0.5, b at -0.5 (relative), both moving +x.
   # rel_pos = (b - a) = (-1,0,0); rel_vel = (b_v - a_v) = (1,0,0) → dot < 0?
   # dot = -1 → is_closing TRUE. Wrong setup. Let me invert.
   # Drone a is left, moving right; drone b is right, moving right faster.
   # They're diverging: rel_pos (b → a) = -x, rel_vel (b → a) = +x.
   # Actually use: a at x=0 moving +x at 1; b at x=2 moving +x at 2. b pulls away.
   drone_a = _stub("a", 0.5, pos=(0, 0, 0), vel=(1, 0, 0))
   drone_b = _stub("b", 0.5, pos=(2, 0, 0), vel=(2, 0, 0))
   # Predicted trajectories: a goes 0→5, b goes 2→12 → gap grows, no intersection.
   # To force min_dist < threshold inside horizon we craft trajectories manually.
   traj_a = np.stack([np.array([t, 0, 0]) for t in [1.0, 2.0, 3.0, 4.0, 5.0]])
   # b moves forward but slowly enough that a catches up briefly within prediction
   traj_b = np.stack([np.array([t, 0, 0]) for t in [2.4, 3.0, 3.6, 4.2, 4.8]])
   # min predicted distance is small (a catches b), but current state says a is
   # behind b moving slower → not actually closing now.
   # Override: make rel_vel diverging in current state.
   drone_a = _stub("a", 0.5, pos=(0.0, 0, 0), vel=(0.5, 0, 0))
   drone_b = _stub("b", 0.5, pos=(2.4, 0, 0), vel=(0.6, 0, 0))
   events = detect_intersections(
      drones=[drone_a, drone_b],
      predicted_trajs={"a": traj_a, "b": traj_b},
      threshold=1.0,
      dt=1.0,
   )
   # Current rel_pos = (2.4, 0, 0), rel_vel = (0.1, 0, 0); dot > 0 → diverging.
   assert is_closing(drone_a, drone_b) is False
   assert events == []
