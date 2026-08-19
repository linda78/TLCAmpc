"""Stateless conflict-detection primitives.

Pure functions — no persistent state. Sibling to
:mod:`drone_sim.simulation.utils.conflict_detection`, which holds the
stateful waypoint-based evasion strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from drone_sim.domain.drone import Drone

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntersectionEvent:
   """A predicted close encounter between two drones over the MPC horizon.

   :param pair: Frozenset of the two drone IDs (order-invariant).
   :param detection_point: Midpoint of the two trajectories at the closest
          approach step.
   :param k_min: Index of the closest-approach step in the horizon.
   :param min_dist: Minimum predicted distance between the two trajectories.
   :param ttc: dict mapping drone_id to time-to-collision in seconds
          (k_min * dt) — symmetric in this baseline implementation,
          kept as a dict for future asymmetric variants.
   """

   pair: frozenset[str]
   detection_point: np.ndarray
   k_min: int
   min_dist: float
   ttc: dict[str, float]


def compute_threshold(drones: Sequence[Drone]) -> float:
   """Trigger threshold ``2 * min(d.safety_zone for d in drones)``."""
   if not drones:
      return 0.0
   return 2.0 * float(min(d.safety_zone for d in drones))


def constant_velocity_trajectory(position: np.ndarray, velocity: np.ndarray, horizon: int, dt: float) -> np.ndarray:
   """Predict an ``(horizon, 3)`` trajectory by constant-velocity extrapolation."""
   pos = np.asarray(position, dtype=float).reshape(3)
   vel = np.asarray(velocity, dtype=float).reshape(3)
   steps = np.arange(1, horizon + 1, dtype=float)[:, None]
   return pos[None, :] + steps * dt * vel[None, :]


def closest_approach(traj_i: np.ndarray, traj_j: np.ndarray) -> tuple[float, int, np.ndarray, np.ndarray]:
   """Find the step of closest approach between two predicted trajectories.

   :param traj_i: Predicted positions for drone i, shape ``(H, 3)``.
   :param traj_j: Predicted positions for drone j, shape ``(H, 3)``.
   :return: Tuple ``(min_dist, k_min, point_i, point_j)`` where ``min_dist``
            is the Euclidean closest-approach distance over the horizon,
            ``k_min`` the step index, and ``point_*`` the corresponding
            position samples.
   """
   ti = np.asarray(traj_i, dtype=float)
   tj = np.asarray(traj_j, dtype=float)
   if ti.shape != tj.shape or ti.ndim != 2 or ti.shape[1] != 3:
      raise ValueError(f"closest_approach expects two (H,3) trajectories, got {ti.shape} and {tj.shape}")
   dists = np.linalg.norm(ti - tj, axis=1)
   k_min = int(np.argmin(dists))
   return float(dists[k_min]), k_min, ti[k_min].copy(), tj[k_min].copy()


def is_closing(drone_i: Drone, drone_j: Drone, eps: float = 1e-9) -> bool:
   """Whether two drones are currently approaching each other.

   Evaluated on the current state (not the predicted trajectory). Sign
   of ``dot(rel_pos, rel_vel)``: negative means closing. Filters out
   pairs whose horizon still dips below threshold even though they have
   already passed each other.
   """
   pi = np.asarray(drone_i.x, dtype=float)[:3]
   pj = np.asarray(drone_j.x, dtype=float)[:3]
   vi = np.asarray(drone_i.x, dtype=float)[3:6]
   vj = np.asarray(drone_j.x, dtype=float)[3:6]
   rel_pos = pj - pi
   rel_vel = vj - vi
   norm = float(np.linalg.norm(rel_pos)) + eps
   closing = float(np.dot(rel_pos, rel_vel) / norm)
   return closing < 0.0


def detect_intersections(drones: Sequence[Drone], predicted_trajs: dict[str, np.ndarray], threshold: float, dt: float) -> list[IntersectionEvent]:
   """Pairs whose predicted trajectories cross within ``threshold``.

   Pairs not currently closing (per :func:`is_closing`) are skipped.

   :param drones: Drones in the scenario.
   :param predicted_trajs: ``drone_id -> (H, 3)``; missing drones skipped.
   :param threshold: From :func:`compute_threshold`.
   :param dt: Simulation step length (for TTC).
   :return: One :class:`IntersectionEvent` per offending pair.
   """
   events: list[IntersectionEvent] = []
   eligible: list[Drone] = [d for d in drones if d.drone_id in predicted_trajs]
   skipped_diverging = 0
   for drone_i, drone_j in combinations(eligible, 2):
      traj_i = predicted_trajs[drone_i.drone_id]
      traj_j = predicted_trajs[drone_j.drone_id]
      min_dist, k_min, p_i, p_j = closest_approach(traj_i, traj_j)
      if min_dist >= threshold:
         continue
      if not is_closing(drone_i, drone_j):
         skipped_diverging += 1
         _log.debug("intersection: SKIP pair=%s diverging (min_dist=%.3f < threshold=%.3f)", sorted((drone_i.drone_id, drone_j.drone_id)), min_dist,
               threshold)
         continue
      detection_point = 0.5 * (p_i + p_j)
      ttc_value = float(k_min) * float(dt)
      events.append(IntersectionEvent(pair=frozenset({drone_i.drone_id, drone_j.drone_id}), detection_point=detection_point, k_min=k_min, min_dist=min_dist,
            ttc={drone_i.drone_id: ttc_value, drone_j.drone_id: ttc_value}))
      _log.info("intersection: DETECT pair=%s min_dist=%.3f threshold=%.3f k_min=%d detection_point=%s", sorted((drone_i.drone_id, drone_j.drone_id)), min_dist,
            threshold, k_min, np.round(detection_point, 3).tolist())
   if _log.isEnabledFor(logging.DEBUG):
      _log.debug("intersection: detect_intersections threshold=%.3f n_eligible=%d n_events=%d n_diverging_skipped=%d", threshold, len(eligible), len(events),
            skipped_diverging)
   return events


def pair_min_distance(traj_i: np.ndarray, traj_j: np.ndarray) -> tuple[float, int]:
   """Minimum distance and its step index between two trajectories.

   Convenience wrapper around :func:`closest_approach` for callers that
   only need scalar outputs (e.g. resolution checks in the state store).
   """
   min_dist, k_min, _, _ = closest_approach(traj_i, traj_j)
   return min_dist, k_min


__all__: Iterable[str] = ("IntersectionEvent", "closest_approach", "compute_threshold", "constant_velocity_trajectory", "detect_intersections", "is_closing",
                          "pair_min_distance",)
