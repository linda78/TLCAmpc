from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from drone_sim.domain.drone import Drone


def points_to_box_dist(
    points: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    eps: float = 1e-3,
) -> np.ndarray:
    """Vectorized smooth point-to-box distance for multiple points.

    :param points: Query points (H, 3).
    :param center: Box center (3,).
    :param half_extents: Box half-sizes per axis (3,).
    :param eps: Regularization constant.
    :return: Smooth distances (H,).
    """
    d = np.abs(points - center) - half_extents
    relu_d = np.maximum(d, 0.0)
    return np.sqrt(np.sum(relu_d ** 2, axis=1) + eps * eps) - eps


def point_to_box_dist(
    point: np.ndarray,
    center: np.ndarray,
    half_extents: np.ndarray,
    eps: float = 1e-3,
) -> float:
    """Smooth point-to-axis-aligned-box distance, SLSQP-compatible.

    Uses a regularized L2 norm of per-axis excess: sqrt(sum(relu(d)^2) + eps^2) - eps
    where d_i = |point_i - center_i| - half_extents_i.

    This is positive outside the box, approaches 0 at the box surface, and
    has a non-zero gradient everywhere (avoiding the flat zero-gradient region
    inside the box that would stall SLSQP).

    :param point: 3D query point.
    :param center: Box center (3,).
    :param half_extents: Box half-sizes per axis (3,).
    :param eps: Regularization constant — controls smoothness near the surface.
    :return: Smooth distance >= 0.
    """
    d = np.abs(point - center) - half_extents         # signed per-axis excess
    relu_d = np.maximum(d, 0.0)                        # only positive excess matters
    return float(np.sqrt(np.dot(relu_d, relu_d) + eps * eps) - eps)


def _velocity_at_step(pred_vel: np.ndarray | None, step: int) -> np.ndarray | None:
   """Extract the velocity vector at a given step, or ``None`` if unavailable."""
   if pred_vel is not None:
      return pred_vel[step]
   return None


def _safety_radius(drone: Drone, velocity: np.ndarray | None) -> float:
   """Compute the safety radius for a drone at a given velocity.

   :param drone: the drone to compute the radius for.
   :param velocity: 3D velocity vector, or ``None`` to use the fixed safety zone.
   :return: adaptive radius when the drone is adaptive and velocity is provided,
            otherwise the fixed ``drone.safety_zone``.
   """
   if velocity is not None and drone.is_adaptive:
      return drone.compute_adaptive_radius(velocity)
   return drone.safety_zone


def _safety_radii(drone: Drone, pred_vel: np.ndarray | None, horizon: int) -> np.ndarray:
   """Compute safety radii for all steps at once.

   :param drone: the drone.
   :param pred_vel: predicted velocities (H, 3) or None.
   :param horizon: number of steps.
   :return: safety radii array (H,).
   """
   if pred_vel is not None and drone.is_adaptive:
      speeds = np.linalg.norm(pred_vel[:horizon], axis=1)
      u_min, u_max = drone.bounds()
      u_max_scalar = float(np.max(np.abs(u_max)))
      s_stop = speeds ** 2 / (2.0 * u_max_scalar)
      return drone.safety_zone + drone.alpha * s_stop
   return np.full(horizon, drone.safety_zone)


class MPCConstraints(ABC):
   """Base class for constraints."""

   def __init__(self, horizon: int):
      self._horizon = horizon

   @abstractmethod
   def label(self) -> str:
      pass


class VelocityConstraints(MPCConstraints):
   def label(self) -> str:
      return "velocity"

   def evaluate_single(self, drone_state: Drone, v_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
      """Evaluate velocity constraints for a single drone.

      :param drone_state: the drone.
      :param v_pred: predicted velocity per time step (shape: (horizon, 3)).
      :param values: combined constraint values to append to.
      :return: updated values list.
      """
      speed_sq = np.sum(v_pred[:self._horizon] ** 2, axis=1)
      result = drone_state.v_max ** 2 - speed_sq
      return np.concatenate([values, result])

   def evaluate_multi(self, drone_states: list[Drone], v_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
      """Evaluate velocity constraints for multiple drones.

      :param drone_states: list of drones.
      :param v_pred: predicted velocity per drone per time step (shape: (num_drones, horizon, 3)).
      :param values: combined constraint values to append to.
      :return: updated values list.
      """
      parts = []
      for drone_idx, drone in enumerate(drone_states):
         speed_sq = np.sum(v_pred[drone_idx, :self._horizon] ** 2, axis=1)
         parts.append(drone.v_max ** 2 - speed_sq)
      return np.concatenate([values, *parts])


class MovingObstacleAvoidanceConstraints(MPCConstraints):
   """Pairwise collision avoidance: at every horizon step the distance of two drones must exceed the sum of their safety radii.

   With ``terminal_constraint`` (default off, coordinator parameter ``terminal_constraint_pairs``) every pair gets one
   additional row for the last horizon step that uses the terminal radius ``Drone.compute_terminal_radius`` (base radius
   plus the full stopping distance plus half a sampling step) of both drones instead of the step radii, the pair
   counterpart of the wall rows in ``RoomConstraints``. The row is appended after the H step rows of that pair. Calls
   without velocities get no terminal rows. In the single-drone form the neighbour's terminal radius is computed with the
   ego drone's parameters ("same drone type"); a neighbour without velocities keeps its last step radius.
   """

   def __init__(self, horizon: int, terminal_constraint: bool = False):
      super().__init__(horizon)
      self._terminal_constraint = terminal_constraint

   def label(self) -> str:
      return "moving_obstacle_avoidance"

   def evaluate_single(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray | None]],
      values: np.ndarray,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      """Evaluate moving obstacle avoidance constraints for a single drone.

      :param drone: the drone.
      :param pred_pos: predicted positions per time step (shape: (horizon, 3)).
      :param neighbor_trajectories: mapping of neighbor id to
             (trajectory (H,3), predicted_velocities (H,3) or None).
             Uses ego drone's config to compute neighbor adaptive radii.
      :param values: combined constraint values to append to.
      :param pred_vel: optional predicted velocities (shape: (horizon, 3)).
                       When provided and the drone is adaptive, per-step adaptive radii are used.
      :return: updated values list.
      """
      result = self._evaluate(drone, pred_pos, neighbor_trajectories, pred_vel=pred_vel)
      return np.concatenate([values, result])

   def evaluate_multi(
      self,
      drones: list[Drone],
      pred_pos: dict[str, np.ndarray],
      values: np.ndarray,
      pred_vel: dict[str, np.ndarray] | None = None,
   ) -> np.ndarray:
      """Evaluate collision avoidance for all drone pairs.

      Uses i<j pairs to avoid duplicates. Includes cons_stop since all drone
      info is available in the multi-drone case.

      :param drones: list of drones.
      :param pred_pos: predicted positions keyed by drone_id.
      :param values: combined constraint values to append to.
      :param pred_vel: optional dict mapping drone_id to predicted velocity arrays
                       (shape per drone: (horizon, 3)). When provided, adaptive
                       drones use per-step adaptive radii.
      :return: updated values list.
      """
      parts = []
      for i in range(len(drones)):
         for j in range(i + 1, len(drones)):
            drone_i = drones[i]
            drone_j = drones[j]
            traj_i = pred_pos[drone_i.drone_id]
            traj_j = pred_pos[drone_j.drone_id]
            vel_i = pred_vel[drone_i.drone_id] if pred_vel is not None else None
            vel_j = pred_vel[drone_j.drone_id] if pred_vel is not None else None

            dists = np.linalg.norm(traj_i - traj_j, axis=1)
            radii_i = _safety_radii(drone_i, vel_i, self._horizon)
            radii_j = _safety_radii(drone_j, vel_j, self._horizon)
            cons_stop = drone_i.cons_stop + drone_j.cons_stop
            result = dists - (radii_i + radii_j + cons_stop)
            if self._terminal_constraint and vel_i is not None and vel_j is not None:
               last = self._horizon - 1
               r_term = drone_i.compute_terminal_radius(vel_i[last]) + drone_j.compute_terminal_radius(vel_j[last])
               result = np.append(result, dists[last] - (r_term + cons_stop))
            parts.append(result)
      return np.concatenate([values, *parts])

   def _evaluate(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray | None]],
      pred_vel: np.ndarray | None = None,
   ):
      """Evaluate collision constraints against neighbor trajectories.

      :param drone: the ego drone.
      :param pred_pos: predicted positions (horizon, 3).
      :param neighbor_trajectories: neighbor data mapping id to
             (trajectory (H,3), predicted_velocities (H,3) or None).
             Neighbor adaptive radii are computed using the ego drone's config
             ("same drone type" assumption).
      :param pred_vel: optional predicted velocities for the ego drone (horizon, 3).
      :return: constraint margin array.
      """
      parts = []
      last = self._horizon - 1
      ego_radii = _safety_radii(drone, pred_vel, self._horizon)
      # For non-adaptive drones, all radii are identical — compute once
      fixed_radii = None if drone.is_adaptive else ego_radii
      r_term_ego = drone.compute_terminal_radius(pred_vel[last]) if self._terminal_constraint and pred_vel is not None else None
      for neighbor_traj, neighbor_vel in neighbor_trajectories.values():
         neighbor_traj = np.asarray(neighbor_traj, dtype=float).reshape((self._horizon, 3))
         dists = np.linalg.norm(pred_pos - neighbor_traj, axis=1)
         neighbor_radii = fixed_radii if fixed_radii is not None else _safety_radii(drone, neighbor_vel, self._horizon)
         result = dists - (ego_radii + neighbor_radii)
         if r_term_ego is not None:
            # neighbour terminal radius with the ego parameters ("same drone type"); without velocities its last step radius
            r_term_nbr = drone.compute_terminal_radius(np.asarray(neighbor_vel, dtype=float)[last]) if neighbor_vel is not None else neighbor_radii[last]
            result = np.append(result, dists[last] - (r_term_ego + r_term_nbr))
         parts.append(result)
      if not parts:
         return np.zeros(0)
      return np.concatenate(parts)


class ObstacleAvoidanceConstraints(MPCConstraints):
   def label(self) -> str:
      return "obstacle_avoidance"

   def evaluate_single(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      obstacles: list[tuple[np.ndarray, np.ndarray]],
      values: np.ndarray,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      """Evaluate obstacle avoidance constraints for a single drone.

      :param drone: the drone.
      :param pred_pos: predicted position per time step (shape: (horizon, 3)).
      :param obstacles: list of obstacles, each a (center, half_extents) tuple.
      :param values: combined constraint values to append to.
      :param pred_vel: optional predicted velocities (shape: (horizon, 3)).
                       When provided and the drone is adaptive, per-step adaptive radii are used.
      :return: updated values list.
      """
      result = self._evaluate(drone, pred_pos, obstacles, pred_vel=pred_vel)
      return np.concatenate([values, result])

   def evaluate_multi(
      self,
      drones: list[Drone],
      pred_pos: dict[str, np.ndarray],
      obstacles: list[tuple[np.ndarray, np.ndarray]],
      values: np.ndarray,
      pred_vel: dict[str, np.ndarray] | None = None,
   ) -> np.ndarray:
      """Evaluate obstacle avoidance constraints for multiple drones.

      :param drones: list of Drones involved.
      :param pred_pos: predicted positions for the drones keyed by drone_id.
      :param obstacles: list of obstacles, each a (center, half_extents) tuple.
      :param values: combined constraint values to append to.
      :param pred_vel: optional dict mapping drone_id to predicted velocity arrays (shape: (horizon, 3)).
      :return: updated values list.
      """
      parts = []
      for drone in drones:
         vel = pred_vel[drone.drone_id] if pred_vel is not None else None
         parts.append(self._evaluate(drone, pred_pos[drone.drone_id], obstacles, pred_vel=vel))
      return np.concatenate([values, *parts])

   def _evaluate(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      obstacles: list[tuple[np.ndarray, np.ndarray]],
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      """Evaluate static obstacle constraints.

      Produces one horizon-length row per obstacle and concatenates them so
      that SLSQP treats each element as a separate inequality constraint.
      With N obstacles and horizon H the returned shape is (N*H,), or (H,)
      when there are no obstacles.

      :param drone: the drone.
      :param pred_pos: predicted positions (horizon, 3).
      :param obstacles: list of (center, half_extents) tuples.
      :param pred_vel: optional predicted velocities (horizon, 3).
      :return: constraint margin array.
      """
      parts = []
      radii = _safety_radii(drone, pred_vel, self._horizon)
      for center, half_extents in obstacles:
         obstacle_center = np.asarray(center, dtype=float).reshape(3)
         obstacle_half_extents = np.asarray(half_extents, dtype=float).reshape(3)
         dists = points_to_box_dist(pred_pos, obstacle_center, obstacle_half_extents)
         row = dists - radii
         parts.append(row)
      if not parts:
         return np.zeros(self._horizon)
      return np.concatenate(parts)


class RoomConstraints(MPCConstraints):
   """Room boundary constraints: the safety sphere of every horizon step must lie inside the room.

   With ``terminal_constraint`` (default on, switchable through the coordinator parameter of the same
   name) every drone gets one additional constraint for the last horizon step: there the sphere uses
   the terminal radius ``Drone.compute_terminal_radius`` (base radius plus the full stopping distance
   plus half a sampling step) instead of the step radius. The step constraints stay untouched, the
   terminal rows are appended after them (6 rows for a box room, 1 row for a sphere room). This
   guarantees that braking to a standstill is a feasible continuation of any plan, so the MPC problem
   stays feasible at the walls. Calls without ``pred_vel`` get no terminal rows.
   """

   def __init__(self, horizon: int, wall_tolerance: float = 0.0, terminal_constraint: bool = True):
      super().__init__(horizon)
      self._wall_tolerance = wall_tolerance
      self._terminal_constraint = terminal_constraint

   def label(self) -> str:
      return "room"

   def _terminal_radius(self, drone: Drone, pred_vel: np.ndarray | None) -> float | None:
      """Terminal radius for the last horizon step, or ``None`` when no terminal constraint applies."""
      if not self._terminal_constraint or pred_vel is None:
         return None
      return drone.compute_terminal_radius(pred_vel[self._horizon - 1])

   def evaluate_single(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      room_max: float,
      room_min: float,
      values: np.ndarray,
      room_is_sphere: bool = False,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      """Evaluate room boundary constraints for a single drone.

      :param drone: the drone.
      :param pred_pos: predicted positions per time step (shape: (horizon, 3)).
      :param room_max: room upper bounds (3,) for box or radius for sphere.
      :param room_min: room lower bounds (3,) for box or unused for sphere.
      :param values: combined constraint values to append to.
      :param room_is_sphere: if True, room_max is treated as sphere radius.
      :param pred_vel: optional predicted velocities (shape: (horizon, 3)).
                       When provided and the drone is adaptive, per-step adaptive radii are used.
      :return: updated values list.
      """
      result = self._evaluate(drone, pred_pos, room_max, room_min, room_is_sphere, pred_vel=pred_vel)
      return np.concatenate([values, result])

   def evaluate_multi(
      self,
      drones: list[Drone],
      pred_pos: dict[str, np.ndarray],
      room_max: float,
      room_min: float,
      values: np.ndarray,
      room_is_sphere: bool = False,
      pred_vel: dict[str, np.ndarray] | None = None,
   ) -> np.ndarray:
      """Evaluate room boundary constraints for multiple drones.

      :param drones: list of Drones involved.
      :param pred_pos: predicted positions for the drones keyed by drone_id.
      :param room_max: room upper bounds (3,) for box or radius for sphere.
      :param room_min: room lower bounds (3,) for box or unused for sphere.
      :param values: combined constraint values to append to.
      :param room_is_sphere: if True, room_max is treated as sphere radius.
      :param pred_vel: optional dict mapping drone_id to predicted velocity arrays (shape: (horizon, 3)).
      :return: updated values list.
      """
      parts = []
      for drone in drones:
         vel = pred_vel[drone.drone_id] if pred_vel is not None else None
         parts.append(self._evaluate(drone, pred_pos[drone.drone_id], room_max, room_min, room_is_sphere, pred_vel=vel))
      return np.concatenate([values, *parts])

   def _evaluate(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      room_max: float,
      room_min: float,
      room_is_sphere: bool = False,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      if room_is_sphere:
         return self._evaluate_sphere(drone, pred_pos, room_radius=room_max, pred_vel=pred_vel)
      return self._evaluate_box(drone, pred_pos, room_max, room_min, pred_vel=pred_vel)

   def _evaluate_box(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      room_max: float,
      room_min: float,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      lower_bounds = np.asarray(room_min, dtype=float).reshape(3)
      upper_bounds = np.asarray(room_max, dtype=float).reshape(3)
      radii = _safety_radii(drone, pred_vel, self._horizon)  # (H,)
      # pred_pos is (H, 3), radii is (H,) -> radii[:, None] is (H, 1)
      radii_col = radii[:, np.newaxis]
      lower_part = pred_pos - radii_col - lower_bounds + self._wall_tolerance  # (H, 3)
      upper_part = upper_bounds - (pred_pos + radii_col) + self._wall_tolerance  # (H, 3)
      # Interleave: for each step, 3 lower then 3 upper
      result = np.empty(self._horizon * 6, dtype=float)
      result[0::6] = lower_part[:, 0]
      result[1::6] = lower_part[:, 1]
      result[2::6] = lower_part[:, 2]
      result[3::6] = upper_part[:, 0]
      result[4::6] = upper_part[:, 1]
      result[5::6] = upper_part[:, 2]

      r_term = self._terminal_radius(drone, pred_vel)
      if r_term is None:
         return result
      # Terminal constraint: the sphere with the full braking radius at the last step lies inside the room
      p_last = pred_pos[self._horizon - 1]
      terminal = np.concatenate([
         p_last - r_term - lower_bounds + self._wall_tolerance,
         upper_bounds - (p_last + r_term) + self._wall_tolerance,
      ])
      return np.concatenate([result, terminal])

   def _evaluate_sphere(
      self,
      drone: Drone,
      pred_pos: np.ndarray,
      room_radius: float,
      pred_vel: np.ndarray | None = None,
   ) -> np.ndarray:
      dists = np.linalg.norm(pred_pos[:self._horizon], axis=1)
      radii = _safety_radii(drone, pred_vel, self._horizon)
      result = room_radius - dists - radii
      r_term = self._terminal_radius(drone, pred_vel)
      if r_term is None:
         return result
      return np.append(result, room_radius - dists[-1] - r_term)
