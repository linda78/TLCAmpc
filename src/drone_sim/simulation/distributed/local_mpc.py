from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from drone_sim.domain.constraints import (MovingObstacleAvoidanceConstraints, ObstacleAvoidanceConstraints, RoomConstraints, VelocityConstraints)
from drone_sim.simulation.utils.cpu_costs import CpuCostLedger

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
   from drone_sim.domain.drone import Drone


def _pad_or_trim_horizon(u: np.ndarray, horizon: int) -> np.ndarray:
   """Pad or trim a control sequence to match the given horizon."""
   u = np.asarray(u, dtype=float).reshape((-1, 3))
   if u.shape[0] < horizon:
      pad = np.tile(u[-1:], (horizon - u.shape[0], 1))
      u = np.concatenate([u, pad], axis=0)
   elif u.shape[0] > horizon:
      u = u[:horizon]
   return u


@dataclass
class LocalMPCSolver:
   """Per-drone MPC solver for distributed optimization.

   Solves a local MPC problem for a single drone, treating neighbor
   predicted trajectories as fixed moving obstacles.

   Used as the primal update step in ADMM-based distributed MPC.
   """

   dt: float
   horizon: int

   # Optimizer settings
   max_iter: int = 50
   f_tol: float = 1e-4
   symmetry_break_eps: float = 0.05  # Random noise magnitude for symmetry breaking
   terminal_constraint: bool = True  # braking-distance terminal radius at the last horizon step (room walls), see RoomConstraints
   terminal_constraint_pairs: bool = False  # the same terminal radius for every drone pair at the last step, see MovingObstacleAvoidanceConstraints

   # Shared CPU-time ledger, see drone_sim.simulation.utils.cpu_costs
   costs: CpuCostLedger = field(default_factory=CpuCostLedger, repr=False)

   def solve(self, drone: Drone, neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray | None]],
             obstacles: list[tuple[np.ndarray, np.ndarray]] | None = None, room_min: np.ndarray | None = None, room_max: np.ndarray | None = None,
             u_prev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
      """ Solve local MPC problem for a single drone.

      :param drone: Drone object with state, route, controller, and physics
      :param neighbor_trajectories: Dict mapping neighbor_id to
             (trajectory (H,3), predicted_velocities (H,3) or None).
      :param obstacles: List of (center, half_extents) static obstacles
      :param room_min: Room lower bounds (3,) or None
      :param room_max: Room upper bounds (3,) or None
      :param u_prev: Previous control sequence (H,3) for warm-start
      :return:
         u_opt: Optimized control sequence (horizon, 3)
         traj_opt: Optimized position trajectory (horizon, 3)
         success: Whether optimization succeeded
         vel_opt: Optimized velocity trajectory (horizon, 3)
      """
      ledger = self.costs

      with ledger.phase("setup"):
         controller = drone.controller
         obstacles = obstacles or []

         u_min, u_max = drone.bounds()
         horizon = self.horizon

         # Initial guess
         if u_prev is not None:
            # Warm-start: shift previous solution
            u0 = np.concatenate([u_prev[1:], u_prev[-1:]], axis=0)
         else:
            u0 = _pad_or_trim_horizon(controller.central_initial_guess(drone), horizon)

         u0 = np.clip(u0, u_min, u_max)

         # Symmetry-breaking noise to prevent collinear deadlocks
         if self.symmetry_break_eps > 0:
            noise = np.random.uniform(-self.symmetry_break_eps, self.symmetry_break_eps, size=u0.shape, )
            u0 = np.clip(u0 + noise, u_min, u_max)

         # Pre-build constraint evaluators (stateless, depend only on horizon)
         collision_c = MovingObstacleAvoidanceConstraints(horizon=horizon, terminal_constraint=self.terminal_constraint_pairs)
         obstacle_c = ObstacleAvoidanceConstraints(horizon=horizon)
         room_c = RoomConstraints(horizon=horizon, terminal_constraint=self.terminal_constraint) if room_min is not None and room_max is not None else None
         velocity_c = VelocityConstraints(horizon=horizon)
         # Extract cost weights for inline computation (avoids redundant predict_trajectory in central_cost)
         qp = np.diag(controller._Qp)  # (3,)
         qv = np.diag(controller._Qv)  # (3,)
         r = np.diag(controller._R)    # (3,)
         lambda_vel = getattr(controller, 'lambda_vel', 0.0)
         p_ref = np.asarray(drone.route.current_ref(), dtype=float).reshape(3)

         # Shared prediction cache: cost() and constraints() are called with the same u_flat
         # by SLSQP on each iteration, so we cache to avoid duplicate predict_trajectory calls.
         _cache_u = [None]
         _cache_result = [None]
         _cache_g = [None]

      def _predict_cached(u_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
         if _cache_u[0] is None or not np.array_equal(u_flat, _cache_u[0]):
            _cache_u[0] = u_flat.copy()
            _cache_result[0] = self._predict_states(drone, u_flat.reshape((horizon, 3)))
            _cache_g[0] = None  # invalidate constraint cache
         return _cache_result[0]

      def cost(u_flat: np.ndarray) -> float:
         with ledger.phase("cost"):
            positions, velocities = _predict_cached(u_flat)
            u = u_flat.reshape((horizon, 3))
            errors = positions - p_ref
            return float(
               np.sum(errors ** 2 * qp)
               + np.sum(velocities ** 2 * qv)
               + np.sum(u ** 2 * r)
               + lambda_vel * np.sum(velocities ** 2)
            )

      def constraints(u_flat: np.ndarray) -> np.ndarray:
         with ledger.phase("constraints"):
            positions, velocities = _predict_cached(u_flat)
            if _cache_g[0] is not None:
               return _cache_g[0]

            parts = []
            c = collision_c._evaluate(drone, positions, neighbor_trajectories, pred_vel=velocities)
            if len(c) > 0:
               parts.append(c)
            c = obstacle_c._evaluate(drone, positions, obstacles, pred_vel=velocities)
            if len(c) > 0:
               parts.append(c)
            if room_c is not None:
               c = room_c._evaluate(drone, positions, room_max, room_min, pred_vel=velocities)
               if len(c) > 0:
                  parts.append(c)
            # Velocity constraints to enforce v_max via SLSQP
            vel_g = velocity_c.evaluate_single(drone, velocities, np.array([], dtype=float))
            if len(vel_g) > 0:
               parts.append(vel_g)

            g = np.concatenate(parts) if parts else np.array([], dtype=float)
            _cache_g[0] = g
            return g

      with ledger.phase("setup"):
         # Build bounds
         axis_bounds = [(float(u_min[a]), float(u_max[a])) for a in range(3)]
         bounds = axis_bounds * horizon

      # Optimize
      with ledger.phase("solver"):
         result = minimize(cost, u0.flatten(), method="SLSQP", bounds=bounds, constraints={"type": "ineq", "fun": constraints},
                           options={"maxiter": self.max_iter, "ftol": self.f_tol, "disp": False})

      with ledger.phase("validate"):
         u_opt = np.clip(result.x.reshape((horizon, 3)), u_min, u_max)

         # Reuse cached prediction/constraints if u_opt matches last evaluation
         u_opt_flat = u_opt.flatten()
         if _cache_u[0] is not None and np.array_equal(u_opt_flat, _cache_u[0]):
            traj_opt, vel_opt = _cache_result[0]
            g = _cache_g[0] if _cache_g[0] is not None else constraints(u_opt_flat)
         else:
            traj_opt, vel_opt = self._predict_states(drone, u_opt)
            g = constraints(u_opt_flat)
         feasible = result.success and (len(g) == 0 or g.min() >= -self.f_tol)

         # Infeasible fallback: decelerate, then hold position
         if not feasible:
            u_opt, traj_opt, vel_opt, feasible = self._infeasible_fallback(drone, u_opt, traj_opt, constraints, u_min, u_max, horizon, )

         if _log.isEnabledFor(logging.DEBUG):
            self._debug_log_feasibility_check(drone, neighbor_trajectories, feasible, result, traj_opt, g)

      return u_opt, traj_opt, feasible, vel_opt

   def _infeasible_fallback(self, drone: Drone, u_opt: np.ndarray, traj_opt: np.ndarray, constraints_fn, u_min: np.ndarray, u_max: np.ndarray,
         horizon: int, ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
      """Try safe fallback controls when the optimizer returns infeasible.

      Fallback order:
      1. Decelerate — brake toward zero velocity
      2. Hold position — zero control (no acceleration, let the other drone resolve)

      :return: (u, trajectory, velocities, feasible) for the best fallback found
      """
      v_current = np.asarray(drone.x, dtype=float)[3:6]

      # Fallback 1: decelerate (brake toward zero velocity)
      u_decel_step = np.clip(-v_current / self.dt, u_min, u_max)
      u_decel = np.tile(u_decel_step, (horizon, 1))
      traj_decel, vel_decel = self._predict_states(drone, u_decel)
      g_decel = constraints_fn(u_decel.flatten())
      if len(g_decel) == 0 or g_decel.min() >= -1e-6:
         _log.debug("  fallback: DECELERATE for %s", drone.drone_id)
         return u_decel, traj_decel, vel_decel, True

      # Fallback 2: zero control (hold — don't accelerate)
      u_zero = np.zeros((horizon, 3))
      traj_zero, vel_zero = self._predict_states(drone, u_zero)
      g_zero = constraints_fn(u_zero.flatten())
      if len(g_zero) == 0 or g_zero.min() >= -1e-6:
         _log.debug("  fallback: HOLD for %s", drone.drone_id)
         return u_zero, traj_zero, vel_zero, True

      # All fallbacks failed — pick the least-violating option
      options = [
         (u_opt, traj_opt, None, constraints_fn(u_opt.flatten())),
         (u_decel, traj_decel, vel_decel, g_decel),
         (u_zero, traj_zero, vel_zero, g_zero),
      ]
      best_u, best_traj, best_vel, _ = max(options, key=lambda o: o[3].min() if len(o[3]) else 0.0)
      if best_vel is None:
         _, best_vel = self._predict_states(drone, best_u)
      _log.debug("  fallback: ALL FAILED for %s, using least-violating", drone.drone_id)
      return best_u, best_traj, best_vel, False

   def _predict_states(self, drone: Drone, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
      """Predict position and velocity trajectories from state and controls.

      :param drone: Drone with initial state and physics model
      :param u: Control sequence (horizon, 3)
      :return: Tuple of (predicted_positions, predicted_velocities), each (horizon, 3)
      """
      return drone.physics.predict_trajectory(drone.x, u)

   def _debug_log_feasibility_check(self, drone: Drone, neighbor_trajectories: dict[str, tuple[np.ndarray, np.ndarray]], feasible: bool, result: OptimizeResult,
                                    traj_opt: np.ndarray, g: np.ndarray):
      _log.debug("SOLVE %s  feasible=%s  opt_success=%s  g_min=%.4f  cost=%.4f", drone.drone_id, feasible, result.success, g.min() if len(g) else float("nan"),
                 result.fun)
      for nid, (ntraj, nvel) in neighbor_trajectories.items():
         ntraj = np.asarray(ntraj, dtype=float).reshape((-1, 3))
         dists = np.linalg.norm(traj_opt - ntraj, axis=1)
         _log.debug("  neighbor=%s  dists=%s  safety_ego=%.2f  safety_nbr=%.2f  threshold=%.2f", nid, np.round(dists, 3), drone.safety_zone, drone.safety_zone,
                    2 * drone.safety_zone)
