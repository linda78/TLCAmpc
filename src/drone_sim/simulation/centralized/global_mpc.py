from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from drone_sim.domain.constraints import (MovingObstacleAvoidanceConstraints, ObstacleAvoidanceConstraints, RoomConstraints, VelocityConstraints,
                                          _safety_radius, _velocity_at_step)
from drone_sim.domain.drone import Drone, has_central_cost
from drone_sim.simulation.distributed.local_mpc import _pad_or_trim_horizon
from drone_sim.simulation.utils.cpu_costs import CpuCostLedger

_log = logging.getLogger(__name__)


@dataclass
class GlobalMPCSolver:
   """Global MPC solver for multi-drone joint optimization.

   Handles the SLSQP optimization core: cost assembly, constraint evaluation,
   state prediction, and symmetry breaking. Analogous to LocalMPCSolver on the
   distributed side.

   Safety constraints use owner-only rule:
       dist(owner, other) >= owner.safety_zone + other.safety_buffer
   """

   dt: float
   horizon: int = 5
   max_iter: int = 120
   f_tol: float = 1e-3
   room_wall_tolerance: float = 0.0
   terminal_constraint: bool = True  # braking-distance terminal radius at the last horizon step (room walls), see RoomConstraints
   terminal_constraint_pairs: bool = False  # the same terminal radius for every drone pair at the last step, see MovingObstacleAvoidanceConstraints
   symmetry_break_accel: float = 0.05

   # Shared CPU-time ledger, see drone_sim.simulation.utils.cpu_costs
   costs: CpuCostLedger = field(default_factory=CpuCostLedger, repr=False)

   # Cached constraint evaluators (created in __post_init__)
   _collision_c: MovingObstacleAvoidanceConstraints = field(init=False, repr=False)
   _obstacle_c: ObstacleAvoidanceConstraints = field(init=False, repr=False)
   _room_c: RoomConstraints = field(init=False, repr=False)
   _velocity_c: VelocityConstraints = field(init=False, repr=False)

   def __post_init__(self) -> None:
      self._collision_c = MovingObstacleAvoidanceConstraints(horizon=self.horizon, terminal_constraint=self.terminal_constraint_pairs)
      self._obstacle_c = ObstacleAvoidanceConstraints(horizon=self.horizon)
      self._room_c = RoomConstraints(horizon=self.horizon, wall_tolerance=self.room_wall_tolerance, terminal_constraint=self.terminal_constraint)
      self._velocity_c = VelocityConstraints(horizon=self.horizon)

   def _pack(self, u: np.ndarray) -> np.ndarray:
      return np.asarray(u, dtype=float).reshape(-1)

   def _unpack(self, u_flat: np.ndarray, num_drones: int) -> np.ndarray:
      return np.asarray(u_flat, dtype=float).reshape((num_drones, self.horizon, 3))

   def _predict_states(self, drones: list[Drone], u: np.ndarray) -> np.ndarray:
      states = np.zeros((len(drones), self.horizon, 6), dtype=float)
      for i, drone in enumerate(drones):
         x = np.asarray(drone.x, dtype=float).reshape(6)
         for k in range(self.horizon):
            x = drone.physics.step(x, u[i, k])
            states[i, k] = x
      return states

   def _apply_symmetry_break(self, u0: np.ndarray) -> np.ndarray:
      """Apply a tiny alternating perturbation per drone to break perfect symmetry.

      Nudges each drone with a small randomized 3D bias whose sign alternates,
      preventing the optimizer from getting stuck at collinear deadlocks.
      """
      eps = self.symmetry_break_accel
      if eps <= 0.0:
         return u0

      num_drones = u0.shape[0]
      if num_drones < 2:
         return u0

      # Random unit-ish direction, alternating sign per drone
      base_vec = np.random.uniform(0.1, 1.0, size=3)
      base_vec /= np.linalg.norm(base_vec)

      signs = np.where(np.arange(num_drones) % 2 == 0, 1.0, -1.0)
      # Shape: (num_drones, 1, 3) for broadcasting over the horizon axis
      deltas = signs[:, None, None] * eps * base_vec[None, None, :]

      return u0 + deltas

   def _constraint_breakdown(self, u_flat: np.ndarray, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, np.ndarray]], room_min: np.ndarray | None,
         room_max: np.ndarray | None) -> dict[str, dict[str, np.ndarray]]:
      """Return per-category constraint margins keyed by label.

      Returns a dict with keys 'collision', 'room', 'velocity', each mapping
      a human-readable label to a margin array (horizon,) where negative values
      indicate constraint violations.
      """
      u = self._unpack(u_flat, len(drones))
      states = self._predict_states(drones, u)
      pred_pos = {d.drone_id: states[i, :, :3] for i, d in enumerate(drones)}
      pred_vel = {d.drone_id: states[i, :, 3:6] for i, d in enumerate(drones)}

      # Collision: one (horizon,) array per i<j pair
      collision: dict[str, np.ndarray] = {}
      for i in range(len(drones)):
         for j in range(i + 1, len(drones)):
            di, dj = drones[i], drones[j]
            pi = pred_pos[di.drone_id]
            pj = pred_pos[dj.drone_id]
            vi = pred_vel[di.drone_id]
            vj = pred_vel[dj.drone_id]
            margins = np.zeros(self.horizon)
            for step in range(self.horizon):
               si = _safety_radius(di, _velocity_at_step(vi, step))
               sj = _safety_radius(dj, _velocity_at_step(vj, step))
               dist = float(np.linalg.norm(pi[step] - pj[step]))
               margins[step] = dist - (si + sj + di.cons_stop + dj.cons_stop)
            collision[f"{di.drone_id} <-> {dj.drone_id}"] = margins

      # Room: one array per drone (horizon*6 per-face margins, plus 6 terminal-radius margins when
      # enabled), evaluated by the same class the solver uses so the breakdown cannot drift
      room: dict[str, np.ndarray] = {}
      if room_min is not None and room_max is not None:
         for d in drones:
            room[d.drone_id] = self._room_c._evaluate(d, pred_pos[d.drone_id], room_max, room_min, pred_vel=pred_vel[d.drone_id])

      # Velocity: one (horizon,) array per drone
      velocity: dict[str, np.ndarray] = {}
      for i, d in enumerate(drones):
         speed_sq = np.sum(states[i, :, 3:6] ** 2, axis=1)
         velocity[d.drone_id] = d.v_max ** 2 - speed_sq

      return {"collision": collision, "room": room, "velocity": velocity}

   def _log_constraint_breakdown(self, u_flat: np.ndarray, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, np.ndarray]], room_min: np.ndarray | None,
         room_max: np.ndarray | None) -> None:
      """Log per-category constraint margins. Violations always logged at WARNING."""
      bd = self._constraint_breakdown(u_flat, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)

      for label, margins in bd.get("collision", {}).items():
         msg = f"  collision {label}: min={margins.min():.3e}  per-step={np.round(margins, 4).tolist()}"
         if margins.min() < 0:
            _log.warning(msg)

      for label, margins in bd.get("obstacle", {}).items():
         msg = f"  obstacle {label}: min={margins.min():.3e}  per-step={np.round(margins, 4).tolist()}"
         if margins.min() < 0:
            _log.warning(msg)

      for drone_id, margins in bd.get("room", {}).items():
         msg = f"  room     {drone_id}: min={margins.min():.3e}"
         if margins.min() < 0:
            _log.warning(msg)

      for drone_id, margins in bd.get("velocity", {}).items():
         msg = f"  velocity {drone_id}: min={margins.min():.3e}"
         if margins.min() < 0:
            _log.warning(msg)

   def _cost(self, u_flat: np.ndarray, *, drones: list[Drone], controllers: list[object], clip_u) -> float:
      with self.costs.phase("cost"):
         u = clip_u(self._unpack(u_flat, len(drones)))
         return float(sum(ctrl.central_cost(u[i], drone) for i, (drone, ctrl) in enumerate(zip(drones, controllers)))) # type: ignore[attr-defined]

   def _constraints(self, u_flat: np.ndarray, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, np.ndarray]], room_min: np.ndarray | None,
                    room_max: np.ndarray | None) -> np.ndarray:
      """Inequality constraints c(u) >= 0 for collision, obstacle, room, and velocity limits."""
      with self.costs.phase("constraints"):
         u = self._unpack(u_flat, len(drones))
         predicted_states = self._predict_states(drones, u)
         pred_pos = {drone.drone_id: predicted_states[i, :, :3] for i, drone in enumerate(drones)}
         pred_vel = {drone.drone_id: predicted_states[i, :, 3:6] for i, drone in enumerate(drones)}
         vals = np.array([], dtype=float)

         vals = self._collision_c.evaluate_multi(drones, pred_pos, vals, pred_vel=pred_vel)
         vals = self._obstacle_c.evaluate_multi(drones, pred_pos, obstacles, vals, pred_vel=pred_vel)

         if room_min is not None and room_max is not None:
            vals = self._room_c.evaluate_multi(drones, pred_pos, room_max, room_min, vals, pred_vel=pred_vel)

         vals = self._velocity_c.evaluate_multi(drones, predicted_states[:, :, 3:6], vals)

         return vals

   def solve(self, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, np.ndarray]], room_min: np.ndarray | None = None,
             room_max: np.ndarray | None = None, u_prev: dict[str, np.ndarray] | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
      """Solve the global MPC optimization for all drones.

      :param drones: List of Drone objects (only those with central_cost are optimized).
      :param obstacles: List of (center, half_extents) static obstacles.
      :param room_min: Room lower bounds (3,) or None.
      :param room_max: Room upper bounds (3,) or None.
      :param u_prev: Previous control sequences keyed by drone_id for warm-start.
      :return: Tuple of (controls_dict, u_sequences_dict) where:
          - controls_dict: {drone_id: first_step_control (3,)}
          - u_sequences_dict: {drone_id: full_u_sequence (horizon, 3)}
      """
      from scipy.optimize import minimize

      with self.costs.phase("setup"):
         controllers = [d.controller for d in drones]
         idx_opt = [i for i, ctrl in enumerate(controllers) if has_central_cost(ctrl)]

         if not idx_opt:
            return {}, {}

         opt_drones = [drones[i] for i in idx_opt]
         opt_controllers = [controllers[i] for i in idx_opt]
         opt_ids = [d.drone_id for d in opt_drones]
         num_optimized = len(opt_drones)
         u_prev = u_prev or {}

         bounds_list = [d.bounds() for d in opt_drones]
         u_mins = np.array([b[0] for b in bounds_list], dtype=float).reshape(num_optimized, 3)
         u_maxs = np.array([b[1] for b in bounds_list], dtype=float).reshape(num_optimized, 3)

         def clip_u(u: np.ndarray) -> np.ndarray:
            return np.clip(u, u_mins[:, None, :], u_maxs[:, None, :])

         u0 = np.zeros((num_optimized, self.horizon, 3), dtype=float)
         if all(did in u_prev for did in opt_ids):
            prev = np.stack([u_prev[did] for did in opt_ids], axis=0)
            u0 = np.concatenate([prev[:, 1:, :], prev[:, -1:, :]], axis=1)
            u0 = self._apply_symmetry_break(u0)
         else:
            # Build per-drone initial guesses, trim/pad to solver horizon
            u_guess = [_pad_or_trim_horizon(ctrl.central_initial_guess(d), self.horizon) for d, ctrl in zip(opt_drones, opt_controllers)] # type: ignore[attr-defined]

            u_guess = self._apply_symmetry_break(np.stack(u_guess, axis=0))

            # Backtrack alpha until initial guess is feasible (or fall back to zeros)
            alpha = 1.0
            _backtrack_feasible = False
            for _ in range(12):
               u0 = clip_u(alpha * u_guess)
               g = self._constraints(self._pack(u0), drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)
               if g.min(initial=0.0) >= 0.0:
                  _backtrack_feasible = True
                  break
               alpha *= 0.5
            else:
               u0 = np.zeros_like(u0)
               _log.warning("GlobalMPC: initial guess infeasible at all scales (alpha 1.0 -> %.4f), falling back to u=zeros", alpha)

            if _backtrack_feasible and alpha < 1.0 - 1e-9:
               _log.debug("GlobalMPC: initial guess needed backtracking to alpha=%.4f", alpha)

         # Per-variable bounds: repeat each drone's (min, max) for every horizon step and axis
         bounds = [(float(u_mins[j, axis]), float(u_maxs[j, axis])) for j in range(num_optimized) for _ in range(self.horizon) for axis in range(3)]

         cons = {"type": "ineq", "fun": lambda u_flat: self._constraints(u_flat, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)}

      with self.costs.phase("solver"):
         res = minimize(lambda u_flat: self._cost(u_flat, drones=opt_drones, controllers=opt_controllers, clip_u=clip_u), self._pack(u0), method="SLSQP",
               bounds=bounds, constraints=[cons], options={"maxiter": self.max_iter, "ftol": self.f_tol, "disp": False})

      with self.costs.phase("validate"):
         _log.debug("GlobalMPC: SLSQP done — success=%s nit=%d nfev=%d fun=%.4f msg='%s'", res.success, res.nit, res.nfev,
               res.fun if np.isfinite(res.fun) else float("nan"), res.message)
         if res.nit >= self.max_iter:
            _log.warning("GlobalMPC: SLSQP hit iteration limit (%d/%d) — solution may be infeasible", res.nit, self.max_iter)
         self._log_constraint_breakdown(res.x, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)

         if not res.success or not np.isfinite(res.fun):
            raise RuntimeError(f"CentralMPCGlobalCoordinator optimization failed: {res.message} (status={res.status})")

         self._validate_constraint_margins(res.x, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)

      u_opt = clip_u(self._unpack(res.x, num_optimized))
      controls = {did: u_opt[k, 0].copy() for k, did in enumerate(opt_ids)}
      sequences = {did: u_opt[k] for k, did in enumerate(opt_ids)}
      return controls, sequences

   def _validate_constraint_margins(self, u_flat: np.ndarray, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, np.ndarray]],
         room_min: np.ndarray | None, room_max: np.ndarray | None) -> None:
      """Raise RuntimeError if the solution violates constraints beyond numerical tolerance."""
      g = self._constraints(u_flat, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)
      min_margin = float(g.min(initial=np.inf)) if g.size else float("inf")
      if not np.isfinite(min_margin):
         raise RuntimeError("CentralMPCGlobalCoordinator produced non-finite constraint margins.")
      if min_margin < -self.f_tol:
         _log.warning("GlobalMPC: constraint violation detected (min margin=%.3e) — breakdown:", min_margin)
         self._log_constraint_breakdown(u_flat, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max)
         raise RuntimeError(f"CentralMPCGlobalCoordinator produced infeasible controls: min constraint margin {min_margin:.3e} < 0.")
