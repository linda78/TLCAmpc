from __future__ import annotations

import logging
import random
import warnings
from dataclasses import dataclass, field
from time import process_time

import numpy as np

_log = logging.getLogger(__name__)

from drone_sim.domain.drone import Drone, has_central_cost
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.distributed.admm_state import ADMMState
from drone_sim.simulation.distributed.local_mpc import LocalMPCSolver, _pad_or_trim_horizon
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMailbox
from drone_sim.simulation.utils.cpu_costs import CpuCostLedger


@register_coordinator("dmpc_admm")
@dataclass
class DistributedMPCCoordinator:
   """Distributed MPC coordinator using ADMM consensus.

   Orchestrates ADMM iterations to produce drone controls by coordinating
   local MPC solvers. Each drone solves its own optimization problem while
   exchanging trajectory information with neighbors to reach consensus on
   collision avoidance constraints.

   This coordinator can replace the central MPC for scalable distributed
   optimization.
   """

   dt: float
   horizon: int = 5
   rho: float = 1.0  # ADMM penalty parameter
   max_admm_iter: int = 50  # Max ADMM iterations per timestep
   primal_tol: float = 1e-3
   dual_tol: float = 1e-3
   comm_radius: float | None = None  # For NeighborGraph
   gauss_seidel: bool = True  # Use Gauss-Seidel updates for symmetry breaking
   stagnation_limit: int = 3  # Break early if no drone makes progress for this many iterations

   # Shared CPU-time ledger, see drone_sim.simulation.utils.cpu_costs
   costs: CpuCostLedger = field(default_factory=CpuCostLedger, repr=False)

   # Internal state (initialized in __post_init__)
   _neighbor_graph: NeighborGraph = field(init=False)
   _mailbox: TrajectoryMailbox = field(init=False)
   _admm_state: ADMMState = field(init=False)
   _u_prev: dict[str, np.ndarray] = field(default_factory=dict)
   _last_iteration_count: int = field(default=0, init=False)
   _last_primal_residual: float = field(default=0.0, init=False)
   _last_dual_residual: float = field(default=0.0, init=False)
   _last_converged: bool = field(default=True, init=False)

   def __post_init__(self) -> None:
      self._neighbor_graph = NeighborGraph(comm_radius=self.comm_radius)
      self._mailbox = TrajectoryMailbox()
      self._admm_state = ADMMState(rho=self.rho, primal_tol=self.primal_tol, dual_tol=self.dual_tol, horizon=self.horizon, )

   def solve_controls(self, *, drones: list[Drone], obstacles: list[tuple[np.ndarray, np.ndarray]], room_min: np.ndarray | None = None,
                      room_max: np.ndarray | None = None, ) -> dict[str, np.ndarray]:
      """Solve for drone controls using distributed ADMM optimization.
      Matches the CentralMPCGlobalCoordinator interface.

      :param drones: List of Drone objects to optimize
      :param obstacles: List of (center, half_extents) static obstacles
      :param room_min: Room lower bounds (3,) or None
      :param room_max: Room upper bounds (3,) or None
      :return: Dict mapping drone_id to control (3,) for first timestep
      """
      ledger = self.costs

      with ledger.phase("setup"):
         # Identify which drones to optimize (must have central_cost interface)
         opt_drones = [d for d in drones if has_central_cost(d.controller)]

         if not opt_drones:
            return {}

         # TODO: get rid of this all over list handlings. Use the drones list instead
         opt_ids = [d.drone_id for d in opt_drones]
         drone_by_id = {d.drone_id: d for d in opt_drones}
         all_drones_by_id = {d.drone_id: d for d in drones}

         # 1. Update neighbor graph from current positions
         positions = {d.drone_id: np.asarray(d.x, dtype=float)[:3] for d in drones}
         self._neighbor_graph.update(positions)

         # 2. Get neighbor pairs and initialize ADMMState
         neighbor_pairs = self._neighbor_graph.get_neighbor_pairs()
         self._admm_state.initialize(neighbor_pairs)

         trajectories, local_solvers = self.init_trajectories(drones)

         # Add static trajectories for non-optimized drones (needed for neighbor pairs)
         for drone in drones:
            if drone.drone_id not in drone_by_id:
               pos = np.asarray(drone.x, dtype=float)[:3]
               trajectories[drone.drone_id] = np.tile(pos, (self.horizon, 1))

         # 3. ADMM iteration loop
         converged = False
         stagnated = False

         # Track controls and velocities across iterations
         controls: dict[str, np.ndarray] = {}
         velocities: dict[str, np.ndarray | None] = {did: None for did in trajectories}

         # Track trajectory changes for stagnation detection
         prev_trajectories: dict[str, np.ndarray] = {did: traj.copy() for did, traj in trajectories.items()}
         stagnation_count = 0

      iteration = 0
      parallel_seconds = 0.0  # sum over the rounds of this step of the slowest local solve, see CpuCostLedger
      for iteration in range(self.max_admm_iter):
         round_max = 0.0
         # Determine drone solving order for this iteration
         with ledger.phase("setup"):
            drone_order = list(opt_ids)
            if self.gauss_seidel:
               if iteration < 3:
                  # First few iterations: random shuffle for initial symmetry breaking
                  random.shuffle(drone_order)
               else:
                  # Later iterations: priority-based ordering (most constrained first)
                  drone_order.sort(key=lambda d: self._compute_priority(d, trajectories, velocities, all_drones_by_id))

         # 3a. Broadcast current trajectories via mailbox (initial state)
         with ledger.phase("exchange"):
            self._mailbox.clear()
            for drone_id in opt_ids:
               self._mailbox.broadcast(sender_id=drone_id, trajectory=trajectories[drone_id], predicted_velocities=None, timestamp=0, neighbor_graph=self._neighbor_graph)

         # 3b. For each drone: receive neighbors, solve local MPC
         if self.gauss_seidel:
            # Gauss-Seidel: immediate updates after each drone solves
            for drone_id in drone_order:
               drone = drone_by_id[drone_id]
               solver = local_solvers[drone_id]

               # Get neighbor trajectories from mailbox (includes any updates)
               with ledger.phase("exchange"):
                  messages = self._mailbox.receive(drone_id)
                  neighbor_trajectories = {
                     sid: (msg.trajectory, msg.predicted_velocities)
                     for sid, msg in messages.items()
                  }

               with ledger.phase("setup"):
                  u_prev = self._warm_start_controls(drone_id, iteration, controls)
               (u_opt, traj_opt, success, vel_opt), seconds = self._solve_local(
                  solver, drone=drone, neighbor_trajectories=neighbor_trajectories,
                  obstacles=obstacles, room_min=room_min, room_max=room_max, u_prev=u_prev,
               )
               round_max = max(round_max, seconds)

               # Immediate update (Gauss-Seidel style)
               trajectories[drone_id] = traj_opt
               controls[drone_id] = u_opt
               velocities[drone_id] = vel_opt

               # Broadcast immediately so next drone sees updated trajectory
               with ledger.phase("exchange"):
                  self._mailbox.broadcast(sender_id=drone_id, trajectory=traj_opt,
                                          predicted_velocities=vel_opt, timestamp=0,
                                          neighbor_graph=self._neighbor_graph)
         else:
            # Jacobi: all drones use stale data, update all at once
            trajectories, controls, velocities, round_max = self._jacobi(drone_order, drone_by_id, local_solvers, iteration, obstacles, room_min, room_max,
                                                                        prev_controls=controls)
         parallel_seconds += round_max

         with ledger.phase("admm_update"):
            # 3c. Stagnation detection — break early if no drone made progress
            stagnated, prev_trajectories, stagnation_count = self._check_stagnation(
               trajectories, prev_trajectories, opt_ids, stagnation_count, iteration,
            )
            if stagnated:
               break

            # 3d. Update z and lambda for all neighbor pairs
            self.update_z(trajectories, velocities, all_drones_by_id, neighbor_pairs)

            # 3e. Check convergence
            if self._admm_state.is_converged(trajectories):
               converged = True
               break

      # Store iteration count for debugging/testing
      self._last_iteration_count = iteration + 1
      ledger.count("admm_iterations", iteration + 1)
      ledger.add_seconds("parallel_step", parallel_seconds)
      ledger.record_max("max_admm_iter_step", iteration + 1)

      # Record final residuals for debugging/visualization
      with ledger.phase("admm_update"):
         primal_res, dual_res = self._admm_state.compute_residuals(trajectories)
      self._last_primal_residual = primal_res
      self._last_dual_residual = dual_res
      self._last_converged = converged

      if _log.isEnabledFor(logging.DEBUG):
         self._debug_log_status(iteration, converged, stagnated, primal_res, dual_res, drones, trajectories, controls)

      # Warn if not converged (but don't fail - use best-effort solution)
      if not converged and not stagnated:
         warnings.warn(f"DistributedMPCCoordinator did not converge after {self.max_admm_iter} "
                       f"iterations (primal/dual residuals may exceed tolerance)", RuntimeWarning, stacklevel=2, )

      # 4. Extract first-step controls and store for warm-start
      result: dict[str, np.ndarray] = {}
      for drone_id in opt_ids:
         u_seq = controls[drone_id]
         self._u_prev[drone_id] = u_seq
         result[drone_id] = u_seq[0].copy()

      return result

   def _warm_start_controls(self, drone_id: str, iteration: int, controls: dict[str, np.ndarray]) -> np.ndarray | None:
      """Pick the best warm-start control sequence for a drone.

      On the first ADMM iteration, reuses the previous timestep's solution.
      On subsequent iterations, reuses the current timestep's last solve.

      :param drone_id: ID of the drone.
      :param iteration: Current ADMM iteration index (0-based).
      :param controls: Controls computed so far in the current timestep.
      :return: Warm-start control sequence (H, 3), or None.
      """
      if iteration > 0 and drone_id in controls:
         return controls[drone_id]
      if iteration == 0 and drone_id in self._u_prev:
         return self._u_prev[drone_id]
      return None

   def _solve_local(self, solver: LocalMPCSolver, **kwargs) -> tuple[tuple[np.ndarray, np.ndarray, bool, np.ndarray | None], float]:
      """Run one local MPC solve and book its inclusive CPU seconds as ``local_solve`` (see CpuCostLedger). Returns (result, seconds)."""
      t0 = process_time()
      result = solver.solve(**kwargs)
      seconds = process_time() - t0
      self.costs.add_seconds("local_solve", seconds)
      self.costs.count("local_solves")
      return result, seconds

   def _jacobi(self, drone_order: list[str], drone_by_id: dict[str, Drone], local_solvers: dict[str, LocalMPCSolver], iteration: int,
               obstacles: list[tuple[np.ndarray, np.ndarray]] | None = None, room_min: np.ndarray | None = None,
               room_max: np.ndarray | None = None,
               prev_controls: dict[str, np.ndarray] | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], float]:
      """Jacobi update: all drones solve using stale neighbor data, then update all at once.

      Returns the new trajectories, controls and velocities plus the CPU seconds of the slowest local solve of this round.
      """
      ledger = self.costs
      new_trajectories: dict[str, np.ndarray] = {}
      new_controls: dict[str, np.ndarray] = {}
      new_velocities: dict[str, np.ndarray] = {}
      round_max = 0.0

      for drone_id in drone_order:
         drone = drone_by_id[drone_id]
         solver = local_solvers[drone_id]

         with ledger.phase("exchange"):
            messages = self._mailbox.receive(drone_id)
            neighbor_trajectories = {
               sid: (msg.trajectory, msg.predicted_velocities)
               for sid, msg in messages.items()
            }

         with ledger.phase("setup"):
            u_prev = self._warm_start_controls(drone_id, iteration, prev_controls or {})
         (u_opt, traj_opt, success, vel_opt), seconds = self._solve_local(
            solver, drone=drone, neighbor_trajectories=neighbor_trajectories,
            obstacles=obstacles, room_min=room_min, room_max=room_max, u_prev=u_prev,
         )
         round_max = max(round_max, seconds)

         new_trajectories[drone_id] = traj_opt
         new_controls[drone_id] = u_opt
         new_velocities[drone_id] = vel_opt

      with ledger.phase("exchange"):
         # Broadcast updated trajectories with velocities (no extra _predict_states call)
         for drone_id in drone_order:
            self._mailbox.broadcast(sender_id=drone_id, trajectory=new_trajectories[drone_id],
                                    predicted_velocities=new_velocities[drone_id], timestamp=0,
                                    neighbor_graph=self._neighbor_graph)

      return new_trajectories, new_controls, new_velocities, round_max

   def update_z(self, trajectories: dict[str, np.ndarray], vel_dict: dict[str, np.ndarray | None],
                all_drones_by_id: dict[str, Drone], neighbor_pairs: list[tuple[str, str]]) -> None:
      for pair in neighbor_pairs:
         id_i, id_j = pair
         traj_i = trajectories[id_i]
         traj_j = trajectories[id_j]

         radii_i = self._compute_safety_radii(all_drones_by_id[id_i], vel_dict.get(id_i))
         radii_j = self._compute_safety_radii(all_drones_by_id[id_j], vel_dict.get(id_j))
         min_dist = radii_i + radii_j

         self._admm_state.update_z(pair, traj_i, traj_j, min_dist)
         self._admm_state.update_lambda(pair, traj_i, traj_j)

   def _check_stagnation(self, trajectories: dict[str, np.ndarray], prev_trajectories: dict[str, np.ndarray],
                          opt_ids: list[str], stagnation_count: int, iteration: int) -> tuple[bool, dict[str, np.ndarray], int]:
      """Check whether ADMM has stagnated (no meaningful trajectory progress).

      :param trajectories: Current trajectories for all drones.
      :param prev_trajectories: Trajectories from the previous iteration.
      :param opt_ids: IDs of drones being optimized.
      :param stagnation_count: Running count of consecutive stagnant iterations.
      :param iteration: Current ADMM iteration index.
      :return: (stagnated, updated_prev_trajectories, updated_stagnation_count).
      """
      max_change = max(float(np.linalg.norm(trajectories[did] - prev_trajectories[did])) for did in opt_ids)
      if max_change < self.primal_tol:
         stagnation_count += 1
      else:
         stagnation_count = 0
      prev_trajectories = {did: trajectories[did].copy() for did in opt_ids}

      if stagnation_count >= self.stagnation_limit:
         warnings.warn(f"ADMM stagnated after {iteration + 1} iterations — no drone made meaningful trajectory progress for {self.stagnation_limit} "
                       f"consecutive iterations", RuntimeWarning, stacklevel=2)
         return True, prev_trajectories, stagnation_count
      return False, prev_trajectories, stagnation_count

   def init_trajectories(self, drones: list[Drone]) -> tuple[dict[str, np.ndarray], dict[str, LocalMPCSolver]]:
      # Initialize trajectories (use warm-start if available)
      trajectories: dict[str, np.ndarray] = {}
      local_solvers: dict[str, LocalMPCSolver] = {}

      for drone in drones:
         controller = drone.controller
         if not has_central_cost(controller):
            continue

         # Create local solver for this drone with bounds from physics
         local_solvers[drone.drone_id] = LocalMPCSolver(dt=self.dt, horizon=self.horizon, costs=self.costs)

         # Initialize trajectory from warm-start or initial guess
         if drone.drone_id in self._u_prev:
            # Warm-start: shift previous solution
            u_prev = self._u_prev[drone.drone_id]
            u0 = np.concatenate([u_prev[1:], u_prev[-1:]], axis=0)
         else:
            # Get initial guess from controller
            u0 = _pad_or_trim_horizon(controller.central_initial_guess(drone), self.horizon)

         # Predict trajectory from controls
         trajectories[drone.drone_id] = local_solvers[drone.drone_id]._predict_states(drone, u0)[0]
      return trajectories, local_solvers

   def get_last_iteration_count(self) -> int:
      """Get the number of ADMM iterations from the last solve.

      Useful for testing warm-start effectiveness.
      """
      return self._last_iteration_count

   def get_last_residuals(self) -> tuple[float, float]:
      """Get (primal_residual, dual_residual) from last solve."""
      return self._last_primal_residual, self._last_dual_residual

   def get_last_converged(self) -> bool:
      """Check if last solve converged."""
      return self._last_converged

   def get_neighbor_pairs(self) -> list[tuple[str, str]]:
      """Get current neighbor pairs for visualization."""
      return self._neighbor_graph.get_neighbor_pairs()

   def _compute_priority(self, drone_id: str, trajectories: dict[str, np.ndarray],
                         vel_dict: dict[str, np.ndarray | None], all_drones_by_id: dict[str, Drone]) -> float:
      """Compute priority score - lower = higher priority (solve first).

      Drones with smaller safety margins to neighbors get higher priority.

      :param drone_id: ID of the drone
      :param trajectories: Current trajectories for all drones
      :param vel_dict: Current velocities for all drones (or None per drone)
      :param all_drones_by_id: All drones by ID for safety radius computation
      :return: Priority score (lower = higher priority = solve first)
      """
      neighbors = self._neighbor_graph.get_neighbors(drone_id)
      if not neighbors:
         return float("inf")

      min_margin = float("inf")
      traj_i = trajectories.get(drone_id)
      if traj_i is None:
         return float("inf")

      radii_i = self._compute_safety_radii(all_drones_by_id[drone_id], vel_dict.get(drone_id))

      for neighbor_id in neighbors:
         traj_j = trajectories.get(neighbor_id)
         if traj_j is None:
            continue
         radii_j = self._compute_safety_radii(all_drones_by_id[neighbor_id], vel_dict.get(neighbor_id))
         min_dist = float(np.mean(radii_i) + np.mean(radii_j))
         dists = np.linalg.norm(traj_i - traj_j, axis=1)
         min_margin = min(min_margin, float(np.min(dists)) - min_dist)

      return min_margin

   def _compute_safety_radii(self, drone: Drone, velocities: np.ndarray | None) -> np.ndarray:
      """Compute per-step safety radii for a drone.

      :param drone: the drone
      :param velocities: predicted velocities (H, 3), or None
      :return: per-step safety radii (H,)
      """
      if velocities is not None and drone.is_adaptive:
         return np.array([drone.compute_adaptive_radius(velocities[step])
                          for step in range(self.horizon)])
      return np.full(self.horizon, drone.safety_zone)

   def _debug_log_status(self, iteration: int, converged: bool, stagnated: bool, primal_res: float, dual_res: float,
                         drones: list[Drone], trajectories: dict[str, np.ndarray], controls: dict[str, np.ndarray]):
      status = "converged" if converged else ("stagnated" if stagnated else "max_iter")
      _log.debug("ADMM done: status=%s  iters=%d  primal=%.4e  dual=%.4e", status, iteration + 1, primal_res, dual_res)
      for i, drone_i in enumerate(drones[:-1]):
         for drone_j in drones[i+1:]:
            dists = np.linalg.norm(trajectories[drone_i.drone_id] - trajectories[drone_j.drone_id], axis=1)
            threshold = drone_i.safety_zone + drone_j.safety_zone
            _log.debug("  pair %s-%s  dists=%s  threshold=%.2f  min_dist=%.3f  violated=%s", drone_i.drone_id, drone_j.drone_id,
                       np.round(dists, 3), threshold, float(dists.min()), dists.min() < threshold)
      for drone in drones:
         u_seq = controls[drone.drone_id]
         _log.debug("  %s  u[0]=%s  traj=%s", drone.drone_id, np.round(u_seq[0], 3),
                    np.round(trajectories[drone.drone_id], 3).tolist())

