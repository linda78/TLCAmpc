from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.config import ColorValue, ScenarioConfig
from drone_sim.domain.constraints import point_to_box_dist
from drone_sim.domain.drone import Drone, GatedDrone, Route
from drone_sim.domain.registry import create_controller, create_coordinator, create_physics

_log = logging.getLogger(__name__)


def _normalize_color(c: ColorValue) -> str | tuple[float, float, float]:
   if isinstance(c, str):
      return c

   rgb = [float(x) for x in c]
   if max(rgb) > 1.0:
      rgb = [x / 255.0 for x in rgb]

   return tuple(min(1.0, max(0.0, x)) for x in rgb)


def _color_to_json(c: str | tuple[float, float, float]) -> str | list[float]:
   if isinstance(c, str):
      return c
   return [float(x) for x in c]


@dataclass
class Simulator:
   dt: float
   physics: object
   drones: list[Drone]
   obstacles: list[tuple[np.ndarray, np.ndarray]]
   room_min: np.ndarray
   room_max: np.ndarray

   # Optional central coordination (e.g. centralized MPC over a subset of drones).
   coordinator: object | None = None

   # Infeasibility flag and reason if the central optimizer cannot find a valid route.
   infeasible: bool = False
   infeasible_reason: str | None = None

   # Visualization traces (in-memory only).
   trace_len: int = 300
   traces: dict[str, list[np.ndarray]] = field(default_factory=dict)

   # Safety-zone collision events for the *current* state.
   # Each entry represents an intruder intersecting a drone's safety sphere.
   last_collisions: list[dict] = field(default_factory=list)

   # Applied first-step control per drone for the *current* step, keyed by
   # drone_id. Mirrors the controls actually applied in step() so callers can
   # measure control energy (sum ||u||^2 dt) without re-deriving from velocity
   # differences (which the wall-clamp would corrupt). Empty before the first step.
   last_controls: dict[str, np.ndarray] = field(default_factory=dict)

   t: float = 0.0

   # Step counter (integer ticks).
   step_count: int = 0

   # Wall-clock seconds spent inside `step()` so far (cumulative).
   compute_time_s: float = 0.0

   # LSTM trajectory prediction infrastructure (None when not configured).
   _lstm_history: object | None = field(default=None, init=False, repr=False)
   _lstm_provider: object | None = field(default=None, init=False, repr=False)
   # BoF (Backoff-Function) trajectory/uncertainty provider (None when not configured).
   _bof_provider: object | None = field(default=None, init=False, repr=False)
   _bof_history: object | None = field(default=None, init=False, repr=False)

   @classmethod
   def from_config(cls, cfg: ScenarioConfig) -> "Simulator":
      # Ensure implementations are registered
      from drone_sim.controllers import central_cost as _central_cost  # noqa: F401
      from drone_sim.physics import linear_kinematics as _  # noqa: F401

      # Build physics lookup: supports single PhysicsSpec or list of PhysicsSpec
      physics_specs = cfg.physics if isinstance(cfg.physics, list) else [cfg.physics]
      physics_by_id: dict[str | None, object] = {}
      first_physics = None
      for ps in physics_specs:
         phys = create_physics({"type": ps.type, "params": {"dt": cfg.dt, **ps.params}})
         physics_by_id[ps.id] = phys
         if first_physics is None:
            first_physics = phys
      physics = first_physics

      drones: list[Drone] = []

      coordinator = None
      if cfg.coordinator is not None:
         coord_params = {"dt": cfg.dt, **cfg.coordinator.params}
         # Pass comm_radius for distributed coordinators
         if cfg.comm_radius is not None:
            coord_params["comm_radius"] = cfg.comm_radius
         coordinator = create_coordinator({"type": cfg.coordinator.type, "params": coord_params})

      # GatedDrone needs an intersection-capable coordinator to flip its
      # wait-state. Warn (don't raise) so misconfigured runs are loud
      # but not blocked.
      has_gated = any(d.type == "gated" for d in cfg.drones)
      if has_gated and coordinator is not None:
         from drone_sim.simulation.intersection.intersection_mixin import (
            IntersectionMixin,
         )
         if not isinstance(coordinator, IntersectionMixin):
            warnings.warn(
               "GatedDrone is only effective with an intersection coordinator "
               "(dmpc_admm_intersection / mpc_central_intersection) — "
               f"current coordinator is {type(coordinator).__name__}. "
               "Wait-state will be set but never honored.",
               RuntimeWarning, stacklevel=2,
            )

      for drone_cfg in cfg.drones:
         spec = drone_cfg.controller or cfg.controller
         controller = create_controller({"type": spec.type, "params": {"dt": cfg.dt, **spec.params}})
         start = np.asarray(drone_cfg.start, dtype=float)
         x0 = np.zeros(6, dtype=float)
         x0[:3] = start

         # Resolve per-drone physics: lookup by drone's physics ID, fall back to first/global
         drone_physics = physics_by_id.get(drone_cfg.physics, physics)

         route = Route(waypoints=[np.asarray(w, dtype=float) for w in drone_cfg.waypoints], target=np.asarray(drone_cfg.target, dtype=float))
         drone_color = _normalize_color(drone_cfg.drone_color)
         safety_color = _normalize_color(drone_cfg.safety_color or drone_cfg.drone_color)
         trace_color = _normalize_color(drone_cfg.trace_color or drone_cfg.drone_color)

         drone_cls = GatedDrone if drone_cfg.type == "gated" else Drone
         drones.append(
            drone_cls(drone_id=drone_cfg.drone_id, radius=drone_cfg.radius, safety_zone=drone_cfg.safety_zone, cons_stop=drone_cfg.cons_stop, color=drone_color,
                  safety_color=safety_color, trace_color=trace_color, controller=controller, physics=drone_physics, x=x0, route=route, alpha=drone_cfg.alpha,
                  safety_zone_mode=drone_cfg.safety_zone_mode))

      obstacles = [(np.asarray(o.center, dtype=float), np.asarray(o.half_extents, dtype=float)) for o in cfg.obstacles]

      if cfg.room is not None:
         room_min = np.asarray(cfg.room.min, dtype=float)
         room_max = np.asarray(cfg.room.max, dtype=float)
      else:
         # Derive a reasonable default room from scenario geometry.
         pts: list[np.ndarray] = []
         for drone_cfg in cfg.drones:
            pts.append(np.asarray(drone_cfg.start, dtype=float))
            pts.extend([np.asarray(w, dtype=float) for w in drone_cfg.waypoints])
            pts.append(np.asarray(drone_cfg.target, dtype=float))
         for c, _he in obstacles:
            pts.append(c)

         if pts:
            stacked = np.stack(pts, axis=0)
            p_min = stacked.min(axis=0)
            p_max = stacked.max(axis=0)
         else:
            p_min = np.zeros(3)
            p_max = np.ones(3)

         margin = 1.0
         room_min = p_min - margin
         room_max = p_max + margin

      sim = cls(dt=cfg.dt, physics=physics, drones=drones, obstacles=obstacles, room_min=room_min, room_max=room_max, coordinator=coordinator)

      # Instantiate LSTM history buffer and provider when model path is configured.
      if cfg.lstm_model_path is not None:
         from pathlib import Path
         from drone_sim.prediction import (TrajectoryHistoryBuffer, LSTMModelLoader, LSTMSafetyZoneProvider, UncertaintyPropagator, )
         _lstm_history = TrajectoryHistoryBuffer(m=20)
         _loader = LSTMModelLoader(Path(cfg.lstm_model_path))
         _propagator = UncertaintyPropagator()
         _horizon = cfg.coordinator.params.get("horizon", 5) if cfg.coordinator else 5
         sim._lstm_history = _lstm_history
         sim._lstm_provider = LSTMSafetyZoneProvider(_loader, _propagator, _lstm_history, horizon=_horizon, # lstm_look_ahead is deprecated!
               look_ahead=cfg.lstm_look_ahead, )

      # Instantiate BoF (Backoff-Function) provider when enabled. Parallel to
      # the LSTM provider. The adapter (library/REST) is selected from config.
      if getattr(cfg, "bof_enabled", False):
         from drone_sim.prediction import (
            TrajectoryHistoryBuffer,
            BoFSafetyZoneProvider,
            BoFLibraryAdapter,
            BoFRestAdapter,
         )
         _mpc_horizon = cfg.coordinator.params.get("horizon", 5) if cfg.coordinator else 5
         _bof_horizon = cfg.bof_horizon
         if _bof_horizon < _mpc_horizon:
            raise ValueError(
               f"bof_horizon ({_bof_horizon}) must be >= MPC horizon ({_mpc_horizon}) — "
               f"the planner needs at least {_mpc_horizon} per-step radii."
            )
         _bof_history = TrajectoryHistoryBuffer(m=cfg.bof_history_size)
         if cfg.bof_backend == "library":
            _bof_adapter = BoFLibraryAdapter(
               horizon=_bof_horizon,
               has_velocity=cfg.bof_has_velocity,
               growth_tau=cfg.bof_growth_tau,
            )
         else:
            # Validator on ScenarioConfig guarantees bof_url is set when backend == "rest".
            _bof_adapter = BoFRestAdapter(
               url=cfg.bof_url,
               horizon=_bof_horizon,
               has_velocity=cfg.bof_has_velocity,
               growth_tau=cfg.bof_growth_tau,
            )
         sim._bof_history = _bof_history
         sim._bof_provider = BoFSafetyZoneProvider(
            adapter=_bof_adapter,
            buffer=_bof_history,
            horizon=_mpc_horizon,
         )

      # Initialize traces with the start positions.
      sim.traces = {d.drone_id: [d.position().copy()] for d in sim.drones}
      sim.last_collisions = sim._compute_collisions()
      return sim

   def _compute_collisions(self) -> list[dict]:
      """Compute safety-zone collision events.

      Drone-drone: collision when Euclidean distance < sum of safety zones.
      Drone-obstacle: collision when point_to_box_dist(drone_pos, box) < drone.safety_zone.
        point_to_box_dist returns 0 inside the box and positive distance outside,
        so drones inside a box are always flagged.
      """
      events: list[dict] = []

      # Drone-drone
      for i, owner in enumerate(self.drones):
         p_owner = owner.position()
         for j, intr in enumerate(self.drones):
            if i == j:
               continue

            dist = float(np.linalg.norm(intr.position() - p_owner))
            threshold = float(owner.safety_zone + intr.safety_zone)

            if dist + 1e-6 <= threshold:
               events.append({"kind": "drone_drone", "owner": owner.drone_id, "intruder": intr.drone_id, "distance": dist, "threshold": threshold})

      # Drone-obstacle (box geometry)
      for owner in self.drones:
         p_owner = owner.position()
         for k, (c, he) in enumerate(self.obstacles):
            dist = point_to_box_dist(p_owner, c, he)
            threshold = float(owner.safety_zone)
            if dist <= threshold:
               events.append({"kind": "drone_obstacle", "owner": owner.drone_id, "obstacle_idx": k, "distance": dist, "threshold": threshold})

      return events

   def step(self) -> None:
      t0 = time.perf_counter()
      try:
         # Reset infeasibility flag at the beginning of the step.
         self.infeasible = False
         self.infeasible_reason = None

         # Compute controls based on current state (synchronous)
         us: list[np.ndarray] = []

         # Refresh route refs first (so all neighbor refs are consistent).
         for d in self.drones:
            d.route.advance_if_reached(d.position())

         positions = [d.position().copy() for d in self.drones]
         velocities = [d.velocity().copy() for d in self.drones]
         prefs = [d.route.current_ref().copy() for d in self.drones]

         # This build expects a centralized MPC coordinator.
         # All Paper/"basic_paper" configs provide `coordinator: {"type": "mpc_central", ...}`.
         # If a scenario omits the coordinator, we fail fast instead of silently running a different control scheme.
         if self.coordinator is None:
            raise RuntimeError("Simulator-step requires a coordinator (centralized MPC). Provide `coordinator` in ScenarioConfig.")

         # Drop BoF predictions cached from the previous step so the GUI sees only the current step's tubes (the provider accumulates across the per-ego-drone calls inside solve_controls).
         if self._bof_provider is not None and hasattr(self._bof_provider, "clear_step_cache"):
            self._bof_provider.clear_step_cache()
         try:
            # solve_controls may swap drone controllers (e.g. GatedDrone
            # parking). Run it before the per-drone fallback so the loop
            # sees the post-swap controller.
            u_by_id = self.coordinator.solve_controls(drones=self.drones, obstacles=self.obstacles, room_min=self.room_min, room_max=self.room_max,
                  lstm_provider=(self._bof_provider or self._lstm_provider), )

         except RuntimeError as exc:
            # Mark the step as infeasible (e.g. walls/obstacles make the optimization problem infeasible) and abort this step without advancing the
            # simulation time.
            self.infeasible = True
            self.infeasible_reason = str(exc)
            return

         # Per-drone fallback for non-opt drones (incl. just-parked
         # GatedDrones whose WaitController emits a brake here).
         for i, d in enumerate(self.drones):
            if d.drone_id in u_by_id:
               us.append(np.asarray(u_by_id[d.drone_id], dtype=float).reshape(3))
               continue
            neighbors = [(positions[j], velocities[j], self.drones[j].radius, self.drones[j].safety_zone, prefs[j],) for j in range(len(self.drones)) if j != i]
            if hasattr(d.controller, "control"):
               u = d.controller.control(d, neighbors, self.obstacles)
            else:
               u = np.zeros(3, dtype=float)
            us.append(np.asarray(u, dtype=float).reshape(3))
            if isinstance(d, GatedDrone) and d.is_waiting:
               _log.info(
                  "simulator: WAIT-BRAKE applied to %s ctrl=%s |v|=%.3f u=%s",
                  d.drone_id, type(d.controller).__name__,
                  float(np.linalg.norm(velocities[i])), np.round(us[i], 3).tolist(),
               )

         # Apply physics updates
         for d, u in zip(self.drones, us, strict=True):
            d.x = d.predict(u)
            # d.x = self.physics.step(d.x, u)

            # Keep the drone inside the room bounds. We clamp the position and zero the velocity component(s) that hit a wall.
            p = d.position()
            p_min = self.room_min + d.radius
            p_max = self.room_max - d.radius

            p_clamped = np.clip(p, p_min, p_max)
            hit = ~np.isclose(p_clamped, p)
            if np.any(hit):
               d.x[:3] = p_clamped
               d.x[3:][hit] = 0.0

            # Append trace point (after clamping).
            trace = self.traces.setdefault(d.drone_id, [])
            trace.append(d.position().copy())
            if len(trace) > self.trace_len:
               del trace[:-self.trace_len]

            # Update LSTM history buffer with the new state after clamping.
            if self._lstm_history is not None:
               self._lstm_history.update(d.drone_id, d.x.copy())
            # Mirror update into the BoF history buffer (independent capacity m).
            if self._bof_history is not None:
               self._bof_history.update(d.drone_id, d.x.copy())

         self.last_collisions = self._compute_collisions()
         # Mirror the applied controls (aligned with self.drones order) so
         # callers can accumulate control energy for this step.
         self.last_controls = {d.drone_id: u for d, u in zip(self.drones, us, strict=True)}

         # Diagnostic: real (not predicted) pairwise distances + wait-state.
         if _log.isEnabledFor(logging.INFO):
            entries = []
            wait_state = {d.drone_id: (isinstance(d, GatedDrone) and d.is_waiting)
                          for d in self.drones}
            for i, di in enumerate(self.drones):
               for dj in self.drones[i + 1:]:
                  dist = float(np.linalg.norm(di.position() - dj.position()))
                  thresh = float(di.safety_zone + dj.safety_zone)
                  marker = " *COLLIDE*" if dist < thresh else ""
                  entries.append(
                     f"{di.drone_id}({'W' if wait_state[di.drone_id] else 'F'})"
                     f"-{dj.drone_id}({'W' if wait_state[dj.drone_id] else 'F'})"
                     f"={dist:.3f}/thresh={thresh:.2f}{marker}"
                  )
            _log.info("simulator: step %d distances %s",
                      self.step_count + 1, "  ".join(entries))

         self.t += self.dt
         self.step_count += 1
      finally:
         self.compute_time_s += time.perf_counter() - t0

   def to_dict(self) -> dict:
      result = {
         "t": self.t,
         "dt": self.dt,
         "room": {"min": self.room_min.tolist(), "max": self.room_max.tolist()},
         "drones": [
            {
               "drone_id": d.drone_id,
               "x": d.x.tolist(),
               "route_idx": d.route.idx,
               "p_ref": d.route.current_ref().tolist(),
               "radius": d.radius,
               "safety_zone": d.safety_zone,
               "adaptive_safety_radius": d.compute_adaptive_radius(d.velocity()) if d.is_adaptive else None,
               "drone_color": _color_to_json(d.color),
               "safety_color": _color_to_json(d.safety_color),
               "trace_color": _color_to_json(d.trace_color),
            }
            for d in self.drones
         ],
         "obstacles": [{"center": c.tolist(), "half_extents": he.tolist()} for c, he in self.obstacles],
         "collisions": list(self.last_collisions),
      }

      # Add ADMM stats if using distributed coordinator
      if hasattr(self.coordinator, "get_last_iteration_count"):
         result["admm_stats"] = {"iteration_count": self.coordinator.get_last_iteration_count(), "primal_residual": self.coordinator.get_last_residuals()[0],
               "dual_residual": self.coordinator.get_last_residuals()[1], "converged": self.coordinator.get_last_converged(),
               "neighbor_pairs": [list(pair) for pair in self.coordinator.get_neighbor_pairs()], }

      return result
