from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.config import ColorValue, ScenarioConfig
from drone_sim.domain.constraints import point_to_box_dist
from drone_sim.domain.drone import Drone, Route
from drone_sim.domain.registry import create_controller, create_coordinator, create_physics


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
   # Camera perception pipeline (all off/None unless the scenario sets `camera_enabled`).
   # `_perception_view_store`, `_perception_mailbox` and `_camera_expose_truth` are read off the
   # simulation by getattr in `api/perception_router.py` — renaming one here silently turns the
   # detector's endpoints back into 409s.
   _perception_mailbox: object | None = field(default=None, init=False, repr=False)
   _perception_view_store: object | None = field(default=None, init=False, repr=False)
   _perception_worker: object | None = field(default=None, init=False, repr=False)
   _camera_model: object | None = field(default=None, init=False, repr=False)
   _camera_feeds_dmpc: bool = field(default=False, init=False, repr=False)
   _camera_rate_steps: int = field(default=1, init=False, repr=False)
   _camera_expose_truth: bool = field(default=False, init=False, repr=False)

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

         # Same fallback shape as the colors above, one level deeper: per-drone override, else the scenario default. Validated at config load.
         model = cfg.drone_model_for(drone_cfg)

         drones.append(
            Drone(drone_id=drone_cfg.drone_id, radius=drone_cfg.radius, safety_zone=drone_cfg.safety_zone, cons_stop=drone_cfg.cons_stop, color=drone_color,
                  safety_color=safety_color, trace_color=trace_color, controller=controller, physics=drone_physics, x=x0, route=route, alpha=drone_cfg.alpha,
                  safety_zone_mode=drone_cfg.safety_zone_mode, model=model))

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

      # Camera perception. Builds the chain capture -> (render) -> detector -> PerceptionMailbox; whether
      # that mailbox then also replaces the DMPC broadcasts is `camera_feeds_dmpc` and is decided per step
      # below. Without `camera_enabled` not a single object here exists and the simulation runs exactly as
      # it did before this feature — the reason every flag defaults to off.
      if getattr(cfg, "camera_enabled", False):
         from drone_sim.perception import (CameraModel, CameraView, CameraViewStore, PerceptionMailbox, PerceptionWorker, StubPerceptionAdapter,
                                           render_fpv_png, )

         sim._camera_model = CameraModel(fov_deg=cfg.camera_fov_deg, range_m=cfg.camera_range)
         sim._perception_mailbox = PerceptionMailbox()
         sim._camera_feeds_dmpc = bool(cfg.camera_feeds_dmpc)
         sim._camera_rate_steps = int(cfg.camera_rate_steps)
         sim._camera_expose_truth = bool(cfg.camera_expose_truth)

         # Obstacles and the drone_id -> DroneModel mapping are constant over a run, so the renderer closes
         # over them once instead of every CameraView carrying them. Only resolved DroneModel objects go in:
         # the closure runs on the worker thread and must not reach back into config or the file system.
         _renderer = None
         if cfg.camera_render_images:
            _obstacles_for_render = sim.obstacles
            _models_for_render = {d.drone_id: d.model for d in sim.drones}

            def _render_view(view: CameraView) -> bytes:
               return render_fpv_png(view, _obstacles_for_render, models=_models_for_render)

            _renderer = _render_view

         # "stub" detects in-process on the worker thread; "rest" only renders and parks the view for the
         # external detector, which pushes its estimates into the same mailbox through the API router.
         # The config validator guarantees `camera_render_images` for the "rest" backend, so the store
         # never fills with image-less views.
         _adapter = StubPerceptionAdapter(noise_sigma=cfg.camera_noise_sigma) if cfg.camera_backend == "stub" else None
         if cfg.camera_backend == "rest":
            sim._perception_view_store = CameraViewStore()

         sim._perception_worker = PerceptionWorker(sim._perception_mailbox, adapter=_adapter, renderer=_renderer,
                                                   view_store=sim._perception_view_store, async_mode=cfg.camera_async)
         sim._perception_worker.start()

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

   def _capture_perception(self) -> None:
      """Take one camera view per drone and hand the whole batch to the perception worker.

      A no-op unless the scenario enabled the camera. Captures run every ``camera_rate_steps``-th step; in
      between, the estimates from the last capture simply stay in the mailbox (``post`` upserts and nothing
      expires), so a slow camera means older neighbor data, never missing data.

      The batch goes over in one :meth:`~drone_sim.perception.worker.PerceptionWorker.submit`, which never
      blocks: asynchronously it is queued (dropping a batch that is still waiting), synchronously it is
      detected right here. Views are self-contained copies, so the worker thread reading one cannot race the
      state updates at the end of this step.
      """
      if self._perception_worker is None or self.step_count % self._camera_rate_steps != 0:
         return

      views = [self._camera_model.capture(d, self.drones, step=self.step_count, sim_time=self.t) for d in self.drones]
      self._perception_worker.submit(views)

   def close(self) -> None:
      """Release the background resources of this simulation. Idempotent.

      Today that is the perception worker thread. It is a daemon, so forgetting this call never keeps the
      process alive — but a GUI that reloads scenarios would accumulate one live thread (and one rendering
      pipeline) per reload, which is why :class:`~drone_sim.gui.direct_backend.DirectBackend` closes the old
      simulation before replacing it.

      The mailbox and the view store survive on purpose: a detector may still be reading them over REST, and
      they hold no thread. A closed simulation can still be stepped, it just stops capturing.
      """
      worker = self._perception_worker
      self._perception_worker = None
      if worker is not None:
         worker.stop()

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

         # Capture before the solve, not after: with `camera_async=false` the worker detects inline, so
         # this step's estimates are in the mailbox by the time the coordinator reads it below.
         self._capture_perception()

         # This build expects a centralized MPC coordinator.
         # All Paper/"basic_paper" configs provide `coordinator: {"type": "mpc_central", ...}`.
         # If a scenario omits the coordinator, we fail fast instead of silently running a different control scheme.
         if self.coordinator is None:
            raise RuntimeError("Simulator-step requires a coordinator (centralized MPC). Provide `coordinator` in ScenarioConfig.")

         # First compute per-drone local controls (used for non-optimized drones, and as a fallback).
         for i, d in enumerate(self.drones):
            neighbors = [(positions[j], velocities[j], self.drones[j].radius, self.drones[j].safety_zone, prefs[j],) for j in range(len(self.drones)) if j != i]

            if hasattr(d.controller, "control"):
               u = d.controller.control(d, neighbors, self.obstacles)
            else:
               u = np.zeros(3, dtype=float)
            us.append(np.asarray(u, dtype=float).reshape(3))

         # Then override optimized drones with coordinator outputs.
         # Drop BoF predictions cached from the previous step so the GUI sees only the current step's tubes (the provider accumulates across the per-ego-drone calls inside solve_controls).
         if self._bof_provider is not None and hasattr(self._bof_provider, "clear_step_cache"):
            self._bof_provider.clear_step_cache()
         try:
            u_by_id = self.coordinator.solve_controls(drones=self.drones, obstacles=self.obstacles, room_min=self.room_min, room_max=self.room_max,
                  lstm_provider=(self._bof_provider or self._lstm_provider),
                  # Only `camera_feeds_dmpc` hands the mailbox over; otherwise perception observes passively and
                  # the coordinators keep negotiating on true states exactly as before.
                  perception_mailbox=(self._perception_mailbox if self._camera_feeds_dmpc else None), )

         except RuntimeError as exc:
            # Mark the step as infeasible (e.g. walls/obstacles make the optimization problem infeasible) and abort this step without advancing the
            # simulation time.
            self.infeasible = True
            self.infeasible_reason = str(exc)
            return

         for i, d in enumerate(self.drones):
            if d.drone_id in u_by_id:
               us[i] = np.asarray(u_by_id[d.drone_id], dtype=float).reshape(3)

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
