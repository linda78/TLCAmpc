from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.config import ColorValue, ScenarioConfig
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
   obstacles: list[tuple[np.ndarray, float]]
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

   @classmethod
   def from_config(cls, cfg: ScenarioConfig) -> "Simulator":
      # Ensure implementations are registered
      from drone_sim.controllers import central_cost as _central_cost  # noqa: F401
      from drone_sim.physics import linear_kinematics as _  # noqa: F401
      from drone_sim.simulation import coordinator as _coord  # noqa: F401

      physics = create_physics({"type": cfg.physics.type, "params": {"dt": cfg.dt, **cfg.physics.params}})

      drones: list[Drone] = []

      coordinator = None
      if cfg.coordinator is not None:
         coordinator = create_coordinator(
               {"type": cfg.coordinator.type, "params": {"dt": cfg.dt, **cfg.coordinator.params}})

      for drone_cfg in cfg.drones:
         spec = drone_cfg.controller or cfg.controller
         controller = create_controller({"type": spec.type, "params": {"dt": cfg.dt, **spec.params}})
         start = np.asarray(drone_cfg.start, dtype=float)
         x0 = np.zeros(6, dtype=float)
         x0[:3] = start

         route = Route(waypoints=[np.asarray(w, dtype=float) for w in drone_cfg.waypoints],
                       target=np.asarray(drone_cfg.target, dtype=float))
         drone_color = _normalize_color(drone_cfg.drone_color)
         safety_color = _normalize_color(drone_cfg.safety_color or drone_cfg.drone_color)
         trace_color = _normalize_color(drone_cfg.trace_color or drone_cfg.drone_color)

         drones.append(Drone(drone_id=drone_cfg.drone_id, radius=drone_cfg.radius, safety_zone=drone_cfg.safety_zone,
                             cons_stop=drone_cfg.cons_stop, v_max=drone_cfg.v_max, color=drone_color, safety_color=safety_color,
                             trace_color=trace_color, controller=controller, x=x0, route=route))

      obstacles = [(np.asarray(o.center, dtype=float), float(o.radius)) for o in cfg.obstacles]

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
         for c, _r in obstacles:
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

      # Initialize traces with the start positions.
      sim.traces = {d.drone_id: [d.position().copy()] for d in sim.drones}
      sim.last_collisions = sim._compute_collisions()
      return sim

   def _compute_collisions(self) -> list[dict]:
      """Compute safety-zone collision events using fixed radii only.

         A collision is reported when another drone enters the owner's safety sphere (radius = owner.safety_zone + intruder.radius)
         or when an obstacle center enters radius = owner.safety_zone + obstacle.radius.
      """
      events: list[dict] = []

      # Drone-drone
      for i, owner in enumerate(self.drones):
         p_owner = owner.position()
         for j, intr in enumerate(self.drones):
            if i == j:
               continue

            dist = float(np.linalg.norm(intr.position() - p_owner))
            threshold = float(owner.safety_zone + intr.radius)

            if dist <= threshold:
               events.append({"kind": "drone_drone", "owner": owner.drone_id, "intruder": intr.drone_id, "distance": dist, "threshold": threshold})

      # Drone-obstacle
      for owner in self.drones:
         p_owner = owner.position()
         for k, (c, r) in enumerate(self.obstacles):
            dist = float(np.linalg.norm(c - p_owner))
            threshold = float(owner.safety_zone + r)
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
            raise RuntimeError(
               "Simulator-step requires a coordinator (centralized MPC). Provide `coordinator` in ScenarioConfig.")

         # First compute per-drone local controls (used for non-optimized drones, and as a fallback).
         for i, d in enumerate(self.drones):
            neighbors = [(positions[j], velocities[j], self.drones[j].radius, self.drones[j].safety_zone, prefs[j],) for
                         j in range(len(self.drones)) if j != i]

            if hasattr(d.controller, "control"):
               u = d.controller.control(d.x, prefs[i], neighbors, self.obstacles, self_radius=d.radius,
                                        self_safety_zone=d.safety_zone)
            else:
               u = np.zeros(3, dtype=float)
            us.append(np.asarray(u, dtype=float).reshape(3))

         # Then override optimized drones with coordinator outputs.
         all_drone_state = {d.drone_id: (positions[i], velocities[i], float(d.radius)) for i, d in
                            enumerate(self.drones)}

         try:
            u_by_id = self.coordinator.solve_controls(drone_ids=[d.drone_id for d in self.drones], xs=[d.x for d in self.drones], prefs=prefs,
                                                      radii=[d.radius for d in self.drones], safety_zones=[d.safety_zone for d in self.drones],
                                                      cons_stops=[d.cons_stop for d in self.drones], v_maxs=[d.v_max for d in self.drones],
                                                      controllers=[d.controller for d in self.drones],
                                                      obstacles=self.obstacles, all_drone_state=all_drone_state, room_min=self.room_min, room_max=self.room_max)

         except RuntimeError as exc:
            # Mark the step as infeasible (e.g. walls/obstacles make the optimization problem infeasible) and abort this step without advancing the simulation time.
            self.infeasible = True
            self.infeasible_reason = str(exc)
            return

         for i, d in enumerate(self.drones):
            if d.drone_id in u_by_id:
               us[i] = np.asarray(u_by_id[d.drone_id], dtype=float).reshape(3)

         # Apply physics updates
         for d, u in zip(self.drones, us, strict=True):
            d.x = self.physics.step(d.x, u)

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

         self.last_collisions = self._compute_collisions()

         self.t += self.dt
         self.step_count += 1
      finally:
         self.compute_time_s += time.perf_counter() - t0

   def to_dict(self) -> dict:
      return {"t": self.t, "dt": self.dt, "room": {"min": self.room_min.tolist(), "max": self.room_max.tolist()},
            "drones": [{"drone_id": d.drone_id, "x": d.x.tolist(), "route_idx": d.route.idx,
                        "p_ref": d.route.current_ref().tolist(), "radius": d.radius, "safety_zone": d.safety_zone,
                        "drone_color": _color_to_json(d.color), "safety_color": _color_to_json(d.safety_color),
                        "trace_color": _color_to_json(d.trace_color)} for d in self.drones],
            "obstacles": [{"center": c.tolist(), "radius": r} for c, r in self.obstacles],
            "collisions": list(self.last_collisions)}
