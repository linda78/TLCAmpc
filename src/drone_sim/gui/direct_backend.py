"""DirectBackend: SimulationBackend implementation wrapping Simulator directly."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from drone_sim.domain.config import ScenarioConfig
from drone_sim.perception import CameraModel, render_fpv_png
from drone_sim.simulation.simulator import Simulator
from drone_sim.gui.backend import (SimulationBackend, SimState, DroneState, StepResult, PredictedTrajectory, )


class DirectBackend(SimulationBackend):
   """Wraps Simulator with no serialization overhead. No Qt dependency."""

   def __init__(self) -> None:
      self._sim: Simulator | None = None
      self._cfg: ScenarioConfig | None = None
      self._config_path: Path | None = None
      # Rebuilt on every load/reset, never reused: CameraModel caches the last heading per drone id
      # for the zero-velocity fallback, and that cache must not outlive the run it was observed in.
      self._camera: CameraModel | None = None
      # Started on the first REST-perception scenario and then kept alive across reloads — the router
      # resolves the simulation per request, so a detector connection survives a reset. See close().
      self._perception_server: object | None = None

   # ------------------------------------------------------------------ #
   # Public API                                                           #
   # ------------------------------------------------------------------ #

   def load_config(self, path: Path) -> SimState:
      cfg_json = json.loads(Path(path).read_text(encoding="utf-8"))
      cfg = ScenarioConfig.model_validate(cfg_json)
      self._cfg = cfg
      self._config_path = Path(path)
      self._sim = Simulator.from_config(cfg)
      self._camera = self._make_camera()
      self._ensure_perception_server(cfg)
      return self._make_sim_state()

   def step(self) -> StepResult:
      if self._sim is None:
         raise RuntimeError("Call load_config() before step()")
      self._sim.step()
      return self._make_step_result()

   def get_state(self) -> SimState:
      if self._sim is None:
         raise RuntimeError("Call load_config() before get_state()")
      return self._make_sim_state()

   def reset(self) -> None:
      if self._cfg is None:
         raise RuntimeError("Call load_config() before reset()")
      # Uses CACHED config — does NOT re-read from disk
      self._sim = Simulator.from_config(self._cfg)
      self._camera = self._make_camera()

   def render_fpv(self, drone_id: str, size: tuple[int, int]) -> bytes | None:
      """Render ``drone_id``'s pinhole camera image of the current simulation state.

      Rendered synchronously on the calling thread — no ``PerceptionWorker`` involved. The worker exists to
      keep a per-step capture of *every* drone off the simulation thread; here exactly one view is drawn on
      demand, well inside a GUI frame budget.

      The camera geometry comes from the scenario's ``camera_fov_deg``/``camera_range`` so this view matches
      what the detector will later work with; ``camera_enabled`` is deliberately not consulted — it gates the
      perception pipeline, not drawing a picture when someone asks for one.
      """
      if self._sim is None:
         raise RuntimeError("Call load_config() before render_fpv()")
      drone = next((d for d in self._sim.drones if d.drone_id == drone_id), None)
      if drone is None:
         return None
      view = self._camera.capture(drone, self._sim.drones, step=self._sim.step_count, sim_time=self._sim.t)
      return render_fpv_png(view, self._sim.obstacles, models={d.drone_id: d.model for d in self._sim.drones}, size=size)

   def close(self) -> None:
      """Stop the perception API thread. Idempotent, safe to call without a loaded config."""
      if self._perception_server is not None:
         self._perception_server.stop()
         self._perception_server = None

   # ------------------------------------------------------------------ #
   # Private helpers                                                      #
   # ------------------------------------------------------------------ #

   def _make_camera(self) -> CameraModel:
      return CameraModel(fov_deg=self._cfg.camera_fov_deg, range_m=self._cfg.camera_range)

   def _ensure_perception_server(self, cfg: ScenarioConfig) -> None:
      """Start the perception REST server once, for scenarios that ask the detector to pull images over HTTP.

      Only ``camera_backend == "rest"`` starts a listener: every other scenario — including every existing one — stays exactly as it was, and two
      GUI instances only collide on ``camera_api_port`` when both actually serve perception.

      Started at most once per backend. A reload never restarts it, and never stops it either: the resolver below hands out ``self._sim``, so the
      running server follows the new simulation on its own and a detector mid-conversation is not disconnected by a reset. Loading a non-perception
      scenario afterwards therefore leaves the server up, answering ``409`` until a perception scenario is loaded again.

      The uvicorn import lives in :meth:`PerceptionApiServer.start`, so the default GUI path never pays for it.
      """
      if self._perception_server is not None or not (cfg.camera_enabled and cfg.camera_backend == "rest"):
         return

      from drone_sim.gui.perception_server import PerceptionApiServer

      server = PerceptionApiServer(lambda: self._sim, port=cfg.camera_api_port)
      server.start()
      self._perception_server = server

   def _make_sim_state(self) -> SimState:
      sim = self._sim
      coordinator_type = (type(sim.coordinator).__name__ if sim.coordinator is not None else "none")
      return SimState(drone_count=len(sim.drones), obstacle_count=len(sim.obstacles), obstacles=sim.obstacles, coordinator_type=coordinator_type,
                      dt=sim.dt, step_count=sim.step_count, room_min=sim.room_min, room_max=sim.room_max, config_path=str(self._config_path) if self._config_path is not None else None,
                      drone_ids=[d.drone_id for d in sim.drones], )

   def _make_step_result(self) -> StepResult:
      sim = self._sim
      drone_states: list[DroneState] = []
      safety_radii: list[float] = []

      for i, d in enumerate(sim.drones):
         vel = d.velocity()
         r =  float(d.compute_adaptive_radius(vel))
         safety_radii.append(r)
         drone_states.append(DroneState(drone_id=d.drone_id, position=d.position(), velocity=vel, radius=d.radius, safety_zone=float(d.safety_zone),
               adaptive_safety_radius=r if d.is_adaptive else None, max_adaptive_safety_radius=d.compute_max_adaptive_radius() if d.is_adaptive else None,
               color=d.color if isinstance(d.color, str) else list(d.color), safety_color=(d.safety_color if isinstance(d.safety_color, str) else list(d.safety_color)),
               trace_color=(d.trace_color if isinstance(d.trace_color, str) else list(d.trace_color))))

      # Destination check — helper takes Drone objects, not DroneState (BACK-01: no sim access in GUI)
      from drone_sim.domain.utils.helper import all_drones_reached_destination
      all_reached = all_drones_reached_destination(sim.drones)

      # ADMM stats — hasattr duck-typing (same pattern as app.py and sim.to_dict())
      admm_iteration_count: int | None = None
      if hasattr(sim.coordinator, "get_last_iteration_count"):
         admm_iteration_count = sim.coordinator.get_last_iteration_count()

      # BoF predictions — surface trajectories + radii from the most recent
      # provider call. LSTM provider doesn't expose trajectories yet, so this
      # is BoF-only for now. predicted_trajectories is the raw BoF output;
      # last_radii is the post-processed (floor/cap) version that the planner
      # actually used as a constraint.
      predictions: list[PredictedTrajectory] = []
      provider = getattr(sim, "_bof_provider", None)
      if provider is not None:
         trajs = getattr(provider, "predicted_trajectories", {})
         radii_by_id = getattr(provider, "last_radii", {})
         drone_by_id = {d.drone_id: d for d in sim.drones}
         for did, traj in trajs.items():
            radii = radii_by_id.get(did)
            drone = drone_by_id.get(did)
            if radii is None or drone is None:
               continue
            safety_color = drone.safety_color if isinstance(drone.safety_color, str) else list(drone.safety_color)
            core_color = drone.color if isinstance(drone.color, str) else list(drone.color)
            predictions.append(PredictedTrajectory(
               drone_id=did,
               points=np.asarray(traj, dtype=float),
               radii=np.asarray(radii, dtype=float),
               color=safety_color,
               inner_radius=float(drone.radius),
               core_color=core_color,
            ))

      return StepResult(drones=drone_states, safety_radii=safety_radii, last_collisions=list(sim.last_collisions), infeasible=bool(sim.infeasible),
            infeasible_reason=sim.infeasible_reason, step_count=sim.step_count, t=float(sim.t), all_reached=all_reached,
            admm_iteration_count=admm_iteration_count, predictions=predictions)
