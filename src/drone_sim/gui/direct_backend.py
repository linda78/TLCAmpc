"""DirectBackend: SimulationBackend implementation wrapping Simulator directly."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator
from drone_sim.gui.backend import (SimulationBackend, SimState, DroneState, StepResult, PredictedTrajectory, )
from tools.franck.compute_path import compute_n_drone_collision_paths
from drone_sim.domain.utils.helper import all_drones_reached_destination

class DirectBackend(SimulationBackend):
   """Wraps Simulator with no serialization overhead. No Qt dependency."""

   def __init__(self) -> None:
      self._sim: Simulator | None = None
      self._cfg: ScenarioConfig | None = None
      self._config_path: Path | None = None
      self._drone_starts: dict[str, np.ndarray] = {}
      self._reference_paths = None
      self._collision_point = None

   # ------------------------------------------------------------------ #
   # Public API                                                           #
   # ------------------------------------------------------------------ #

   def load_config(self, path: Path) -> SimState:
      cfg_json = json.loads(Path(path).read_text(encoding="utf-8"))
      cfg = ScenarioConfig.model_validate(cfg_json)
      self._cfg = cfg
      self._config_path = Path(path)
      self._sim = Simulator.from_config(cfg)           
      start_positions = [d.start_position if d.start_position is not None else d.position() for d in self._sim.drones]
      self._reference_paths, self._collision_point = compute_n_drone_collision_paths(positions=start_positions, cube_side=6.0, T=500)
      
      return self._make_sim_state()

   def step(self) -> StepResult:
      if self._sim is None:
         raise RuntimeError("Call load_config() before step()")      

      if not all_drones_reached_destination(self._sim.drones):
         self._sim.step()  
         
      return self._make_step_result()

   def initial_result(self) -> StepResult:
      if self._sim is None:
          raise RuntimeError("Call load_config() first")

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
            
      start_positions = [np.asarray(d.start_position if d.start_position is not None else d.position(), dtype=float).copy() for d in self._sim.drones]
      self._reference_paths, self._collision_point = (compute_n_drone_collision_paths(positions=start_positions, cube_side=6.0, T=500,))

   # ------------------------------------------------------------------ #
   # Private helpers                                                      #
   # ------------------------------------------------------------------ #

   def _make_sim_state(self) -> SimState:
      sim = self._sim
      coordinator_type = (type(sim.coordinator).__name__ if sim.coordinator is not None else "none")
      return SimState(drone_count=len(sim.drones), obstacle_count=len(sim.obstacles), obstacles=sim.obstacles, coordinator_type=coordinator_type,
                      dt=sim.dt, step_count=sim.step_count, room_min=sim.room_min, room_max=sim.room_max, start_positions=[d.start_position for d in sim.drones],
                      target_positions=[d.route.target for d in sim.drones], reference_paths=self._reference_paths, collision_point=self._collision_point,
                      config_path=str(self._config_path) if self._config_path is not None else None, )

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
               trace_color=(d.trace_color if isinstance(d.trace_color, str) else list(d.trace_color)), start=self._drone_starts.get(d.drone_id, np.zeros(3)), target=np.asarray(d.route.target, dtype=float),))


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
            infeasible_reason=sim.infeasible_reason, step_count=sim.step_count, t=float(sim.t), all_reached=all_reached, admm_iteration_count=admm_iteration_count, 
            predictions=predictions, reference_paths=self._reference_paths, collision_point=self._collision_point, start_positions=[d.start_position for d in sim.drones], 
            target_positions=[d.route.target for d in sim.drones],)

