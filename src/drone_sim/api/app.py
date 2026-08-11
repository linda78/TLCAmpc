from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import Response

from drone_sim.api.perception_router import build_perception_router
from drone_sim.api.render import render_png
from drone_sim.api.schemas import ConfigResponse, HealthResponse, StateResponse, StepResponse
from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator

app = FastAPI(title="DroneSim 3D (REST)")

_sim: Simulator | None = None

# Perception endpoints for the external video detector. The lambda is read on every request, so the router always sees whatever /config installed last.
app.include_router(build_perception_router(lambda: _sim))


@app.get("/swagger", include_in_schema=False)
def swagger_ui() -> object:
   # FastAPI already serves Swagger UI at /docs.
   # This is just a more explicit alias some users expect.
   return get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
   return HealthResponse(status="ok")


@app.post("/config", response_model=ConfigResponse, tags=["simulation"])
def load_config(cfg: ScenarioConfig) -> ConfigResponse:
   global _sim
   # The outgoing simulation may own a perception worker thread; without this, every POST /config would leave
   # one more of them capturing and rendering for a scenario nobody looks at any more (risk R2).
   if _sim is not None:
      _sim.close()
   _sim = Simulator.from_config(cfg)
   return ConfigResponse(status="loaded", num_drones=len(_sim.drones))


@app.post("/step", response_model=StepResponse, tags=["simulation"])
def step(n: int = 1) -> StepResponse:
   if _sim is None:
      raise HTTPException(status_code=400, detail="No scenario configured. POST /config first.")
   for _ in range(n):
      _sim.step()
      if _sim.infeasible:
         # Central MPC reported an infeasible optimization problem (e.g. walls or obstacles blocking all routes).
         # Surface this as a structured status instead of a generic 500 error.
         return StepResponse(status="infeasible", t=_sim.t, dt=_sim.dt, detail=_sim.infeasible_reason)
   return StepResponse(status="stepped", t=_sim.t, dt=_sim.dt)


@app.get("/state", response_model=StateResponse, tags=["simulation"])
def state() -> StateResponse:
   if _sim is None:
      raise HTTPException(status_code=400, detail="No scenario configured. POST /config first.")
   return StateResponse.model_validate(_sim.to_dict())


@app.get("/render", tags=["visualization"], responses={200: {"content": {"image/png": {}}}})
def render(width: int = 900, height: int = 700, dpi: int = 120, elev: float = 20.0, azim: float = -60.0,
           trace_len: int = 300) -> Response:
   """Render the current state as a PNG image."""

   if _sim is None:
      raise HTTPException(status_code=400, detail="No scenario configured. POST /config first.")

   if trace_len < 0:
      raise HTTPException(status_code=422, detail="trace_len must be >= 0")

   traces = {d.drone_id: _sim.traces.get(d.drone_id, [])[-trace_len:] for i, d in enumerate(_sim.drones)}

   # Compute per-drone safety zone radius and wireframe alpha.
   # compute_adaptive_radius handles both adaptive (velocity-dependent) and fixed drones.
   safety_zones: list[float] = []
   safety_alphas: list[float] = []
   for d in _sim.drones:
       vel = d.velocity()
       safety_zones.append(float(d.compute_adaptive_radius(vel)))
       if d.is_adaptive:
           speed_ratio = min(float(np.linalg.norm(vel)) / d.v_max, 1.0) if d.v_max > 0 else 0.0
           safety_alphas.append(0.3 + 0.7 * speed_ratio)
       else:
           safety_alphas.append(0.8)

   max_safety_zones: list[float] = [
       float(d.compute_max_adaptive_radius()) for d in _sim.drones
   ]

   # Extract ADMM visualization data if using distributed coordinator
   neighbor_links = None
   admm_iteration_count = None
   admm_converged = None

   if hasattr(_sim.coordinator, "get_neighbor_pairs"):
      # Build neighbor links as index pairs (not drone_id pairs)
      drone_id_to_idx = {d.drone_id: i for i, d in enumerate(_sim.drones)}
      id_pairs = _sim.coordinator.get_neighbor_pairs()
      neighbor_links = [
         (drone_id_to_idx[id_i], drone_id_to_idx[id_j])
         for id_i, id_j in id_pairs
         if id_i in drone_id_to_idx and id_j in drone_id_to_idx
      ]
      admm_iteration_count = _sim.coordinator.get_last_iteration_count()
      admm_converged = _sim.coordinator.get_last_converged()

   png = render_png(room_min=_sim.room_min, room_max=_sim.room_max, drones=_sim.drones, drone_traces=traces, obstacles=_sim.obstacles,
                    step_count=_sim.step_count, compute_time_s=_sim.compute_time_s, neighbor_links=neighbor_links,
                    admm_iteration_count=admm_iteration_count, admm_converged=admm_converged,
                    safety_alphas=safety_alphas, width=width, height=height, dpi=dpi, elev=elev, azim=azim)

   return Response(content=png, media_type="image/png")
