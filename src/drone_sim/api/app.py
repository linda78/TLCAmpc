from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import Response

from drone_sim.api.render import render_png
from drone_sim.api.schemas import ConfigResponse, HealthResponse, StateResponse, StepResponse
from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator

app = FastAPI(title="DroneSim 3D (REST)")

_sim: Simulator | None = None


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

   traces = [[] if trace_len == 0 else _sim.traces.get(d.drone_id, [])[-trace_len:] for d in _sim.drones]

   # Use fixed safety zones as defined on each drone (no velocity-dependent adjustment).
   safety_zones = [float(d.safety_zone) for d in _sim.drones]

   png = render_png(room_min=_sim.room_min, room_max=_sim.room_max, drone_positions=[d.position() for d in _sim.drones],
                    drone_radii=[d.radius for d in _sim.drones], drone_safety_zones=safety_zones,
                    drone_colors=[d.color for d in _sim.drones], safety_colors=[d.safety_color for d in _sim.drones],
                    trace_colors=[d.trace_color for d in _sim.drones], drone_traces=traces, obstacles=_sim.obstacles,
                    step_count=_sim.step_count, compute_time_s=_sim.compute_time_s, room_radius=_sim.room_radius,
                    width=width, height=height, dpi=dpi, elev=elev, azim=azim)

   return Response(content=png, media_type="image/png")
