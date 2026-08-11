from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
   status: str


class ConfigResponse(BaseModel):
   status: str
   num_drones: int


class StepResponse(BaseModel):
   status: str
   t: float
   dt: float
   # Optional detail about the step outcome (e.g. infeasibility reason).
   detail: str | None = None


class DroneState(BaseModel):
   drone_id: str
   x: list[float] = Field(..., min_length=6, max_length=6)
   route_idx: int
   p_ref: list[float] = Field(..., min_length=3, max_length=3)
   radius: float
   safety_zone: float
   adaptive_safety_radius: float | None = None

   drone_color: str | list[float]
   safety_color: str | list[float]
   trace_color: str | list[float]


class ObstacleState(BaseModel):
   center: list[float] = Field(..., min_length=3, max_length=3)
   half_extents: list[float] = Field(..., min_length=3, max_length=3)


class RoomState(BaseModel):
   min: list[float] = Field(..., min_length=3, max_length=3)
   max: list[float] = Field(..., min_length=3, max_length=3)


class CollisionEvent(BaseModel):
   kind: str
   owner: str

   # drone_drone
   intruder: str | None = None

   # drone_obstacle
   obstacle_idx: int | None = None

   distance: float
   threshold: float


class ADMMStats(BaseModel):
   """ADMM statistics from distributed MPC coordinator."""

   iteration_count: int
   primal_residual: float
   dual_residual: float
   converged: bool
   neighbor_pairs: list[list[str]] = Field(default_factory=list)


class StateResponse(BaseModel):
   t: float
   dt: float
   room: RoomState
   drones: list[DroneState]
   obstacles: list[ObstacleState]
   collisions: list[CollisionEvent] = []
   admm_stats: ADMMStats | None = None  # Only present with DMPC coordinator


# ---------------------------------------------------------------------------
# Perception (see drone_sim.api.perception_router)
# ---------------------------------------------------------------------------


class PerceptionVisibleDrone(BaseModel):
   """Ground truth of one neighbor inside the pulled view.

   Only sent when ``camera_expose_truth`` is on — it is a calibration/debugging aid, not part of the normal exchange: deriving positions from pixels
   *is* the detector's job. Deliberately without velocity; the DMPC bridge derives that by finite difference over consecutive estimates.
   """

   drone_id: str
   position: list[float] = Field(..., min_length=3, max_length=3)
   radius: float


class PerceptionViewResponse(BaseModel):
   """One rendered camera view, ready for the detector to process.

   ``captured_step`` / ``captured_time`` are the capture token: the detector echoes both back in :class:`PerceptionEstimatesRequest`, which is what
   anchors an estimate on the simulation time axis.
   """

   observer_id: str
   captured_step: int
   captured_time: float
   image_png_base64: str
   visible: list[PerceptionVisibleDrone] | None = None


class PerceptionViewInfo(BaseModel):
   """Index entry of the cheap poll endpoint — capture token only, never image data."""

   observer_id: str
   captured_step: int
   captured_time: float


class PerceptionViewsResponse(BaseModel):
   views: list[PerceptionViewInfo] = Field(default_factory=list)


class PerceptionEstimate(BaseModel):
   """One estimated neighbor position as reported by the detector.

   ``drone_id`` identifies the *observed* neighbor (data association is assumed solved, UVDAR-style); the observing drone is named once per request.
   """

   drone_id: str
   position: list[float] = Field(..., min_length=3, max_length=3)
   sigma: float | None = None


class PerceptionEstimatesRequest(BaseModel):
   """A detector's answer for exactly one ego drone at exactly one capture.

   ``captured_step`` and ``captured_time`` are **required** on purpose: they come from the pulled view and must be echoed unchanged. Without them the
   simulation time axis breaks silently and the bridge's finite-difference velocities turn into noise.
   """

   observer_id: str
   captured_step: int
   captured_time: float
   estimates: list[PerceptionEstimate] = Field(default_factory=list)


class PerceptionEstimatesResponse(BaseModel):
   status: str
   accepted: int
