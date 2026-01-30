from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

ColorValue = str | list[float]


class PhysicsSpec(BaseModel):
   type: str
   params: dict[str, Any] = Field(default_factory=dict)


class ControllerSpec(BaseModel):
   type: str
   params: dict[str, Any] = Field(default_factory=dict)


class DroneConfig(BaseModel):
   drone_id: str
   start: list[float] = Field(..., min_length=3, max_length=3)
   waypoints: list[list[float]] = Field(default_factory=list)
   target: list[float] = Field(..., min_length=3, max_length=3)

   # Optional per-drone controller override (otherwise ScenarioConfig.controller is used).
   controller: ControllerSpec | None = None

   # Drone physical radius (used for room clamping and visualization).
   radius: float = 0.2

   # Visualization / safety bubble radius around the drone.
   safety_zone: float = 1.0

   # Conservative stopping addition, like it is shown in the paper
   cons_stop: float = 0.0

   # Maximum velocity magnitude (m/s) for trajectory optimization constraint.
   v_max: float = 5.0

   # Colors used by the renderer. Each field accepts either:
   # - a matplotlib-compatible color string (e.g. "red", "tab:blue", "#ff00aa")
   # - an RGB list [r,g,b] either in 0..1 or 0..255.
   drone_color: ColorValue = "tab:blue"
   # If omitted, the renderer uses the drone_color.
   safety_color: ColorValue | None = None
   # If omitted, the renderer uses the drone_color.
   trace_color: ColorValue | None = None


class ObstacleConfig(BaseModel):
   center: list[float] = Field(..., min_length=3, max_length=3)
   radius: float


class RoomConfig(BaseModel):
   """Axis-aligned room bounds used for visualization (and later constraints).
   """

   min: list[float] = Field(..., min_length=3, max_length=3)
   max: list[float] = Field(..., min_length=3, max_length=3)


class ScenarioConfig(BaseModel):
   dt: float = 0.1
   physics: PhysicsSpec

   # Default controller used for drones that do not define DroneConfig.controller.
   controller: ControllerSpec

   # The simulation step can be coordinated centrally (e.g. centralized MPC over a subset of drones) while still allowing per-drone controllers.
   # This is optional for now, cause we want to make changes in central_cost later, so this is not needed any longer.
   # Nevertheless, maybe there will also be an completely global coordinated system later, this can be used also.
   coordinator: ControllerSpec | None = None

   drones: list[DroneConfig]
   obstacles: list[ObstacleConfig] = Field(default_factory=list)

   # Optional visualization bounds.
   room: RoomConfig | None = None
