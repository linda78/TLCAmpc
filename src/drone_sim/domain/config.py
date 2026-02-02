from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

ColorValue = str | list[float]


class PhysicsConfig(BaseModel):
   """Named physics model configuration.

   Allows defining multiple physics configurations with different parameters
   that can be referenced by drones via the physics ID.
   """
   id: str
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

   # Reference to a physics configuration by ID. Required field.
   physics: str

   # Optional per-drone controller override (otherwise ScenarioConfig.controller is used).
   controller: ControllerSpec | None = None

   # Drone physical radius (used for room clamping and visualization).
   radius: float = 0.2

   # Visualization / safety bubble radius around the drone.
   # Set to None for adaptive safety zones.
   safety_zone: float | None = 1.0

   # Conservative stopping addition, like it is shown in the paper
   cons_stop: float = 0.0

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
   """Room bounds used for visualization and constraints.

   Supports two mutually exclusive geometries:
   - Rectangular (box): specify `min` and `max` (axis-aligned bounds)
   - Spherical: specify `r` (radius, center is always [0,0,0])

   Exactly one geometry must be specified. If both or neither are provided,
   a validation error is raised.
   """

   # Rectangular room bounds (axis-aligned box)
   min: list[float] | None = Field(default=None, min_length=3, max_length=3)
   max: list[float] | None = Field(default=None, min_length=3, max_length=3)

   # Spherical room radius (center is always [0,0,0])
   r: float | None = None

   @model_validator(mode="after")
   def validate_geometry(self) -> "RoomConfig":
      """Ensure exactly one room geometry is specified."""
      has_box = self.min is not None and self.max is not None
      has_partial_box = (self.min is not None) != (self.max is not None)
      has_sphere = self.r is not None

      if has_partial_box:
         raise ValueError("Rectangular room requires both 'min' and 'max' to be specified")

      if has_box and has_sphere:
         raise ValueError(
            "Room geometry is ambiguous: both rectangular (min/max) and spherical (r) "
            "parameters are specified. Please use only one geometry type."
         )

      if not has_box and not has_sphere:
         raise ValueError(
            "Room geometry not specified: provide either rectangular bounds (min/max) "
            "or spherical radius (r)."
         )

      return self

   def is_spherical(self) -> bool:
      """Return True if this room uses spherical geometry."""
      return self.r is not None


class ScenarioConfig(BaseModel):
   dt: float = 0.1

   # Named physics configurations that drones can reference by ID.
   physics: list[PhysicsConfig]

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

   @model_validator(mode="after")
   def validate_physics_references(self) -> "ScenarioConfig":
      """Ensure all drone physics IDs reference existing physics configurations."""
      if not self.physics:
         raise ValueError("At least one physics configuration must be defined")

      physics_ids = {p.id for p in self.physics}

      # Check for duplicate physics IDs
      if len(physics_ids) != len(self.physics):
         raise ValueError("Physics IDs must be unique")

      for drone in self.drones:
         if drone.physics not in physics_ids:
            raise ValueError(
               f"Drone '{drone.drone_id}' references unknown physics ID '{drone.physics}'. "
               f"Available physics IDs: {sorted(physics_ids)}"
            )

      return self
