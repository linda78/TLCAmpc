from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ColorValue = str | list[float]


class PhysicsSpec(BaseModel):
   type: str
   id: str | None = None
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

   # Physics model ID referencing a PhysicsSpec by name (None = use global physics).
   physics: str | None = None

   # Drone physical radius (used for room clamping and visualization).
   radius: float = 0.2

   # Visualization / safety bubble radius around the drone.
   safety_zone: float = 1.0

   # Conservative stopping addition, like it is shown in the paper
   cons_stop: float = 0.0

   # Adaptive safety zone parameter. When set, the drone uses a velocity-dependent
   # safety radius: r(t) = r_min + alpha * ||v||^2 / (2 * U_max).
   # When None, the fixed safety_zone is used instead.
   alpha: float | None = None

   # Safety zone mode. "fixed" uses the static safety_zone, "adaptive" uses
   # velocity-dependent radius (requires alpha), "lstm" uses LSTM-predicted radii.
   # Default "fixed" is backward compatible with all existing configs.
   safety_zone_mode: Literal["fixed", "adaptive", "lstm"] = "fixed"

   # Colors used by the renderer. Each field accepts either:
   # - a matplotlib-compatible color string (e.g. "red", "tab:blue", "#ff00aa")
   # - an RGB list [r,g,b] either in 0..1 or 0..255.
   drone_color: ColorValue = "tab:blue"
   # If omitted, the renderer uses the drone_color.
   safety_color: ColorValue | None = None
   # If omitted, the renderer uses the drone_color.
   trace_color: ColorValue | None = None

   @model_validator(mode="after")
   def _validate_alpha(self) -> DroneConfig:
      if self.alpha is not None and self.alpha <= 0:
         raise ValueError("alpha must be positive when set")
      return self


class ObstacleConfig(BaseModel):
   center: list[float] = Field(..., min_length=3, max_length=3)
   half_extents: list[float] = Field(..., min_length=3, max_length=3)


class RoomConfig(BaseModel):
   """Axis-aligned room bounds used for visualization (and later constraints).
   """

   min: list[float] = Field(..., min_length=3, max_length=3)
   max: list[float] = Field(..., min_length=3, max_length=3)


class ScenarioConfig(BaseModel):
   dt: float = 0.1
   physics: PhysicsSpec | list[PhysicsSpec]

   # Default controller used for drones that do not define DroneConfig.controller.
   controller: ControllerSpec

   # Optional coordinator used for distributed MPC neighbor discovery.
   coordinator: ControllerSpec | None = None

   # Communication radius for distributed MPC neighbor discovery.
   # When None, all drones are considered neighbors (backward compatible with centralized mode).
   # When set to a positive float, only drones within this distance are neighbors.
   comm_radius: float | None = None

   drones: list[DroneConfig]
   obstacles: list[ObstacleConfig] = Field(default_factory=list)

   # Optional visualization bounds.
   room: RoomConfig | None = None

   # Path to a trained LSTM model checkpoint (.pt file). Required when any
   # drone has safety_zone_mode="lstm". Defaults to None for backward compat.
   lstm_model_path: str | None = None

   # Look-ahead step index for LSTM uncertainty extraction (1-indexed).
   # When None, uses sigma[:horizon] (first H steps — small near-term sigma).
   # When set to N, uses sigma[N-1] (0-indexed) tiled across all H steps.
   # sigma grows with autoregressive step count: look_ahead=20 gives ~T/4
   # accumulated uncertainty, typically 3-5x larger than first-step sigma.
   # Start with 20 for H=4, T=80 models; increase if floor still dominates.
   lstm_look_ahead: int | None = None

   # Enable the BoF (Backoff-Function) safety-zone provider as a parallel
   # alternative to the LSTM provider. When True, BoFSafetyZoneProvider is
   # instantiated and supplied to coordinators in place of the LSTM one.
   # The actual tool call lives in BoFSafetyZoneProvider._call_bof_tool.
   bof_enabled: bool = False

   # Which BoF adapter to use. "library" calls Brian's BoFPredictor in-process
   # (default — no network IO in the MPC hot path). "rest" POSTs each history
   # window to a Flask endpoint at ``bof_url``.
   bof_backend: Literal["library", "rest"] = "library"

   # Endpoint base URL for the REST adapter, e.g. "http://localhost:5005".
   # Required when ``bof_backend == "rest"``.
   bof_url: str | None = None

   # When True, forward the velocity columns of the history buffer to BoF
   # (sends 6 columns instead of 3). The returned trajectory is sliced back
   # to (H, 3) by the adapter regardless. Default is positions only.
   bof_has_velocity: bool = False

   # History window size m (rows of [px,py,pz,vx,vy,vz]) the BoF predictor
   # expects per neighbor. Drives TrajectoryHistoryBuffer capacity.
   bof_history_size: int = 100

   # How many steps BoF predicts into the future per call. Independent of
   # the MPC horizon (which controls how far the planner enforces
   # constraints). The provider takes the first ``coordinator.params.horizon``
   # radii as the planner's per-step constraint and caches the full
   # ``bof_horizon``-length trajectory + radii for the GUI prediction tube.
   # Default 50 ≈ 5 s of lookahead at dt=0.1; bump to 100 for longer tubes.
   bof_horizon: int = 50

   # Time-decay constant for BoF's per-step σ growth. ``None`` (default)
   # leaves the predictor's σ flat over the horizon — radii are constant.
   # When set to a positive float, BoF scales radii by ``sqrt(1 + k/tau)``
   # at horizon step k, producing growing-funnel uncertainty that flows
   # through to both the MPC constraint and the GUI tube. Smaller tau =
   # faster growth (tau≈horizon/4 for visibly cone-shaped tubes).
   bof_growth_tau: float | None = None

   # Enable the camera perception pipeline (View → capture → external
   # predictor → PerceptionMailbox). When False (default) no captures are
   # taken and the simulator behaves exactly as before — all camera_* fields
   # below are validated but otherwise inert until the simulator wiring lands.
   camera_enabled: bool = False

   # Horizontal field of view of the simulated camera in degrees. Neighbors
   # outside the FOV cone are not observed. Must be in (0, 360].
   camera_fov_deg: float = 90.0

   # Maximum sensing distance in meters. Neighbors farther away than this
   # are not observed regardless of FOV. Must be positive.
   camera_range: float = 10.0

   # Which capture backend to use. "stub" synthesizes detections in-process
   # from true sim state (default — no network IO). "rest" POSTs each capture
   # to an external predictor service at ``camera_url``.
   camera_backend: Literal["stub", "rest"] = "stub"

   # Endpoint base URL for the REST backend, e.g. "http://localhost:5006".
   # Required when ``camera_backend == "rest"``.
   camera_url: str | None = None

   # Standard deviation of Gaussian position noise added to observed
   # neighbor positions (meters). 0.0 (default) = perfect observations.
   camera_noise_sigma: float = 0.0

   # Capture cadence: one capture every N sim steps. 1 (default) captures
   # every step; larger values model slower camera/predictor rates.
   camera_rate_steps: int = 1

   # When True, render a pseudo-FPV PNG per capture and attach it to the
   # View for debugging/visualization. Requires ``camera_enabled``.
   camera_render_images: bool = False

   # When True (default), captures are processed asynchronously off the MPC
   # hot path. Set False to run inline for deterministic tests.
   camera_async: bool = True

   # When True, the DMPC neighbor mailbox is fed from camera perception
   # instead of true-state communication broadcasts. Default False keeps the
   # mailbox fed from broadcasts as before. Requires ``camera_enabled``.
   camera_feeds_dmpc: bool = False

   @model_validator(mode="after")
   def _validate_bof(self) -> ScenarioConfig:
      if self.bof_backend == "rest" and not self.bof_url:
         raise ValueError("bof_url must be set when bof_backend='rest'")
      if self.bof_history_size <= 0:
         raise ValueError("bof_history_size must be positive")
      if self.bof_horizon <= 0:
         raise ValueError("bof_horizon must be positive")
      if self.bof_growth_tau is not None and self.bof_growth_tau <= 0:
         raise ValueError("bof_growth_tau must be positive when set")
      return self

   @model_validator(mode="after")
   def _validate_camera(self) -> ScenarioConfig:
      if self.camera_backend == "rest" and not self.camera_url:
         raise ValueError("camera_url must be set when camera_backend='rest'")
      if not 0 < self.camera_fov_deg <= 360:
         raise ValueError("camera_fov_deg must be in (0, 360]")
      if self.camera_range <= 0:
         raise ValueError("camera_range must be positive")
      if self.camera_rate_steps < 1:
         raise ValueError("camera_rate_steps must be at least 1")
      if self.camera_noise_sigma < 0:
         raise ValueError("camera_noise_sigma must be non-negative")
      if self.camera_feeds_dmpc and not self.camera_enabled:
         raise ValueError("camera_feeds_dmpc requires camera_enabled")
      if self.camera_render_images and not self.camera_enabled:
         raise ValueError("camera_render_images requires camera_enabled")
      return self
