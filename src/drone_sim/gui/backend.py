from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DroneState:
    drone_id: str
    position: np.ndarray             # [x, y, z]
    velocity: np.ndarray             # [vx, vy, vz]
    radius: float
    safety_zone: float
    adaptive_safety_radius: float | None   # None for non-adaptive drones
    max_adaptive_safety_radius: float | None
    color: str | list[float]
    safety_color: str | list[float]
    trace_color: str | list[float]


@dataclass
class PredictedTrajectory:
    """Per-step prediction emitted by an uncertainty provider (BoF / future LSTM-traj).

    ``points`` and ``radii`` are aligned row-by-row: ``radii[k]`` is the safety
    radius the planner applied to ``points[k]``. ``inner_radius`` is the
    drone's physical body radius — constant along the path, used for the
    inner (core) tube.
    """
    drone_id: str                    # the predicted neighbor's id
    points: np.ndarray               # (H, 3) predicted positions over the horizon
    radii: np.ndarray                # (H,) post-processed safety radii (floor/cap applied)
    color: str | list[float]         # neighbor's safety color (outer tube tint)
    inner_radius: float              # drone.radius — the body, not the safety zone
    core_color: str | list[float]    # neighbor's drone color (inner tube tint)


@dataclass
class StepResult:
    drones: list[DroneState]
    safety_radii: list[float]         # current effective safety radius per drone
    last_collisions: list[dict]       # raw collision events from Simulator
    infeasible: bool
    infeasible_reason: str | None
    step_count: int
    t: float
    all_reached: bool = False              # True when all drones are at their destination
    admm_iteration_count: int | None = None  # None for non-ADMM coordinators
    predictions: list[PredictedTrajectory] = field(default_factory=list)


@dataclass
class SimState:
    drone_count: int
    obstacle_count: int
    obstacles: list[tuple[np.ndarray, np.ndarray]]
    coordinator_type: str             # type(coordinator).__name__ or "none"
    dt: float
    step_count: int
    room_min: np.ndarray
    room_max: np.ndarray
    config_path: str | None = None   # absolute path to loaded JSON; None before first load
    drone_ids: list[str] = field(default_factory=list)  # ids in scenario order, for view/drone pickers before the first step


class SimulationBackend(ABC):
    """Abstract contract for simulation backends used by GUI widgets."""

    @abstractmethod
    def load_config(self, path: Path) -> SimState:
        """Load a scenario JSON and return the initial simulation state."""
        ...

    @abstractmethod
    def step(self) -> StepResult:
        """Advance one simulation tick and return the resulting state."""
        ...

    @abstractmethod
    def get_state(self) -> SimState:
        """Return current simulation metadata (does not advance the simulation)."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation to initial state from the last loaded config."""
        ...

    @abstractmethod
    def render_fpv(self, drone_id: str, size: tuple[int, int]) -> bytes | None:
        """Render what one drone's camera sees right now, as PNG bytes of exactly ``size`` pixels.

        Rendering lives behind the backend because it needs the actual ``Drone`` objects — their resolved
        display model and the obstacle list — none of which :class:`DroneState` carries (BACK-01: the GUI
        never touches the simulation).

        :param drone_id: Observing drone.
        :param size: ``(width, height)`` in pixels.
        :return: PNG bytes, or ``None`` when no drone with that id exists.
        """
        ...
