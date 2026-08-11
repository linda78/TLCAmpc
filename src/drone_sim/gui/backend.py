from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drone_sim.domain.drone_model import DroneModel


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
    # What the *scenario* says this drone looks like, resolved by the config layer — display data like the
    # colors above. The runtime override (set_drone_model_override) is deliberately not folded in here: it is
    # view state that outlives any single StepResult, and a result cached from before the override would
    # otherwise redraw with the model the user just replaced. None for backends that resolve no models.
    model: DroneModel | None = None


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
class PerceivedMarker:
    """One neighbor position as a camera *estimated* it — the display counterpart to the drone's true position.

    Read next to the drone it names: the marker sits where ``observer_id``'s camera believes ``observed_id``
    is, the sphere sits where it actually is, and the gap between them is the estimation error. Several
    observers seeing the same neighbor produce several markers for it.

    One estimate per ``(observer, observed)`` pair, taken from the perception mailbox: it holds the newest
    estimate until a newer one replaces it, so a neighbor that has left every field of view keeps its last
    marker rather than disappearing (see ``PerceptionMailbox.post``).
    """
    observer_id: str                 # the drone whose camera produced the estimate
    observed_id: str                 # the drone the estimate is about
    position: np.ndarray             # (3,) estimated world position
    sigma: float | None              # 1-sigma uncertainty in meters, None when the detector reports none


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
    perceived: list[PerceivedMarker] = field(default_factory=list)  # empty whenever the scenario runs without a camera


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
    def set_drone_model_override(self, model: DroneModel | None) -> None:
        """Draw *every* drone as ``model`` from now on, ignoring what the scenario configured. ``None`` restores the config.

        This is the runtime counterpart to ``drone_model``/``drone_model_path``: a way to try meshes out without
        writing a scenario file for each one. It is deliberately fleet-wide — while it is set the swarm is
        homogeneous — and deliberately temporary:

        * ``load_config`` drops it, because the scenario just said what its drones look like.
        * ``reset`` keeps it, because a reset repeats the run, not the choice of what to look at.

        Implementations must apply it to *all* views they serve, :meth:`render_fpv` included. An override that
        reaches only some of them is exactly what this method exists to prevent.

        :param model: Resolved model to draw every drone with, or ``None`` to fall back to each drone's own.
        """
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

    def close(self) -> None:
        """Release everything the backend started behind the scenes (threads, sockets). Idempotent.

        Called when the window closes. Deliberately concrete and empty rather than abstract: a backend
        that owns nothing but the simulation object has nothing to release, and should not be forced to
        say so.
        """
        return
