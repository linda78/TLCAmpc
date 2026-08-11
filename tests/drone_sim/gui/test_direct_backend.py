"""Pure Python unit tests for DirectBackend — no Qt dependency."""
from __future__ import annotations

import json
import threading
from pathlib import Path
import numpy as np

import pytest

from drone_sim.domain.drone_model import resolve_drone_model
from drone_sim.gui.backend import DroneState, SimState, StepResult
from drone_sim.gui.direct_backend import DirectBackend

MINIMAL_CONFIG = {
    "dt": 0.1,
    "physics": {"type": "linear_kinematics", "v_max": 2.0, "u_min": -1.0, "u_max": 1.0},
    "controller": {"type": "mpc_agent"},
    "coordinator": {"type": "mpc_central"},
    "drones": [
        {
            "drone_id": "d1",
            "start": [0.0, 0.0, 5.0],
            "target": [5.0, 5.0, 5.0],
        }
    ],
    "room": {"min": [-10.0, -10.0, 0.0], "max": [10.0, 10.0, 10.0]},
}


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "test_config.json"
    p.write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    return p


@pytest.fixture
def loaded_backend(config_file: Path) -> DirectBackend:
    backend = DirectBackend()
    backend.load_config(config_file)
    return backend


# --- load_config ---

def test_load_config_returns_sim_state(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert isinstance(state, SimState)


def test_load_config_drone_count(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.drone_count == 1


def test_load_config_step_count_zero(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.step_count == 0


def test_load_config_obstacle_count(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.obstacle_count == 0


def test_load_config_dt(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.dt == pytest.approx(0.1)


def test_load_config_coordinator_type_is_string(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert isinstance(state.coordinator_type, str)
    assert len(state.coordinator_type) > 0


def test_load_config_room_bounds_are_lists(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert isinstance(state.room_min, np.ndarray)
    assert isinstance(state.room_max, np.ndarray)
    assert len(state.room_min) == 3
    assert len(state.room_max) == 3


# --- step ---

def test_step_returns_step_result(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result, StepResult)


def test_step_increments_step_count(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert result.step_count == 1


def test_step_drone_count(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert len(result.drones) == 1


def test_step_drone_state_types(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    d = result.drones[0]
    assert isinstance(d, DroneState)
    assert isinstance(d.drone_id, str)
    assert isinstance(d.position, np.ndarray)
    assert isinstance(d.velocity, np.ndarray)
    assert len(d.position) == 3
    assert len(d.velocity) == 3


def test_step_position_values_are_numpy(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    d = result.drones[0]
    assert all(isinstance(v, (float, np.floating)) for v in d.position), "position elements must be float"
    assert all(isinstance(v, (float, np.floating)) for v in d.velocity), "velocity elements must be float"


def test_step_safety_radii_length(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert len(result.safety_radii) == 1
    assert isinstance(result.safety_radii[0], float)


def test_step_infeasible_is_bool(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result.infeasible, bool)


def test_step_last_collisions_is_list(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result.last_collisions, list)


def test_step_t_is_float(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result.t, float)


# --- get_state ---

def test_get_state_returns_sim_state(loaded_backend: DirectBackend) -> None:
    state = loaded_backend.get_state()
    assert isinstance(state, SimState)


def test_get_state_does_not_advance_step_count(loaded_backend: DirectBackend) -> None:
    loaded_backend.get_state()
    loaded_backend.get_state()
    state = loaded_backend.get_state()
    assert state.step_count == 0


# --- reset ---

def test_reset_after_steps_returns_step_count_to_zero(loaded_backend: DirectBackend) -> None:
    loaded_backend.step()
    loaded_backend.step()
    loaded_backend.reset()
    state = loaded_backend.get_state()
    assert state.step_count == 0


def test_reset_does_not_require_file_path(loaded_backend: DirectBackend, tmp_path: Path) -> None:
    """reset() must use cached config — original file can be deleted."""
    loaded_backend.step()
    # Delete the original file to prove reset does not re-read disk
    for f in tmp_path.iterdir():
        f.unlink()
    loaded_backend.reset()  # must not raise
    state = loaded_backend.get_state()
    assert state.step_count == 0


# --- all_reached ---

def test_step_all_reached_is_bool(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert isinstance(result.all_reached, bool)


def test_step_all_reached_false_on_first_step(loaded_backend: DirectBackend) -> None:
    """Drone starts far from destination — should not be reached on step 1."""
    result = loaded_backend.step()
    assert result.all_reached is False


# --- admm_iteration_count ---

def test_step_admm_iteration_count_none_for_central_coordinator(loaded_backend: DirectBackend) -> None:
    """MINIMAL_CONFIG uses mpc_central coordinator — not ADMM — so admm_iteration_count must be None."""
    result = loaded_backend.step()
    assert result.admm_iteration_count is None


def test_step_admm_iteration_count_type_is_int_or_none(loaded_backend: DirectBackend) -> None:
    result = loaded_backend.step()
    assert result.admm_iteration_count is None or isinstance(result.admm_iteration_count, int)


# --- SimState.config_path ---

def test_load_config_sets_config_path(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.config_path is not None
    assert isinstance(state.config_path, str)


def test_load_config_path_ends_with_filename(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.config_path.endswith("test_config.json")


def test_config_path_none_before_load() -> None:
    """get_state() before load_config() should raise RuntimeError, but the field default is None."""
    # We only test the dataclass default — a fresh SimState() would have config_path=None.
    from drone_sim.gui.backend import SimState
    s = SimState(
        drone_count=0,
        obstacle_count=0,
        obstacles=[],
        coordinator_type="none",
        dt=0.1,
        step_count=0,
        room_min=np.zeros(3),
        room_max=np.ones(3)
    )
    assert s.config_path is None
    assert s.drone_ids == []


# --- SimState.drone_ids ---

TWO_DRONE_CONFIG = {
    **MINIMAL_CONFIG,
    "drones": [
        {"drone_id": "d1", "start": [0.0, 0.0, 5.0], "target": [8.0, 0.0, 5.0]},
        {"drone_id": "d2", "start": [8.0, 0.0, 5.0], "target": [0.0, 0.0, 5.0]},
    ],
}


@pytest.fixture
def two_drone_backend(tmp_path: Path) -> DirectBackend:
    p = tmp_path / "two_drones.json"
    p.write_text(json.dumps(TWO_DRONE_CONFIG), encoding="utf-8")
    backend = DirectBackend()
    backend.load_config(p)
    return backend


def test_load_config_reports_drone_ids(two_drone_backend: DirectBackend) -> None:
    state = two_drone_backend.get_state()
    assert state.drone_ids == ["d1", "d2"]


def test_drone_ids_length_matches_drone_count(two_drone_backend: DirectBackend) -> None:
    """The GUI fills its view picker from drone_ids before the first step, where drone_count is all it had."""
    state = two_drone_backend.get_state()
    assert len(state.drone_ids) == state.drone_count


def test_drone_ids_available_before_first_step(config_file: Path) -> None:
    backend = DirectBackend()
    state = backend.load_config(config_file)
    assert state.drone_ids == ["d1"]


# --- render_fpv ---

def test_render_fpv_returns_png_bytes(two_drone_backend: DirectBackend) -> None:
    png = two_drone_backend.render_fpv("d1", (160, 120))
    assert isinstance(png, bytes)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_fpv_unknown_drone_returns_none(two_drone_backend: DirectBackend) -> None:
    assert two_drone_backend.render_fpv("nope", (160, 120)) is None


def test_render_fpv_works_without_camera_enabled(two_drone_backend: DirectBackend) -> None:
    """camera_enabled gates the perception pipeline, not drawing a picture on request — TWO_DRONE_CONFIG leaves it off."""
    assert two_drone_backend._cfg.camera_enabled is False
    assert two_drone_backend.render_fpv("d1", (160, 120)) is not None


def test_render_fpv_after_steps(two_drone_backend: DirectBackend) -> None:
    two_drone_backend.step()
    two_drone_backend.step()
    assert two_drone_backend.render_fpv("d2", (160, 120)) is not None


def test_render_fpv_after_reset(two_drone_backend: DirectBackend) -> None:
    """reset() replaces the Simulator — the camera must follow, or capture() reads drones from the old run."""
    two_drone_backend.step()
    two_drone_backend.reset()
    assert two_drone_backend.render_fpv("d1", (160, 120)) is not None


def test_render_fpv_before_load_raises() -> None:
    backend = DirectBackend()
    with pytest.raises(RuntimeError):
        backend.render_fpv("d1", (160, 120))


# --- set_drone_model_override ---
#
# The runtime "OBJ Model" button. The bug it fixes: the button used to reach only the external 3D view,
# so the drone view kept drawing spheres while the overview drew a mesh.

OBJ_MODEL = resolve_drone_model("obj", "drone_costum_0_0_5.obj")


def _captured_models(backend: DirectBackend, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Render one frame with the PNG writer stubbed out, and return the ``models`` mapping it was handed."""
    seen: dict = {}

    def fake_render(view, obstacles, *, models=None, size=(320, 240)):
        seen.update(models or {})
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr("drone_sim.gui.direct_backend.render_fpv_png", fake_render)
    backend.render_fpv("d1", (160, 120))
    return seen


def test_model_override_defaults_to_none(two_drone_backend: DirectBackend) -> None:
    assert two_drone_backend._model_override is None


def test_without_override_fpv_uses_configured_models(two_drone_backend: DirectBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    """TWO_DRONE_CONFIG sets no drone_model, so every drone resolves to a sphere — and that is what gets drawn."""
    models = _captured_models(two_drone_backend, monkeypatch)
    assert set(models) == {"d1", "d2"}
    assert all(m.kind == "sphere" for m in models.values())


def test_override_applies_to_every_drone(two_drone_backend: DirectBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fleet-wide by design: pressing the button once makes the whole swarm homogeneous."""
    two_drone_backend.set_drone_model_override(OBJ_MODEL)
    models = _captured_models(two_drone_backend, monkeypatch)
    assert set(models) == {"d1", "d2"}
    assert all(m is OBJ_MODEL for m in models.values())


def test_clearing_override_restores_configured_models(two_drone_backend: DirectBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    two_drone_backend.set_drone_model_override(OBJ_MODEL)
    two_drone_backend.set_drone_model_override(None)
    models = _captured_models(two_drone_backend, monkeypatch)
    assert all(m.kind == "sphere" for m in models.values())


def test_override_changes_the_rendered_image(two_drone_backend: DirectBackend) -> None:
    """End-to-end through the real renderer: a mesh is not the same picture as a sphere."""
    sphere_png = two_drone_backend.render_fpv("d1", (160, 120))
    two_drone_backend.set_drone_model_override(OBJ_MODEL)
    obj_png = two_drone_backend.render_fpv("d1", (160, 120))
    assert obj_png != sphere_png


def test_load_config_drops_override(two_drone_backend: DirectBackend, config_file: Path) -> None:
    """A newly loaded scenario has just said what its drones look like — the previous pick no longer applies."""
    two_drone_backend.set_drone_model_override(OBJ_MODEL)
    two_drone_backend.load_config(config_file)
    assert two_drone_backend._model_override is None


def test_reset_keeps_override(two_drone_backend: DirectBackend, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reset repeats the run, not the choice of what to look at."""
    two_drone_backend.set_drone_model_override(OBJ_MODEL)
    two_drone_backend.step()
    two_drone_backend.reset()
    assert two_drone_backend._model_override is OBJ_MODEL
    assert all(m is OBJ_MODEL for m in _captured_models(two_drone_backend, monkeypatch).values())


# --- perception lifecycle (risk R2: one worker thread per reload) ---

CAMERA_CONFIG = {
    **TWO_DRONE_CONFIG,
    "camera_enabled": True,
    "camera_async": True,
}


@pytest.fixture
def camera_config_file(tmp_path: Path) -> Path:
    p = tmp_path / "camera.json"
    p.write_text(json.dumps(CAMERA_CONFIG), encoding="utf-8")
    return p


def _worker_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "PerceptionWorker"]


def test_load_config_starts_one_worker(camera_config_file: Path) -> None:
    backend = DirectBackend()
    try:
        backend.load_config(camera_config_file)
        assert len(_worker_threads()) == 1
    finally:
        backend.close()


def test_reloading_does_not_accumulate_workers(camera_config_file: Path) -> None:
    """R2: the daemon flag only guarantees the process can exit, not that the old worker stops working."""
    backend = DirectBackend()
    try:
        for _ in range(3):
            backend.load_config(camera_config_file)

        assert len(_worker_threads()) == 1
    finally:
        backend.close()


def test_reset_does_not_accumulate_workers(camera_config_file: Path) -> None:
    backend = DirectBackend()
    try:
        backend.load_config(camera_config_file)
        backend.step()
        for _ in range(3):
            backend.reset()

        assert len(_worker_threads()) == 1
    finally:
        backend.close()


def test_reset_leaves_a_working_camera(camera_config_file: Path) -> None:
    """Closing the outgoing simulation must not cost the incoming one its perception."""
    backend = DirectBackend()
    try:
        backend.load_config(camera_config_file)
        backend.reset()
        backend.step()

        worker = backend._sim._perception_worker
        assert worker is not None and worker.is_running is True
    finally:
        backend.close()


def test_close_stops_the_worker(camera_config_file: Path) -> None:
    backend = DirectBackend()
    backend.load_config(camera_config_file)

    backend.close()

    assert _worker_threads() == []


def test_close_is_idempotent_with_a_camera(camera_config_file: Path) -> None:
    backend = DirectBackend()
    backend.load_config(camera_config_file)

    backend.close()
    backend.close()  # must not raise

    assert _worker_threads() == []


def test_close_without_a_camera_scenario(config_file: Path) -> None:
    backend = DirectBackend()
    backend.load_config(config_file)

    backend.close()  # must not raise

    assert backend._sim is not None  # the state stays readable after the window closes
