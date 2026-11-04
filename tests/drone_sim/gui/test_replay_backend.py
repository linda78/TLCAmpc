"""Pure Python unit tests for ReplayBackend — no Qt dependency."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from drone_sim.gui.backend import SimState, StepResult
from drone_sim.gui.replay_backend import REQUIRED_COLUMNS, ReplayBackend

U_MAX = 3.0
CONFIG = {
    "dt": 0.1,
    "physics": {"type": "linear_kinematics", "params": {"v_max": 2.5, "u_min": [-U_MAX] * 3, "u_max": [U_MAX] * 3}},
    "controller": {"type": "mpc_agent"},
    "coordinator": {"type": "mpc_central"},
    "obstacles": [{"center": [20.0, 0.0, 0.0], "half_extents": [1.0, 1.0, 1.0]}],
    "drones": [
        {"drone_id": "d1", "start": [0.0, 0.0, 0.0], "target": [4.0, 0.0, 0.0], "safety_zone": 1.0, "alpha": 0.5, "drone_color": "#00FFFF"},
        {"drone_id": "d2", "start": [0.0, 5.0, 0.0], "target": [4.0, 5.0, 0.0], "safety_zone": 1.52},
    ],
    "room": {"min": [-10.0, -10.0, -10.0], "max": [30.0, 10.0, 10.0]},
}
# three frames; d1 flies to its target, d2 stays; last frame has both at their targets
FRAMES = [
    (0.0, {"d1": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), "d2": ([0.0, 5.0, 0.0], [0.0, 0.0, 0.0])}),
    (0.1, {"d1": ([2.0, 0.0, 0.0], [2.0, 0.0, 0.0]), "d2": ([0.0, 5.0, 0.0], [0.0, 0.0, 0.0])}),
    (0.2, {"d1": ([4.0, 0.0, 0.0], [0.0, 0.0, 0.0]), "d2": ([4.0, 5.0, 0.0], [0.0, 0.0, 0.0])}),
]


def _write_csv(path: Path, frames=FRAMES, columns=("step", "t", "drone_id", "x", "y", "z", "vx", "vy", "vz", "speed"), shuffle: bool = False) -> Path:
    rows = []
    for i, (t, states) in enumerate(frames):
        for drone_id, (p, v) in states.items():
            rows.append({"step": i, "t": t, "drone_id": drone_id, "x": p[0], "y": p[1], "z": p[2], "vx": v[0], "vy": v[1], "vz": v[2],
                         "speed": float(np.linalg.norm(v))})
    if shuffle:
        rows = rows[::-1]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "scenario.json"
    p.write_text(json.dumps(CONFIG), encoding="utf-8")
    return p


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    return _write_csv(tmp_path / "run.csv")


@pytest.fixture
def backend(config_file: Path, csv_file: Path) -> ReplayBackend:
    b = ReplayBackend()
    b.load_config(config_file)
    b.load_csv(csv_file)
    return b


# --- loading ---

def test_load_config_returns_sim_state_without_frames(config_file: Path) -> None:
    b = ReplayBackend()
    state = b.load_config(config_file)
    assert isinstance(state, SimState)
    assert state.drone_count == 2 and state.obstacle_count == 1
    assert b.frame_count == 0
    assert "replay" in state.coordinator_type


def test_load_csv_requires_config(csv_file: Path) -> None:
    with pytest.raises(RuntimeError):
        ReplayBackend().load_csv(csv_file)


def test_step_requires_csv(config_file: Path) -> None:
    b = ReplayBackend()
    b.load_config(config_file)
    with pytest.raises(RuntimeError):
        b.step()


def test_load_csv_groups_rows_into_frames(backend: ReplayBackend) -> None:
    assert backend.frame_count == 3
    assert "run.csv" in backend.get_state().coordinator_type


def test_missing_columns_raise_value_error(config_file: Path, tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "no_velocity.csv", columns=("t", "drone_id", "x", "y", "z"))
    b = ReplayBackend()
    b.load_config(config_file)
    with pytest.raises(ValueError, match="vx"):
        b.load_csv(path)


def test_required_columns_constant() -> None:
    assert REQUIRED_COLUMNS == ("drone_id", "t", "x", "y", "z", "vx", "vy", "vz")


def test_unknown_drone_raises_value_error(config_file: Path, tmp_path: Path) -> None:
    frames = [(0.0, {"d1": ([0.0] * 3, [0.0] * 3), "d2": ([0.0, 5.0, 0.0], [0.0] * 3), "ghost": ([1.0] * 3, [0.0] * 3)})]
    path = _write_csv(tmp_path / "ghost.csv", frames=frames)
    b = ReplayBackend()
    b.load_config(config_file)
    with pytest.raises(ValueError, match="ghost"):
        b.load_csv(path)


def test_frame_without_every_drone_raises_value_error(config_file: Path, tmp_path: Path) -> None:
    frames = [(0.0, {"d1": ([0.0] * 3, [0.0] * 3), "d2": ([0.0, 5.0, 0.0], [0.0] * 3)}), (0.1, {"d1": ([1.0, 0.0, 0.0], [0.0] * 3)})]
    path = _write_csv(tmp_path / "gap.csv", frames=frames)
    b = ReplayBackend()
    b.load_config(config_file)
    with pytest.raises(ValueError, match="d2"):
        b.load_csv(path)


def test_rows_in_any_order_and_extra_columns(config_file: Path, tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "shuffled.csv", shuffle=True)
    b = ReplayBackend()
    b.load_config(config_file)
    b.load_csv(path)
    first = b.step()
    assert first.t == pytest.approx(0.0)
    assert first.step_count == 0


# --- stepping ---

def test_first_step_is_frame_zero(backend: ReplayBackend) -> None:
    result = backend.step()
    assert isinstance(result, StepResult)
    assert result.step_count == 0
    assert result.t == pytest.approx(0.0)
    assert result.end_of_data is False
    assert result.infeasible is False
    by_id = {d.drone_id: d for d in result.drones}
    np.testing.assert_allclose(by_id["d1"].position, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(by_id["d2"].position, [0.0, 5.0, 0.0])


def test_steps_follow_the_frames_and_flag_the_last(backend: ReplayBackend) -> None:
    backend.step()
    second = backend.step()
    third = backend.step()
    assert second.step_count == 1 and second.t == pytest.approx(0.1) and second.end_of_data is False
    np.testing.assert_allclose({d.drone_id: d for d in second.drones}["d1"].position, [2.0, 0.0, 0.0])
    np.testing.assert_allclose({d.drone_id: d for d in second.drones}["d1"].velocity, [2.0, 0.0, 0.0])
    assert third.step_count == 2 and third.t == pytest.approx(0.2) and third.end_of_data is True


def test_step_after_the_end_repeats_the_last_frame(backend: ReplayBackend) -> None:
    for _ in range(3):
        backend.step()
    again = backend.step()
    assert again.step_count == 2 and again.end_of_data is True
    np.testing.assert_allclose({d.drone_id: d for d in again.drones}["d1"].position, [4.0, 0.0, 0.0])


def test_adaptive_radius_from_recorded_velocity(backend: ReplayBackend) -> None:
    backend.step()
    second = backend.step()
    by_id = {d.drone_id: d for d in second.drones}
    expected = 1.0 + 0.5 * 2.0 ** 2 / (2.0 * U_MAX)
    assert by_id["d1"].adaptive_safety_radius == pytest.approx(expected)
    assert by_id["d2"].adaptive_safety_radius is None
    assert second.safety_radii == pytest.approx([expected, 1.52])


def test_all_reached_on_the_final_frame(backend: ReplayBackend) -> None:
    results = [backend.step() for _ in range(3)]
    assert [r.all_reached for r in results] == [False, False, True]


def test_collisions_from_recorded_positions(config_file: Path, tmp_path: Path) -> None:
    frames = [(0.0, {"d1": ([0.0, 0.0, 0.0], [0.0] * 3), "d2": ([0.0, 5.0, 0.0], [0.0] * 3)}),
              (0.1, {"d1": ([0.0, 4.0, 0.0], [0.0] * 3), "d2": ([0.0, 5.0, 0.0], [0.0] * 3)})]  # 1 m apart, spheres 1.0 + 1.52
    b = ReplayBackend()
    b.load_config(config_file)
    b.load_csv(_write_csv(tmp_path / "hit.csv", frames=frames))
    assert b.step().last_collisions == []
    assert any(e["kind"] == "drone_drone" for e in b.step().last_collisions)


# --- state and reset ---

def test_get_state_does_not_advance(backend: ReplayBackend) -> None:
    backend.step()
    backend.step()
    assert backend.get_state().step_count == 1
    assert backend.step().step_count == 2


def test_get_state_carries_config_path_and_room(backend: ReplayBackend, config_file: Path) -> None:
    state = backend.get_state()
    assert state.config_path == str(config_file)
    assert list(state.room_min) == [-10.0, -10.0, -10.0]


def test_reset_returns_to_the_first_frame(backend: ReplayBackend) -> None:
    for _ in range(3):
        backend.step()
    backend.reset()
    first = backend.step()
    assert first.step_count == 0 and first.t == pytest.approx(0.0) and first.end_of_data is False
    np.testing.assert_allclose({d.drone_id: d for d in first.drones}["d1"].position, [0.0, 0.0, 0.0])


def test_load_config_again_drops_the_frames(backend: ReplayBackend, config_file: Path) -> None:
    backend.load_config(config_file)
    assert backend.frame_count == 0
    with pytest.raises(RuntimeError):
        backend.step()


# --- seeking ---

def test_seek_jumps_to_the_frame_and_clamps(backend: ReplayBackend) -> None:
    mid = backend.seek(1)
    assert mid.step_count == 1 and mid.t == pytest.approx(0.1) and mid.end_of_data is False
    assert backend.frame_index == 1
    np.testing.assert_allclose({d.drone_id: d for d in mid.drones}["d1"].position, [2.0, 0.0, 0.0])
    last = backend.seek(99)
    assert last.step_count == 2 and last.end_of_data is True
    first = backend.seek(-5)
    assert first.step_count == 0 and backend.frame_index == 0


def test_step_continues_after_seek(backend: ReplayBackend) -> None:
    backend.seek(0)
    assert backend.step().step_count == 1
    backend.seek(2)
    assert backend.step().step_count == 2  # stays on the last frame


def test_seek_requires_csv(config_file: Path) -> None:
    b = ReplayBackend()
    b.load_config(config_file)
    with pytest.raises(RuntimeError):
        b.seek(0)


def test_frame_index_before_first_step_is_zero(backend: ReplayBackend) -> None:
    assert backend.frame_index == 0


def test_positions_up_to_gives_the_recorded_path(backend: ReplayBackend) -> None:
    backend.seek(1)
    traces = backend.positions_up_to()
    assert traces["d1"] == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    assert traces["d2"] == [[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]
    assert backend.positions_up_to(0)["d1"] == [[0.0, 0.0, 0.0]]
    assert len(backend.positions_up_to(10)["d1"]) == 3  # clamped to the last frame
