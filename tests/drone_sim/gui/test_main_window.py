"""Smoke tests for MainWindow — run headless via QT_QPA_PLATFORM=offscreen."""
import os

import numpy as np

os.environ["QT_API"] = "pyside6"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from drone_sim.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    """Single QApplication for all tests in this module."""
    existing = QApplication.instance()
    if existing:
        yield existing
    else:
        a = QApplication(sys.argv)
        yield a
        # Do NOT call a.quit() — other tests may reuse the app


@pytest.fixture
def window(app):
    """Fresh MainWindow for each test."""
    w = MainWindow()
    yield w
    w.close()


class TestMainWindowScaffold:
    def test_window_creates_without_crash(self, window):
        assert window is not None

    def test_canvas_attribute_exists(self, window):
        assert hasattr(window, "_canvas")

    def test_ax_is_3d(self, window):
        from mpl_toolkits.mplot3d import Axes3D
        assert isinstance(window._ax, Axes3D)

    def test_playing_starts_false(self, window):
        assert window._playing is False

    def test_interval_ms_default(self, window):
        assert window._interval_ms == 100

    def test_traces_starts_empty(self, window):
        assert window._traces == {}

    def test_step_label_initial_text(self, window):
        assert "Step: 0" in window._step_label.text()

    def test_speed_slider_range(self, window):
        assert window._speed_slider.minimum() == 10
        assert window._speed_slider.maximum() == 500

    def test_speed_slider_default_value(self, window):
        assert window._speed_slider.value() == 100


class TestSpeedSlot:
    def test_speed_changed_updates_interval_ms(self, window):
        window._speed_slider.setValue(250)
        assert window._interval_ms == 250

    def test_speed_changed_updates_label(self, window):
        window._speed_slider.setValue(300)
        assert "300" in window._speed_label.text()


class TestKeyboardShortcut:
    def test_spacebar_shortcut_is_registered(self, window):
        # QShortcut fires regardless of child focus; verify one is registered on the window.
        from PySide6.QtGui import QKeySequence, QShortcut
        shortcuts = window.findChildren(QShortcut)
        keys = [s.key() for s in shortcuts]
        assert QKeySequence(Qt.Key.Key_Space) in keys, "Spacebar QShortcut not found on MainWindow"

    def test_toggle_play_pause_starts_play_when_paused(self, window, monkeypatch):
        import drone_sim.gui.main_window as m
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))
        window._playing = False
        window._toggle_play_pause()
        assert window._playing is True
        assert len(fired) == 1

    def test_toggle_play_pause_pauses_when_playing(self, window):
        window._playing = True
        window._toggle_play_pause()
        assert window._playing is False


class TestPauseStopsPlayback:
    def test_pause_sets_playing_false(self, window):
        window._playing = True
        window._on_pause()
        assert window._playing is False

    def test_play_sets_playing_true(self, window, monkeypatch):
        # Monkeypatch QTimer.singleShot to avoid actual timer firing
        import drone_sim.gui.main_window as m
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))
        window._playing = False
        window._on_play()
        assert window._playing is True
        assert len(fired) == 1  # singleShot called once to start the loop

    def test_play_while_playing_is_noop(self, window, monkeypatch):
        import drone_sim.gui.main_window as m
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))
        window._playing = True
        window._on_play()
        assert len(fired) == 0  # already playing — no additional singleShot


class TestRunToCompletionFlag:
    def test_run_to_completion_starts_false(self, window):
        assert window._run_to_completion is False

    def test_on_run_to_completion_sets_flag(self, window, monkeypatch):
        import drone_sim.gui.main_window as m
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: None))
        window._on_run_to_completion()
        assert window._run_to_completion is True
        assert window._playing is True

    def test_on_pause_clears_run_to_completion_flag(self, window):
        window._run_to_completion = True
        window._playing = True
        window._on_pause()
        assert window._run_to_completion is False
        assert window._playing is False

    def test_max_run_steps_is_5000(self):
        from drone_sim.gui.main_window import _MAX_RUN_STEPS
        assert _MAX_RUN_STEPS == 5000


class TestCollisionIndicator:
    def test_collision_label_exists(self, window):
        assert hasattr(window, "_collision_label")

    def test_collision_label_default_text(self, window):
        assert window._collision_label.text() == "No Collisions"

    def test_update_collision_indicator_no_collisions(self, window):
        from drone_sim.gui.backend import StepResult, DroneState
        result = StepResult(
            drones=[],
            safety_radii=[],
            last_collisions=[],
            infeasible=False,
            infeasible_reason=None,
            step_count=1,
            t=0.1,
            all_reached=False,
            admm_iteration_count=None,
        )
        window._update_collision_indicator(result)
        assert window._collision_label.text() == "No Collisions"

    def test_update_collision_indicator_with_collisions(self, window):
        from drone_sim.gui.backend import StepResult
        result = StepResult(
            drones=[],
            safety_radii=[],
            last_collisions=[{"drone_a": "d1", "drone_b": "d2"}],
            infeasible=False,
            infeasible_reason=None,
            step_count=1,
            t=0.1,
            all_reached=False,
            admm_iteration_count=None,
        )
        window._update_collision_indicator(result)
        assert window._collision_label.text() == "COLLISION"


class TestAdmmIndicator:
    def test_admm_label_exists(self, window):
        assert hasattr(window, "_admm_label")

    def test_admm_label_hidden_by_default(self, window):
        # Use isHidden() — isVisible() depends on parent being shown (offscreen headless)
        assert window._admm_label.isHidden()

    def test_update_admm_indicator_none_keeps_hidden(self, window):
        from drone_sim.gui.backend import StepResult
        result = StepResult(
            drones=[],
            safety_radii=[],
            last_collisions=[],
            infeasible=False,
            infeasible_reason=None,
            step_count=1,
            t=0.1,
            all_reached=False,
            admm_iteration_count=None,
        )
        window._update_admm_indicator(result)
        assert window._admm_label.isHidden()

    def test_update_admm_indicator_int_shows_label(self, window):
        from drone_sim.gui.backend import StepResult
        result = StepResult(
            drones=[],
            safety_radii=[],
            last_collisions=[],
            infeasible=False,
            infeasible_reason=None,
            step_count=1,
            t=0.1,
            all_reached=False,
            admm_iteration_count=7,
        )
        window._update_admm_indicator(result)
        # Use not isHidden() — isVisible() depends on parent being shown (offscreen headless)
        assert not window._admm_label.isHidden()
        assert "7" in window._admm_label.text()


class TestInfoPanel:
    def test_info_path_label_exists(self, window):
        assert hasattr(window, "_info_path_label")

    def test_info_coord_label_exists(self, window):
        assert hasattr(window, "_info_coord_label")

    def test_info_dt_label_exists(self, window):
        assert hasattr(window, "_info_dt_label")

    def test_info_drones_label_exists(self, window):
        assert hasattr(window, "_info_drones_label")

    def test_info_obstacles_label_exists(self, window):
        assert hasattr(window, "_info_obstacles_label")

    def test_on_config_loaded_updates_labels(self, window):
        from drone_sim.gui.backend import SimState
        state = SimState(
            drone_count=3,
            obstacle_count=2,
            obstacles=[],
            coordinator_type="DistributedMPCCoordinator",
            dt=0.2,
            step_count=0,
            room_min=np.asarray([-10.0, -10.0, 0.0]),
            room_max=np.asarray([10.0, 10.0, 10.0]),
            config_path="/some/path/MyScenario.json",
        )
        window._on_config_loaded(state)
        assert window._info_path_label.text() == "MyScenario.json"
        assert "DistributedMPCCoordinator" in window._info_coord_label.text()
        assert "3" in window._info_drones_label.text()
        assert "2" in window._info_obstacles_label.text()

    def test_on_config_loaded_sets_tooltip_to_full_path(self, window):
        from drone_sim.gui.backend import SimState
        state = SimState(
            drone_count=1,
            obstacle_count=0,
            obstacles=[],
            coordinator_type="CentralMPCGlobalCoordinator",
            dt=0.1,
            step_count=0,
            room_min=np.asarray([-5.0, -5.0, 0.0]),
            room_max=np.asarray([5.0, 5.0, 5.0]),
            config_path="/full/absolute/path/config.json",
        )
        window._on_config_loaded(state)
        assert window._info_path_label.toolTip() == "/full/absolute/path/config.json"


class TestReplay:
    def test_replay_button_exists(self, window):
        assert window._btn_replay.text() == "Replay CSV"

    def test_tick_stops_on_end_of_data(self, window, monkeypatch):
        import drone_sim.gui.main_window as m
        from drone_sim.gui.backend import SimState, StepResult
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))

        class _LastFrame:
            def step(self):
                return StepResult(drones=[], safety_radii=[], last_collisions=[], infeasible=False, infeasible_reason=None, step_count=3, t=0.3,
                                  end_of_data=True)

            def get_state(self):
                return SimState(drone_count=0, obstacle_count=0, obstacles=[], coordinator_type="replay: run.csv", dt=0.1, step_count=3,
                                room_min=np.asarray([-1.0, -1.0, -1.0]), room_max=np.asarray([1.0, 1.0, 1.0]))

        window._backend = _LastFrame()
        window._playing = True
        window._tick()
        assert window._playing is False
        assert all(fn != window._tick for _, fn in fired)  # not rescheduled (the canvas may queue its own draw_idle)

    def test_open_replay_switches_to_replay_backend(self, window, monkeypatch, tmp_path):
        import json
        import drone_sim.gui.main_window as m
        from drone_sim.gui.replay_backend import ReplayBackend
        from tests.drone_sim.gui.test_replay_backend import CONFIG, _write_csv
        cfg = tmp_path / "scenario.json"
        cfg.write_text(json.dumps(CONFIG), encoding="utf-8")
        run = _write_csv(tmp_path / "run.csv")
        answers = iter([(str(run), "CSV Files (*.csv)"), (str(cfg), "JSON Files (*.json)")])
        monkeypatch.setattr(m.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: next(answers)))

        window._on_open_replay()

        assert isinstance(window._backend, ReplayBackend)
        assert "Step: 0" in window._step_label.text()
        assert window._info_coord_label.text() == "replay: run.csv"
        assert window._info_path_label.text() == "scenario.json"

    def test_open_scenario_after_replay_returns_to_direct_backend(self, window, monkeypatch, tmp_path):
        import json
        import drone_sim.gui.main_window as m
        from drone_sim.gui.direct_backend import DirectBackend
        from drone_sim.gui.replay_backend import ReplayBackend
        from tests.drone_sim.gui.test_replay_backend import CONFIG
        cfg = tmp_path / "scenario.json"
        cfg.write_text(json.dumps(CONFIG), encoding="utf-8")
        window._backend = ReplayBackend()
        monkeypatch.setattr(m.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(cfg), "JSON Files (*.json)")))

        window._on_open_file()

        assert isinstance(window._backend, DirectBackend)
        assert "Step: 1" in window._step_label.text()

    def test_cancelled_dialog_keeps_backend(self, window, monkeypatch):
        import drone_sim.gui.main_window as m
        before = window._backend
        monkeypatch.setattr(m.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))
        window._on_open_replay()
        assert window._backend is before


def _fake_state(room_min=(-10.0, -10.0, -10.0), room_max=(30.0, 10.0, 10.0)):
    from drone_sim.gui.backend import SimState
    return SimState(drone_count=0, obstacle_count=0, obstacles=[], coordinator_type="x", dt=0.1, step_count=0,
                    room_min=np.asarray(room_min), room_max=np.asarray(room_max))


def _fake_result():
    from drone_sim.gui.backend import StepResult
    return StepResult(drones=[], safety_radii=[], last_collisions=[], infeasible=False, infeasible_reason=None, step_count=0, t=0.0)


class _StaticBackend:
    """Backend stand-in for view tests: a room, no drones."""

    def get_state(self):
        return _fake_state()

    def step(self):
        return _fake_result()


def _open_replay(window, monkeypatch, tmp_path):
    """Load the three-frame test recording of test_replay_backend into the window through the (monkeypatched) dialogs."""
    import json
    import drone_sim.gui.main_window as m
    from tests.drone_sim.gui.test_replay_backend import CONFIG, _write_csv
    cfg = tmp_path / "scenario.json"
    cfg.write_text(json.dumps(CONFIG), encoding="utf-8")
    run = _write_csv(tmp_path / "run.csv")
    answers = iter([(str(run), "CSV Files (*.csv)"), (str(cfg), "JSON Files (*.json)")])
    monkeypatch.setattr(m.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: next(answers)))
    window._on_open_replay()
    return cfg


class TestViewKeys:
    def test_view_shortcuts_are_registered(self, window):
        from PySide6.QtGui import QKeySequence, QShortcut
        keys = [s.key() for s in window.findChildren(QShortcut)]
        for key in ("+", "-", "=", "w", "a", "s", "d", ",", ".", "Left", "Right", "Shift+Left", "Shift+Right", "Home", "End"):
            assert QKeySequence(key) in keys, f"no shortcut for {key}"

    def test_keys_hint_label_exists(self, window):
        assert "w/a/s/d" in window._keys_label.text()

    def test_zoom_keys_scale_and_clamp(self, window):
        import drone_sim.gui.main_window as m
        window._zoom_by(m._ZOOM_STEP)
        assert window._zoom == pytest.approx(0.85)
        window._zoom_by(1.0 / m._ZOOM_STEP)
        assert window._zoom == pytest.approx(1.0)
        for _ in range(50):
            window._zoom_by(m._ZOOM_STEP)
        assert window._zoom == pytest.approx(0.2)
        for _ in range(50):
            window._zoom_by(1.0 / m._ZOOM_STEP)
        assert window._zoom == pytest.approx(5.0)

    def test_pan_keys_shift_the_axes_within_the_figure(self, window):
        import drone_sim.gui.main_window as m
        window._backend = _StaticBackend()
        window._last_result = _fake_result()
        window._pan_by(m._PAN_FRACTION, 0.0)  # d: picture to the right
        window._pan_by(0.0, -m._PAN_FRACTION)  # s: picture down
        np.testing.assert_allclose(window._pan, [m._PAN_FRACTION, -m._PAN_FRACTION])
        pos = window._ax.get_position(original=True)
        assert pos.x0 == pytest.approx(m._PAN_FRACTION) and pos.y0 == pytest.approx(-m._PAN_FRACTION)
        assert pos.width == pytest.approx(1.0) and pos.height == pytest.approx(1.0)
        window._pan_by(-m._PAN_FRACTION, m._PAN_FRACTION)
        np.testing.assert_allclose(window._pan, [0.0, 0.0])

    def test_zoom_keeps_the_room_as_axis_limits(self, window):
        window._backend = _StaticBackend()
        window._last_result = _fake_result()
        window._zoom_by(0.5)
        assert window._ax.get_xlim3d() == pytest.approx((-10.0, 30.0))  # the grey panes stay on the room walls
        assert window._ax.get_ylim3d() == pytest.approx((-10.0, 10.0))
        assert window._ax.get_zlim3d() == pytest.approx((-10.0, 10.0))

    def test_zoom_is_a_pure_magnification(self, window):
        """Box enlargement and focal length scale together: every point projects to twice its coordinates at zoom 0.5."""
        from mpl_toolkits.mplot3d import proj3d
        window._backend = _StaticBackend()
        window._last_result = _fake_result()
        window._ax.view_init(elev=30, azim=-60)
        xs = np.array([-10.0, 30.0, 5.0, 12.0, -3.0])
        ys = np.array([-10.0, 10.0, 2.0, -7.0, 9.0])
        zs = np.array([-10.0, 10.0, -4.0, 8.0, 0.0])
        window._apply_view()
        base = np.array(proj3d.proj_transform(xs, ys, zs, window._ax.get_proj())[:2])
        window._zoom_by(0.5)
        zoomed = np.array(proj3d.proj_transform(xs, ys, zs, window._ax.get_proj())[:2])
        np.testing.assert_allclose(zoomed, 2.0 * base, atol=1e-9)

    def test_right_drag_zooms_and_middle_drag_pans(self, window):
        import drone_sim.gui.main_window as m
        window._backend = _StaticBackend()
        window._last_result = _fake_result()

        class _Evt:
            def __init__(self, button, x, y):
                self.button, self.x, self.y = button, x, y

        window._on_button_press(_Evt(3, 100.0, 100.0))
        window._on_mouse_move(_Evt(3, 100.0, 100.0 + m._DRAG_ZOOM_PIXELS))  # upwards: twice as close
        assert window._zoom == pytest.approx(0.5)
        window._on_mouse_move(_Evt(3, 100.0, 100.0))  # back down
        assert window._zoom == pytest.approx(1.0)
        window._on_button_release(_Evt(3, 100.0, 100.0))
        assert window._drag is None
        bbox = window._canvas.figure.bbox
        window._on_button_press(_Evt(2, 0.0, 0.0))
        window._on_mouse_move(_Evt(2, bbox.width / 4, bbox.height / 2))
        np.testing.assert_allclose(window._pan, [0.25, 0.5])
        window._on_button_release(_Evt(2, 0.0, 0.0))

    def test_left_button_does_not_start_a_drag(self, window):
        class _Evt:
            button, x, y = 1, 10.0, 10.0

        window._on_button_press(_Evt())
        assert window._drag is None

    def test_mplot3d_pan_and_zoom_buttons_are_off(self, window):
        assert window._ax._rotate_btn == [1]
        assert window._ax._pan_btn == [] and window._ax._zoom_btn == []

    def test_scroll_uses_the_same_zoom_steps(self, window):
        import drone_sim.gui.main_window as m

        class _Evt:
            button = "up"

        window._on_scroll(_Evt())
        assert window._zoom == pytest.approx(m._ZOOM_STEP)

    def test_load_resets_zoom_and_pan(self, window, monkeypatch, tmp_path):
        window._zoom = 0.3
        window._pan = np.ones(2)
        _open_replay(window, monkeypatch, tmp_path)
        assert window._zoom == 1.0
        np.testing.assert_allclose(window._pan, [0.0, 0.0])

    def test_reset_clears_pan(self, window, monkeypatch, tmp_path):
        _open_replay(window, monkeypatch, tmp_path)
        window._pan = np.ones(2)
        window._on_reset()
        np.testing.assert_allclose(window._pan, [0.0, 0.0])

    def test_screenshot_snapshot_records_the_pan(self, window, monkeypatch, tmp_path):
        import json
        _open_replay(window, monkeypatch, tmp_path)
        window._pan = np.asarray([0.1, 0.2])
        monkeypatch.chdir(tmp_path)
        window._on_screenshot()
        snapshot = json.loads(next((tmp_path / "screenshots").glob("*.json")).read_text())
        assert snapshot["view"]["pan"] == [0.1, 0.2]


class TestReplayControls:
    def test_replay_box_hidden_until_a_replay_is_loaded(self, window, monkeypatch, tmp_path):
        assert window._replay_box.isHidden()
        _open_replay(window, monkeypatch, tmp_path)
        assert not window._replay_box.isHidden()
        assert window._frame_slider.maximum() == 2
        assert window._frame_slider.value() == 0
        assert window._frame_label.text() == "frame 0 / 2"

    def test_open_scenario_hides_the_replay_box(self, window, monkeypatch, tmp_path):
        import drone_sim.gui.main_window as m
        cfg = _open_replay(window, monkeypatch, tmp_path)
        monkeypatch.setattr(m.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(cfg), "JSON Files (*.json)")))
        window._on_open_file()
        assert window._replay_box.isHidden()

    def test_slider_seeks_and_rebuilds_the_traces(self, window, monkeypatch, tmp_path):
        _open_replay(window, monkeypatch, tmp_path)
        window._frame_slider.setValue(2)
        assert "Step: 2" in window._step_label.text()
        assert window._backend.frame_index == 2
        assert window._traces["d1"] == [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        assert window._frame_label.text() == "frame 2 / 2"
        window._frame_slider.setValue(1)
        assert "Step: 1" in window._step_label.text()
        assert len(window._traces["d1"]) == 2

    def test_frame_keys(self, window, monkeypatch, tmp_path):
        _open_replay(window, monkeypatch, tmp_path)
        window._step_frames(1)
        assert window._backend.frame_index == 1 and window._frame_slider.value() == 1
        window._step_frames(10)
        assert window._backend.frame_index == 2  # clamped
        window._step_frames(-10)
        assert window._backend.frame_index == 0
        window._seek_end()
        assert window._backend.frame_index == 2
        window._seek(0)
        assert window._backend.frame_index == 0

    def test_frame_keys_without_replay_are_ignored(self, window):
        window._step_frames(1)
        window._seek_end()
        window._seek(0)
        assert "Step: 0" in window._step_label.text()

    def test_faster_and_slower_cycle_the_factor(self, window):
        import drone_sim.gui.main_window as m
        assert window._speed_factor_label.text() == "×1"
        window._on_faster()
        window._on_faster()
        assert window._speed_factor_label.text() == "×4"
        assert window._frames_per_tick() == 4
        for _ in range(10):
            window._on_faster()
        assert window._replay_speed == m._REPLAY_SPEEDS[-1]
        for _ in range(20):
            window._on_slower()
        assert window._replay_speed == m._REPLAY_SPEEDS[0]
        assert window._speed_factor_label.text() == "×0.125"
        assert window._frames_per_tick() == 1

    def test_slow_factor_stretches_the_tick_interval_only_in_replay(self, window, monkeypatch, tmp_path):
        window._on_slower()  # ×0.5
        assert window._tick_interval_ms() == 100  # DirectBackend: the factor does not apply
        _open_replay(window, monkeypatch, tmp_path)
        assert window._tick_interval_ms() == 200
        window._on_faster()
        window._on_faster()  # ×2
        assert window._tick_interval_ms() == 100

    def test_tick_advances_by_the_factor_and_stops_at_the_end(self, window, monkeypatch, tmp_path):
        import drone_sim.gui.main_window as m
        _open_replay(window, monkeypatch, tmp_path)
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))
        window._on_faster()  # ×2
        window._playing = True
        window._tick()
        assert window._backend.frame_index == 2
        assert len(window._traces["d1"]) == 3  # the skipped frame is in the trace
        assert window._playing is False  # last frame reached
        assert all(fn != window._tick for _, fn in fired)

    def test_tick_at_factor_one_steps_one_frame(self, window, monkeypatch, tmp_path):
        import drone_sim.gui.main_window as m
        _open_replay(window, monkeypatch, tmp_path)
        fired = []
        monkeypatch.setattr(m.QTimer, "singleShot", staticmethod(lambda ms, fn: fired.append((ms, fn))))
        window._playing = True
        window._tick()
        assert window._backend.frame_index == 1
        assert window._frame_slider.value() == 1
        assert window._playing is True
        assert [ms for ms, fn in fired if fn == window._tick] == [100]
