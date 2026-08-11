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


class TestFpvView:
    """The 'External view <-> View from drone X' switch: combo, stacked widget, and what each side must not disturb."""

    FPV_CONFIG = {
        "dt": 0.1,
        "physics": {"type": "linear_kinematics", "v_max": 2.0, "u_min": -1.0, "u_max": 1.0},
        "controller": {"type": "mpc_agent"},
        "coordinator": {"type": "mpc_central"},
        "camera_fov_deg": 120.0,
        "camera_range": 15.0,
        "drones": [
            {"drone_id": "d1", "start": [0.0, 0.0, 5.0], "target": [8.0, 0.0, 5.0]},
            {"drone_id": "d2", "start": [8.0, 0.0, 5.0], "target": [0.0, 0.0, 5.0]},
        ],
        "room": {"min": [-10.0, -10.0, 0.0], "max": [10.0, 10.0, 10.0]},
    }

    @pytest.fixture
    def loaded_window(self, window, tmp_path):
        """MainWindow with a two-drone scenario loaded — mirrors _on_open_file without the file dialog."""
        import json
        from pathlib import Path

        path = Path(tmp_path) / "fpv_scenario.json"
        path.write_text(json.dumps(self.FPV_CONFIG), encoding="utf-8")
        window._backend.load_config(path)
        result = window._backend.step()
        window._redraw(result)
        window._on_config_loaded(window._backend.get_state())
        return window

    def test_combo_starts_with_external_view_only(self, window):
        assert window._view_combo.count() == 1
        assert window._view_combo.itemData(0) is None

    def test_current_fpv_drone_is_none_by_default(self, window):
        assert window._current_fpv_drone() is None

    def test_stack_starts_on_canvas(self, window):
        assert window._view_stack.currentWidget() is window._canvas

    def test_config_load_populates_combo(self, loaded_window):
        assert loaded_window._view_combo.count() == 3  # external + d1 + d2
        assert loaded_window._view_combo.itemData(0) is None
        assert loaded_window._view_combo.itemData(1) == "d1"
        assert loaded_window._view_combo.itemData(2) == "d2"

    def test_reload_does_not_accumulate_entries(self, loaded_window):
        loaded_window._on_config_loaded(loaded_window._backend.get_state())
        assert loaded_window._view_combo.count() == 3

    def test_selecting_drone_switches_to_fpv_label(self, loaded_window):
        loaded_window._view_combo.setCurrentIndex(1)
        assert loaded_window._current_fpv_drone() == "d1"
        assert loaded_window._view_stack.currentWidget() is loaded_window._fpv_label

    def test_selecting_drone_renders_a_pixmap(self, loaded_window):
        loaded_window._view_combo.setCurrentIndex(1)
        pixmap = loaded_window._fpv_label.pixmap()
        assert not pixmap.isNull()
        assert pixmap.width() > 0 and pixmap.height() > 0

    def test_switching_between_two_drones_repaints(self, loaded_window):
        """Both drone views share one widget, so a repaint guard keyed on the widget would skip this switch."""
        loaded_window._view_combo.setCurrentIndex(1)
        from_d1 = loaded_window._fpv_label.pixmap().toImage()

        loaded_window._view_combo.setCurrentIndex(2)

        assert loaded_window._current_fpv_drone() == "d2"
        assert loaded_window._fpv_label.pixmap().toImage() != from_d1

    def test_switching_back_shows_canvas(self, loaded_window):
        loaded_window._view_combo.setCurrentIndex(2)
        loaded_window._view_combo.setCurrentIndex(0)
        assert loaded_window._current_fpv_drone() is None
        assert loaded_window._view_stack.currentWidget() is loaded_window._canvas

    def test_round_trip_preserves_orbit_angle(self, loaded_window):
        """A detour through the FPV view must leave the external view exactly where the user parked it.

        Currently this holds for free — matplotlib 3.10's ``cla()`` no longer resets ``elev``/``azim``, so
        even the 3D path would preserve them. The test pins the contract for whoever next touches either
        branch of ``_redraw`` (or upgrades matplotlib).
        """
        loaded_window._ax.view_init(elev=33.0, azim=57.0)
        elev, azim = loaded_window._ax.elev, loaded_window._ax.azim

        loaded_window._view_combo.setCurrentIndex(1)
        loaded_window._backend.step()
        loaded_window._redraw(loaded_window._last_result)
        loaded_window._view_combo.setCurrentIndex(0)

        assert loaded_window._ax.elev == pytest.approx(elev)
        assert loaded_window._ax.azim == pytest.approx(azim)

    def test_stepping_in_fpv_mode_updates_the_image(self, loaded_window):
        """_tick() calls _redraw() unconditionally — in FPV mode that must render, not fall through to the 3D path."""
        loaded_window._view_combo.setCurrentIndex(1)
        before = loaded_window._fpv_label.pixmap().toImage()
        for _ in range(15):
            loaded_window._redraw(loaded_window._backend.step())
        assert loaded_window._fpv_label.pixmap().toImage() != before

    def test_record_button_disabled_in_fpv_mode(self, loaded_window):
        """The video writer grabs the matplotlib figure, which is hidden in FPV mode."""
        loaded_window._view_combo.setCurrentIndex(1)
        assert loaded_window._btn_record.isEnabled() is False
        loaded_window._view_combo.setCurrentIndex(0)
        assert loaded_window._btn_record.isEnabled() is True

    def test_screenshot_in_fpv_mode_writes_png(self, loaded_window, tmp_path, monkeypatch):
        from pathlib import Path

        monkeypatch.chdir(tmp_path)
        loaded_window._view_combo.setCurrentIndex(1)
        loaded_window._on_screenshot()

        written = list((Path(tmp_path) / "screenshots").glob("*.png"))
        assert len(written) == 1
        assert "fpv_d1" in written[0].name
        assert written[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert not list((Path(tmp_path) / "screenshots").glob("*.json"))


class TestObjModelOverride:
    """The 'OBJ Model' button as a temporary override of the config — for *every* view, not just the external one.

    Before this, the button fed a path the external view read directly while the drone view read ``Drone.model``
    from the config, so the same neighbor could be a mesh in one view and a sphere in the other. Both now start
    from the drone's configured model and let ``_model_override`` win over it.
    """

    ASSET = "drone_costum_0_0_5.obj"  # small enough to parse per frame in a test

    @pytest.fixture
    def loaded_window(self, window, tmp_path):
        import json
        from pathlib import Path

        path = Path(tmp_path) / "obj_scenario.json"
        path.write_text(json.dumps(TestFpvView.FPV_CONFIG), encoding="utf-8")
        window._backend.load_config(path)
        window._redraw(window._backend.step())
        window._on_config_loaded(window._backend.get_state())
        return window

    @staticmethod
    def _pick(monkeypatch, path_str: str) -> None:
        """Make the next file dialog return ``path_str`` ('' = the user pressed Cancel)."""
        import drone_sim.gui.main_window as m
        monkeypatch.setattr(m.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (path_str, "")))

    @staticmethod
    def _asset_path() -> str:
        from drone_sim.domain.drone_model import ASSETS_DIR
        return str(ASSETS_DIR / TestObjModelOverride.ASSET)

    def test_no_override_by_default(self, loaded_window):
        assert loaded_window._model_override is None
        assert loaded_window._backend._model_override is None
        assert loaded_window._obj_model_label.text() == "Model: scatter"

    def test_button_sets_override_on_the_backend(self, loaded_window, monkeypatch):
        self._pick(monkeypatch, self._asset_path())
        loaded_window._on_select_obj_model()

        override = loaded_window._backend._model_override
        assert override is not None
        assert override.kind == "obj"
        assert override.path.name == self.ASSET

    def test_button_updates_external_view_state_and_label(self, loaded_window, monkeypatch):
        self._pick(monkeypatch, self._asset_path())
        loaded_window._on_select_obj_model()

        assert loaded_window._model_override is not None
        assert loaded_window._model_override.path.name == self.ASSET
        assert "drone_costum_0_0_5" in loaded_window._obj_model_label.text()

    def test_button_reaches_the_drone_view(self, loaded_window, monkeypatch):
        """The actual fix: pressing the button while a drone view is showing must change that image."""
        loaded_window._view_combo.setCurrentIndex(1)
        before = loaded_window._fpv_label.pixmap().toImage()

        self._pick(monkeypatch, self._asset_path())
        loaded_window._on_select_obj_model()

        assert loaded_window._fpv_label.pixmap().toImage() != before

    def test_cancel_clears_the_override(self, loaded_window, monkeypatch):
        self._pick(monkeypatch, self._asset_path())
        loaded_window._on_select_obj_model()

        self._pick(monkeypatch, "")
        loaded_window._on_select_obj_model()

        assert loaded_window._model_override is None
        assert loaded_window._backend._model_override is None
        assert loaded_window._obj_model_label.text() == "Model: scatter"

    def test_unreadable_obj_warns_and_keeps_the_current_pick(self, loaded_window, monkeypatch, tmp_path):
        """A broken file is an error, not a request to go back to spheres — and it must not reach the renderer."""
        import drone_sim.gui.main_window as m
        warnings = []
        monkeypatch.setattr(m.QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a)))

        self._pick(monkeypatch, self._asset_path())
        loaded_window._on_select_obj_model()
        good = loaded_window._backend._model_override

        self._pick(monkeypatch, str(tmp_path / "does_not_exist.obj"))
        loaded_window._on_select_obj_model()

        assert len(warnings) == 1
        assert loaded_window._backend._model_override is good
        assert loaded_window._model_override.path.name == self.ASSET

    def test_reset_keeps_the_override(self, loaded_window, monkeypatch):
        """Reset repeats the run, not the choice of what to look at (existing behaviour, now on both views)."""
        self._pick(monkeypatch, self._asset_path())
        loaded_window._on_select_obj_model()

        loaded_window._on_reset()

        assert loaded_window._model_override is not None
        assert loaded_window._backend._model_override is not None

    def test_loading_a_config_drops_the_override(self, loaded_window, monkeypatch, tmp_path):
        import json
        from pathlib import Path

        self._pick(monkeypatch, self._asset_path())
        loaded_window._on_select_obj_model()

        path = Path(tmp_path) / "reload.json"
        path.write_text(json.dumps(TestFpvView.FPV_CONFIG), encoding="utf-8")
        self._pick(monkeypatch, str(path))
        loaded_window._on_open_file()

        assert loaded_window._model_override is None
        assert loaded_window._backend._model_override is None
        assert loaded_window._obj_model_label.text() == "Model: scatter"


class TestPerceivedMarkers:
    """The 'x' per camera position estimate: purely additive, and invisible for every scenario without a camera."""

    @pytest.fixture
    def loaded_window(self, window, tmp_path):
        """A window with a scenario loaded — needed by the two tests that go through the real drawing paths."""
        import json
        from pathlib import Path

        path = Path(tmp_path) / "marker_scenario.json"
        path.write_text(json.dumps(TestFpvView.FPV_CONFIG), encoding="utf-8")
        window._backend.load_config(path)
        window._on_config_loaded(window._backend.get_state())
        return window

    @staticmethod
    def _marker(observed_id: str, position, *, observer_id: str = "d1", sigma=None):
        from drone_sim.gui.backend import PerceivedMarker
        return PerceivedMarker(observer_id=observer_id, observed_id=observed_id, position=np.asarray(position, dtype=float), sigma=sigma)

    @staticmethod
    def _drone(drone_id: str, color: str):
        from drone_sim.gui.backend import DroneState
        return DroneState(drone_id=drone_id, position=np.zeros(3), velocity=np.zeros(3), radius=0.3, safety_zone=1.0, adaptive_safety_radius=None,
                          max_adaptive_safety_radius=None, color=color, safety_color=color, trace_color=color)

    @staticmethod
    def _result(perceived, drones=()):
        from drone_sim.gui.backend import StepResult
        return StepResult(drones=list(drones), safety_radii=[], last_collisions=[], infeasible=False, infeasible_reason=None, step_count=1, t=0.1,
                          perceived=list(perceived))

    class _RecordingAx:
        """Stand-in for the 3D axes — the helper only ever scatters, so recording that call is the whole contract."""

        def __init__(self):
            self.calls = []

        def scatter(self, xs, ys, zs, **kwargs):
            self.calls.append(((xs[0], ys[0], zs[0]), kwargs))

    def test_no_estimates_draws_nothing(self):
        from drone_sim.gui.main_window import _draw_perceived_markers
        ax = self._RecordingAx()
        _draw_perceived_markers(ax, self._result([]))
        assert ax.calls == []

    def test_one_scatter_per_estimate_at_its_position(self):
        from drone_sim.gui.main_window import _draw_perceived_markers
        ax = self._RecordingAx()
        _draw_perceived_markers(ax, self._result([self._marker("d2", [1.0, 2.0, 3.0]), self._marker("d1", [4.0, 5.0, 6.0], observer_id="d2")],
                                                 drones=[self._drone("d1", "red"), self._drone("d2", "blue")]))

        assert [pos for pos, _ in ax.calls] == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        assert all(kwargs["marker"] == "x" for _, kwargs in ax.calls)

    def test_marker_takes_the_observed_drones_color(self):
        """Colored by who is being estimated: the cross then reads as an offset from that drone's own sphere."""
        from drone_sim.gui.main_window import _draw_perceived_markers
        ax = self._RecordingAx()
        _draw_perceived_markers(ax, self._result([self._marker("d2", [1.0, 2.0, 3.0], observer_id="d1")],
                                                 drones=[self._drone("d1", "red"), self._drone("d2", "blue")]))

        assert ax.calls[0][1]["c"] == ["blue"]

    def test_two_observers_of_one_neighbor_give_two_markers(self):
        from drone_sim.gui.main_window import _draw_perceived_markers
        ax = self._RecordingAx()
        _draw_perceived_markers(ax, self._result([self._marker("d3", [1.0, 0.0, 0.0], observer_id="d1"),
                                                  self._marker("d3", [1.1, 0.0, 0.0], observer_id="d2")],
                                                 drones=[self._drone("d3", "green")]))

        assert len(ax.calls) == 2
        assert all(kwargs["c"] == ["green"] for _, kwargs in ax.calls)

    def test_estimate_about_an_unknown_drone_still_draws(self):
        """Never drop an estimate just because its subject is gone — it gets the fallback color and stays visible."""
        from drone_sim.gui.main_window import _PERCEIVED_MARKER_FALLBACK_COLOR, _draw_perceived_markers
        ax = self._RecordingAx()
        _draw_perceived_markers(ax, self._result([self._marker("ghost", [1.0, 0.0, 0.0])], drones=[self._drone("d1", "red")]))

        assert ax.calls[0][1]["c"] == [_PERCEIVED_MARKER_FALLBACK_COLOR]

    def test_redraw_adds_one_artist_per_estimate(self, loaded_window):
        """Through the real axes: the markers reach the live view, and only when there are estimates."""
        drones = [self._drone("d1", "red"), self._drone("d2", "blue")]

        loaded_window._redraw(self._result([], drones=drones))
        without = len(loaded_window._ax.collections)

        loaded_window._redraw(self._result([self._marker("d2", [1.0, 2.0, 3.0]), self._marker("d1", [4.0, 5.0, 6.0], observer_id="d2")], drones=drones))

        assert len(loaded_window._ax.collections) == without + 2

    def test_checkbox_is_on_by_default(self, window):
        """A scenario only produces estimates when it was configured to, so whoever loaded one wants to see them."""
        assert window._perceived_check.isChecked() is True

    def test_unchecking_removes_the_markers(self, loaded_window):
        drones = [self._drone("d1", "red"), self._drone("d2", "blue")]
        result = self._result([self._marker("d2", [1.0, 2.0, 3.0]), self._marker("d1", [4.0, 5.0, 6.0], observer_id="d2")], drones=drones)

        loaded_window._redraw(result)
        with_markers = len(loaded_window._ax.collections)
        loaded_window._perceived_check.setChecked(False)

        assert len(loaded_window._ax.collections) == with_markers - 2

    def test_unchecking_repaints_immediately(self, loaded_window, monkeypatch):
        """Toggling while paused must not wait for the next tick — that would look like a dead checkbox."""
        redrawn = []
        monkeypatch.setattr(type(loaded_window), "_redraw", lambda self, result: redrawn.append(result))
        loaded_window._last_result = self._result([self._marker("d2", [1.0, 2.0, 3.0])], drones=[self._drone("d2", "blue")])

        loaded_window._perceived_check.setChecked(False)

        assert len(redrawn) == 1

    def test_toggling_without_a_result_does_not_raise(self, window):
        window._perceived_check.setChecked(False)  # nothing loaded yet

    def test_rechecking_brings_them_back(self, loaded_window):
        drones = [self._drone("d1", "red"), self._drone("d2", "blue")]
        loaded_window._redraw(self._result([self._marker("d2", [1.0, 2.0, 3.0])], drones=drones))
        with_markers = len(loaded_window._ax.collections)

        loaded_window._perceived_check.setChecked(False)
        loaded_window._perceived_check.setChecked(True)

        assert len(loaded_window._ax.collections) == with_markers

    def test_checkbox_survives_a_layout_rebuild(self, loaded_window):
        """The responsive layout reparents every control — one forgotten widget disappears from the window."""
        loaded_window._perceived_check.setChecked(False)
        loaded_window._rebuild_layout()

        assert loaded_window._perceived_check.isChecked() is False
        assert loaded_window._perceived_check.parent() is not None

    def test_unchecked_screenshot_omits_the_markers(self, loaded_window, tmp_path, monkeypatch):
        import drone_sim.gui.main_window as m
        monkeypatch.chdir(tmp_path)
        drawn = []
        monkeypatch.setattr(m, "_draw_perceived_markers", lambda ax, result: drawn.append(result.perceived))

        loaded_window._last_result = self._result([self._marker("d2", [1.0, 2.0, 3.0])], drones=[self._drone("d2", "blue")])
        loaded_window._perceived_check.setChecked(False)
        loaded_window._on_screenshot()

        assert drawn == []

    def test_screenshot_draws_the_markers_too(self, loaded_window, tmp_path, monkeypatch):
        """The saved picture is what is on screen — including the crosses."""
        import drone_sim.gui.main_window as m
        monkeypatch.chdir(tmp_path)
        drawn = []
        monkeypatch.setattr(m, "_draw_perceived_markers", lambda ax, result: drawn.append(result.perceived))

        loaded_window._last_result = self._result([self._marker("d2", [1.0, 2.0, 3.0])], drones=[self._drone("d2", "blue")])
        loaded_window._on_screenshot()

        assert len(drawn) == 1 and len(drawn[0]) == 1


class TestShutdown:
    """closeEvent releases what the backend started in the background."""

    def test_close_calls_backend_close(self, window):
        calls = []
        window._backend.close = lambda: calls.append(1)

        window.close()

        assert calls == [1]

    def test_close_stops_playback(self, window):
        window._playing = True

        window.close()

        assert window._playing is False

    def test_close_without_config_does_not_raise(self, window):
        window.close()  # no scenario ever loaded
