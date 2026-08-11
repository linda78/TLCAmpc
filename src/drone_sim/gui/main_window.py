from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from drone_sim.domain.drone_model import DroneModel, resolve_drone_model
from drone_sim.gui.backend import StepResult, SimState
from drone_sim.gui.direct_backend import DirectBackend
from drone_sim.api.utils.render_helper import draw_room_wireframe, draw_sphere_wireframe, draw_trace, draw_obstacles, draw_obj_mesh, draw_prediction_tube

_MAX_RUN_STEPS = 5000  # run-to-completion cap (matches paper2_tools/scenarios.py default)
_PREDICTION_TUBE_SAMPLES = 50           # arc-length samples per BoF prediction tube
_PREDICTION_TUBE_OUTER_ALPHA = 0.03     # safety-zone tube
_PREDICTION_TUBE_INNER_ALPHA = 0.08     # core (drone-radius) tube
_PREDICTION_TUBE_CENTERLINE_ALPHA = 0.02  # dashed centerline opacity (drawn once)

_EXTERNAL_VIEW_LABEL = "External view"  # first combo entry — the orbiting 3D scene, itemData None
_FPV_MIN_SIZE = (320, 240)              # floor for the render size; the label reports garbage before the first layout pass
_FPV_SCREENSHOT_SIZE = (1280, 960)      # FPV screenshots are rendered fresh, not scaled up from the on-screen frame

def _coerce_color(color):
   """Convert color list/tuple to a tuple matplotlib accepts (str passes through)."""
   if isinstance(color, (list, tuple)):
      return tuple(float(c) for c in color[:3])
   return color


def _draw_ghost_max_sphere(ax: object, is_adaptive: bool, position, safety_zone:float, max_radius:float, safety_color):
   if is_adaptive:
      pos = np.asarray(position, dtype=float).reshape(3)

      # only draw if meaningfully larger
      if max_radius > safety_zone + 1e-4:
         draw_sphere_wireframe(ax, pos, max_radius, color=safety_color, alpha=0.12, lw=0.4)


class MainWindow(QMainWindow):
    """Main application window: 3D canvas + playback controls."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Drone Simulation")

        self._backend = DirectBackend()
        self._playing: bool = False
        self._interval_ms: int = 100
        self._traces: dict[str, list[list[float]]] = {}
        self._zoom: float = 1.0  # <1 zoomed in, >1 zoomed out; applied to room limits
        self._run_to_completion: bool = False
        # GUI-side mirror of the backend's fleet-wide model override — the external view draws straight from it,
        # the drone view goes through the backend. None = no override, i.e. whatever the scenario configured.
        self._obj_path: Path | None = None  # path to .obj file for 3D drone model
        self._obj_scale: float = 0.3  # scale of the OBJ model in world units
        self._last_result: StepResult | None = None
        # Recording state (live MP4/GIF capture of the canvas)
        self._recording: bool = False
        self._video_writer: object | None = None
        self._video_path: Path | None = None
        self._video_frames: int = 0

        # ---- Canvas ----
        fig = Figure()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._canvas.setMinimumSize(400, 300)
        self._ax = fig.add_subplot(111, projection="3d")

        # ---- FPV view ----
        # The camera image is a plain pixmap, not a matplotlib artist, so it gets its own widget and the two
        # views are swapped by a stack. Keeping the canvas alive (rather than rebuilding it) is what lets the
        # orbit angle of the 3D scene survive a trip through the FPV view.
        self._fpv_label = QLabel("No camera view yet")
        self._fpv_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fpv_label.setMinimumSize(400, 300)
        self._fpv_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self._canvas)
        self._view_stack.addWidget(self._fpv_label)

        self._view_combo = QComboBox()
        self._view_combo.addItem(_EXTERNAL_VIEW_LABEL, None)  # itemData carries the drone id, None = external

        # ---- Controls ----
        self._btn_open = QPushButton("Open")
        self._btn_play = QPushButton("Play")
        self._btn_pause = QPushButton("Pause")
        self._btn_reset = QPushButton("Reset")
        self._btn_run_to_end = QPushButton("Run to Completion")
        self._btn_screenshot = QPushButton("Screenshot")
        self._btn_record = QPushButton("Record")
        self._btn_obj_model = QPushButton("OBJ Model")
        self._obj_model_label = QLabel("Model: scatter")

        # ---- Status widgets ----
        self._collision_label = QLabel("No Collisions")
        self._collision_label.setStyleSheet("color: green;")

        self._admm_label = QLabel("")
        self._admm_label.setVisible(False)

        # ---- Scenario info panel ----
        info_box = QGroupBox("Scenario Info")
        _form = QFormLayout()
        self._info_path_label = QLabel("—")
        self._info_coord_label = QLabel("—")
        self._info_dt_label = QLabel("—")
        self._info_drones_label = QLabel("—")
        self._info_obstacles_label = QLabel("—")
        _form.addRow("Config:", self._info_path_label)
        _form.addRow("Coordinator:", self._info_coord_label)
        _form.addRow("dt:", self._info_dt_label)
        _form.addRow("Drones:", self._info_drones_label)
        _form.addRow("Obstacles:", self._info_obstacles_label)
        info_box.setLayout(_form)
        self._info_box = info_box

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(10, 500)
        self._speed_slider.setValue(100)
        self._speed_label = QLabel("100 ms")

        # Responsive layout: controls left (vertical) when wide, bottom (horizontal) when tall
        self._controls_max_width = 180  # max width for controls panel in landscape mode
        self._is_landscape: bool = False  # track current layout mode

        # Create central widget and main layout placeholder
        self._central = QWidget()
        self._main_layout: QHBoxLayout | QVBoxLayout | None = None
        self._ctrl_container: QWidget | None = None
        self.setCentralWidget(self._central)
        self._rebuild_layout()

        # ---- Status bar ----
        self._step_label = QLabel("Step: 0 | t: 0.00 s")
        self.statusBar().addPermanentWidget(self._step_label)

        # ---- Signal connections ----
        self._btn_open.clicked.connect(self._on_open_file)
        self._btn_play.clicked.connect(self._on_play)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_run_to_end.clicked.connect(self._on_run_to_completion)
        self._btn_screenshot.clicked.connect(self._on_screenshot)
        self._btn_record.clicked.connect(self._on_record)
        self._btn_obj_model.clicked.connect(self._on_select_obj_model)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

        # ---- Keyboard shortcuts ----
        # QShortcut fires regardless of which child widget has focus (unlike keyPressEvent
        # which only fires when MainWindow itself has focus — canvas steals focus after orbit).
        spacebar = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        spacebar.activated.connect(self._toggle_play_pause)

        # ---- Scroll zoom ----
        # Axes3D scroll zoom is not wired automatically in the Qt backend without a
        # NavigationToolbar. Connect it manually via mpl_connect.
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

    # ------------------------------------------------------------------ #
    # Playback slots                                                       #
    # ------------------------------------------------------------------ #

    def _on_open_file(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Open Scenario JSON", "configs", "JSON Files (*.json)")
        if not path_str:
            return
        self._backend.load_config(Path(path_str))
        self._playing = False
        self._traces = {}
        self._zoom = 1.0
        # load_config already dropped the override inside the backend; mirror that here. Not via
        # _apply_drone_model_override — that would repaint the *previous* scenario's result against the new sim.
        self._show_drone_model_override(None)
        # Call step() once to get initial drone positions for first render.
        # (SimState from load_config has counts only, no per-drone positions.)
        # Research note (open question 3): simplest approach is one probe step on load.
        result = self._backend.step()
        for drone in result.drones:
            self._traces.setdefault(drone.drone_id, []).append(drone.position.tolist())
        self._redraw(result)
        self._update_step_label(result)
        self._update_collision_indicator(result)
        state = self._backend.get_state()
        self._on_config_loaded(state)

    def _on_play(self) -> None:
        if not self._playing:
            self._playing = True
            QTimer.singleShot(0, self._tick)  # fire immediately for responsiveness

    def _on_pause(self) -> None:
        self._playing = False  # _tick() checks this flag and exits without rescheduling
        self._run_to_completion = False  # Pause cancels run-to-completion (research Q3 resolution)

    def _toggle_play_pause(self) -> None:
        if self._playing:
            self._on_pause()
        else:
            self._on_play()

    def _on_reset(self) -> None:
        self._playing = False
        self._traces = {}  # CRITICAL: clear GUI traces BEFORE backend reset (pitfall 7)
        self._zoom = 1.0
        self._backend.reset()
        # Draw initial frame after reset via one probe step.
        # Intentional: reset always shows step 1 (research open question 3).
        result = self._backend.step()
        for drone in result.drones:
            self._traces.setdefault(drone.drone_id, []).append(drone.position.tolist())
        self._redraw(result)
        self._update_step_label(result)
        self._update_collision_indicator(result)
        # NOTE: _on_config_loaded is NOT called on reset — config metadata is unchanged

    def _on_run_to_completion(self) -> None:
        """Start playing and auto-stop when all drones reach destination or max steps hit."""
        self._run_to_completion = True
        self._on_play()

    def _on_config_loaded(self, state: SimState) -> None:
        """Update scenario info panel after load_config(). NOT called on reset — config metadata unchanged."""
        path_str = state.config_path or "—"
        fname = Path(path_str).name if state.config_path else "—"
        self._info_path_label.setText(fname)
        self._info_path_label.setToolTip(path_str)
        self._info_coord_label.setText(state.coordinator_type)
        self._info_dt_label.setText(f"{state.dt} s")
        self._info_drones_label.setText(str(state.drone_count))
        self._info_obstacles_label.setText(str(state.obstacle_count))
        self._populate_view_combo(state.drone_ids)

    # ------------------------------------------------------------------ #
    # View selector (external 3D scene <-> FPV of one drone)               #
    # ------------------------------------------------------------------ #

    def _current_fpv_drone(self) -> str | None:
        """Drone id selected in the view combo, or ``None`` while the external 3D view is showing."""
        return self._view_combo.currentData()

    def _populate_view_combo(self, drone_ids: list[str]) -> None:
        """Refill the view selector for a freshly loaded scenario, resetting to the external view."""
        blocked = self._view_combo.blockSignals(True)
        self._view_combo.clear()
        self._view_combo.addItem(_EXTERNAL_VIEW_LABEL, None)
        for drone_id in drone_ids:
            self._view_combo.addItem(f"View from {drone_id}", drone_id)
        self._view_combo.setCurrentIndex(0)
        self._view_combo.blockSignals(blocked)
        self._on_view_changed()  # signals were blocked — apply the reset to external view by hand

    def _on_view_changed(self, _index: int = 0) -> None:
        fpv_id = self._current_fpv_drone()
        # The video writer is bound to the matplotlib figure. In FPV mode that figure is hidden and stops being
        # redrawn, so a running recording would keep grabbing a frozen scene nobody is looking at.
        if fpv_id is not None and self._recording:
            self._stop_recording()
        self._btn_record.setEnabled(fpv_id is None)
        self._view_stack.setCurrentWidget(self._canvas if fpv_id is None else self._fpv_label)
        if self._last_result is not None:
            self._redraw(self._last_result)

    def _update_collision_indicator(self, result: StepResult) -> None:
        if result.last_collisions:
            self._collision_label.setText("COLLISION")
            self._collision_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self._collision_label.setText("No Collisions")
            self._collision_label.setStyleSheet("color: green;")

    def _update_admm_indicator(self, result: StepResult) -> None:
        if result.admm_iteration_count is None:
            self._admm_label.setVisible(False)
        else:
            self._admm_label.setVisible(True)
            self._admm_label.setText(f"ADMM: {result.admm_iteration_count} iters")

    def _tick(self) -> None:
        if not self._playing:
            return  # Pause was pressed — do not reschedule
        result = self._backend.step()
        if result.infeasible:
            self._playing = False
            self._run_to_completion = False
            reason = result.infeasible_reason or "Solver reported infeasible controls."
            QMessageBox.warning(self, "Solver Infeasible", reason)
            return  # do NOT reschedule (pitfall 6)
        for drone in result.drones:
            self._traces.setdefault(drone.drone_id, []).append(drone.position)
        self._redraw(result)
        self._update_step_label(result)
        self._update_collision_indicator(result)
        self._update_admm_indicator(result)

        # Run-to-completion termination check:
        if self._run_to_completion:
            if result.all_reached or result.step_count >= _MAX_RUN_STEPS:
                self._playing = False
                self._run_to_completion = False
                return  # do NOT reschedule (auto-stop)

        # Reschedule AFTER current tick completes — prevents event accumulation (locked decision)
        QTimer.singleShot(self._interval_ms, self._tick)

    def _redraw(self, result: StepResult) -> None:
        self._last_result = result

        # FPV branches out before the 3D path rather than after it: the whole scene below — wireframes,
        # spheres, traces, prediction tubes — would otherwise be rebuilt every tick into a canvas the stack
        # is not showing. Placing the branch above the cla()/view_init block also keeps the external view's
        # orbit angle out of reach of the FPV path entirely.
        fpv_id = self._current_fpv_drone()
        if fpv_id is not None:
            self._draw_fpv(fpv_id)
            return

        # Save orbit angle BEFORE cla() — ax.cla() resets elev/azim to defaults (pitfall 3)
        elev = self._ax.elev
        azim = self._ax.azim

        self._ax.cla()

        # Restore orbit angle IMMEDIATELY after cla()
        self._ax.view_init(elev=elev, azim=azim)

        sim_state = self._backend.get_state()
        room_min = sim_state.room_min
        room_max = sim_state.room_max

        draw_room_wireframe(self._ax, room_min, room_max)

        draw_obstacles(self._ax, sim_state.obstacles)

        # Draw drones
        for drone in result.drones:
            pos = drone.position
            color = drone.color
            safety_r = (drone.adaptive_safety_radius if drone.adaptive_safety_radius is not None else drone.safety_zone)
            safety_color = drone.safety_color

            if self._obj_path is not None:
                self._obj_scale = drone.radius
                draw_obj_mesh(self._ax, pos, self._obj_path, scale=self._obj_scale, color=color if isinstance(color, str) else "steelblue", alpha=0.8)
            else:
                self._ax.scatter([pos[0]], [pos[1]], [pos[2]], s=80, c=[color] if isinstance(color, str) else [color], depthshade=True, label=drone.drone_id)
                draw_sphere_wireframe(self._ax, pos, safety_r, color=safety_color, alpha=0.6, lw=0.6)
                # Ghost sphere: maximum adaptive radius (only when larger than current zone)
                _draw_ghost_max_sphere(self._ax, drone.adaptive_safety_radius is not None, pos, drone.adaptive_safety_radius, drone.max_adaptive_safety_radius,
                                       safety_color)

            trace = self._traces.get(drone.drone_id, [])
            if trace:
                draw_trace(self._ax, trace, drone.trace_color)

        # BoF prediction tubes (one per drone with a fresh prediction this step).
        # Outer = post-processed safety radius (barely visible halo).
        # Inner = drone body radius (slightly more present, with centerline).
        for pred in result.predictions:
            outer_color = _coerce_color(pred.color)
            inner_color = _coerce_color(pred.core_color)
            # DIRTY FIX: halve drone.radius for the inner tube, the rendered tube ends up ~2x too wide compared to drone.radius. Real cause not found yet.
            # Revert this /2 once the underlying radius/diameter bug is fixed.
            inner_radii = np.full(pred.radii.shape, pred.inner_radius / 2.0)

            draw_prediction_tube(
                self._ax, pred.points, pred.radii,
                color=outer_color,
                n_samples=_PREDICTION_TUBE_SAMPLES,
                alpha=_PREDICTION_TUBE_OUTER_ALPHA,
                draw_centerline=False,
            )
            draw_prediction_tube(
                self._ax, pred.points, inner_radii,
                color=inner_color,
                n_samples=_PREDICTION_TUBE_SAMPLES,
                alpha=_PREDICTION_TUBE_INNER_ALPHA,
                centerline_alpha=_PREDICTION_TUBE_CENTERLINE_ALPHA,
                draw_centerline=True,
            )


        # Set axis limits from room bounds, scaled by zoom level.
        # self._zoom is stored on self (not ax) so ax.cla() never resets it.
        cx = (room_min[0] + room_max[0]) / 2
        cy = (room_min[1] + room_max[1]) / 2
        cz = (room_min[2] + room_max[2]) / 2
        hx = (room_max[0] - room_min[0]) / 2 * self._zoom
        hy = (room_max[1] - room_min[1]) / 2 * self._zoom
        hz = (room_max[2] - room_min[2]) / 2 * self._zoom
        self._ax.set_xlim(cx - hx, cx + hx)
        self._ax.set_ylim(cy - hy, cy + hy)
        self._ax.set_zlim(cz - hz, cz + hz)
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")

        box = [room_max[i] - room_min[i] for i in range(3)]
        if all(b > 0 for b in box):
            self._ax.set_box_aspect(box)

        # Use draw_idle — NOT draw() — to avoid blocking the event loop (pitfall)
        self._canvas.draw_idle()

        # Video recording: grab_frame() forces a sync draw on the figure, so it
        # captures the just-rendered scene independently of draw_idle's async pump.
        if self._recording:
            self._capture_frame()

    def _draw_fpv(self, drone_id: str) -> None:
        """Render the drone's camera image at the label's own pixel size, so the view scales with the window."""
        size = self._fpv_label.size()
        width = max(_FPV_MIN_SIZE[0], size.width())
        height = max(_FPV_MIN_SIZE[1], size.height())

        png = self._backend.render_fpv(drone_id, (width, height))
        if png is None:
            self._fpv_label.setPixmap(QPixmap())
            self._fpv_label.setText(f"No camera view for {drone_id}")
            return

        pixmap = QPixmap()
        pixmap.loadFromData(png, "PNG")
        self._fpv_label.setPixmap(pixmap)

    def _update_step_label(self, result: StepResult) -> None:
        self._step_label.setText(f"Step: {result.step_count} | t: {result.t:.2f} s")

    # ------------------------------------------------------------------ #
    # OBJ model selection                                                  #
    # ------------------------------------------------------------------ #

    def _on_select_obj_model(self) -> None:
        """Pick an .obj file to draw *every* drone with, in every view. 'Cancel' reverts to the configured models.

        The override is temporary by design — it exists so meshes can be tried out without writing a scenario
        file per model, and it is dropped again the next time a config is loaded (but survives a reset).

        The path goes through :func:`resolve_drone_model`, the same check the config layer runs, so an
        unreadable .obj surfaces as one dialog here rather than a log line per rendered frame. A rejected file
        leaves the current selection alone: it is an error, not a request to go back to spheres.
        """
        utils_dir = str(Path(__file__).resolve().parent.parent / "resources" / "assets")
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select OBJ Model", utils_dir, "OBJ Files (*.obj)"
        )
        if not path_str:
            self._apply_drone_model_override(None)
            return

        try:
            model = resolve_drone_model("obj", path_str)
        except ValueError as exc:
            QMessageBox.warning(self, "OBJ Model", f"Cannot use this model:\n{exc}")
            return

        self._apply_drone_model_override(model)

    def _apply_drone_model_override(self, model: DroneModel | None) -> None:
        """Push the override to the backend, update the GUI's own copy, and repaint whichever view is showing.

        Repainting via ``_redraw`` rather than ``draw_idle`` is what makes the change visible immediately in
        *both* views: ``draw_idle`` only re-renders the artists already on the 3D canvas, and would not touch
        the drone view at all.
        """
        self._backend.set_drone_model_override(model)
        self._show_drone_model_override(model)
        if self._last_result is not None:
            self._redraw(self._last_result)
        else:
            self._canvas.draw_idle()

    def _show_drone_model_override(self, model: DroneModel | None) -> None:
        """Update the GUI-side copy of the override (path for the external view, label for the user). No repaint."""
        self._obj_path = model.path if model is not None else None
        self._obj_model_label.setText(f"Model: {self._obj_path.stem}" if self._obj_path is not None else "Model: scatter")

    # ------------------------------------------------------------------ #
    # Screenshot                                                           #
    # ------------------------------------------------------------------ #

    def _on_screenshot(self) -> None:
        result = self._last_result
        if result is None:
            QMessageBox.warning(self, "Screenshot", "No simulation loaded yet.")
            return

        # The block below rebuilds the 3D scene from scratch — in FPV mode that would save a picture of
        # something other than what is on screen.
        fpv_id = self._current_fpv_drone()
        if fpv_id is not None:
            self._save_fpv_screenshot(fpv_id, result)
            return

        sim_state = self._backend.get_state()

        # Create a separate figure so the live canvas is not disturbed
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=self._ax.elev, azim=self._ax.azim)

        draw_room_wireframe(ax, sim_state.room_min, sim_state.room_max)
        draw_obstacles(ax, sim_state.obstacles)

        for drone in result.drones:
            pos = drone.position
            color = drone.color
            safety_r = (drone.adaptive_safety_radius if drone.adaptive_safety_radius is not None else drone.safety_zone)
            safety_color = drone.safety_color

            if self._obj_path is not None:
                draw_obj_mesh(ax, pos, self._obj_path, scale=drone.radius, color=color if isinstance(color, str) else "steelblue", alpha=0.8)
                # Invisible scatter for legend entry with drone color
                ax.scatter([pos[0]], [pos[1]], [pos[2]], s=40, c=[color] if isinstance(color, str) else [color], alpha=0.0, label=drone.drone_id)
            else:
                ax.scatter([pos[0]], [pos[1]], [pos[2]], s=80, c=[color] if isinstance(color, str) else [color], depthshade=True, label=drone.drone_id)
                draw_sphere_wireframe(ax, pos, safety_r, color=safety_color, alpha=0.6, lw=0.6)
                _draw_ghost_max_sphere(ax, drone.adaptive_safety_radius is not None, pos, drone.adaptive_safety_radius, drone.max_adaptive_safety_radius, safety_color)

            trace = self._traces.get(drone.drone_id, [])
            if trace:
                draw_trace(ax, trace, drone.trace_color)

        # BoF prediction tubes (mirror live view: outer safety halo + inner body tube).
        #  TODO: refactor, it should not be needed to implement that stuff twice!
        for pred in result.predictions:
            outer_color = _coerce_color(pred.color)
            inner_color = _coerce_color(pred.core_color)
            # DIRTY FIX: halve drone.radius for the inner tube, the rendered tube ends up ~2x too wide compared to drone.radius. Real cause not found yet.
            # Revert this /2 once the underlying radius/diameter bug is fixed.
            inner_radii = np.full(pred.radii.shape, pred.inner_radius / 2.0)
            draw_prediction_tube(
                ax, pred.points, pred.radii,
                color=outer_color,
                n_samples=_PREDICTION_TUBE_SAMPLES,
                alpha=_PREDICTION_TUBE_OUTER_ALPHA,
                draw_centerline=False,
            )
            draw_prediction_tube(
                ax, pred.points, inner_radii,
                color=inner_color,
                n_samples=_PREDICTION_TUBE_SAMPLES,
                alpha=_PREDICTION_TUBE_INNER_ALPHA,
                centerline_alpha=_PREDICTION_TUBE_CENTERLINE_ALPHA,
                draw_centerline=True,
            )

        # Axis limits (match live view zoom)
        room_min, room_max = sim_state.room_min, sim_state.room_max
        cx = (room_min[0] + room_max[0]) / 2
        cy = (room_min[1] + room_max[1]) / 2
        cz = (room_min[2] + room_max[2]) / 2
        hx = (room_max[0] - room_min[0]) / 2 * self._zoom
        hy = (room_max[1] - room_min[1]) / 2 * self._zoom
        hz = (room_max[2] - room_min[2]) / 2 * self._zoom
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        ax.set_zlim(cz - hz, cz + hz)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        box = [room_max[i] - room_min[i] for i in range(3)]
        if all(b > 0 for b in box):
            ax.set_box_aspect(box)

        # Scenario name (lower-left) and timestamp (lower-right)
        scenario_name = Path(sim_state.config_path).stem if sim_state.config_path else "unknown"
        fig.text(0.02, 0.02, scenario_name, fontsize=9, ha="left", va="bottom")
        fig.text(0.98, 0.02, f"t = {result.t:.2f} s (step {result.step_count})", fontsize=9, ha="right", va="bottom")

        # Save
        out_dir = Path("screenshots")
        out_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scenario_name}_{result.step_count}_{timestamp}.png"
        path = out_dir / filename
        fig.savefig(str(path), dpi=150, bbox_inches="tight")

        # Save data snapshot for later re-rendering
        snapshot = {
            "scenario_name": scenario_name,
            "config_path": sim_state.config_path,
            "step_count": result.step_count,
            "t": result.t,
            "dt": sim_state.dt,
            "room_min": sim_state.room_min.tolist(),
            "room_max": sim_state.room_max.tolist(),
            "obstacles": [(c.tolist(), e.tolist()) for c, e in sim_state.obstacles],
            "view": {"elev": self._ax.elev, "azim": self._ax.azim, "zoom": self._zoom},
            "drones": [
                {
                    "drone_id": d.drone_id,
                    "position": d.position.tolist(),
                    "velocity": d.velocity.tolist(),
                    "radius": d.radius,
                    "safety_zone": d.safety_zone,
                    "adaptive_safety_radius": d.adaptive_safety_radius,
                    "max_adaptive_safety_radius": d.max_adaptive_safety_radius,
                    "color": d.color if isinstance(d.color, str) else list(d.color),
                    "safety_color": d.safety_color if isinstance(d.safety_color, str) else list(d.safety_color),
                    "trace_color": d.trace_color if isinstance(d.trace_color, str) else list(d.trace_color),
                    "trace": [p.tolist() if hasattr(p, "tolist") else p for p in self._traces.get(d.drone_id, [])],
                }
                for d in result.drones
            ],
        }
        snapshot_path = path.with_suffix(".json")
        snapshot_path.write_text(json.dumps(snapshot, indent=2))

        self._step_label.setText(f"Saved: {path.name}")

    def _save_fpv_screenshot(self, drone_id: str, result: StepResult) -> None:
        """Write the drone's camera image to ``screenshots/``, re-rendered at print size.

        No JSON snapshot alongside it, unlike the 3D screenshot: an FPV frame is reproduced by loading the
        same scenario and stepping to the same step, not from a handful of positions.
        """
        png = self._backend.render_fpv(drone_id, _FPV_SCREENSHOT_SIZE)
        if png is None:
            QMessageBox.warning(self, "Screenshot", f"No camera view for {drone_id}.")
            return

        sim_state = self._backend.get_state()
        scenario_name = Path(sim_state.config_path).stem if sim_state.config_path else "unknown"
        out_dir = Path("screenshots")
        out_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{scenario_name}_fpv_{drone_id}_{result.step_count}_{timestamp}.png"
        path.write_bytes(png)

        self._step_label.setText(f"Saved: {path.name}")

    # ------------------------------------------------------------------ #
    # Video recording                                                    #
    # ------------------------------------------------------------------ #

    def _on_record(self) -> None:
        """Toggle live video recording of the canvas. MP4 via ffmpeg if
        available, otherwise GIF via Pillow. Captures one frame per redraw."""
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._last_result is None:
            QMessageBox.warning(self, "Record", "No simulation loaded yet.")
            return

        from matplotlib.animation import FFMpegWriter, PillowWriter

        sim_state = self._backend.get_state()
        scenario_name = Path(sim_state.config_path).stem if sim_state.config_path else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("screenshots")
        out_dir.mkdir(exist_ok=True)

        # Frame rate matches current playback speed so the video plays back at
        # the same pace the user sees on screen. Clamped to a sane range.
        fps = max(2, min(60, int(round(1000.0 / max(self._interval_ms, 1)))))

        if FFMpegWriter.isAvailable():
            self._video_path = out_dir / f"{scenario_name}_{timestamp}.mp4"
            writer = FFMpegWriter(fps=fps, bitrate=2400)
        else:
            self._video_path = out_dir / f"{scenario_name}_{timestamp}.gif"
            writer = PillowWriter(fps=fps)

        # FFMpegWriter.setup() opens the encoder pipe and binds it to the figure.
        writer.setup(self._canvas.figure, str(self._video_path), dpi=120)
        self._video_writer = writer
        self._video_frames = 0
        self._recording = True
        self._btn_record.setText("Stop Rec")
        self._btn_record.setStyleSheet("color: red; font-weight: bold;")
        self._step_label.setText(f"Recording -> {self._video_path.name} @ {fps} fps")
        # Capture the current frame immediately so the first redraw isn't missed.
        self._capture_frame()

    def _stop_recording(self) -> None:
        if not self._recording or self._video_writer is None:
            return
        try:
            self._video_writer.finish()
        except Exception as exc:
            QMessageBox.warning(self, "Record", f"Failed to finalize video: {exc}")
        self._recording = False
        self._btn_record.setText("Record")
        self._btn_record.setStyleSheet("")
        if self._video_path is not None:
            self._step_label.setText(f"Saved: {self._video_path.name} ({self._video_frames} frames)")
        self._video_writer = None
        self._video_path = None
        self._video_frames = 0

    def _capture_frame(self) -> None:
        """Grab the current canvas as one video frame. No-op when not recording."""
        if not self._recording or self._video_writer is None:
            return
        try:
            self._video_writer.grab_frame()
            self._video_frames += 1
        except Exception as exc:
            # Stop on first failure so we don't spam the user with errors.
            self._recording = False
            self._btn_record.setText("Record")
            self._btn_record.setStyleSheet("")
            QMessageBox.warning(self, "Record", f"Frame capture failed, recording stopped: {exc}")

    # ------------------------------------------------------------------ #
    # Speed slider                                                        #
    # ------------------------------------------------------------------ #

    def _on_speed_changed(self, value: int) -> None:
        self._interval_ms = value
        self._speed_label.setText(f"{value} ms")

    # ------------------------------------------------------------------ #
    # Scroll zoom                                                          #
    # ------------------------------------------------------------------ #

    def _on_scroll(self, event) -> None:
        """Zoom 3D view by scaling room limits stored in self._zoom.
        Stored in Python (not on ax) so ax.cla() in _redraw() cannot reset it."""
        if event.button == "up":
            self._zoom = max(0.2, self._zoom * 0.85)
        elif event.button == "down":
            self._zoom = min(5.0, self._zoom * (1.0 / 0.85))
        self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Shutdown                                                             #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:
        """Shut down what the backend started in the background before the window goes away.

        Currently that is the perception REST server thread. Its daemon flag would collect it eventually, but
        only at interpreter exit — which keeps the port bound for as long as the process lingers. A running
        recording is finalized here too, so closing mid-capture leaves a playable file instead of a stub."""
        self._playing = False
        if self._recording:
            self._stop_recording()
        self._backend.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    # Responsive layout                                                    #
    # ------------------------------------------------------------------ #

    def resizeEvent(self, event) -> None:
        """Rebuild layout when aspect ratio crosses landscape/portrait threshold."""
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        landscape = w > h
        if landscape != self._is_landscape:
            self._is_landscape = landscape
            self._rebuild_layout()

    def _rebuild_layout(self) -> None:
        """Rebuild the main layout based on current aspect ratio."""
        # Remove old layout if exists
        if self._main_layout is not None:
            # Reparent widgets before deleting layout
            self._view_stack.setParent(None)  # takes canvas + FPV label with it — both stay children of the stack
            self._view_combo.setParent(None)
            self._btn_open.setParent(None)
            self._btn_play.setParent(None)
            self._btn_pause.setParent(None)
            self._btn_reset.setParent(None)
            self._btn_run_to_end.setParent(None)
            self._btn_screenshot.setParent(None)
            self._btn_record.setParent(None)
            self._btn_obj_model.setParent(None)
            self._obj_model_label.setParent(None)
            self._speed_slider.setParent(None)
            self._speed_label.setParent(None)
            self._collision_label.setParent(None)
            self._admm_label.setParent(None)
            self._info_box.setParent(None)
            # Delete old layout
            QWidget().setLayout(self._central.layout())

        w = self.width()
        h = self.height()
        landscape = w > h
        self._is_landscape = landscape

        if landscape:
            # Landscape: controls on left, vertical, max width limited
            ctrl_layout = QVBoxLayout()
            ctrl_layout.addWidget(self._btn_open)
            ctrl_layout.addWidget(self._btn_play)
            ctrl_layout.addWidget(self._btn_pause)
            ctrl_layout.addWidget(self._btn_reset)
            ctrl_layout.addWidget(self._btn_run_to_end)
            ctrl_layout.addWidget(self._btn_screenshot)
            ctrl_layout.addWidget(self._btn_record)
            ctrl_layout.addSpacing(10)
            ctrl_layout.addWidget(self._btn_obj_model)
            ctrl_layout.addWidget(self._obj_model_label)
            ctrl_layout.addSpacing(10)
            ctrl_layout.addWidget(QLabel("View:"))
            ctrl_layout.addWidget(self._view_combo)
            ctrl_layout.addSpacing(15)
            ctrl_layout.addWidget(self._info_box)
            ctrl_layout.addWidget(self._collision_label)
            ctrl_layout.addWidget(self._admm_label)
            ctrl_layout.addStretch()
            ctrl_layout.addWidget(QLabel("Speed:"))
            self._speed_slider.setOrientation(Qt.Orientation.Horizontal)
            ctrl_layout.addWidget(self._speed_slider)
            ctrl_layout.addWidget(self._speed_label)

            ctrl_container = QWidget()
            ctrl_container.setLayout(ctrl_layout)
            ctrl_container.setMaximumWidth(self._controls_max_width)
            ctrl_container.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)

            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(2, 2, 2, 2)
            main_layout.setSpacing(4)
            main_layout.addWidget(ctrl_container)
            main_layout.addWidget(self._view_stack, stretch=1)
            self._view_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else:
            # Portrait: controls at bottom, horizontal
            ctrl_layout = QHBoxLayout()
            ctrl_layout.addWidget(self._btn_open)
            ctrl_layout.addWidget(self._btn_play)
            ctrl_layout.addWidget(self._btn_pause)
            ctrl_layout.addWidget(self._btn_reset)
            ctrl_layout.addWidget(self._btn_run_to_end)
            ctrl_layout.addWidget(self._btn_screenshot)
            ctrl_layout.addWidget(self._btn_record)
            ctrl_layout.addWidget(self._btn_obj_model)
            ctrl_layout.addWidget(QLabel("View:"))
            ctrl_layout.addWidget(self._view_combo)
            ctrl_layout.addStretch()
            ctrl_layout.addWidget(QLabel("Speed:"))
            self._speed_slider.setOrientation(Qt.Orientation.Horizontal)
            ctrl_layout.addWidget(self._speed_slider)
            ctrl_layout.addWidget(self._speed_label)

            ctrl_container = QWidget()
            ctrl_container.setLayout(ctrl_layout)

            status_row = QHBoxLayout()
            status_row.addWidget(self._info_box)
            status_row.addWidget(self._obj_model_label)
            status_row.addStretch()
            status_row.addWidget(self._collision_label)
            status_row.addWidget(self._admm_label)

            main_layout = QVBoxLayout()
            main_layout.setContentsMargins(2, 2, 2, 2)
            main_layout.setSpacing(4)
            main_layout.addWidget(self._view_stack, stretch=1)
            main_layout.addLayout(status_row)
            main_layout.addWidget(ctrl_container)
            self._view_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._central.setLayout(main_layout)
        self._main_layout = main_layout
        self._ctrl_container = ctrl_container
