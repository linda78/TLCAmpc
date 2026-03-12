#!/usr/bin/env python3
"""
tpt_video.py — Generate an MP4 of the drone simulation with live BOF
Trajectory Prediction Tubes (TPTs) overlaid.

Follows exactly the same approach as:
    python -m tools.live_view --config configs/... --obj-name drone_costum_0_0_5.obj

But additionally:
  - Feeds drone positions to the BOF server every step
  - Draws live TPT prediction tubes on top of the 3D scene
  - Saves the result as an MP4

Usage (run from inside TLAmpc1/ with its venv active):
    python tpt_video.py

    # Use a specific config:
    python tpt_video.py --config configs/conflict_evasion/Converging3DroneCentral.json

    # Narrow gap long room (5 drones):
    python tpt_video.py --config configs/narrow_gap/ThickGap5DronesAdaptive.json --steps 400

    # Quick test run:
    python tpt_video.py --steps 100 --output ~/Desktop/test.mp4

Requirements:
    - Run from inside TLAmpc1/ with its venv active  (source .venv/bin/activate)
    - BOF server running in another terminal          (python run_server.py)
    - ffmpeg installed                               (brew install ffmpeg)
"""

from __future__ import annotations

import argparse
import io
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests

# ── Make drone_sim importable when run from TLAmpc1/ root ────────────────────
_HERE = Path(__file__).resolve().parent
for _p in [_HERE / "src", _HERE / "../src", _HERE]:
    if (_p / "drone_sim").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from drone_sim.domain.config import ScenarioConfig
from drone_sim.simulation.simulator import Simulator
def all_drones_reached_destination(drones) -> bool:
    """Inline equivalent of drone_sim.domain.utils.helper.all_drones_reached_destination."""
    return all(d.route.target_reached(d.position()) for d in drones)

# Must import these explicitly so their @register_coordinator decorators fire.
# Wrapped in try/except in case your installed package is missing newer modules
# (fix: run  pip install -e .  again inside TLAmpc1/ to refresh the install).
for _mod in [
    "drone_sim.simulation.centralized.coordinator",
    "drone_sim.simulation.centralized.conflict_evasion_coordinator",
    "drone_sim.simulation.distributed.distributed_coordinator",
    "drone_sim.simulation.distributed.conflict_evasion_coordinator",
    "drone_sim.simulation.distributed.threaded_coordinator",
]:
    try:
        __import__(_mod)
    except ModuleNotFoundError:
        pass  # not available in this install — skip silently

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tpt_video")

# ── Defaults ──────────────────────────────────────────────────────────────────
BOF_URL           = "http://localhost:5003"
DEFAULT_CONFIG    = "configs/conflict_evasion/Converging3DroneCentral.json"
DEFAULT_OBJ       = "drone_costum_0_0_5.obj"
DEFAULT_STEPS     = 300
DEFAULT_FPS       = 15
DEFAULT_OUTPUT    = "tpt_analysis.mp4"
DEFAULT_TRACE_LEN = 300
TPT_EVERY         = 5      # request TPTs from BOF every N steps
TUBE_ALPHA        = 0.80   # opacity of prediction line
SPHERE_ALPHA      = 0.12   # opacity of uncertainty spheres


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post(url, payload=None, timeout=10.0):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _get(url, timeout=30.0):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── BOF helpers ───────────────────────────────────────────────────────────────

def push_positions(sim: Simulator, id_map: dict[str, int], sim_t: float):
    positions = []
    for d in sim.drones:
        int_id = id_map[d.drone_id]
        p = d.position()
        positions.append({
            "drone_id": int_id,
            "time":     float(sim_t),
            "pos_x":    float(p[0]),
            "pos_y":    float(p[1]),
            "pos_z":    float(p[2]),
        })
    _post(f"{BOF_URL}/api/positions", positions)


def fetch_tpts(id_map: dict[str, int]) -> dict[int, dict]:
    ids_csv = ",".join(str(i) for i in id_map.values())
    resp = _get(f"{BOF_URL}/api/tpt_batch?drone_ids={ids_csv}", timeout=30.0)
    if not resp or resp.get("status") != "OK":
        return {}
    return {
        int(k): v
        for k, v in resp["results"].items()
        if v.get("status") == "OK"
    }


# ── Frame renderer ────────────────────────────────────────────────────────────

def render_frame(
    sim: Simulator,
    tpts: dict[int, dict],
    id_map: dict[str, int],
    obj_path: str | None,
    trace_len: int,
    width: int, height: int, dpi: int,
    elev: float, azim: float,
) -> bytes:
    """Render one frame: base simulation + TPT overlays."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from drone_sim.api.utils.render_helper import (
        draw_room_wireframe, draw_obstacles,
        draw_trace, draw_sphere_wireframe,
        draw_ghost_max_sphere, draw_obj_mesh,
    )

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    # ── Room & obstacles ──────────────────────────────────────────────────────
    draw_room_wireframe(ax, sim.room_min, sim.room_max)
    draw_obstacles(ax, sim.obstacles)

    # ── Per-drone: body, safety zone, trace, TPT tube ─────────────────────────
    traces = {
        d.drone_id: sim.traces.get(d.drone_id, [])[-trace_len:]
        for d in sim.drones
    }

    for i, drone in enumerate(sim.drones):
        pos   = np.asarray(drone.position(), dtype=float)
        vel   = drone.velocity()
        s_rad = float(drone.compute_adaptive_radius(vel))

        alpha_sphere = 0.8
        if drone.is_adaptive and drone.v_max > 0:
            speed_ratio  = min(float(np.linalg.norm(vel)) / drone.v_max, 1.0)
            alpha_sphere = 0.3 + 0.7 * speed_ratio

        # Drone 3D model or fallback scatter
        if obj_path:
            draw_obj_mesh(ax, pos, obj_path, scale=drone.radius * 0.5 * 10,
                          color=drone.color, alpha=0.85)
        else:
            ax.scatter([pos[0]], [pos[1]], [pos[2]],
                       s=max(20.0, drone.radius * 300.0),
                       c=[drone.color], depthshade=True, label=drone.drone_id)

        # Safety zone
        draw_sphere_wireframe(ax, pos, radius=s_rad,
                              color=drone.safety_color, alpha=alpha_sphere, lw=0.6)

        # Historical trace
        draw_trace(ax, traces.get(drone.drone_id, []), drone.trace_color)

        # ── TPT overlay ───────────────────────────────────────────────────────
        int_id = id_map.get(drone.drone_id)
        tpt    = tpts.get(int_id) if int_id is not None else None

        if tpt:
            center = tpt.get("center", [])
            bounds = tpt.get("bounds", [])

            if center:
                cx = [p["x"] for p in center]
                cy = [p["y"] for p in center]
                cz = [p["z"] for p in center]

                # Dashed prediction centre line
                ax.plot(cx, cy, cz,
                        color=drone.trace_color,
                        linewidth=2.0,
                        linestyle="--",
                        alpha=TUBE_ALPHA,
                        label=f"{drone.drone_id} TPT")

                # Uncertainty spheres at sparse intervals
                step_sz = max(1, len(bounds) // 8)
                for b in bounds[::step_sz]:
                    mx = (b["x_low"] + b["x_high"]) / 2
                    my = (b["y_low"] + b["y_high"]) / 2
                    mz = (b["z_low"] + b["z_high"]) / 2
                    rx = (b["x_high"] - b["x_low"]) / 2
                    ry = (b["y_high"] - b["y_low"]) / 2
                    rz = (b["z_high"] - b["z_low"]) / 2
                    avg_r = (rx + ry + rz) / 3
                    if avg_r > 0.005:
                        draw_sphere_wireframe(
                            ax, np.array([mx, my, mz]),
                            radius=avg_r,
                            color=drone.trace_color,
                            alpha=SPHERE_ALPHA, lw=0.4,
                        )

    # ── Axes, title ───────────────────────────────────────────────────────────
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(float(sim.room_min[0]), float(sim.room_max[0]))
    ax.set_ylim(float(sim.room_min[1]), float(sim.room_max[1]))
    ax.set_zlim(float(sim.room_min[2]), float(sim.room_max[2]))

    box = np.asarray(sim.room_max) - np.asarray(sim.room_min)
    if np.all(np.isfinite(box)) and np.all(box > 0):
        ax.set_box_aspect((float(box[0]), float(box[1]), float(box[2])))

    ax.view_init(elev=float(elev), azim=float(azim))

    n_tpt     = len(tpts)
    n_drones  = len(sim.drones)
    tpt_label = (f"  |  TPT: {n_tpt}/{n_drones} drones"
                 if n_tpt else "  |  TPT: buffering...")
    ax.set_title(f"Step: {sim.step_count}  |  t: {sim.t:.1f}s{tpt_label}", fontsize=9)

    if sim.drones:
        ax.legend(loc="upper right", fontsize=6, ncol=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")

    import matplotlib.pyplot as plt
    plt.close(fig)

    return buf.getvalue()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global BOF_URL
    ap = argparse.ArgumentParser(
        description="Drone simulation MP4 with live BOF TPT overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config",    default=DEFAULT_CONFIG,  help="Scenario JSON path (relative to TLAmpc1/)")
    ap.add_argument("--obj-name",  default=DEFAULT_OBJ,     help="OBJ drone model filename (inside resources/assets/)")
    ap.add_argument("--steps",     type=int,   default=DEFAULT_STEPS,  help="Max simulation steps")
    ap.add_argument("--trace-len", type=int,   default=DEFAULT_TRACE_LEN, help="Trace history length")
    ap.add_argument("--fps",       type=float, default=DEFAULT_FPS,    help="Output video FPS")
    ap.add_argument("--output",    default=DEFAULT_OUTPUT,             help="Output MP4 path")
    ap.add_argument("--bof",       default=BOF_URL,                    help="BOF server URL")
    ap.add_argument("--width",     type=int,   default=900)
    ap.add_argument("--height",    type=int,   default=700)
    ap.add_argument("--dpi",       type=int,   default=100)
    ap.add_argument("--elev",      type=float, default=20.0)
    ap.add_argument("--azim",      type=float, default=-60.0)
    args = ap.parse_args()

    BOF_URL = args.bof

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not shutil.which("ffmpeg"):
        log.error("ffmpeg not found. Install with:  brew install ffmpeg")
        sys.exit(1)

    health = _get(f"{BOF_URL}/health", timeout=5.0)
    if health is None:
        log.error("BOF server not responding at %s\n"
                  "Start it:  cd BoF/.../python_scripts && python run_server.py", BOF_URL)
        sys.exit(1)
    log.info("BOF server OK: %s", health)

    # ── Load scenario ─────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _HERE / config_path
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    import json
    with open(config_path) as f:
        cfg_json = json.load(f)
    cfg = ScenarioConfig.model_validate(cfg_json)
    sim = Simulator.from_config(cfg)

    # OBJ model path — same logic as tools/live_view.py
    obj_path = None
    if args.obj_name:
        obj_path = str(_HERE / "src" / "drone_sim" / "resources" / "assets" / args.obj_name)
        if not Path(obj_path).exists():
            log.warning("OBJ model not found at %s — rendering without 3D model", obj_path)
            obj_path = None

    id_map = {d.drone_id: i for i, d in enumerate(sim.drones)}
    log.info("Loaded: %s  (%d drones)", config_path.name, len(sim.drones))

    # ── Reset BOF ─────────────────────────────────────────────────────────────
    _post(f"{BOF_URL}/api/reset_all", {})
    log.info("BOF history reset.")

    # ── Render loop ───────────────────────────────────────────────────────────
    frame_dir  = Path(tempfile.mkdtemp(prefix="tpt_frames_"))
    frame_idx  = 0
    current_tpts: dict[int, dict] = {}

    log.info("Rendering up to %d steps → %s", args.steps, frame_dir)

    for step in range(1, args.steps + 1):

        # Step simulation (same as live_view.py)
        sim.step()
        if sim.infeasible:
            log.warning("Infeasible at step %d (%s) — stopping.", step, sim.infeasible_reason)
            break

        # Push positions to BOF
        push_positions(sim, id_map, sim.t)

        # Fetch TPTs every N steps
        if step % TPT_EVERY == 0:
            new_tpts = fetch_tpts(id_map)
            if new_tpts:
                current_tpts = new_tpts

        # Render and save frame
        png = render_frame(
            sim, current_tpts, id_map, obj_path,
            trace_len=args.trace_len,
            width=args.width, height=args.height, dpi=args.dpi,
            elev=args.elev, azim=args.azim,
        )
        (frame_dir / f"frame_{frame_idx:05d}.png").write_bytes(png)
        frame_idx += 1

        if step % 25 == 0:
            log.info("Step %3d/%d | t=%5.1fs | TPTs: %d/%d drones",
                     step, args.steps, sim.t, len(current_tpts), len(sim.drones))

        # Stop when all drones reach their target
        if all_drones_reached_destination(sim.drones):
            log.info("All drones reached target at step %d.", step)
            break

    # ── Assemble MP4 ──────────────────────────────────────────────────────────
    log.info("Assembling MP4 from %d frames ...", frame_idx)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(args.fps)),
        "-i", str(frame_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(frame_dir, ignore_errors=True)

    if result.returncode != 0:
        log.error("ffmpeg failed:\n%s", result.stderr[-1000:])
        sys.exit(1)

    log.info("Done!  ->  %s  (%.1f s at %g fps)", output.resolve(),
             frame_idx / args.fps, args.fps)


if __name__ == "__main__":
    main()