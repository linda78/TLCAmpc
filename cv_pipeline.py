#!/usr/bin/env python3
"""
cv_pipeline.py  —  Step 4: Full Pipeline Integration
=====================================================
Wires the computer-vision drone-detection module to the BOF trajectory
prediction server and exposes the results through its own REST API.

Architecture
------------
  TLCAmpc sim  ──GET /render──►  [CV Pipeline]  ──POST /api/positions──►  BOF server
  TLCAmpc sim  ──GET /state ──►  [CV Pipeline]
                                      │
                              GET /tpt_batch ◄──────────────────────────  BOF server
                                      │
                              ◄── GET /tpts ── external consumer (optimisers)

Detection modes
---------------
  "gt"   – ground-truth: reads world positions directly from GET /state.
            Use this when the CV model is not yet connected or for testing.
  "cv"   – computer-vision: fetches PNG from GET /render, runs your
            drone-detection model, back-projects pixels to world coords,
            then feeds to BOF.  Plug your own detector in detect_drones_cv().

Usage
-----
  # Terminal 1 — TLCAmpc (port 8000)
  cd /Users/brianvisas/TLCAmpc
  uvicorn drone_sim.api.app:app --reload

  # Terminal 2 — BOF server (port 5003)
  cd /Users/brianvisas/BoF/python_scripts
  python run_server.py

  # Terminal 3 — CV pipeline (port 7000)
  python cv_pipeline.py --config /Users/brianvisas/TLCAmpc/configs/conflict_evasion/Converging3DroneCentral.json

  # Then poll TPTs:
  curl "http://localhost:7000/tpts"
  curl "http://localhost:7000/tpts?drone_ids=0,1,2"

CLI flags
---------
  --sim-url     TLCAmpc base URL            (default: http://localhost:8000)
  --bof-url     BOF server base URL         (default: http://localhost:5003)
  --port        This server's port          (default: 7000)
  --config      Path to scenario JSON       (required)
  --steps       Total sim steps to run      (default: 0 = run forever)
  --poll-hz     How often to poll sim (Hz)  (default: 10)
  --mode        Detection mode: gt | cv     (default: gt)
  --log         JSONL log file path         (default: cv_pipeline_log.jsonl)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cv_pipeline")

# ---------------------------------------------------------------------------
# Runtime config (filled in main())
# ---------------------------------------------------------------------------
SIM_URL: str = "http://localhost:8000"
BOF_URL: str = "http://localhost:5003"
POLL_HZ: float = 10.0
DETECTION_MODE: str = "gt"          # "gt" or "cv"
MAX_STEPS: int = 0                  # 0 = run forever

# ---------------------------------------------------------------------------
# Shared state (written by background loop, read by API)
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_latest_tpts: Dict = {}             # drone_id_str -> tpt dict
_latest_detections: List[Dict] = [] # list of {drone_id, t, x, y, z, source}
_step_count: int = 0
_running: bool = False

# ---------------------------------------------------------------------------
# Camera / projection helpers
# ---------------------------------------------------------------------------

def build_projection_matrix(
    room_min: List[float],
    room_max: List[float],
    img_w: int = 900,
    img_h: int = 700,
    elev: float = 20.0,
    azim: float = -60.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build an approximate camera projection from matplotlib's 3-D view params.

    This constructs the same view-projection matrix that matplotlib uses so
    we can back-project detected pixel centroids to world coordinates.

    Returns
    -------
    P   : (3, 4) projection matrix  (homogeneous)
    Pinv: (4, 3) pseudo-inverse     (for back-projection)
    meta: dict with room bounds and image size
    """
    room_min_np = np.array(room_min, dtype=float)
    room_max_np = np.array(room_max, dtype=float)
    room_center = (room_min_np + room_max_np) / 2.0
    room_size = np.max(room_max_np - room_min_np)

    # Build look-at matrix (same convention as ax.view_init)
    elev_r = np.radians(elev)
    azim_r = np.radians(azim)

    # Camera position on a sphere around the room centre
    eye = room_center + room_size * 1.8 * np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r),
    ])

    # Build orthonormal camera frame
    fwd = room_center - eye
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up_true = np.cross(right, fwd)

    # 4×4 view matrix
    V = np.eye(4)
    V[0, :3] = right
    V[1, :3] = up_true
    V[2, :3] = -fwd
    V[0, 3] = -right @ eye
    V[1, 3] = -up_true @ eye
    V[2, 3] = fwd @ eye

    # Simple orthographic projection into [-1,1]^2
    scale = 1.0 / (room_size * 0.7)
    Proj = np.array([
        [scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0,     0, 1],   # discard depth; keep homogeneous
    ], dtype=float)

    # Viewport to pixel coords
    Vp = np.array([
        [img_w / 2,        0, img_w / 2],
        [       0, -img_h / 2, img_h / 2],
        [       0,         0,         1],
    ], dtype=float)

    # Full 3×4 matrix: pixel = Vp @ Proj @ V @ [X,Y,Z,1]^T
    P = Vp @ Proj @ V  # (3, 4)
    Pinv = np.linalg.pinv(P)  # (4, 3)

    meta = {
        "room_min": room_min,
        "room_max": room_max,
        "img_w": img_w,
        "img_h": img_h,
        "elev": elev,
        "azim": azim,
        "room_center": room_center.tolist(),
    }
    return P, Pinv, meta


def backproject_pixel_to_world(
    px: float, py: float,
    Pinv: np.ndarray,
    assumed_z: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Back-project a pixel (px, py) to approximate world (X, Y, Z).

    With only a 2-D detection we lose depth.  Two strategies:
      1. assumed_z: fix Z = assumed_z (e.g. average flight altitude).
      2. Ray-casting: find world point on the Z=assumed_z plane along the ray.

    Returns (world_x, world_y, world_z).
    """
    # Homogeneous pixel
    p_h = np.array([px, py, 1.0])

    # Lift to 4-D homogeneous world direction
    w_h = Pinv @ p_h     # (4,)
    if abs(w_h[3]) > 1e-9:
        world = w_h[:3] / w_h[3]
    else:
        world = w_h[:3]

    if assumed_z is not None:
        world[2] = assumed_z

    return float(world[0]), float(world[1]), float(world[2])


# ---------------------------------------------------------------------------
# Ground-truth detector  (mode="gt")
# ---------------------------------------------------------------------------

def detect_drones_gt(sim_state: Dict, t: float) -> List[Dict]:
    """
    Extract drone positions directly from the simulation state JSON.

    This is the reference implementation for testing—no actual CV is used.
    The positions come from DroneState.x = [px, py, pz, vx, vy, vz].
    """
    detections = []
    for drone in sim_state.get("drones", []):
        raw_id = drone["drone_id"]                  # e.g. "drone_0"
        numeric_id = int(raw_id.replace("-", "_").split("_")[-1])  # drone_0 or drone-0 → 0
        x_vec = drone["x"]                          # [px, py, pz, vx, vy, vz]
        detections.append({
            "drone_id": numeric_id,
            "t": t,
            "pos_x": float(x_vec[0]),
            "pos_y": float(x_vec[1]),
            "pos_z": float(x_vec[2]),
            "source": "gt",
        })
    return detections


# ---------------------------------------------------------------------------
# CV detector  (mode="cv")  — PLUG YOUR MODEL HERE
# ---------------------------------------------------------------------------

def detect_drones_cv(
    png_bytes: bytes,
    sim_state: Dict,
    t: float,
    proj_cache: Dict,
) -> List[Dict]:
    """
    Detect drones in a PNG frame and return world-coordinate positions.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  HOW TO PLUG IN YOUR CV MODEL                                        │
    │                                                                      │
    │  1. Replace the YOLO/detector stub below with your actual model.     │
    │  2. Your detector should return bounding boxes in pixel space:       │
    │       [(label, cx_px, cy_px, w_px, h_px, confidence), ...]          │
    │  3. The back-projection code below converts pixel centroids to       │
    │     approximate world (X, Y, Z) using the same camera matrix that    │
    │     matplotlib used to render the image.                             │
    │  4. If you have a depth estimate from your detector, pass it as      │
    │     assumed_z instead of using the room mid-height default.          │
    └──────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    png_bytes   : raw PNG bytes from GET /render
    sim_state   : latest /state response (for room bounds + fallback)
    t           : current simulation time
    proj_cache  : dict maintained by caller (stores P, Pinv, meta keyed by
                  "matrix")  — rebuilt when room bounds change

    Returns
    -------
    List of detection dicts with keys:
        drone_id, t, pos_x, pos_y, pos_z, source="cv"
    """
    from PIL import Image  # pillow must be installed

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img_w, img_h = img.size

    room = sim_state.get("room", {})
    room_min = room.get("min", [0, 0, 0])
    room_max = room.get("max", [1, 1, 1])

    # (Re)build projection matrix if room bounds changed
    cache_key = f"{room_min}_{room_max}_{img_w}_{img_h}"
    if proj_cache.get("key") != cache_key:
        P, Pinv, meta = build_projection_matrix(
            room_min, room_max, img_w=img_w, img_h=img_h
        )
        proj_cache["P"] = P
        proj_cache["Pinv"] = Pinv
        proj_cache["meta"] = meta
        proj_cache["key"] = cache_key
        log.info("Projection matrix rebuilt for room %s → %s", room_min, room_max)

    Pinv = proj_cache["Pinv"]
    assumed_z = float((room_min[2] + room_max[2]) / 2.0)   # mid-height default

    # ── REPLACE THIS BLOCK WITH YOUR ACTUAL DETECTOR ────────────────────────
    #
    #   Example with a YOLO-style model:
    #
    #   import your_detector
    #   results = your_detector.run(np.array(img))
    #   raw_boxes = [
    #       (r.label, r.cx, r.cy, r.w, r.h, r.conf)
    #       for r in results if r.label == "drone"
    #   ]
    #
    #   For now we fall back to ground-truth to keep the pipeline runnable
    #   even without a trained model:
    raw_boxes = []   # → triggers fallback below
    # ────────────────────────────────────────────────────────────────────────

    if not raw_boxes:
        # Fallback: use ground-truth positions so the pipeline keeps running
        log.debug("CV detector returned no boxes — falling back to GT positions")
        return detect_drones_gt(sim_state, t)

    # Assign detections to drone IDs by nearest existing drone (simple IoU-free
    # nearest-centroid tracker — replace with your tracker if available)
    gt_drones = {
        int(d["drone_id"].replace("-", "_").split("_")[-1]): np.array(d["x"][:3])
        for d in sim_state.get("drones", [])
    }

    detections = []
    for i, (label, cx_px, cy_px, w_px, h_px, conf) in enumerate(raw_boxes):
        wx, wy, wz = backproject_pixel_to_world(cx_px, cy_px, Pinv, assumed_z)

        # Assign to nearest ground-truth drone (simple nearest-neighbour)
        if gt_drones:
            candidate = np.array([wx, wy, wz])
            drone_id = min(
                gt_drones.keys(),
                key=lambda did: float(np.linalg.norm(gt_drones[did] - candidate)),
            )
        else:
            drone_id = i

        detections.append({
            "drone_id": drone_id,
            "t": t,
            "pos_x": wx,
            "pos_y": wy,
            "pos_z": wz,
            "source": "cv",
            "confidence": float(conf),
            "pixel_cx": float(cx_px),
            "pixel_cy": float(cy_px),
        })

    return detections


# ---------------------------------------------------------------------------
# BOF helpers
# ---------------------------------------------------------------------------

def push_positions_to_bof(detections: List[Dict]) -> bool:
    """POST a batch of detected positions to the BOF server."""
    if not detections:
        return True
    payload = [
        {
            "drone_id": d["drone_id"],
            "time":     d["t"],
            "pos_x":    d["pos_x"],
            "pos_y":    d["pos_y"],
            "pos_z":    d["pos_z"],
        }
        for d in detections
    ]
    try:
        r = requests.post(f"{BOF_URL}/api/positions", json=payload, timeout=2.0)
        return r.status_code == 200
    except Exception as e:
        log.warning("BOF push failed: %s", e)
        return False


def fetch_tpts(drone_ids: List[int]) -> Dict:
    """GET TPT batch from the BOF server."""
    if not drone_ids:
        return {}
    ids_str = ",".join(str(i) for i in drone_ids)
    try:
        r = requests.get(f"{BOF_URL}/api/tpt_batch", params={"drone_ids": ids_str}, timeout=3.0)
        if r.status_code == 200:
            return r.json().get("results", {})
    except Exception as e:
        log.warning("BOF TPT fetch failed: %s", e)
    return {}


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def pipeline_loop(config_path: str, log_path: str) -> None:
    """
    Background thread: steps the simulation, detects drones, feeds BOF,
    retrieves TPTs, and updates shared state.
    """
    global _latest_tpts, _latest_detections, _step_count, _running

    log.info("Pipeline starting — config: %s", config_path)

    # ── 1. Load scenario into TLCAmpc ──────────────────────────────────────
    with open(config_path) as f:
        scenario = json.load(f)

    r = requests.post(f"{SIM_URL}/config", json=scenario, timeout=10)
    r.raise_for_status()
    log.info("Scenario loaded: %s", r.json())

    # Reset BOF history
    requests.post(f"{BOF_URL}/api/reset_all", timeout=5)
    log.info("BOF history cleared")

    proj_cache: Dict = {}   # cache for CV projection matrix
    step = 0
    interval = 1.0 / max(POLL_HZ, 0.1)

    with open(log_path, "w") as log_file:
        _running = True

        while _running:
            if MAX_STEPS > 0 and step >= MAX_STEPS:
                log.info("Reached max steps (%d) — stopping loop", MAX_STEPS)
                _running = False
                break

            t_start = time.perf_counter()

            # ── 2. Step simulation ─────────────────────────────────────────
            try:
                step_resp = requests.post(f"{SIM_URL}/step", params={"n": 1}, timeout=3)
                step_data = step_resp.json()
                sim_t = float(step_data.get("t", step * interval))
            except Exception as e:
                log.warning("Sim step failed: %s", e)
                time.sleep(interval)
                continue

            if step_data.get("status") == "infeasible":
                log.warning("Sim reports infeasible at step %d — stopping", step)
                _running = False
                break

            # ── 3. Fetch state + (optionally) frame ────────────────────────
            try:
                state_resp = requests.get(f"{SIM_URL}/state", timeout=3)
                sim_state = state_resp.json()
            except Exception as e:
                log.warning("State fetch failed: %s", e)
                time.sleep(interval)
                continue

            drone_ids = [
                int(d["drone_id"].replace("-", "_").split("_")[-1])
                for d in sim_state.get("drones", [])
            ]

            # ── 4. Detect drones ───────────────────────────────────────────
            if DETECTION_MODE == "cv":
                try:
                    png_resp = requests.get(
                        f"{SIM_URL}/render",
                        params={"width": 900, "height": 700},
                        timeout=5,
                    )
                    png_bytes = png_resp.content
                    detections = detect_drones_cv(png_bytes, sim_state, sim_t, proj_cache)
                except Exception as e:
                    log.warning("CV detection failed: %s — falling back to GT", e)
                    detections = detect_drones_gt(sim_state, sim_t)
            else:
                detections = detect_drones_gt(sim_state, sim_t)

            # ── 5. Push positions to BOF ───────────────────────────────────
            push_ok = push_positions_to_bof(detections)

            # ── 6. Fetch TPTs ──────────────────────────────────────────────
            tpts = fetch_tpts(drone_ids)
            ready_ids = [k for k, v in tpts.items() if v.get("status") == "OK"]

            # ── 7. Update shared state ─────────────────────────────────────
            with _state_lock:
                _latest_tpts = tpts
                _latest_detections = detections
                _step_count = step

            # ── 8. Log ─────────────────────────────────────────────────────
            record = {
                "step": step,
                "sim_t": sim_t,
                "mode": DETECTION_MODE,
                "n_detections": len(detections),
                "bof_push_ok": push_ok,
                "tpt_ready": ready_ids,
                "n_tpt_ready": len(ready_ids),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            if step % 50 == 0:
                log.info(
                    "Step %4d | t=%.2f | detections=%d | TPT ready: %s",
                    step, sim_t, len(detections), ready_ids or "none yet",
                )

            step += 1

            # ── 9. Rate-limit ──────────────────────────────────────────────
            elapsed = time.perf_counter() - t_start
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    log.info("Pipeline loop finished after %d steps", step)


# ---------------------------------------------------------------------------
# FastAPI — exposes TPTs to external consumers (e.g. drone optimisers)
# ---------------------------------------------------------------------------
api = FastAPI(title="CV Pipeline — Drone TPT Service")


@api.get("/health")
def health():
    with _state_lock:
        return {
            "status": "ok",
            "running": _running,
            "step": _step_count,
            "detection_mode": DETECTION_MODE,
            "sim_url": SIM_URL,
            "bof_url": BOF_URL,
        }


@api.get("/tpts")
def get_tpts(drone_ids: Optional[str] = Query(None, description="Comma-separated ids, e.g. 0,1,2")):
    """
    Return the latest Trajectory Prediction Tubes for all (or specified) drones.

    This is the primary endpoint for drone optimisers to consume.
    Response mirrors the BOF /api/tpt_batch format so downstream code
    can be swapped between direct-BOF and pipeline modes transparently.
    """
    with _state_lock:
        tpts = dict(_latest_tpts)

    if drone_ids is not None and drone_ids.strip():
        ids = {s.strip() for s in drone_ids.split(",")}
        tpts = {k: v for k, v in tpts.items() if k in ids}

    n_ready = sum(1 for v in tpts.values() if v.get("status") == "OK")
    return {
        "status": "OK",
        "step": _step_count,
        "detection_mode": DETECTION_MODE,
        "n_drones": len(tpts),
        "n_ready": n_ready,
        "results": tpts,
    }


@api.get("/detections")
def get_detections():
    """Return the latest per-drone detections (world coordinates)."""
    with _state_lock:
        return {
            "status": "OK",
            "step": _step_count,
            "mode": DETECTION_MODE,
            "detections": list(_latest_detections),
        }


@api.get("/tpts/{drone_id}")
def get_tpt_single(drone_id: int):
    """Return the TPT for a single drone."""
    with _state_lock:
        result = _latest_tpts.get(str(drone_id))
    if result is None:
        return JSONResponse(status_code=404, content={"status": "NOT_FOUND", "drone_id": drone_id})
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global SIM_URL, BOF_URL, POLL_HZ, DETECTION_MODE, MAX_STEPS

    parser = argparse.ArgumentParser(description="Step 4 CV Pipeline — wires CV detection to BOF TPT server")
    parser.add_argument("--sim-url",  default="http://localhost:8000", help="TLCAmpc base URL")
    parser.add_argument("--bof-url",  default="http://localhost:5003", help="BOF server base URL")
    parser.add_argument("--port",     type=int, default=7001,          help="This server's port")
    parser.add_argument("--config",   required=True,                   help="Path to scenario JSON config")
    parser.add_argument("--steps",    type=int, default=0,             help="Steps to run (0=forever)")
    parser.add_argument("--poll-hz",  type=float, default=10.0,        help="Polling frequency in Hz")
    parser.add_argument("--mode",     default="gt", choices=["gt", "cv"], help="Detection mode")
    parser.add_argument("--log",      default="cv_pipeline_log.jsonl", help="JSONL log file path")
    args = parser.parse_args()

    SIM_URL         = args.sim_url.rstrip("/")
    BOF_URL         = args.bof_url.rstrip("/")
    POLL_HZ         = args.poll_hz
    DETECTION_MODE  = args.mode
    MAX_STEPS       = args.steps

    log.info("=" * 60)
    log.info("CV Pipeline (Step 4)")
    log.info("  Sim:   %s", SIM_URL)
    log.info("  BOF:   %s", BOF_URL)
    log.info("  Mode:  %s", DETECTION_MODE)
    log.info("  Steps: %s", MAX_STEPS if MAX_STEPS > 0 else "∞")
    log.info("  Port:  %d", args.port)
    log.info("=" * 60)

    # Health checks
    for name, url in [("TLCAmpc", SIM_URL), ("BOF", BOF_URL)]:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            log.info("✓ %s is up: %s", name, r.json())
        except Exception as e:
            log.error("✗ %s not reachable at %s: %s", name, url, e)
            raise SystemExit(1)

    # Start pipeline in background thread
    t = threading.Thread(
        target=pipeline_loop,
        args=(args.config, args.log),
        daemon=True,
        name="pipeline-loop",
    )
    t.start()
    log.info("Pipeline loop started in background thread")

    # Start REST API (blocks)
    log.info("Starting REST API on port %d …", args.port)
    uvicorn.run(api, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()