"""Tests for drone_sim.gui.perception_server — the second host of the perception router.

These start a real uvicorn thread on an ephemeral port; the router itself is covered by
``tests/drone_sim/api/test_perception_router.py``. What is tested here is the lifecycle around it: start/stop
idempotency, no leaked threads, a failed bind that stays visible, and the resolver following the *current*
simulation across a reset.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest

from drone_sim.gui.direct_backend import DirectBackend
from drone_sim.gui.perception_server import PerceptionApiServer
from drone_sim.perception.camera import CameraView
from drone_sim.perception.view_store import CameraViewStore

import numpy as np

REST_CONFIG = {
    "dt": 0.1,
    "physics": {"type": "linear_kinematics", "v_max": 2.0, "u_min": -1.0, "u_max": 1.0},
    "controller": {"type": "mpc_agent"},
    "coordinator": {"type": "mpc_central"},
    "drones": [{"drone_id": "d1", "start": [0.0, 0.0, 5.0], "target": [5.0, 5.0, 5.0]}],
    "room": {"min": [-10.0, -10.0, 0.0], "max": [10.0, 10.0, 10.0]},
    "camera_enabled": True,
    "camera_backend": "rest",
    "camera_render_images": True,
}


def free_port() -> int:
    """An unused TCP port on loopback. Racy in principle, fine for a single test process."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeSim:
    """Only what the router duck-types off the simulation."""

    def __init__(self, view_store=None) -> None:
        self._perception_view_store = view_store
        self._perception_mailbox = None
        self._camera_expose_truth = False


@pytest.fixture
def server():
    """A stopped-on-teardown server on a free port, resolving to whatever the test puts in ``holder``."""
    holder: dict[str, object] = {"sim": None}
    srv = PerceptionApiServer(lambda: holder["sim"], port=free_port())
    srv.holder = holder  # test-only handle for swapping the simulation
    yield srv
    srv.stop()


def get(server: PerceptionApiServer, path: str) -> httpx.Response:
    return httpx.get(f"http://{server.host}:{server.port}{path}", timeout=5.0)


# --- lifecycle ---

def test_not_running_before_start(server: PerceptionApiServer) -> None:
    assert server.is_running() is False


def test_start_makes_it_run(server: PerceptionApiServer) -> None:
    server.start()
    assert server.wait_started(timeout=5.0) is True
    assert server.is_running() is True


def test_start_is_idempotent(server: PerceptionApiServer) -> None:
    server.start()
    server.wait_started(timeout=5.0)
    before = [t for t in threading.enumerate() if t.name.startswith("perception-api-")]
    server.start()
    after = [t for t in threading.enumerate() if t.name.startswith("perception-api-")]

    assert len(after) == len(before) == 1


def test_stop_joins_the_thread(server: PerceptionApiServer) -> None:
    server.start()
    server.wait_started(timeout=5.0)
    server.stop()

    assert server.is_running() is False


def test_stop_without_start_is_a_noop(server: PerceptionApiServer) -> None:
    server.stop()  # must not raise
    assert server.is_running() is False


def test_stop_is_idempotent(server: PerceptionApiServer) -> None:
    server.start()
    server.wait_started(timeout=5.0)
    server.stop()
    server.stop()  # must not raise

    assert server.is_running() is False


def test_start_stop_leaves_no_thread_behind(server: PerceptionApiServer) -> None:
    """R9: a GUI session that opens and closes the server must not accumulate threads."""
    before = len(threading.enumerate())
    server.start()
    server.wait_started(timeout=5.0)
    server.stop()

    assert len(threading.enumerate()) <= before


def test_restart_after_stop_works(server: PerceptionApiServer) -> None:
    server.start()
    server.wait_started(timeout=5.0)
    server.stop()
    server.start()

    assert server.wait_started(timeout=5.0) is True


# --- serving ---

def test_serves_the_perception_router(server: PerceptionApiServer) -> None:
    server.start()
    assert server.wait_started(timeout=5.0)

    response = get(server, "/perception/views")

    assert response.status_code == 409  # no simulation behind the resolver yet


def test_resolver_is_read_per_request(server: PerceptionApiServer) -> None:
    """The whole point of the indirection: a simulation swapped in after start is served without a restart."""
    server.start()
    assert server.wait_started(timeout=5.0)
    assert get(server, "/perception/views").status_code == 409

    store = CameraViewStore()
    store.put(CameraView(observer_id="d1", step=3, sim_time=0.3, position=np.zeros(3), view_dir=np.array([1.0, 0.0, 0.0]),
                         fov_deg=90.0, range_m=10.0, image_png=b"png"))
    server.holder["sim"] = FakeSim(view_store=store)

    response = get(server, "/perception/views")

    assert response.status_code == 200
    assert response.json()["views"] == [{"observer_id": "d1", "captured_step": 3, "captured_time": 0.3}]


def test_binds_loopback_by_default(server: PerceptionApiServer) -> None:
    assert server.host == "127.0.0.1"


# --- failure modes ---

def test_occupied_port_is_reported_not_swallowed(caplog) -> None:
    """R12: uvicorn exits the thread via SystemExit on a bind failure — that must not look like a healthy server."""
    port = free_port()
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)

    srv = PerceptionApiServer(lambda: None, port=port)
    try:
        with caplog.at_level(logging.ERROR, logger="drone_sim.gui.perception_server"):
            srv.start()
            started = srv.wait_started(timeout=5.0)

        assert started is False
        assert any("already in use" in r.message or "could not bind" in r.message for r in caplog.records)
    finally:
        srv.stop()
        blocker.close()


# --- DirectBackend wiring ---

@pytest.fixture
def rest_config_file(tmp_path: Path) -> Path:
    cfg = dict(REST_CONFIG, camera_api_port=free_port())
    p = tmp_path / "rest_camera.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


@pytest.fixture
def plain_config_file(tmp_path: Path) -> Path:
    cfg = {k: v for k, v in REST_CONFIG.items() if not k.startswith("camera_")}
    p = tmp_path / "plain.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_plain_scenario_starts_no_server(plain_config_file: Path) -> None:
    """Every existing scenario stays exactly as it was — no listener, no port taken."""
    backend = DirectBackend()
    try:
        backend.load_config(plain_config_file)
        assert backend._perception_server is None
    finally:
        backend.close()


def test_rest_scenario_starts_a_server(rest_config_file: Path) -> None:
    backend = DirectBackend()
    try:
        backend.load_config(rest_config_file)
        assert backend._perception_server is not None
        assert backend._perception_server.wait_started(timeout=5.0) is True
    finally:
        backend.close()


def test_backend_server_answers_over_http(rest_config_file: Path) -> None:
    """End to end through the GUI host: the simulator's view store is what the detector reaches over HTTP."""
    backend = DirectBackend()
    try:
        backend.load_config(rest_config_file)
        server = backend._perception_server
        assert server.wait_started(timeout=5.0)

        # Loaded but not stepped: the store exists (200, not the pre-wiring 409) and is simply empty.
        response = get(server, "/perception/views")
        assert response.status_code == 200
        assert response.json()["views"] == []

        backend.step()

        # The worker renders on its own thread, so the capture appears shortly after the step, not during it.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not get(server, "/perception/views").json()["views"]:
            time.sleep(0.02)

        assert [v["observer_id"] for v in get(server, "/perception/views").json()["views"]] == ["d1"]
    finally:
        backend.close()


def test_reload_keeps_the_same_server(rest_config_file: Path) -> None:
    backend = DirectBackend()
    try:
        backend.load_config(rest_config_file)
        first = backend._perception_server
        backend.load_config(rest_config_file)

        assert backend._perception_server is first
    finally:
        backend.close()


def test_reset_keeps_the_server(rest_config_file: Path) -> None:
    """A reset replaces the Simulator; the detector connection must survive it."""
    backend = DirectBackend()
    try:
        backend.load_config(rest_config_file)
        server = backend._perception_server
        backend.reset()

        assert backend._perception_server is server
        assert server.is_running() is True
    finally:
        backend.close()


def test_close_stops_the_server(rest_config_file: Path) -> None:
    backend = DirectBackend()
    backend.load_config(rest_config_file)
    server = backend._perception_server
    backend.close()

    assert server.is_running() is False
    assert backend._perception_server is None


def test_close_is_idempotent(rest_config_file: Path) -> None:
    backend = DirectBackend()
    backend.load_config(rest_config_file)
    backend.close()
    backend.close()  # must not raise


def test_close_without_load_config_is_a_noop() -> None:
    DirectBackend().close()  # must not raise


def test_repeated_load_does_not_leak_threads(rest_config_file: Path) -> None:
    """R9, through the backend: reloading a scenario must not stack up uvicorn threads."""
    backend = DirectBackend()
    try:
        for _ in range(3):
            backend.load_config(rest_config_file)
        assert len([t for t in threading.enumerate() if t.name.startswith("perception-api-")]) == 1
    finally:
        backend.close()
