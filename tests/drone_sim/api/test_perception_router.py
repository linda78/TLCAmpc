"""Tests for drone_sim.api.perception_router — the REST seam the external video detector drives.

The router reaches the perception parts by ``getattr`` on the simulation, so these tests use a stand-in object carrying exactly those attributes
instead of a full ``Simulator``. That is the whole contract: whatever the simulator wiring installs under ``_perception_view_store`` /
``_perception_mailbox`` / ``_camera_expose_truth`` is what the endpoints serve.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from drone_sim.api.perception_router import build_perception_router
from drone_sim.perception.camera import CameraView, VisibleDrone
from drone_sim.perception.mailbox import PerceptionMailbox
from drone_sim.perception.view_store import CameraViewStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n-not-really-a-png-but-bytes-are-bytes"


class FakeSim:
   """Stand-in for the attributes the router duck-types off the simulation."""

   def __init__(self, *, view_store=None, mailbox=None, expose_truth=False) -> None:
      self._perception_view_store = view_store
      self._perception_mailbox = mailbox
      self._camera_expose_truth = expose_truth


def make_view(observer_id: str = "d1", *, step: int = 42, sim_time: float = 4.2, image: bytes | None = PNG_BYTES,
              visible: list[VisibleDrone] | None = None) -> CameraView:
   return CameraView(observer_id=observer_id, step=step, sim_time=sim_time, position=np.array([1.0, 2.0, 3.0]),
                     view_dir=np.array([1.0, 0.0, 0.0]), fov_deg=90.0, range_m=10.0,
                     visible=visible if visible is not None else [], image_png=image)


def make_client(sim=None) -> TestClient:
   """Mount the router on a bare app whose resolver hands out ``sim``."""
   app = FastAPI()
   app.include_router(build_perception_router(lambda: sim))
   return TestClient(app)


@pytest.fixture
def store() -> CameraViewStore:
   return CameraViewStore()


@pytest.fixture
def mailbox() -> PerceptionMailbox:
   return PerceptionMailbox()


# =============================================================================
# GET /perception/view/{drone_id}
# =============================================================================
class TestGetView:
   """Tests for the image pull."""

   def test_no_simulation_returns_409(self) -> None:
      response = make_client(None).get("/perception/view/d1")

      assert response.status_code == 409
      assert "No simulation running" in response.json()["detail"]

   def test_perception_inactive_returns_409(self) -> None:
      """A running simulation without the REST perception path is a config state, not a missing resource."""
      response = make_client(FakeSim()).get("/perception/view/d1")

      assert response.status_code == 409
      assert "view store" in response.json()["detail"].lower()

   def test_unknown_observer_returns_404(self, store: CameraViewStore) -> None:
      store.put(make_view("d1"))
      response = make_client(FakeSim(view_store=store)).get("/perception/view/d99")

      assert response.status_code == 404
      assert "d99" in response.json()["detail"]

   def test_image_survives_base64_roundtrip(self, store: CameraViewStore) -> None:
      store.put(make_view("d1", image=PNG_BYTES))
      response = make_client(FakeSim(view_store=store)).get("/perception/view/d1")

      assert response.status_code == 200
      assert base64.b64decode(response.json()["image_png_base64"]) == PNG_BYTES

   def test_capture_token_is_echoed_from_the_view(self, store: CameraViewStore) -> None:
      """R10: the token the detector must send back originates here."""
      store.put(make_view("d1", step=7, sim_time=0.7))
      data = make_client(FakeSim(view_store=store)).get("/perception/view/d1").json()

      assert data["observer_id"] == "d1"
      assert data["captured_step"] == 7
      assert data["captured_time"] == pytest.approx(0.7)

   def test_visible_is_null_without_truth_flag(self, store: CameraViewStore) -> None:
      store.put(make_view("d1", visible=[VisibleDrone("d2", np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]), 0.2)]))
      data = make_client(FakeSim(view_store=store, expose_truth=False)).get("/perception/view/d1").json()

      assert data["visible"] is None

   def test_visible_is_populated_with_truth_flag(self, store: CameraViewStore) -> None:
      store.put(make_view("d1", visible=[VisibleDrone("d2", np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.0, 0.0]), 0.25)]))
      data = make_client(FakeSim(view_store=store, expose_truth=True)).get("/perception/view/d1").json()

      assert data["visible"] == [{"drone_id": "d2", "position": [1.0, 2.0, 3.0], "radius": 0.25}]

   def test_visible_never_carries_velocity(self, store: CameraViewStore) -> None:
      """Handing over velocity would skip the very step the bridge derives by finite difference."""
      store.put(make_view("d1", visible=[VisibleDrone("d2", np.array([1.0, 0.0, 0.0]), np.array([9.0, 9.0, 9.0]), 0.2)]))
      data = make_client(FakeSim(view_store=store, expose_truth=True)).get("/perception/view/d1").json()

      assert "velocity" not in data["visible"][0]

   def test_view_without_image_returns_409(self, store: CameraViewStore) -> None:
      store.put(make_view("d1", image=None))
      response = make_client(FakeSim(view_store=store)).get("/perception/view/d1")

      assert response.status_code == 409
      assert "camera_render_images" in response.json()["detail"]

   def test_repeated_pull_returns_the_same_capture(self, store: CameraViewStore) -> None:
      """The store is a slot, not a queue — reading does not consume."""
      store.put(make_view("d1", step=3))
      client = make_client(FakeSim(view_store=store))

      assert client.get("/perception/view/d1").json() == client.get("/perception/view/d1").json()

   def test_pull_follows_the_store_after_a_new_capture(self, store: CameraViewStore) -> None:
      sim = FakeSim(view_store=store)
      client = make_client(sim)
      store.put(make_view("d1", step=1))
      assert client.get("/perception/view/d1").json()["captured_step"] == 1

      store.put(make_view("d1", step=2))
      assert client.get("/perception/view/d1").json()["captured_step"] == 2


# =============================================================================
# GET /perception/views
# =============================================================================
class TestListViews:
   """Tests for the cheap index poll."""

   def test_no_simulation_returns_409(self) -> None:
      assert make_client(None).get("/perception/views").status_code == 409

   def test_empty_store_returns_empty_list(self, store: CameraViewStore) -> None:
      response = make_client(FakeSim(view_store=store)).get("/perception/views")

      assert response.status_code == 200
      assert response.json()["views"] == []

   def test_lists_one_entry_per_observer_with_capture_token(self, store: CameraViewStore) -> None:
      store.put(make_view("d1", step=5, sim_time=0.5))
      store.put(make_view("d2", step=5, sim_time=0.5))
      views = make_client(FakeSim(view_store=store)).get("/perception/views").json()["views"]

      assert {v["observer_id"] for v in views} == {"d1", "d2"}
      assert all(v["captured_step"] == 5 for v in views)

   def test_index_never_carries_image_data(self, store: CameraViewStore) -> None:
      """The point of this endpoint is polling without paying for N base64 PNGs."""
      store.put(make_view("d1"))
      body = make_client(FakeSim(view_store=store)).get("/perception/views").text

      assert "image" not in body
      assert base64.b64encode(PNG_BYTES).decode("ascii") not in body


# =============================================================================
# POST /perception/estimates
# =============================================================================
class TestPostEstimates:
   """Tests for the estimate push."""

   def test_no_simulation_returns_409(self) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1, "estimates": []}
      assert make_client(None).post("/perception/estimates", json=payload).status_code == 409

   def test_perception_inactive_returns_409(self) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1, "estimates": []}
      response = make_client(FakeSim()).post("/perception/estimates", json=payload)

      assert response.status_code == 409
      assert "mailbox" in response.json()["detail"].lower()

   def test_estimate_lands_in_the_mailbox(self, mailbox: PerceptionMailbox) -> None:
      payload = {"observer_id": "d1", "captured_step": 42, "captured_time": 4.2,
                 "estimates": [{"drone_id": "d2", "position": [1.0, 2.0, 3.0], "sigma": 0.1}]}
      response = make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload)

      assert response.status_code == 200
      assert response.json() == {"status": "accepted", "accepted": 1}
      estimate = mailbox.latest("d1")["d2"]
      assert estimate.observer_id == "d1"
      assert estimate.observed_id == "d2"
      np.testing.assert_allclose(estimate.position, [1.0, 2.0, 3.0])
      assert estimate.sigma == pytest.approx(0.1)

   def test_capture_token_is_stored_verbatim(self, mailbox: PerceptionMailbox) -> None:
      """R10: without the echoed token the simulation time axis silently breaks."""
      payload = {"observer_id": "d1", "captured_step": 42, "captured_time": 4.2,
                 "estimates": [{"drone_id": "d2", "position": [1.0, 2.0, 3.0]}]}
      make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload)

      estimate = mailbox.latest("d1")["d2"]
      assert estimate.captured_step == 42
      assert estimate.captured_time == pytest.approx(4.2)

   def test_received_time_is_stamped_by_the_mailbox(self, mailbox: PerceptionMailbox) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1,
                 "estimates": [{"drone_id": "d2", "position": [0.0, 0.0, 0.0]}]}
      make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload)

      assert mailbox.latest("d1")["d2"].received_time > 0.0

   def test_missing_sigma_becomes_none(self, mailbox: PerceptionMailbox) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1,
                 "estimates": [{"drone_id": "d2", "position": [0.0, 0.0, 0.0]}]}
      make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload)

      assert mailbox.latest("d1")["d2"].sigma is None

   def test_several_estimates_in_one_push(self, mailbox: PerceptionMailbox) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1,
                 "estimates": [{"drone_id": "d2", "position": [1.0, 0.0, 0.0]}, {"drone_id": "d3", "position": [0.0, 1.0, 0.0]}]}
      response = make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload)

      assert response.json()["accepted"] == 2
      assert set(mailbox.latest("d1")) == {"d2", "d3"}

   def test_consecutive_pushes_build_history(self, mailbox: PerceptionMailbox) -> None:
      """Two entries with different capture times is what the bridge's finite difference needs."""
      client = make_client(FakeSim(mailbox=mailbox))
      for step, x in ((1, 0.0), (2, 1.0)):
         client.post("/perception/estimates", json={"observer_id": "d1", "captured_step": step, "captured_time": step * 0.1,
                                                    "estimates": [{"drone_id": "d2", "position": [x, 0.0, 0.0]}]})

      history = mailbox.history("d1", "d2")
      assert [e.captured_step for e in history] == [1, 2]

   def test_empty_batch_is_accepted(self, mailbox: PerceptionMailbox) -> None:
      """A detector that sees nothing must be able to say so; upsert semantics mean nothing is deleted."""
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1, "estimates": []}
      response = make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload)

      assert response.status_code == 200
      assert response.json()["accepted"] == 0
      assert mailbox.latest("d1") == {}

   def test_push_without_prior_capture_is_accepted(self, mailbox: PerceptionMailbox) -> None:
      """The detector is allowed to lag behind the simulation."""
      payload = {"observer_id": "d7", "captured_step": 1, "captured_time": 0.1,
                 "estimates": [{"drone_id": "d2", "position": [0.0, 0.0, 0.0]}]}
      response = make_client(FakeSim(view_store=CameraViewStore(), mailbox=mailbox)).post("/perception/estimates", json=payload)

      assert response.status_code == 200
      assert "d2" in mailbox.latest("d7")

   def test_unknown_drone_id_is_passed_through(self, mailbox: PerceptionMailbox) -> None:
      """Transport, not tracker: a false positive must reach whoever debugs the detector."""
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1,
                 "estimates": [{"drone_id": "ghost", "position": [0.0, 0.0, 0.0]}]}
      make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload)

      assert "ghost" in mailbox.latest("d1")

   def test_short_position_returns_422(self, mailbox: PerceptionMailbox) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1, "estimates": [{"drone_id": "d2", "position": [1.0, 2.0]}]}
      response = make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload)

      assert response.status_code == 422
      assert mailbox.latest("d1") == {}

   def test_long_position_returns_422(self, mailbox: PerceptionMailbox) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1,
                 "estimates": [{"drone_id": "d2", "position": [1.0, 2.0, 3.0, 4.0]}]}
      assert make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload).status_code == 422

   def test_missing_captured_step_returns_422(self, mailbox: PerceptionMailbox) -> None:
      """R10 again, from the other side: the token is required, not defaulted."""
      payload = {"observer_id": "d1", "captured_time": 0.1, "estimates": [{"drone_id": "d2", "position": [0.0, 0.0, 0.0]}]}
      assert make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload).status_code == 422

   def test_missing_captured_time_returns_422(self, mailbox: PerceptionMailbox) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "estimates": [{"drone_id": "d2", "position": [0.0, 0.0, 0.0]}]}
      assert make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload).status_code == 422

   def test_missing_observer_id_returns_422(self, mailbox: PerceptionMailbox) -> None:
      payload = {"captured_step": 1, "captured_time": 0.1, "estimates": [{"drone_id": "d2", "position": [0.0, 0.0, 0.0]}]}
      assert make_client(FakeSim(mailbox=mailbox)).post("/perception/estimates", json=payload).status_code == 422


# =============================================================================
# Round trip and host wiring
# =============================================================================
class TestRoundTrip:
   """Pull and push together, the way the detector uses them."""

   def test_pulled_token_pushed_back_reaches_the_mailbox(self, store: CameraViewStore, mailbox: PerceptionMailbox) -> None:
      store.put(make_view("d1", step=11, sim_time=1.1))
      client = make_client(FakeSim(view_store=store, mailbox=mailbox))

      pulled = client.get("/perception/view/d1").json()
      client.post("/perception/estimates", json={"observer_id": pulled["observer_id"], "captured_step": pulled["captured_step"],
                                                 "captured_time": pulled["captured_time"],
                                                 "estimates": [{"drone_id": "d2", "position": [1.0, 1.0, 1.0]}]})

      estimate = mailbox.latest("d1")["d2"]
      assert (estimate.captured_step, estimate.captured_time) == (11, pytest.approx(1.1))


class TestAppWiring:
   """The router is mounted on the REST app (drone_sim.api.app)."""

   def test_views_endpoint_is_registered(self, client) -> None:
      assert client.get("/perception/views").status_code == 409

   def test_view_endpoint_is_registered(self, client) -> None:
      assert client.get("/perception/view/d1").status_code == 409

   def test_estimates_endpoint_is_registered(self, client) -> None:
      payload = {"observer_id": "d1", "captured_step": 1, "captured_time": 0.1, "estimates": []}
      assert client.post("/perception/estimates", json=payload).status_code == 409

   def test_endpoints_appear_in_openapi(self, client) -> None:
      paths = client.get("/openapi.json").json()["paths"]

      assert {"/perception/views", "/perception/view/{drone_id}", "/perception/estimates"} <= set(paths)

   def test_existing_endpoints_are_untouched(self, client) -> None:
      assert client.get("/health").status_code == 200

   def test_router_follows_the_configured_simulation(self, configured_client) -> None:
      """With a scenario loaded but perception off, the answer changes from "no simulation" to "not active"."""
      response = configured_client.get("/perception/views")

      assert response.status_code == 409
      assert "No simulation running" not in response.json()["detail"]
