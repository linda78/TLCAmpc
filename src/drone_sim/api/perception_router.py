"""REST seam of the perception pipeline: TLCAmpc serves, the detector drives.

The (separately developed) video detector is the **client** of both directions — there is no detector service for the simulation to call. It pulls a
rendered camera image per ego drone and pushes the resulting position estimates back:

.. code-block:: text

    for every drone i:
      step k -> CameraModel.capture(i) -> worker renders PNG -> CameraViewStore[i]
                                                                     |  GET  /perception/view/i
                                                                Videodetection
                                                                     |  POST /perception/estimates  (observer_id=i, capture token echoed)
                                                                PerceptionMailbox[i][*] -> bridge -> DMPC inbox of i

**One drone, one image, one prediction.** The detector always talks about exactly one ego drone; nothing is aggregated across the fleet. There is
deliberately no bulk endpoint handing out every image at once — with N drones that would be N base64 PNGs in one answer, and the detector processes
one ego view after another anyway.

Two hosts, one router. The endpoints must work in the REST run (:mod:`drone_sim.api.app`, simulation driven by ``POST /config`` + ``POST /step``) *and*
in the GUI run (:class:`~drone_sim.gui.perception_server.PerceptionApiServer`, simulation driven by the play button). Those are two independent
``Simulator`` owners, so the router binds to neither: :func:`build_perception_router` takes a **resolver function** and asks it for the current
simulation on every request.

The perception parts are read off the simulation by ``getattr`` — the same duck typing the BoF provider uses. The simulator wiring that sets
``_perception_view_store`` / ``_perception_mailbox`` / ``_camera_expose_truth`` is step 8 of the perception plan; until it lands, every endpoint here
answers ``409`` instead of failing at import time.

The push bypasses adapter and worker entirely and writes straight into the :class:`~drone_sim.perception.mailbox.PerceptionMailbox`. That is safe
without extra work — the mailbox is ``RLock``-protected and uvicorn serves requests from a threadpool — and an adapter in between would be a prop.

Status codes:

* ``409`` — no simulation loaded, or perception not active for it (a *configuration* state; retrying the same request does not help until the
  scenario changes),
* ``404`` — the simulation runs, but this observer has no capture waiting yet (transient: retry later),
* ``422`` — malformed payload, e.g. a position of length 2 or a missing capture token.
"""
from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from fastapi import APIRouter, HTTPException

from drone_sim.api.schemas import (PerceptionEstimatesRequest, PerceptionEstimatesResponse, PerceptionViewInfo, PerceptionViewResponse,
                                   PerceptionViewsResponse, PerceptionVisibleDrone)
from drone_sim.perception.mailbox import PositionEstimate

if TYPE_CHECKING:
   from drone_sim.perception.camera import CameraView
   from drone_sim.perception.mailbox import PerceptionMailbox
   from drone_sim.perception.view_store import CameraViewStore
   from drone_sim.simulation.simulator import Simulator

_log = logging.getLogger(__name__)

_NO_SIM = "No simulation running. POST /config first (REST run), or load a scenario (GUI run)."
_NO_VIEW_STORE = "Perception view store not active. Requires camera_enabled with camera_backend='rest' in the scenario config."
_NO_MAILBOX = "Perception mailbox not active. Requires camera_enabled in the scenario config."
_NO_IMAGE = "Capture carries no rendered image. Requires camera_render_images in the scenario config."


def _require_sim(resolve_sim: Callable[[], Simulator | None]) -> Simulator:
   """Current simulation, or ``409``.

   :param resolve_sim: Host-provided accessor for the running simulation.
   :return: The simulation.
   :raises HTTPException: ``409`` when no scenario is loaded.
   """
   sim = resolve_sim()
   if sim is None:
      raise HTTPException(status_code=409, detail=_NO_SIM)
   return sim


def _require_view_store(sim: Simulator) -> CameraViewStore:
   """View store of ``sim``, or ``409``.

   :param sim: Running simulation.
   :return: The store the perception worker drops rendered views into.
   :raises HTTPException: ``409`` when the REST perception path is not wired up for this scenario.
   """
   store = getattr(sim, "_perception_view_store", None)
   if store is None:
      raise HTTPException(status_code=409, detail=_NO_VIEW_STORE)
   return store


def _require_mailbox(sim: Simulator) -> PerceptionMailbox:
   """Perception mailbox of ``sim``, or ``409``.

   :param sim: Running simulation.
   :return: The mailbox the DMPC bridge reads from.
   :raises HTTPException: ``409`` when perception is off for this scenario.
   """
   mailbox = getattr(sim, "_perception_mailbox", None)
   if mailbox is None:
      raise HTTPException(status_code=409, detail=_NO_MAILBOX)
   return mailbox


def _view_payload(view: CameraView, *, expose_truth: bool) -> PerceptionViewResponse:
   """Serialize one capture for the detector.

   The image travels base64-encoded inside the JSON body rather than as raw ``image/png`` so that image and capture token arrive in **one** answer:
   the detector copies ``captured_step`` / ``captured_time`` verbatim into its push instead of parsing headers. The ~33 % encoding overhead on a
   320x240 PNG is irrelevant next to that.

   :param view: Capture taken from the view store.
   :param expose_truth: Whether to attach the ground truth of the visible neighbors.
   :return: The response body.
   :raises HTTPException: ``409`` when the capture carries no rendered image.
   """
   if not view.image_png:
      raise HTTPException(status_code=409, detail=_NO_IMAGE)

   visible = None
   if expose_truth:
      visible = [PerceptionVisibleDrone(drone_id=v.drone_id, position=[float(c) for c in np.asarray(v.position, dtype=float).reshape(3)],
                                        radius=float(v.radius)) for v in view.visible]

   return PerceptionViewResponse(observer_id=view.observer_id, captured_step=int(view.step), captured_time=float(view.sim_time),
                                 image_png_base64=base64.b64encode(view.image_png).decode("ascii"), visible=visible)


def build_perception_router(resolve_sim: Callable[[], Simulator | None]) -> APIRouter:
   """Build the ``/perception`` router bound to one host's simulation accessor.

   :param resolve_sim: Called on every request; returns the currently running simulation or ``None``. Must stay valid across simulation reloads —
      a GUI reset replaces the ``Simulator`` object, and the resolver is what keeps an established detector connection pointing at the live one.
   :return: Router with the three perception endpoints, ready for ``app.include_router``.
   """
   router = APIRouter(prefix="/perception", tags=["perception"])

   @router.get("/views", response_model=PerceptionViewsResponse)
   def list_views() -> PerceptionViewsResponse:
      """Which observers currently have a capture waiting, and from which simulation instant.

      Cheap index poll — it never returns image data. It answers "who is there to fetch?" so a detector need not guess drone ids; whoever knows them
      already (from the scenario config or ``GET /state``) can ignore this endpoint entirely.
      """
      store = _require_view_store(_require_sim(resolve_sim))

      views: list[PerceptionViewInfo] = []
      for observer_id in store.observers():
         view = store.latest(observer_id)
         # A concurrent clear() (simulation reload) can empty a slot between the two calls — skip rather than report a hole.
         if view is None:
            continue
         views.append(PerceptionViewInfo(observer_id=view.observer_id, captured_step=int(view.step), captured_time=float(view.sim_time)))

      return PerceptionViewsResponse(views=views)

   @router.get("/view/{drone_id}", response_model=PerceptionViewResponse)
   def get_view(drone_id: str) -> PerceptionViewResponse:
      """Latest capture of one ego drone, image included.

      Exactly one view is kept per drone and it is overwritten on every capture, so a detector polling slower than the simulation captures skips
      captures instead of working through a backlog — a stale image is worse than a missed one, since its estimates would enter the DMPC as if they
      described the present.

      :param drone_id: Observing (ego) drone.
      :raises HTTPException: ``404`` when this observer has not captured yet.
      """
      sim = _require_sim(resolve_sim)
      view = _require_view_store(sim).latest(drone_id)
      if view is None:
         raise HTTPException(status_code=404, detail=f"No capture available for drone '{drone_id}'.")
      return _view_payload(view, expose_truth=bool(getattr(sim, "_camera_expose_truth", False)))

   @router.post("/estimates", response_model=PerceptionEstimatesResponse)
   def post_estimates(payload: PerceptionEstimatesRequest) -> PerceptionEstimatesResponse:
      """Accept one detector answer and store it in the perception mailbox.

      Written straight into the mailbox — no adapter, no worker. ``received_time`` is stamped by ``post()``; the capture token from the payload
      becomes the simulation timestamp of every estimate in the batch.

      Nothing is filtered: estimates for drone ids the simulation has never heard of are stored as they arrive. This is transport, not a tracker —
      false positives belong in front of whoever debugs the detector, not in a silently discarded list. A push for an observer that has no capture
      on record is accepted too, so a detector lagging behind the simulation is not punished for it.
      """
      mailbox = _require_mailbox(_require_sim(resolve_sim))

      estimates = [PositionEstimate(observer_id=payload.observer_id, observed_id=e.drone_id, position=np.asarray(e.position, dtype=float),
                                    captured_step=int(payload.captured_step), captured_time=float(payload.captured_time),
                                    sigma=float(e.sigma) if e.sigma is not None else None) for e in payload.estimates]
      mailbox.post(payload.observer_id, estimates)

      # One line per drone per capture — guarded so the default INFO log stays clean.
      if _log.isEnabledFor(logging.DEBUG):
         _log.debug("Perception push: observer=%s step=%d t=%.3f -> %d estimates %s", payload.observer_id, payload.captured_step,
                    payload.captured_time, len(estimates), [e.observed_id for e in estimates])

      return PerceptionEstimatesResponse(status="accepted", accepted=len(estimates))

   return router
