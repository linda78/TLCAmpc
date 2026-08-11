"""Perception REST endpoints for the GUI run.

:mod:`drone_sim.api.app` already serves the perception router, but only for simulations driven through ``POST /config`` + ``POST /step``. A GUI run has
its own :class:`~drone_sim.simulation.simulator.Simulator`, owned by :class:`~drone_sim.gui.direct_backend.DirectBackend` and stepped by the play
button — so it needs its own HTTP host for the same router. Same endpoints, same contract, different owner of the simulation.

The server runs uvicorn in a daemon thread. Daemon is a backstop only: :meth:`PerceptionApiServer.stop` shuts it down cooperatively via
``uvicorn.Server.should_exit``, exactly like the perception worker's stop flag.

Lifecycle in the GUI: started on the first ``load_config`` of a scenario that asks for the REST perception path, then kept alive across config reloads
and resets. That is deliberate — the router resolves the simulation per request, so a detector connection survives a reset instead of having to
reconnect. It is stopped when the window closes.

Bound to ``127.0.0.1`` by default: the detector is expected to run next to the simulation. Pass ``host="0.0.0.0"`` to accept connections from another
machine — that exposes the camera images and the DMPC mailbox to the whole network, so it is an explicit decision, not a default.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
   from uvicorn import Server

   from drone_sim.simulation.simulator import Simulator

_log = logging.getLogger(__name__)

_READY_POLL_S = 0.01


class PerceptionApiServer:
   """uvicorn server in a daemon thread, serving the ``/perception`` endpoints of one GUI session.

   Both :meth:`start` and :meth:`stop` are idempotent, so the GUI can call them from wherever the lifecycle happens to lead without tracking state.

   :param resolve_sim: Called per request; returns the currently running simulation or ``None``. Keep this pointing at the backend's *current*
      simulation (e.g. ``lambda: backend._sim``) rather than capturing one object — the point of the indirection is surviving a reload.
   :param port: TCP port to bind, from ``ScenarioConfig.camera_api_port``.
   :param host: Interface to bind; see the module docstring before widening it.
   """

   def __init__(self, resolve_sim: Callable[[], Simulator | None], *, port: int = 5006, host: str = "127.0.0.1") -> None:
      self._resolve_sim = resolve_sim
      self._port = int(port)
      self._host = str(host)
      # Built by the serving thread, not by start() — see start() on why the uvicorn import is not on the Qt thread.
      self._server: Server | None = None
      self._thread: threading.Thread | None = None
      # Bridges the gap between "stop() was called" and "the thread got far enough to have a server to stop".
      self._stop_requested = threading.Event()

   @property
   def port(self) -> int:
      """Bound TCP port."""
      return self._port

   @property
   def host(self) -> str:
      """Bound interface."""
      return self._host

   def is_running(self) -> bool:
      """Whether the serving thread is alive.

      Note that a thread can be alive but not yet accepting connections; :meth:`wait_started` is the readiness check.

      :return: ``True`` while the uvicorn thread runs.
      """
      return self._thread is not None and self._thread.is_alive()

   def start(self) -> None:
      """Start serving in a daemon thread; a no-op when already started.

      Returns as soon as the thread is spawned. The ``fastapi``/``uvicorn`` imports and the app construction happen *inside* the thread, not here:
      importing fastapi alone costs a few hundred milliseconds, and this method is called from ``load_config`` on the Qt main thread, where that is a
      visible freeze the moment a REST-perception scenario is opened. A GUI run without that path never imports them at all.
      """
      if self._thread is not None:
         return

      self._stop_requested.clear()   # a previous stop() must not shut the new thread down before it binds
      self._thread = threading.Thread(target=self._serve, name=f"perception-api-{self._port}", daemon=True)
      self._thread.start()
      _log.info("Perception API server starting on http://%s:%d/perception", self._host, self._port)

   def wait_started(self, timeout: float = 5.0) -> bool:
      """Block until the server accepts connections.

      Meant for tests and for scripted runs. The GUI does not call it — a detector that polls before the socket is up simply retries.

      :param timeout: Seconds to wait.
      :return: ``True`` once uvicorn reports itself started, ``False`` on timeout or when the thread died on the way up (e.g. port in use).
      """
      deadline = time.monotonic() + timeout
      while time.monotonic() < deadline:
         if self._is_serving():
            return True
         if self._thread is not None and not self._thread.is_alive():
            return False
         time.sleep(_READY_POLL_S)
      # One last look: the server may have come up inside the final poll interval.
      return self._is_serving()

   def _is_serving(self) -> bool:
      """Whether uvicorn reports itself bound and accepting. ``False`` while the thread is still constructing it."""
      return self._server is not None and bool(self._server.started)

   def stop(self, timeout: float = 2.0) -> None:
      """Shut the server down cooperatively and join its thread; a no-op when not running.

      :param timeout: Seconds to wait for the thread to finish. A thread that outlives it is logged and left to the daemon backstop — the GUI must
         not hang on closing because a detector holds a connection open.
      """
      self._stop_requested.set()
      if self._server is not None:
         # uvicorn's documented way to stop a server from outside its own thread.
         self._server.should_exit = True

      thread = self._thread
      if thread is not None:
         thread.join(timeout)
         if thread.is_alive():
            _log.warning("Perception API server thread did not stop within %.1fs; leaving it to the daemon backstop", timeout)
         else:
            _log.info("Perception API server on %s:%d stopped", self._host, self._port)

      self._thread = None
      self._server = None

   def _serve(self) -> None:
      """Thread body: build the app, then run uvicorn and make sure a failed bind is visible.

      The imports and the app construction live here so the caller of :meth:`start` — the Qt main thread — never pays for them.

      A port collision (two GUI instances, same ``camera_api_port``) makes uvicorn log the OS error and call ``sys.exit(1)``, which raises
      ``SystemExit`` — a ``BaseException`` that would otherwise end this thread silently and leave the GUI looking healthy while nothing listens.
      """
      import uvicorn
      from fastapi import FastAPI

      from drone_sim.api.perception_router import build_perception_router

      app = FastAPI(title="DroneSim perception (GUI)")
      app.include_router(build_perception_router(self._resolve_sim))

      # log_level="warning" keeps one access-log line per detector poll out of the terminal the GUI was started from.
      server = uvicorn.Server(uvicorn.Config(app, host=self._host, port=self._port, log_level="warning"))
      self._server = server

      # A stop() that arrived while the imports were still running found no server to signal; honour it here instead of binding the port anyway.
      if self._stop_requested.is_set():
         _log.info("Perception API server on %s:%d stopped before it began serving", self._host, self._port)
         return

      try:
         server.run()
      except SystemExit:
         _log.error("Perception API server could not bind %s:%d — port already in use? (another GUI instance, or camera_api_port clashing with the "
                    "REST API)", self._host, self._port)
      except Exception:
         _log.exception("Perception API server on %s:%d stopped with an error", self._host, self._port)
