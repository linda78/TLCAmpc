"""Transport from the simulation's camera captures to the perception sinks, without ever blocking the simulation.

The simulator produces one :class:`~drone_sim.perception.camera.CameraView` per drone per capture step and hands the whole batch to
:meth:`PerceptionWorker.submit`. Everything that may be *slow* — rendering the FPV image, running an in-process detector — happens behind that call, on
the worker's own thread. What comes out the other side goes to up to two sinks, which may both be active at once:

* **adapter sink** — ``adapter.detect(view)`` is called and the resulting estimates are posted to the
  :class:`~drone_sim.perception.mailbox.PerceptionMailbox`. This is the in-process path (stub detector today, the real detector as a library later).
* **view store sink** — the View is dropped into the :class:`~drone_sim.perception.view_store.CameraViewStore`, where the API layer offers it to an
  out-of-process detector. That detector pushes its estimates into the mailbox itself, through the REST router — not through this module.

At least one sink must be configured; a worker with neither would render images and throw them away.

**Drop-oldest, never block.** The queue holds exactly one batch (``maxsize=1``). A :meth:`submit` that finds it full discards the *waiting* batch and
enqueues the new one, so the simulation never waits for a slow detector and the worker never works on a stale capture while a fresher one waits behind
it. Losing captures is free downstream: :meth:`PerceptionMailbox.post` upserts, so a neighbor simply keeps its last estimate. A whole step's batch is
dropped as a unit, so the estimates the DMPC sees always belong to *one* simulation instant.

**Two modes.**

* ``async_mode=True`` (default) — a daemon thread drains the queue. Estimates lag the simulation by at least one step; the DMPC bridge compensates
  partly by extrapolating (risk R7 of the perception plan).
* ``async_mode=False`` — :meth:`submit` does the work inline on the calling thread. No thread, no queue, nothing dropped: the estimates of step *k*
  are in the mailbox before the solve of step *k*. This is the mode for deterministic tests.

**Failures are isolated, never fatal.** Every sink call is wrapped: a detector that raises costs one View, not the worker thread and not the
simulation. That holds in both modes — ``camera_async`` must not change error semantics — so a broken detector shows up as a log line and empty
estimates, in the same place either way.

Thread model: :meth:`submit` may be called from any single producer (the simulator's main thread); :meth:`start` and :meth:`stop` belong to the owner
of the worker (simulator / GUI backend) and are not meant to race each other.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
   from drone_sim.perception.adapter import PerceptionAdapter
   from drone_sim.perception.camera import CameraView
   from drone_sim.perception.mailbox import PerceptionMailbox
   from drone_sim.perception.view_store import CameraViewStore

_log = logging.getLogger(__name__)

# How long the worker thread waits for a batch before re-checking the stop flag. Bounds the shutdown delay of an idle worker.
_POLL_TIMEOUT_S = 0.1

# Grace period stop() gives the thread to finish what it is doing.
_DEFAULT_STOP_TIMEOUT_S = 2.0


class PerceptionWorker:
   """Moves camera views to the perception sinks, asynchronously by default.

   :param mailbox: Mailbox the adapter's estimates are posted to. Unused when no ``adapter`` is configured, but always required — the simulator owns
      exactly one mailbox per run either way, and the REST detector posts into that same instance.
   :param adapter: In-process detector, anything satisfying :class:`~drone_sim.perception.adapter.PerceptionAdapter`. ``None`` disables the adapter
      sink.
   :param renderer: Called as ``renderer(view) -> bytes | None`` before the sinks; the result is assigned to ``view.image_png``. ``None`` leaves the
      View image-less. This is where the (comparatively expensive) FPV rendering happens, off the simulation thread.
   :param view_store: Store the rendered Views are offered from. ``None`` disables the view store sink.
   :param async_mode: ``True`` runs the sinks on a worker thread, ``False`` inline in :meth:`submit`.
   :raises ValueError: If neither ``adapter`` nor ``view_store`` is given — such a worker would have nothing to do with its Views.
   """

   def __init__(self, mailbox: PerceptionMailbox, *, adapter: PerceptionAdapter | None = None,
                renderer: Callable[[CameraView], bytes | None] | None = None, view_store: CameraViewStore | None = None,
                async_mode: bool = True) -> None:
      if adapter is None and view_store is None:
         raise ValueError("PerceptionWorker needs at least one sink: adapter (in-process detector) or view_store (REST pull)")

      self._mailbox = mailbox
      self._adapter = adapter
      self._renderer = renderer
      self._view_store = view_store
      self._async_mode = bool(async_mode)

      # Depth 1 on purpose: the queue is a hand-over slot, not a buffer. See the module docstring on drop-oldest.
      self._queue: queue.Queue[list[CameraView]] = queue.Queue(maxsize=1)
      # One flag for one state: the worker thread polls it to leave its loop, start()/submit() read it to
      # refuse work after a shutdown. Set in synchronous mode too, where there is no thread to notice it.
      self._stop_event = threading.Event()
      self._thread: threading.Thread | None = None

      _log.info("PerceptionWorker ready: async=%s sinks=%s renderer=%s", self._async_mode,
                "+".join(name for name, sink in (("adapter", adapter), ("view_store", view_store)) if sink is not None),
                renderer is not None)

   @property
   def is_running(self) -> bool:
      """Whether the worker thread is alive.

      Always ``False`` in synchronous mode, where there is no thread. Meant for diagnostics and for tests guarding against thread leaks.
      """
      return self._thread is not None and self._thread.is_alive()

   def start(self) -> None:
      """Spawn the worker thread.

      A no-op in synchronous mode and on repeated calls. The thread is a ``daemon``: :meth:`stop` is the intended shutdown, but a GUI reload that
      forgets one must not keep the process alive.

      :raises RuntimeError: If called after :meth:`stop`. A stopped worker is not restartable — build a new one, as the simulator does per run.
      """
      if not self._async_mode:
         return
      if self._stop_event.is_set():
         raise RuntimeError("PerceptionWorker.start() after stop(): a stopped worker is not restartable, build a new one")
      if self._thread is not None:
         return

      self._thread = threading.Thread(target=self._run, name="PerceptionWorker", daemon=True)
      self._thread.start()

   def submit(self, views: Sequence[CameraView]) -> None:
      """Hand one capture step's Views to the worker. **Never blocks.**

      Asynchronously, the batch is enqueued and a batch still waiting is dropped (see the module docstring). Synchronously, the Views are processed
      right here, in order, before the call returns.

      After :meth:`stop`, submitted Views are discarded — in **both** modes, so that a shut-down worker means the same thing either way: an
      asynchronous one has nobody left to process them, and a synchronous one that kept working past its own shutdown would be a trap for whoever
      wired the lifecycle up.

      :param views: Views of one capture step, typically one per drone. An empty sequence is a no-op.
      """
      if not views:
         return

      if self._stop_event.is_set():
         _log.debug("Perception submit after stop: %d views discarded", len(views))
         return

      batch = list(views)

      if not self._async_mode:
         for view in batch:
            self._process_view(view)
         return

      try:
         self._queue.put_nowait(batch)
      except queue.Full:
         self._drop_and_enqueue(batch)

   def stop(self, timeout: float = _DEFAULT_STOP_TIMEOUT_S) -> None:
      """Signal the worker thread and wait for it, at most ``timeout`` seconds. Idempotent.

      A thread that is inside a slow detector call finishes that call first, so the wait can time out; that is logged and accepted rather than
      escalated — the thread is a daemon and holds nothing the simulation needs.

      :param timeout: Seconds to wait for the thread to finish.
      """
      self._stop_event.set()
      if not self._async_mode or self._thread is None:
         return

      self._thread.join(timeout=timeout)
      if self._thread.is_alive():
         _log.warning("PerceptionWorker thread still running after %.1f s — a detector call is likely still blocked; it is a daemon thread and will "
                      "not keep the process alive", timeout)

   def _drop_and_enqueue(self, batch: list[CameraView]) -> None:
      """Replace the waiting batch with ``batch`` — the drop-oldest half of :meth:`submit`.

      Split out so :meth:`submit` reads as "enqueue, or make room and enqueue". The second ``put_nowait`` cannot realistically fail: only ``submit``
      fills the queue, and there is one producer. It is guarded anyway, because losing the *newest* batch is the harmless outcome and blocking the
      simulation to avoid it would not be.

      :param batch: Views to enqueue in place of whatever is waiting.
      """
      try:
         dropped = self._queue.get_nowait()
      except queue.Empty:
         # The worker drained the queue between the failed put and here — nothing to drop after all.
         dropped = None

      if dropped is not None and _log.isEnabledFor(logging.DEBUG):
         _log.debug("Perception queue full: dropping batch of %d views (step=%s) for a fresher one (step=%s)", len(dropped), dropped[0].step,
                    batch[0].step)

      try:
         self._queue.put_nowait(batch)
      except queue.Full:
         _log.debug("Perception queue refilled concurrently: dropping the newest batch of %d views", len(batch))

   def _run(self) -> None:
      """Worker thread body: drain batches until stopped.

      The ``timeout`` on the get is what makes the shutdown cooperative — an idle worker notices the stop flag within ``_POLL_TIMEOUT_S``. The flag is
      re-checked between Views as well, so stopping during a batch of many drones does not have to wait out the whole batch. The half-processed batch
      is simply abandoned: the mailbox upserts, so what did get through stays valid.
      """
      _log.debug("PerceptionWorker thread started")
      while not self._stop_event.is_set():
         try:
            batch = self._queue.get(timeout=_POLL_TIMEOUT_S)
         except queue.Empty:
            continue

         for view in batch:
            if self._stop_event.is_set():
               break
            self._process_view(view)
      _log.debug("PerceptionWorker thread stopped")

   def _process_view(self, view: CameraView) -> None:
      """Render one View and push it into every configured sink, isolating each step's failures.

      Order is render -> view store -> adapter. The store comes first because its ``put`` is instant while ``detect`` may block for a long time; a
      REST detector should not wait behind an in-process one when both are wired up. That is safe because the adapter contract declares the View
      read-only.

      :param view: One drone's capture. ``image_png`` is assigned here; the rest of the View is passed through untouched.
      """
      if self._renderer is not None:
         try:
            view.image_png = self._renderer(view)
         except Exception:
            _log.exception("Perception renderer failed for observer=%s step=%d — continuing with an image-less view", view.observer_id, view.step)

      if self._view_store is not None:
         try:
            self._view_store.put(view)
         except Exception:
            _log.exception("Perception view store failed for observer=%s step=%d", view.observer_id, view.step)

      if self._adapter is not None:
         try:
            self._mailbox.post(view.observer_id, self._adapter.detect(view))
         except Exception:
            _log.exception("Perception adapter failed for observer=%s step=%d — no estimates for this view", view.observer_id, view.step)
