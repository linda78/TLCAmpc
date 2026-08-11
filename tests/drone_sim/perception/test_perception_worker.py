"""Tests for drone_sim.perception.worker module.

The worker is the one place in the perception pipeline where a slow detector could stall the simulation, so the promises worth pinning are:

- **synchronous mode is deterministic** — after submit() returns, the estimates of that step are in the mailbox,
- **asynchronous mode never blocks** — a batch waiting behind a busy worker is dropped in favour of the fresher one,
- **failures are isolated** — a raising renderer/detector costs one View, not the thread and not the run,
- **shutdown is clean** — stop() joins, is idempotent, and leaves no thread behind (thread leak on GUI reload, risk R2).

Timing rules used throughout: no test sleeps a fixed amount and hopes. Blocking is done with ``threading.Event`` gates that the test controls, and
every wait for the worker is a poll against a deadline (:func:`_wait_until`).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from drone_sim.perception.camera import CameraView, VisibleDrone
from drone_sim.perception.mailbox import PerceptionMailbox, PositionEstimate
from drone_sim.perception.view_store import CameraViewStore
from drone_sim.perception.worker import PerceptionWorker

# Generous upper bound for "the worker got round to it"; every use polls, so a healthy run never spends anywhere near this.
_DEADLINE_S = 5.0


def _view(*, observer="d1", step=0, sim_time=0.0, visible=(("d2", (1.0, 0.0, 0.0)),)) -> CameraView:
   """Build a CameraView carrying one visible neighbor per entry in ``visible``."""
   return CameraView(observer_id=observer, step=step, sim_time=sim_time, position=np.zeros(3), view_dir=np.array([1.0, 0.0, 0.0]),
                     fov_deg=90.0, range_m=10.0,
                     visible=[VisibleDrone(drone_id=did, position=np.array(pos, dtype=float), velocity=np.zeros(3), radius=0.1)
                              for did, pos in visible])


def _wait_until(predicate, timeout: float = _DEADLINE_S) -> bool:
   """Poll ``predicate`` until it holds or ``timeout`` seconds have passed.

   :param predicate: Zero-argument callable checked repeatedly.
   :param timeout: Seconds to keep polling.
   :return: Whether the predicate held before the deadline.
   """
   deadline = time.monotonic() + timeout
   while time.monotonic() < deadline:
      if predicate():
         return True
      time.sleep(0.002)
   return bool(predicate())


class _RecordingAdapter:
   """Detector double that records every View it saw and answers with one estimate per visible neighbor.

   Optionally blocks inside ``detect`` on a gate the test controls — that is how the drop-oldest test pins the worker in place without sleeping.

   :param gate: Event the first ``detect`` call waits for; ``None`` never blocks.
   """

   def __init__(self, gate: threading.Event | None = None) -> None:
      self.seen: list[tuple[str, int]] = []
      self.entered = threading.Event()
      self._gate = gate
      self._lock = threading.Lock()

   def detect(self, view: CameraView) -> list[PositionEstimate]:
      """Record the capture token and mirror ``view.visible`` back as estimates."""
      self.entered.set()
      if self._gate is not None:
         self._gate.wait(timeout=_DEADLINE_S)
      with self._lock:
         self.seen.append((view.observer_id, view.step))
      return [PositionEstimate(observer_id=view.observer_id, observed_id=visible.drone_id, position=np.array(visible.position, dtype=float),
                               captured_step=view.step, captured_time=view.sim_time) for visible in view.visible]

   def steps(self) -> list[int]:
      """Step indices of the Views processed so far, in processing order."""
      with self._lock:
         return [step for _, step in self.seen]


class _RaisingAdapter:
   """Detector double that always raises, to check the worker survives a broken detector."""

   def __init__(self) -> None:
      self.calls = 0

   def detect(self, view: CameraView) -> list[PositionEstimate]:
      """Count the call and fail."""
      self.calls += 1
      raise RuntimeError("detector exploded")


class TestConstruction:
   """Tests for the sink requirement and the constructor's bookkeeping."""

   def test_no_sink_raises(self):
      """Test a worker without adapter and without view store is rejected instead of silently doing nothing."""
      with pytest.raises(ValueError, match="at least one sink"):
         PerceptionWorker(PerceptionMailbox())

   def test_adapter_only_is_allowed(self):
      """Test the in-process path needs no view store."""
      assert PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter()) is not None

   def test_view_store_only_is_allowed(self):
      """Test the REST path needs no adapter — the detector posts its estimates itself."""
      assert PerceptionWorker(PerceptionMailbox(), view_store=CameraViewStore()) is not None

   def test_both_sinks_are_allowed(self):
      """Test adapter and view store may run side by side."""
      assert PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter(), view_store=CameraViewStore()) is not None

   def test_fresh_worker_is_not_running(self):
      """Test the thread is spawned by start(), not by the constructor."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter())
      assert worker.is_running is False


class TestSyncMode:
   """Tests for async_mode=False, the deterministic path used by tests and by camera_async=false runs."""

   def test_submit_processes_inline(self):
      """Test the estimates of a step are in the mailbox by the time submit() returns."""
      mailbox = PerceptionMailbox()
      worker = PerceptionWorker(mailbox, adapter=_RecordingAdapter(), async_mode=False)

      worker.submit([_view(observer="d1", step=7)])

      assert set(mailbox.latest("d1")) == {"d2"}
      assert mailbox.latest("d1")["d2"].captured_step == 7

   def test_start_spawns_no_thread(self):
      """Test start() is a no-op without a thread to speak of."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter(), async_mode=False)

      worker.start()

      assert worker.is_running is False

   def test_whole_batch_is_processed_in_order(self):
      """Test every drone of one capture step is handled, in submission order — nothing is dropped inline."""
      adapter = _RecordingAdapter()
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter, async_mode=False)

      worker.submit([_view(observer="d1", step=1), _view(observer="d2", step=1), _view(observer="d3", step=1)])

      assert [observer for observer, _ in adapter.seen] == ["d1", "d2", "d3"]

   def test_nothing_is_dropped_across_submits(self):
      """Test back-to-back submits all land — the drop-oldest queue is not in play here."""
      adapter = _RecordingAdapter()
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter, async_mode=False)

      for step in range(5):
         worker.submit([_view(step=step)])

      assert adapter.steps() == [0, 1, 2, 3, 4]

   def test_empty_batch_is_a_noop(self):
      """Test submitting no Views touches neither adapter nor mailbox."""
      adapter = _RecordingAdapter()
      mailbox = PerceptionMailbox()
      worker = PerceptionWorker(mailbox, adapter=adapter, async_mode=False)

      worker.submit([])

      assert adapter.seen == []
      assert mailbox.latest("d1") == {}

   def test_stop_is_a_noop(self):
      """Test stop() on a synchronous worker returns without complaint, repeatedly."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter(), async_mode=False)

      worker.stop()
      worker.stop()

      assert worker.is_running is False

   def test_submit_after_stop_is_discarded(self):
      """Test a stopped worker means the same in both modes — no inline processing past shutdown either."""
      adapter = _RecordingAdapter()
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter, async_mode=False)
      worker.stop()

      worker.submit([_view(step=9)])

      assert adapter.steps() == []


class TestSinks:
   """Tests for what each configured sink receives, run synchronously so the assertions need no waiting."""

   def test_renderer_result_is_attached_before_the_sinks(self):
      """Test the rendered PNG bytes reach both the stored View and the detector."""
      store = CameraViewStore()
      adapter = _RecordingAdapter()
      seen_images: list[bytes | None] = []

      def detect(view):
         seen_images.append(view.image_png)
         return adapter.detect(view)

      worker = PerceptionWorker(PerceptionMailbox(), adapter=type("A", (), {"detect": staticmethod(detect)})(), view_store=store,
                                renderer=lambda view: b"PNG-" + view.observer_id.encode(), async_mode=False)

      worker.submit([_view(observer="d1")])

      assert store.latest("d1").image_png == b"PNG-d1"
      assert seen_images == [b"PNG-d1"]

   def test_without_renderer_the_view_stays_image_less(self):
      """Test the worker invents no image when no renderer is configured."""
      store = CameraViewStore()
      worker = PerceptionWorker(PerceptionMailbox(), view_store=store, async_mode=False)

      worker.submit([_view(observer="d1")])

      assert store.latest("d1").image_png is None

   def test_view_store_sink_alone_leaves_the_mailbox_empty(self):
      """Test the REST path posts nothing itself — the external detector does that through the API."""
      mailbox = PerceptionMailbox()
      store = CameraViewStore()
      worker = PerceptionWorker(mailbox, view_store=store, async_mode=False)

      worker.submit([_view(observer="d1")])

      assert store.latest("d1") is not None
      assert mailbox.latest("d1") == {}

   def test_both_sinks_receive_the_same_view(self):
      """Test a worker with both sinks feeds mailbox and store from one capture."""
      mailbox = PerceptionMailbox()
      store = CameraViewStore()
      worker = PerceptionWorker(mailbox, adapter=_RecordingAdapter(), view_store=store, async_mode=False)

      worker.submit([_view(observer="d1", step=4)])

      assert store.latest("d1").step == 4
      assert mailbox.latest("d1")["d2"].captured_step == 4

   def test_estimates_are_posted_under_the_observing_drone(self):
      """Test each View's estimates land under its own observer, not merged into one bucket."""
      mailbox = PerceptionMailbox()
      worker = PerceptionWorker(mailbox, adapter=_RecordingAdapter(), async_mode=False)

      worker.submit([_view(observer="d1", visible=(("d2", (1.0, 0.0, 0.0)),)),
                     _view(observer="d2", visible=(("d1", (0.0, 1.0, 0.0)),))])

      assert set(mailbox.latest("d1")) == {"d2"}
      assert set(mailbox.latest("d2")) == {"d1"}


class TestAsyncMode:
   """Tests for the default path: a worker thread between simulation and sinks."""

   def test_start_then_submit_reaches_the_mailbox(self):
      """Test a submitted batch is processed by the thread within the deadline."""
      mailbox = PerceptionMailbox()
      worker = PerceptionWorker(mailbox, adapter=_RecordingAdapter())
      worker.start()
      try:
         worker.submit([_view(observer="d1", step=2)])

         assert _wait_until(lambda: "d2" in mailbox.latest("d1"))
         assert mailbox.latest("d1")["d2"].captured_step == 2
      finally:
         worker.stop()

   def test_start_makes_the_worker_running(self):
      """Test is_running reflects the spawned thread."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter())
      worker.start()
      try:
         assert worker.is_running is True
      finally:
         worker.stop()

   def test_start_is_idempotent(self):
      """Test calling start() twice does not spawn a second worker thread."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter())
      before = len(threading.enumerate())
      worker.start()
      worker.start()
      try:
         assert len(threading.enumerate()) == before + 1
      finally:
         worker.stop()

   def test_submit_before_start_is_processed_after_start(self):
      """Test an early submit is not lost — it waits in the queue for the thread."""
      mailbox = PerceptionMailbox()
      worker = PerceptionWorker(mailbox, adapter=_RecordingAdapter())

      worker.submit([_view(observer="d1", step=1)])
      worker.start()
      try:
         assert _wait_until(lambda: "d2" in mailbox.latest("d1"))
      finally:
         worker.stop()

   def test_view_store_sink_runs_on_the_thread_too(self):
      """Test the REST path is asynchronous as well: the rendered View shows up in the store without the caller waiting."""
      store = CameraViewStore()
      worker = PerceptionWorker(PerceptionMailbox(), view_store=store, renderer=lambda view: b"PNG")
      worker.start()
      try:
         worker.submit([_view(observer="d1", step=5)])

         assert _wait_until(lambda: store.latest("d1") is not None)
         assert store.latest("d1").image_png == b"PNG"
      finally:
         worker.stop()


class TestDropOldest:
   """Tests for the promise that a slow detector never stalls the simulation."""

   def test_waiting_batch_is_dropped_for_a_fresher_one(self):
      """Test a batch queued behind a busy worker is replaced by the next one instead of queueing up.

      The gate pins the worker inside the first detect(), so the queue state is fully determined: batch 0 is being worked on, batch 1 waits, batch 2
      evicts it. Batch 1 is the one the simulation is allowed to lose.
      """
      gate = threading.Event()
      adapter = _RecordingAdapter(gate=gate)
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter)
      worker.start()
      try:
         worker.submit([_view(step=0)])
         assert adapter.entered.wait(timeout=_DEADLINE_S), "worker never picked up the first batch"

         worker.submit([_view(step=1)])
         worker.submit([_view(step=2)])
         gate.set()

         assert _wait_until(lambda: adapter.steps() == [0, 2])
      finally:
         gate.set()
         worker.stop()

   def test_submit_does_not_block_on_a_busy_worker(self):
      """Test submitting while the detector is stuck returns immediately rather than waiting for a free slot."""
      gate = threading.Event()
      adapter = _RecordingAdapter(gate=gate)
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter)
      worker.start()
      try:
         worker.submit([_view(step=0)])
         assert adapter.entered.wait(timeout=_DEADLINE_S), "worker never picked up the first batch"

         started = time.monotonic()
         for step in range(1, 21):
            worker.submit([_view(step=step)])
         elapsed = time.monotonic() - started

         # Twenty submits against a queue that has been full since the second one; anything near the gate timeout means submit() waited.
         assert elapsed < 1.0
      finally:
         gate.set()
         worker.stop()

   def test_batches_are_dropped_whole(self):
      """Test the unit of dropping is the capture step, so surviving estimates never mix two simulation instants."""
      gate = threading.Event()
      adapter = _RecordingAdapter(gate=gate)
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter)
      worker.start()
      try:
         worker.submit([_view(observer="d1", step=0)])
         assert adapter.entered.wait(timeout=_DEADLINE_S), "worker never picked up the first batch"

         worker.submit([_view(observer="d1", step=1), _view(observer="d2", step=1)])
         worker.submit([_view(observer="d1", step=2), _view(observer="d2", step=2)])
         gate.set()

         assert _wait_until(lambda: adapter.steps() == [0, 2, 2])
         assert {step for _, step in adapter.seen} == {0, 2}
      finally:
         gate.set()
         worker.stop()


class TestFailureIsolation:
   """Tests that a broken detector or renderer costs one View, not the pipeline."""

   def test_raising_adapter_keeps_the_worker_alive(self):
      """Test the thread survives a detector exception and processes the next batch."""
      adapter = _RaisingAdapter()
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter)
      worker.start()
      try:
         worker.submit([_view(step=0)])
         assert _wait_until(lambda: adapter.calls == 1)

         worker.submit([_view(step=1)])

         assert _wait_until(lambda: adapter.calls == 2)
         assert worker.is_running is True
      finally:
         worker.stop()

   def test_raising_adapter_posts_nothing(self):
      """Test a failed detection leaves the mailbox untouched rather than posting half a result."""
      mailbox = PerceptionMailbox()
      worker = PerceptionWorker(mailbox, adapter=_RaisingAdapter(), async_mode=False)

      worker.submit([_view(observer="d1")])

      assert mailbox.latest("d1") == {}

   def test_raising_adapter_does_not_propagate_in_sync_mode(self):
      """Test error semantics do not change with camera_async: an inline detector failure does not reach the simulation loop either."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RaisingAdapter(), async_mode=False)

      worker.submit([_view()])

   def test_raising_renderer_still_feeds_the_sinks(self):
      """Test a rendering failure degrades to an image-less View instead of dropping the capture."""
      mailbox = PerceptionMailbox()
      store = CameraViewStore()

      def renderer(view):
         raise RuntimeError("renderer exploded")

      worker = PerceptionWorker(mailbox, adapter=_RecordingAdapter(), view_store=store, renderer=renderer, async_mode=False)

      worker.submit([_view(observer="d1")])

      assert store.latest("d1").image_png is None
      assert set(mailbox.latest("d1")) == {"d2"}

   def test_one_failing_view_does_not_skip_the_rest_of_the_batch(self):
      """Test the other drones of the same capture step are still processed."""
      mailbox = PerceptionMailbox()
      inner = _RecordingAdapter()

      class _SelectiveAdapter:
         def detect(self, view):
            if view.observer_id == "d1":
               raise RuntimeError("detector exploded")
            return inner.detect(view)

      worker = PerceptionWorker(mailbox, adapter=_SelectiveAdapter(), async_mode=False)

      worker.submit([_view(observer="d1"), _view(observer="d2")])

      assert mailbox.latest("d1") == {}
      assert set(mailbox.latest("d2")) == {"d2"}


class TestLifecycle:
   """Tests for start/stop bookkeeping — the GUI reloads configs, and every leaked thread would stay for the session (risk R2)."""

   def test_stop_joins_the_thread(self):
      """Test the worker thread is gone once stop() returns."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter())
      worker.start()

      worker.stop()

      assert worker.is_running is False

   def test_stop_is_idempotent(self):
      """Test a second stop() is harmless — Simulator.close() may be called more than once."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter())
      worker.start()

      worker.stop()
      worker.stop()

      assert worker.is_running is False

   def test_stop_without_start_is_harmless(self):
      """Test closing a worker that was never started does not raise."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter())

      worker.stop()

      assert worker.is_running is False

   def test_start_after_stop_raises(self):
      """Test a stopped worker refuses to restart instead of silently staying dead."""
      worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter())
      worker.start()
      worker.stop()

      with pytest.raises(RuntimeError, match="not restartable"):
         worker.start()

   def test_submit_after_stop_is_discarded(self):
      """Test Views submitted after shutdown are dropped rather than piling up unprocessed."""
      adapter = _RecordingAdapter()
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter)
      worker.start()
      worker.stop()

      worker.submit([_view(step=9)])

      assert adapter.steps() == []

   def test_repeated_start_stop_cycles_leak_no_threads(self):
      """Test five worker generations, as a GUI reload would create them, leave no thread behind."""
      before = len(threading.enumerate())

      for _ in range(5):
         worker = PerceptionWorker(PerceptionMailbox(), adapter=_RecordingAdapter())
         worker.start()
         worker.submit([_view()])
         worker.stop()

      assert _wait_until(lambda: len(threading.enumerate()) == before)

   def test_stop_returns_while_a_detector_is_stuck(self):
      """Test a blocked detector cannot make shutdown hang: stop() gives up after its timeout and logs.

      The thread stays alive here on purpose — it is a daemon, released as soon as the gate opens in the cleanup below.
      """
      gate = threading.Event()
      adapter = _RecordingAdapter(gate=gate)
      worker = PerceptionWorker(PerceptionMailbox(), adapter=adapter)
      worker.start()
      try:
         worker.submit([_view()])
         assert adapter.entered.wait(timeout=_DEADLINE_S), "worker never picked up the batch"

         started = time.monotonic()
         worker.stop(timeout=0.2)
         elapsed = time.monotonic() - started

         assert elapsed < 1.0
      finally:
         gate.set()
         worker.stop()
