"""Tests for drone_sim.perception.view_store module.

The central promise is the overwrite: one current view per observing drone, no queue and no history. Everything else here guards the polling
contract the API layer depends on (unknown observer -> None, observers() lists what is fetchable) and the single-writer/many-reader threading model.
"""

from __future__ import annotations

import threading

import numpy as np

from drone_sim.perception.camera import CameraView, VisibleDrone
from drone_sim.perception.view_store import CameraViewStore


def _view(*, observer="ego", step=0, sim_time=0.0, image_png=None, visible=()) -> CameraView:
   """Build a CameraView with the fields the store actually cares about."""
   return CameraView(observer_id=observer, step=step, sim_time=sim_time, position=np.zeros(3), view_dir=np.array([1.0, 0.0, 0.0]),
                     fov_deg=90.0, range_m=10.0,
                     visible=[VisibleDrone(drone_id=did, position=np.array(pos, dtype=float), velocity=np.zeros(3), radius=0.1)
                              for did, pos in visible],
                     image_png=image_png)


class TestOverwriteSemantics:
   """Tests for the one-image-per-drone promise."""

   def test_put_then_latest(self):
      """Test a stored view comes back for its observer."""
      store = CameraViewStore()
      store.put(_view(observer="d1", step=3))

      assert store.latest("d1").step == 3

   def test_second_put_replaces_the_first(self):
      """Test a fresh capture overwrites the previous one instead of queueing behind it."""
      store = CameraViewStore()
      store.put(_view(observer="d1", step=3, sim_time=0.3, image_png=b"old"))
      store.put(_view(observer="d1", step=4, sim_time=0.4, image_png=b"new"))

      latest = store.latest("d1")
      assert latest.step == 4
      assert latest.sim_time == 0.4
      assert latest.image_png == b"new"

   def test_many_puts_keep_exactly_one_view(self):
      """Test nothing accumulates: ten captures leave one slot holding the last."""
      store = CameraViewStore()
      for step in range(10):
         store.put(_view(observer="d1", step=step))

      assert store.observers() == ["d1"]
      assert store.latest("d1").step == 9

   def test_observers_are_independent(self):
      """Test overwriting one drone's view leaves the other drones alone."""
      store = CameraViewStore()
      store.put(_view(observer="d1", step=1))
      store.put(_view(observer="d2", step=1))
      store.put(_view(observer="d1", step=2))

      assert store.latest("d1").step == 2
      assert store.latest("d2").step == 1

   def test_out_of_order_put_still_wins(self):
      """Test put() does not compare step indices — the last writer wins, by design.

      The single writer cannot reorder captures, so this documents the intended behaviour rather than guarding against a real case.
      """
      store = CameraViewStore()
      store.put(_view(observer="d1", step=9))
      store.put(_view(observer="d1", step=4))

      assert store.latest("d1").step == 4


class TestPolling:
   """Tests for what the API layer reads."""

   def test_latest_unknown_observer_is_none(self):
      """Test an observer that never captured yields None rather than raising."""
      assert CameraViewStore().latest("nobody") is None

   def test_observers_empty_initially(self):
      """Test a fresh store advertises nothing."""
      assert CameraViewStore().observers() == []

   def test_observers_lists_every_capturing_drone(self):
      """Test observers() names exactly the drones with a fetchable view."""
      store = CameraViewStore()
      store.put(_view(observer="d1"))
      store.put(_view(observer="d2"))

      assert set(store.observers()) == {"d1", "d2"}

   def test_observers_is_a_copy(self):
      """Test mutating the returned list cannot corrupt the store."""
      store = CameraViewStore()
      store.put(_view(observer="d1"))

      store.observers().append("ghost")

      assert store.observers() == ["d1"]

   def test_clear_drops_everything(self):
      """Test clear() empties the store, e.g. on a simulation reload."""
      store = CameraViewStore()
      store.put(_view(observer="d1"))
      store.put(_view(observer="d2"))

      store.clear()

      assert store.observers() == []
      assert store.latest("d1") is None


class TestThreading:
   """Smoke test for the one-writer/many-reader model."""

   def test_concurrent_put_and_poll(self):
      """Test readers polling during a stream of writes never see a broken store."""
      store = CameraViewStore()
      errors: list[BaseException] = []
      stop = threading.Event()

      def writer():
         try:
            for step in range(300):
               store.put(_view(observer="d1", step=step))
         except BaseException as exc:  # noqa: BLE001 - surfaced via the errors list
            errors.append(exc)
         finally:
            stop.set()

      def reader():
         try:
            while not stop.is_set():
               view = store.latest("d1")
               # Either nothing yet, or a fully formed view — never a half-written one.
               assert view is None or (view.observer_id == "d1" and 0 <= view.step < 300)
         except BaseException as exc:  # noqa: BLE001 - surfaced via the errors list
            errors.append(exc)

      threads = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
      for thread in threads:
         thread.start()
      for thread in threads:
         thread.join(timeout=10.0)

      assert not any(t.is_alive() for t in threads)
      assert errors == []
      assert store.latest("d1").step == 299
