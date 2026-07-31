"""Tests for drone_sim.perception.mailbox module.

Covers PositionEstimate defaults and PerceptionMailbox behaviour:
- upsert semantics of post() (an empty batch deletes nothing)
- the two time axes (captured_* from the simulation, received_time from time.monotonic())
- history ordering (oldest first) and maxlen, which the DMPC bridge depends on
- clearing and thread safety
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from drone_sim.perception.mailbox import PerceptionMailbox, PositionEstimate


def _est(observer="ego", observed="d1", pos=(1.0, 2.0, 3.0), step=0, t=0.0, sigma=None) -> PositionEstimate:
   """Build a PositionEstimate with convenient defaults."""
   return PositionEstimate(observer_id=observer, observed_id=observed, position=np.array(pos, dtype=float),
                           captured_step=step, captured_time=t, sigma=sigma)


class TestPositionEstimate:
   """Tests for the PositionEstimate dataclass."""

   def test_defaults(self):
      """Test received_time and sigma default to 0.0 / None before the post."""
      estimate = _est()
      assert estimate.received_time == 0.0
      assert estimate.sigma is None


class TestPerceptionMailboxBasic:
   """Tests for post() / latest() basics."""

   @pytest.fixture
   def mailbox(self) -> PerceptionMailbox:
      """Create an empty PerceptionMailbox with default history length."""
      return PerceptionMailbox()

   def test_post_then_latest(self, mailbox: PerceptionMailbox):
      """Test a posted estimate shows up under latest(observer)[observed]."""
      mailbox.post("ego", [_est(observed="d1", pos=(1.0, 2.0, 3.0))])

      latest = mailbox.latest("ego")
      assert set(latest) == {"d1"}
      np.testing.assert_array_equal(latest["d1"].position, np.array([1.0, 2.0, 3.0]))

   def test_latest_overwrites_per_observed(self, mailbox: PerceptionMailbox):
      """Test two posts for the same pair keep one entry - the newest position wins."""
      mailbox.post("ego", [_est(observed="d1", pos=(1.0, 0.0, 0.0), step=0)])
      mailbox.post("ego", [_est(observed="d1", pos=(2.0, 0.0, 0.0), step=1)])

      latest = mailbox.latest("ego")
      assert len(latest) == 1
      np.testing.assert_array_equal(latest["d1"].position, np.array([2.0, 0.0, 0.0]))
      assert latest["d1"].captured_step == 1

   def test_latest_keeps_observed_apart(self, mailbox: PerceptionMailbox):
      """Test two observed ids in one post both end up in the dict."""
      mailbox.post("ego", [_est(observed="d1", pos=(1.0, 0.0, 0.0)), _est(observed="d2", pos=(0.0, 1.0, 0.0))])

      latest = mailbox.latest("ego")
      assert set(latest) == {"d1", "d2"}
      np.testing.assert_array_equal(latest["d2"].position, np.array([0.0, 1.0, 0.0]))

   def test_observers_are_independent(self, mailbox: PerceptionMailbox):
      """Test a post by one observer does not show up for another observer."""
      mailbox.post("ego-a", [_est(observed="d1")])

      assert set(mailbox.latest("ego-a")) == {"d1"}
      assert mailbox.latest("ego-b") == {}

   def test_latest_unknown_observer_empty(self, mailbox: PerceptionMailbox):
      """Test latest() returns an empty dict for an unknown observer."""
      result = mailbox.latest("nobody")
      assert isinstance(result, dict)
      assert result == {}

   def test_latest_is_shallow_copy(self, mailbox: PerceptionMailbox):
      """Test mutating the returned dict leaves the mailbox untouched."""
      mailbox.post("ego", [_est(observed="d1")])

      returned = mailbox.latest("ego")
      returned.clear()

      assert set(mailbox.latest("ego")) == {"d1"}

   def test_empty_post_is_noop(self, mailbox: PerceptionMailbox):
      """Test an empty post deletes nothing - upsert semantics, not replace."""
      mailbox.post("ego", [_est(observed="d1", pos=(1.0, 0.0, 0.0))])
      mailbox.post("ego", [])

      latest = mailbox.latest("ego")
      assert set(latest) == {"d1"}
      np.testing.assert_array_equal(latest["d1"].position, np.array([1.0, 0.0, 0.0]))
      assert len(mailbox.history("ego", "d1")) == 1

   def test_position_is_copied(self, mailbox: PerceptionMailbox):
      """Test mutating the source array after the post does not change the stored position."""
      source = np.array([1.0, 2.0, 3.0])
      estimate = PositionEstimate(observer_id="ego", observed_id="d1", position=source, captured_step=0, captured_time=0.0)
      mailbox.post("ego", [estimate])

      source[0] = 99.0

      np.testing.assert_array_equal(mailbox.latest("ego")["d1"].position, np.array([1.0, 2.0, 3.0]))

   def test_post_assigns_observer_id(self, mailbox: PerceptionMailbox):
      """Test the observer_id parameter wins over the one carried by the estimate."""
      mailbox.post("ego", [_est(observer="stale-id", observed="d1")])

      assert mailbox.latest("ego")["d1"].observer_id == "ego"


class TestPerceptionMailboxStamping:
   """Tests for the two time axes: received_time vs. captured_*."""

   @pytest.fixture
   def mailbox(self) -> PerceptionMailbox:
      """Create an empty PerceptionMailbox with default history length."""
      return PerceptionMailbox()

   def test_received_time_stamped(self, mailbox: PerceptionMailbox):
      """Test post() stamps received_time from time.monotonic()."""
      before = time.monotonic()
      mailbox.post("ego", [_est(observed="d1")])
      after = time.monotonic()

      assert before <= mailbox.latest("ego")["d1"].received_time <= after

   def test_batch_shares_one_received_time(self, mailbox: PerceptionMailbox):
      """Test all estimates of one post share the same received_time."""
      mailbox.post("ego", [_est(observed="d1"), _est(observed="d2"), _est(observed="d3")])

      latest = mailbox.latest("ego")
      stamps = {latest[observed].received_time for observed in ("d1", "d2", "d3")}
      assert len(stamps) == 1

   def test_captured_fields_untouched(self, mailbox: PerceptionMailbox):
      """Test captured_step/captured_time survive the post unchanged - the axes must not mix."""
      mailbox.post("ego", [_est(observed="d1", step=7, t=0.7)])

      stored = mailbox.latest("ego")["d1"]
      assert stored.captured_step == 7
      assert stored.captured_time == 0.7
      assert stored.received_time != 0.7


class TestPerceptionMailboxHistory:
   """Tests for history() ordering, length and separation."""

   @pytest.fixture
   def mailbox(self) -> PerceptionMailbox:
      """Create an empty PerceptionMailbox with default history length."""
      return PerceptionMailbox()

   def test_history_is_oldest_first(self, mailbox: PerceptionMailbox):
      """Test history returns oldest first, newest last - the contract the bridge relies on."""
      for i, t in enumerate((0.0, 0.1, 0.2)):
         mailbox.post("ego", [_est(observed="d1", step=i, t=t)])

      assert [e.captured_time for e in mailbox.history("ego", "d1")] == [0.0, 0.1, 0.2]

   def test_history_respects_maxlen(self):
      """Test history keeps at most history_len entries and drops the oldest."""
      mailbox = PerceptionMailbox(history_len=2)
      for i, t in enumerate((0.0, 0.1, 0.2)):
         mailbox.post("ego", [_est(observed="d1", step=i, t=t)])

      assert [e.captured_time for e in mailbox.history("ego", "d1")] == [0.1, 0.2]

   def test_history_unknown_pair_empty(self, mailbox: PerceptionMailbox):
      """Test history() returns an empty list for an unknown pair."""
      assert mailbox.history("ego", "d1") == []

   def test_history_is_a_copy(self, mailbox: PerceptionMailbox):
      """Test appending to the returned list leaves the internal deque untouched."""
      mailbox.post("ego", [_est(observed="d1")])

      returned = mailbox.history("ego", "d1")
      returned.append(_est(observed="d1"))

      assert len(mailbox.history("ego", "d1")) == 1

   def test_history_separate_per_pair(self, mailbox: PerceptionMailbox):
      """Test histories of two observed neighbors do not mix."""
      mailbox.post("ego", [_est(observed="d1", t=0.0), _est(observed="d2", t=0.0)])
      mailbox.post("ego", [_est(observed="d1", t=0.1)])

      assert len(mailbox.history("ego", "d1")) == 2
      assert len(mailbox.history("ego", "d2")) == 1


class TestPerceptionMailboxClearing:
   """Tests for clear() and clear_observer()."""

   @pytest.fixture
   def mailbox(self) -> PerceptionMailbox:
      """Create an empty PerceptionMailbox with default history length."""
      return PerceptionMailbox()

   def test_clear_wipes_latest_and_history(self, mailbox: PerceptionMailbox):
      """Test clear() drops both the latest estimates and the history."""
      mailbox.post("ego", [_est(observed="d1")])

      mailbox.clear()

      assert mailbox.latest("ego") == {}
      assert mailbox.history("ego", "d1") == []

   def test_clear_observer_only_affects_one(self, mailbox: PerceptionMailbox):
      """Test clear_observer() leaves the other observers intact."""
      mailbox.post("ego-a", [_est(observed="d1")])
      mailbox.post("ego-b", [_est(observed="d1")])

      mailbox.clear_observer("ego-a")

      assert mailbox.latest("ego-a") == {}
      assert mailbox.history("ego-a", "d1") == []
      assert set(mailbox.latest("ego-b")) == {"d1"}
      assert len(mailbox.history("ego-b", "d1")) == 1

   def test_clear_observer_removes_history_rows(self, mailbox: PerceptionMailbox):
      """Test clear_observer() removes the history rows too, not only the latest dict."""
      mailbox.post("ego", [_est(observed="d1"), _est(observed="d2")])

      mailbox.clear_observer("ego")

      assert mailbox.history("ego", "d1") == []
      assert mailbox.history("ego", "d2") == []

   def test_clear_observer_unknown_is_noop(self, mailbox: PerceptionMailbox):
      """Test clear_observer() on an unknown observer raises nothing."""
      mailbox.clear_observer("nobody")

      assert mailbox.latest("nobody") == {}


class TestPerceptionMailboxValidation:
   """Tests for constructor validation."""

   @pytest.mark.parametrize("history_len", [0, -1])
   def test_history_len_below_one_raises(self, history_len: int):
      """Test history_len < 1 raises ValueError."""
      with pytest.raises(ValueError, match="history_len"):
         PerceptionMailbox(history_len=history_len)

   def test_history_len_one_is_valid(self):
      """Test history_len == 1 is accepted (velocity estimation off, but legal)."""
      mailbox = PerceptionMailbox(history_len=1)
      mailbox.post("ego", [_est(observed="d1", t=0.0)])
      mailbox.post("ego", [_est(observed="d1", t=0.1)])

      assert [e.captured_time for e in mailbox.history("ego", "d1")] == [0.1]


class TestPerceptionMailboxConcurrency:
   """Thread safety tests with concurrent operations."""

   @pytest.fixture
   def mailbox(self) -> PerceptionMailbox:
      """Create an empty PerceptionMailbox with default history length."""
      return PerceptionMailbox()

   def test_concurrent_posts_no_data_loss(self, mailbox: PerceptionMailbox):
      """Test 4 threads each post 10 estimates - every observer keeps its last estimate."""
      num_observers = 4
      posts_per_observer = 10

      def poster(observer_id: str):
         for i in range(posts_per_observer):
            mailbox.post(observer_id, [_est(observed="d1", step=i, t=0.1 * i)])
            time.sleep(0.001)  # Small delay to interleave

      threads = [threading.Thread(target=poster, args=(f"ego-{i}",)) for i in range(num_observers)]
      for thread in threads:
         thread.start()
      for thread in threads:
         thread.join(timeout=5.0)

      for i in range(num_observers):
         latest = mailbox.latest(f"ego-{i}")
         assert set(latest) == {"d1"}
         assert latest["d1"].captured_step == posts_per_observer - 1

   def test_concurrent_post_and_read(self, mailbox: PerceptionMailbox):
      """Test 2 posters + 2 readers running simultaneously - no exceptions, readers get valid data."""
      errors: list[str] = []
      num_iterations = 50

      def poster(observer_id: str):
         try:
            for i in range(num_iterations):
               mailbox.post(observer_id, [_est(observed="d1", step=i, t=0.1 * i),
                                          _est(observed="d2", step=i, t=0.1 * i)])
               time.sleep(0.001)
         except Exception as e:
            errors.append(f"Poster {observer_id}: {e}")

      def reader(observer_id: str):
         try:
            for _ in range(num_iterations):
               for estimate in mailbox.latest(observer_id).values():
                  assert isinstance(estimate, PositionEstimate)
                  assert estimate.position.shape == (3,)
               for entry in mailbox.history(observer_id, "d1"):
                  assert entry.position.shape == (3,)
               time.sleep(0.001)
         except Exception as e:
            errors.append(f"Reader {observer_id}: {e}")

      threads = []
      for i in range(2):
         threads.append(threading.Thread(target=poster, args=(f"ego-{i}",)))
         threads.append(threading.Thread(target=reader, args=(f"ego-{i}",)))

      for thread in threads:
         thread.start()
      for thread in threads:
         thread.join(timeout=10.0)

      assert len(errors) == 0, f"Errors occurred: {errors}"