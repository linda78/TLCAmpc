"""Tests for drone_sim.perception.bridge and the deliver() methods it feeds.

Covers the seam between the perception mailbox and the DMPC trajectory mailboxes:
- the constant-velocity extrapolation (finite difference over captured_time, k+1 spacing, tiled velocities)
- the degenerate cases that must NOT invent motion (single estimate, identical captured_time)
- delivery visible through TrajectoryMailbox.receive() and ThreadSafeMailbox.receive_latest()
- R1: only "monotonic" timestamps survive AsyncLocalSolver._filter_stale, "step" ones are dropped
- the two new deliver() methods themselves, including overwrite-per-sender and concurrent delivery
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.domain.drone import Drone, Route
from drone_sim.perception.bridge import feed_trajectory_mailbox
from drone_sim.perception.mailbox import PerceptionMailbox, PositionEstimate
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics
from drone_sim.simulation.distributed.async_local_solver import AsyncLocalSolver
from drone_sim.simulation.distributed.local_mpc import LocalMPCSolver
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph
from drone_sim.simulation.distributed.threaded_mailbox import ThreadSafeMailbox
from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMailbox, TrajectoryMessage


def _est(observed="d2", pos=(0.0, 0.0, 0.0), step=0, t=0.0) -> PositionEstimate:
   """Build a PositionEstimate; observer_id is overwritten by post() anyway."""
   return PositionEstimate(observer_id="", observed_id=observed, position=np.array(pos, dtype=float), captured_step=step, captured_time=t)


def _msg(drone_id="d1", timestamp=0) -> TrajectoryMessage:
   """Build a minimal TrajectoryMessage for the deliver() tests."""
   return TrajectoryMessage(drone_id=drone_id, trajectory=np.zeros((3, 3)), predicted_velocities=None, timestamp=timestamp)


def _async_solver(stale_threshold_sec: float = 1.0) -> AsyncLocalSolver:
   """Build a real AsyncLocalSolver just to exercise its _filter_stale (no thread is ever started)."""
   physics = LinearKinematicsPhysics(dt=0.1, v_max=5.0)
   drone = Drone(drone_id="d1", radius=0.1, safety_zone=1.0, cons_stop=0.0, color="tab:blue", safety_color="tab:cyan", trace_color="tab:blue",
                 controller=CentralMPCAgent(dt=0.1, horizon=3), physics=physics, x=np.zeros(6), route=Route(waypoints=[], target=np.zeros(3)),
                 alpha=None)
   return AsyncLocalSolver(drone=drone, mailbox=ThreadSafeMailbox(), neighbor_graph=NeighborGraph(comm_radius=None),
                           local_solver=LocalMPCSolver(dt=0.1, horizon=3), stale_threshold_sec=stale_threshold_sec)


class TestVelocityEstimation:
   """Tests for the finite difference over the perception history."""

   def test_finite_difference_from_two_estimates(self):
      """Test velocity is (p_new - p_old) / (t_new - t_old) from the two newest history entries."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(pos=(0.0, 0.0, 0.0), step=0, t=0.0)])
      perception.post("d1", [_est(pos=(0.2, 0.4, 0.0), step=1, t=0.1)])
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=3, dt=0.1, timestamp_mode="step")

      velocities = mailbox.receive("d1")["d2"].predicted_velocities
      np.testing.assert_allclose(velocities[0], np.array([2.0, 4.0, 0.0]))

   def test_finite_difference_uses_captured_time_not_dt(self):
      """Test the difference divides by the capture interval, so a slow camera (camera_rate_steps > 1) still gives the right speed."""
      perception = PerceptionMailbox()
      # Two captures 0.5 s apart while the simulation runs at dt=0.1 - only captured_time may decide the speed.
      perception.post("d1", [_est(pos=(0.0, 0.0, 0.0), step=0, t=0.0)])
      perception.post("d1", [_est(pos=(1.0, 0.0, 0.0), step=5, t=0.5)])
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="step")

      np.testing.assert_allclose(mailbox.receive("d1")["d2"].predicted_velocities[0], np.array([2.0, 0.0, 0.0]))

   def test_single_estimate_gives_zero_velocity(self):
      """Test a first sighting produces a hovering neighbor instead of an invented motion."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(pos=(1.0, 2.0, 3.0), step=0, t=0.0)])
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=4, dt=0.1, timestamp_mode="step")

      message = mailbox.receive("d1")["d2"]
      np.testing.assert_array_equal(message.predicted_velocities, np.zeros((4, 3)))
      # A zero velocity means a constant trajectory: every step sits on the estimated position.
      np.testing.assert_allclose(message.trajectory, np.tile(np.array([1.0, 2.0, 3.0]), (4, 1)))

   def test_identical_captured_time_gives_zero_velocity(self):
      """Test two estimates of the same capture are skipped instead of dividing by zero."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(pos=(0.0, 0.0, 0.0), step=0, t=0.0)])
      perception.post("d1", [_est(pos=(5.0, 0.0, 0.0), step=0, t=0.0)])
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=3, dt=0.1, timestamp_mode="step")

      np.testing.assert_array_equal(mailbox.receive("d1")["d2"].predicted_velocities, np.zeros((3, 3)))

   def test_skips_back_to_the_newest_differing_capture(self):
      """Test a duplicate captured_time on top does not blind the difference - it walks back to the next distinct capture."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(pos=(0.0, 0.0, 0.0), step=0, t=0.0)])
      perception.post("d1", [_est(pos=(0.1, 0.0, 0.0), step=1, t=0.1)])
      perception.post("d1", [_est(pos=(0.1, 0.0, 0.0), step=1, t=0.1)])  # re-post of the same capture
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="step")

      np.testing.assert_allclose(mailbox.receive("d1")["d2"].predicted_velocities[0], np.array([1.0, 0.0, 0.0]))


class TestExtrapolation:
   """Tests for the shape and spacing of the extrapolated horizon."""

   @pytest.fixture
   def perception(self) -> PerceptionMailbox:
      """Perception mailbox where d1 saw d2 move at 1 m/s along +x, ending at the origin."""
      mailbox = PerceptionMailbox()
      mailbox.post("d1", [_est(pos=(-0.1, 0.0, 0.0), step=0, t=0.0)])
      mailbox.post("d1", [_est(pos=(0.0, 0.0, 0.0), step=1, t=0.1)])
      return mailbox

   def test_trajectory_starts_one_step_ahead(self, perception: PerceptionMailbox):
      """Test trajectory[k] = p + v*dt*(k+1), i.e. index 0 is the state after one step (like _predict_states)."""
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=3, dt=0.1, timestamp_mode="step")

      trajectory = mailbox.receive("d1")["d2"].trajectory
      expected = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
      np.testing.assert_allclose(trajectory, expected)

   def test_shapes_match_horizon(self, perception: PerceptionMailbox):
      """Test both arrays are (H, 3) and the velocity is tiled unchanged over the horizon."""
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=5, dt=0.1, timestamp_mode="step")

      message = mailbox.receive("d1")["d2"]
      assert message.trajectory.shape == (5, 3)
      assert message.predicted_velocities.shape == (5, 3)
      np.testing.assert_allclose(message.predicted_velocities, np.tile(np.array([1.0, 0.0, 0.0]), (5, 1)))

   def test_horizon_one(self, perception: PerceptionMailbox):
      """Test the smallest legal horizon still produces a (1, 3) trajectory one step ahead."""
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=1, dt=0.1, timestamp_mode="step")

      np.testing.assert_allclose(mailbox.receive("d1")["d2"].trajectory, np.array([[0.1, 0.0, 0.0]]))

   def test_rejects_invalid_horizon(self, perception: PerceptionMailbox):
      """Test a horizon below 1 is rejected rather than silently producing empty trajectories."""
      with pytest.raises(ValueError, match="horizon"):
         feed_trajectory_mailbox(perception=perception, trajectory_mailbox=TrajectoryMailbox(), receiver_ids=["d1"], horizon=0, dt=0.1,
                                 timestamp_mode="step")

   def test_rejects_unknown_timestamp_mode(self, perception: PerceptionMailbox):
      """Test an unknown timestamp_mode raises instead of defaulting to the one that breaks _filter_stale."""
      with pytest.raises(ValueError, match="timestamp_mode"):
         feed_trajectory_mailbox(perception=perception, trajectory_mailbox=TrajectoryMailbox(), receiver_ids=["d1"], horizon=3, dt=0.1,
                                 timestamp_mode="wallclock")  # type: ignore[arg-type]


class TestDelivery:
   """Tests for what lands in which inbox, on both mailbox types."""

   def test_sync_mailbox_delivery_visible_via_receive(self):
      """Test the estimates of an observer land in that observer's own inbox, keyed by the observed drone."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(observed="d2", pos=(1.0, 0.0, 0.0)), _est(observed="d3", pos=(0.0, 1.0, 0.0))])
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1", "d2"], horizon=3, dt=0.1,
                              timestamp_mode="step", step=7)

      inbox = mailbox.receive("d1")
      assert set(inbox) == {"d2", "d3"}
      assert inbox["d2"].drone_id == "d2"
      # d2 observed nothing, so it plans as if alone.
      assert mailbox.receive("d2") == {}

   def test_threaded_mailbox_delivery_visible_via_receive_latest(self):
      """Test the same feed works against ThreadSafeMailbox without any change at the call site."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(observed="d2", pos=(1.0, 0.0, 0.0))])
      mailbox = ThreadSafeMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=3, dt=0.1, timestamp_mode="monotonic")

      inbox = mailbox.receive_latest("d1")
      assert set(inbox) == {"d2"}
      np.testing.assert_allclose(inbox["d2"].trajectory[0], np.array([1.0, 0.0, 0.0]))

   def test_no_estimates_delivers_nothing(self):
      """Test an observer with an empty field of view produces no messages at all."""
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=PerceptionMailbox(), trajectory_mailbox=mailbox, receiver_ids=["d1", "d2"], horizon=3, dt=0.1,
                              timestamp_mode="step")

      assert mailbox.receive("d1") == {}
      assert mailbox.receive("d2") == {}

   def test_routing_bypasses_neighbor_graph(self):
      """Test delivery happens on camera visibility alone - an empty NeighborGraph would have routed nothing."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(observed="d2")])
      mailbox = TrajectoryMailbox()
      # Not passed anywhere on purpose: the bridge has no NeighborGraph parameter, and this documents why.
      graph = NeighborGraph(comm_radius=0.01)
      graph.update({"d1": np.zeros(3), "d2": np.array([100.0, 0.0, 0.0])})
      assert graph.get_neighbors("d1") == set()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=3, dt=0.1, timestamp_mode="step")

      assert "d2" in mailbox.receive("d1")

   def test_repeated_feed_overwrites_previous_message(self):
      """Test a second feed replaces the neighbor's message instead of accumulating stale ones."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(observed="d2", pos=(0.0, 0.0, 0.0), step=0, t=0.0)])
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="step", step=0)
      perception.post("d1", [_est(observed="d2", pos=(1.0, 0.0, 0.0), step=1, t=0.1)])
      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="step", step=1)

      inbox = mailbox.receive("d1")
      assert len(inbox) == 1
      assert inbox["d2"].timestamp == 1
      np.testing.assert_allclose(inbox["d2"].trajectory[0], np.array([2.0, 0.0, 0.0]))

   def test_safety_zone_radius_best_effort(self):
      """Test known ids get their radius stamped on, unknown ids leave the field None."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(observed="d2"), _est(observed="d3")])
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="step",
                              safety_zone_by_id={"d2": 1.25})

      inbox = mailbox.receive("d1")
      assert inbox["d2"].safety_zone_radius == 1.25
      assert inbox["d3"].safety_zone_radius is None

   def test_safety_zone_none_leaves_field_unset(self):
      """Test omitting the mapping entirely is legal and leaves the field None."""
      perception = PerceptionMailbox()
      perception.post("d1", [_est(observed="d2")])
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="step")

      assert mailbox.receive("d1")["d2"].safety_zone_radius is None


class TestTimestampModes:
   """Tests for R1 - the clock the messages are stamped with decides whether the async solver sees them at all."""

   @pytest.fixture
   def perception(self) -> PerceptionMailbox:
      """Perception mailbox with one estimate for d1."""
      mailbox = PerceptionMailbox()
      mailbox.post("d1", [_est(observed="d2", pos=(1.0, 0.0, 0.0))])
      return mailbox

   def test_step_mode_stamps_the_step_index(self, perception: PerceptionMailbox):
      """Test timestamp_mode='step' writes the simulation step, exactly as the synchronous coordinator expects."""
      mailbox = TrajectoryMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="step",
                              step=42)

      assert mailbox.receive("d1")["d2"].timestamp == 42

   def test_monotonic_mode_ignores_step_and_stamps_the_clock(self, perception: PerceptionMailbox):
      """Test timestamp_mode='monotonic' samples time.monotonic() at delivery time and ignores the step argument."""
      before = time.monotonic()
      mailbox = ThreadSafeMailbox()

      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="monotonic",
                              step=42)

      after = time.monotonic()
      timestamp = mailbox.receive_latest("d1")["d2"].timestamp
      assert before <= timestamp <= after

   def test_monotonic_messages_survive_the_stale_filter(self, perception: PerceptionMailbox):
      """Test R1: monotonic-stamped messages pass AsyncLocalSolver._filter_stale."""
      mailbox = ThreadSafeMailbox()
      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="monotonic")

      kept = _async_solver()._filter_stale(mailbox.receive_latest("d1"))

      assert set(kept) == {"d2"}

   def test_step_messages_are_dropped_by_the_stale_filter(self, perception: PerceptionMailbox):
      """Test R1, the failure mode: a step index looks like an age of hours, so every message would be filtered away."""
      mailbox = ThreadSafeMailbox()
      feed_trajectory_mailbox(perception=perception, trajectory_mailbox=mailbox, receiver_ids=["d1"], horizon=2, dt=0.1, timestamp_mode="step",
                              step=3)

      kept = _async_solver()._filter_stale(mailbox.receive_latest("d1"))

      assert kept == {}


class TestTrajectoryMailboxDeliver:
   """Tests for TrajectoryMailbox.deliver()."""

   @pytest.fixture
   def mailbox(self) -> TrajectoryMailbox:
      """Create an empty TrajectoryMailbox."""
      return TrajectoryMailbox()

   def test_deliver_files_under_sender_id(self, mailbox: TrajectoryMailbox):
      """Test the message is keyed by its own drone_id, not by the receiver."""
      mailbox.deliver("d2", _msg(drone_id="d1"))

      assert set(mailbox.receive("d2")) == {"d1"}
      assert mailbox.receive("d1") == {}

   def test_deliver_overwrites_same_sender(self, mailbox: TrajectoryMailbox):
      """Test a second message from the same sender replaces the first."""
      mailbox.deliver("d2", _msg(drone_id="d1", timestamp=0))
      mailbox.deliver("d2", _msg(drone_id="d1", timestamp=1))

      inbox = mailbox.receive("d2")
      assert len(inbox) == 1
      assert inbox["d1"].timestamp == 1

   def test_deliver_keeps_senders_apart(self, mailbox: TrajectoryMailbox):
      """Test two senders coexist in one inbox."""
      mailbox.deliver("d3", _msg(drone_id="d1"))
      mailbox.deliver("d3", _msg(drone_id="d2"))

      assert set(mailbox.receive("d3")) == {"d1", "d2"}

   def test_deliver_coexists_with_broadcast(self, mailbox: TrajectoryMailbox):
      """Test broadcast-routed and directly delivered messages share the same inbox."""
      graph = NeighborGraph(comm_radius=None)
      graph.update({"d1": np.zeros(3), "d2": np.array([1.0, 0.0, 0.0])})
      mailbox.broadcast("d1", np.zeros((3, 3)), None, 0, graph)
      mailbox.deliver("d2", _msg(drone_id="d3"))

      assert set(mailbox.receive("d2")) == {"d1", "d3"}

   def test_clear_removes_delivered_messages(self, mailbox: TrajectoryMailbox):
      """Test the per-step clear() of the DMPC loop also wipes bridge messages."""
      mailbox.deliver("d2", _msg(drone_id="d1"))

      mailbox.clear()

      assert mailbox.receive("d2") == {}


class TestThreadSafeMailboxDeliver:
   """Tests for ThreadSafeMailbox.deliver()."""

   @pytest.fixture
   def mailbox(self) -> ThreadSafeMailbox:
      """Create an empty ThreadSafeMailbox."""
      return ThreadSafeMailbox()

   def test_deliver_files_under_sender_id(self, mailbox: ThreadSafeMailbox):
      """Test the message reaches exactly the named receiver, keyed by its own drone_id."""
      mailbox.deliver("d2", _msg(drone_id="d1"))

      assert set(mailbox.receive_latest("d2")) == {"d1"}
      assert mailbox.receive_latest("d1") == {}

   def test_deliver_overwrites_same_sender(self, mailbox: ThreadSafeMailbox):
      """Test a second message from the same sender replaces the first."""
      mailbox.deliver("d2", _msg(drone_id="d1", timestamp=0))
      mailbox.deliver("d2", _msg(drone_id="d1", timestamp=1))

      inbox = mailbox.receive_latest("d2")
      assert len(inbox) == 1
      assert inbox["d1"].timestamp == 1

   def test_deliver_wakes_a_waiting_thread(self, mailbox: ThreadSafeMailbox):
      """Test deliver() notifies the receiver's Condition, so a solver thread parked in wait_for_update() runs."""
      received: dict[str, TrajectoryMessage] = {}

      def waiter():
         if mailbox.wait_for_update("d2", timeout=2.0):
            received.update(mailbox.receive_latest("d2"))

      thread = threading.Thread(target=waiter)
      thread.start()
      time.sleep(0.1)  # let the waiter reach the Condition

      mailbox.deliver("d2", _msg(drone_id="d1", timestamp=99))

      thread.join(timeout=3.0)
      assert not thread.is_alive()
      assert received["d1"].timestamp == 99

   def test_deliver_does_not_wake_other_receivers(self, mailbox: ThreadSafeMailbox):
      """Test only the addressed receiver's Condition is notified."""
      notified = []

      def waiter():
         notified.append(mailbox.wait_for_update("d3", timeout=0.5))

      thread = threading.Thread(target=waiter)
      thread.start()
      time.sleep(0.1)

      mailbox.deliver("d2", _msg(drone_id="d1"))

      thread.join(timeout=3.0)
      assert notified == [False]

   def test_concurrent_deliver_no_data_loss(self, mailbox: ThreadSafeMailbox):
      """Test 4 threads delivering 25 messages each leave one latest message per sender and raise nothing."""
      errors: list[str] = []

      def deliverer(sender_id: str):
         try:
            for i in range(25):
               mailbox.deliver("receiver", _msg(drone_id=sender_id, timestamp=i))
         except Exception as exc:  # pragma: no cover - a failure here is the point of the test
            errors.append(f"{sender_id}: {exc}")

      threads = [threading.Thread(target=deliverer, args=(f"sender-{i}",)) for i in range(4)]
      for thread in threads:
         thread.start()
      for thread in threads:
         thread.join(timeout=10.0)

      assert errors == []
      inbox = mailbox.receive_latest("receiver")
      assert set(inbox) == {f"sender-{i}" for i in range(4)}
      assert all(message.timestamp == 24 for message in inbox.values())

   def test_concurrent_deliver_and_read(self, mailbox: ThreadSafeMailbox):
      """Test a reader polling while writers deliver never sees a half-built inbox."""
      errors: list[str] = []
      stop = threading.Event()

      def writer(sender_id: str):
         try:
            for i in range(50):
               mailbox.deliver("reader", _msg(drone_id=sender_id, timestamp=i))
         except Exception as exc:  # pragma: no cover - a failure here is the point of the test
            errors.append(f"writer {sender_id}: {exc}")

      def reader():
         try:
            while not stop.is_set():
               for message in mailbox.receive_latest("reader").values():
                  assert isinstance(message, TrajectoryMessage)
                  assert message.trajectory.shape == (3, 3)
         except Exception as exc:  # pragma: no cover - a failure here is the point of the test
            errors.append(f"reader: {exc}")

      reader_thread = threading.Thread(target=reader)
      reader_thread.start()
      writers = [threading.Thread(target=writer, args=(f"writer-{i}",)) for i in range(2)]
      for thread in writers:
         thread.start()
      for thread in writers:
         thread.join(timeout=10.0)
      stop.set()
      reader_thread.join(timeout=10.0)

      assert errors == []

   def test_clear_removes_delivered_messages(self, mailbox: ThreadSafeMailbox):
      """Test clear() wipes directly delivered messages like broadcast ones."""
      mailbox.deliver("d2", _msg(drone_id="d1"))

      mailbox.clear()

      assert mailbox.receive_latest("d2") == {}
