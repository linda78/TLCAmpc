"""Thread-safe mailbox for asynchronous trajectory exchange.

Provides thread-safe message passing for v3.0 threaded distributed MPC.
Each drone has an inbox storing the latest message from each sender.
Supports non-blocking read, non-blocking post, and blocking wait-for-update.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drone_sim.simulation.distributed.trajectory_exchange import TrajectoryMessage


class ThreadSafeMailbox:
    """Thread-safe mailbox for asynchronous trajectory exchange.

    Each drone has an inbox storing the latest message from each sender.
    Supports non-blocking read, non-blocking post, and blocking wait-for-update.

    Thread safety:
    - Uses threading.RLock to protect _inboxes dict structure
    - Uses per-drone threading.Condition for event notification
    - Lock hierarchy: never hold _lock while notifying Conditions to avoid deadlock
    """

    def __init__(self) -> None:
        """Initialize empty mailbox with thread safety primitives."""
        self._lock = threading.RLock()
        self._inboxes: dict[str, dict[str, TrajectoryMessage]] = {}
        self._conditions: dict[str, threading.Condition] = {}

    def broadcast(
        self,
        sender_id: str,
        message: TrajectoryMessage,
        neighbor_ids: set[str],
    ) -> None:
        """Broadcast trajectory message to all neighbors (non-blocking).

        Posts message to each neighbor's inbox. Latest message from same sender
        overwrites previous. Notifies waiting threads on each neighbor's Condition.

        :param sender_id: ID of the sending drone
        :param message: TrajectoryMessage to broadcast
        :param neighbor_ids: Set of neighbor drone IDs to receive message
        """
        self._post(sender_id, message, neighbor_ids)

    def deliver(self, receiver_id: str, message: TrajectoryMessage) -> None:
        """Deliver one message directly into a single drone's inbox, bypassing the NeighborGraph (non-blocking).

        Used by the perception bridge (:mod:`drone_sim.perception.bridge`), where the
        routing decision has already been made by the camera: a neighbor is in an
        observer's estimates precisely because that observer saw it, so re-running the
        geometric comm-radius test on ground truth would only second-guess the sensor.
        Otherwise this behaves exactly like a one-recipient broadcast -- the message is
        filed under ``message.drone_id``, a newer message from the same sender overwrites
        the previous one, and a thread parked in wait_for_update for this receiver is
        woken.

        :param receiver_id: ID of the receiving drone
        :param message: TrajectoryMessage to file under its own ``drone_id``
        """
        # A one-recipient post in which the sender is the observed drone itself. It goes through the
        # same primitive as broadcast() rather than through broadcast() itself, so that the two stay
        # distinguishable to anyone watching the public methods -- which is the negotiation-versus-
        # observation distinction this class is asked about.
        self._post(message.drone_id, message, (receiver_id,))

    def _post(self, sender_id: str, message: TrajectoryMessage, receiver_ids: Iterable[str]) -> None:
        """File one message into every named inbox and wake whoever waits on them.

        The single place that touches ``_inboxes`` and ``_conditions`` together, and therefore the
        single place that has to honour the lock hierarchy of this class: collect the conditions
        while holding ``_lock``, notify after releasing it.

        :param sender_id: Key the message is filed under in each inbox; the latest overwrites the previous.
        :param message: TrajectoryMessage to file
        :param receiver_ids: Inboxes to file it into
        """
        conditions_to_notify = []

        with self._lock:
            for receiver_id in receiver_ids:
                if receiver_id not in self._inboxes:
                    self._inboxes[receiver_id] = {}
                self._inboxes[receiver_id][sender_id] = message

                # Collect conditions to notify (but don't notify under lock)
                if receiver_id in self._conditions:
                    conditions_to_notify.append(self._conditions[receiver_id])

        # Notify OUTSIDE the main lock to avoid deadlock
        for condition in conditions_to_notify:
            with condition:
                condition.notify_all()

    def receive_latest(self, receiver_id: str) -> dict[str, TrajectoryMessage]:
        """Receive latest messages for a drone (non-blocking).

        Returns a shallow copy of the receiver's inbox dict so callers
        can't corrupt internal state.

        :param receiver_id: ID of the receiving drone
        :return: Dict mapping sender_id to TrajectoryMessage. Empty dict if no messages.
        """
        with self._lock:
            if receiver_id not in self._inboxes:
                return {}
            # Return shallow copy to prevent caller from modifying internal state
            return self._inboxes[receiver_id].copy()

    def wait_for_update(
        self,
        receiver_id: str,
        timeout: float | None = None,
    ) -> bool:
        """Wait for new message to arrive (blocking).

        Uses per-drone threading.Condition to block until broadcast notifies.
        Caller should call receive_latest() after this returns True to get messages.
        This two-step design avoids holding the lock while the caller processes messages.

        :param receiver_id: ID of the receiving drone
        :param timeout: Maximum time to wait in seconds (None = wait forever)
        :return: True if notified, False if timed out
        """
        # Lazy-create Condition for this drone
        with self._lock:
            if receiver_id not in self._conditions:
                self._conditions[receiver_id] = threading.Condition()
            condition = self._conditions[receiver_id]

        # Wait on the Condition (releases its internal lock while waiting)
        with condition:
            return condition.wait(timeout=timeout)

    def clear(self) -> None:
        """Clear all messages from all inboxes.

        Thread-safe. Called at start of each timestep to reset for new iteration.
        """
        with self._lock:
            self._inboxes.clear()
            # Keep conditions alive for reuse

    def clear_drone(self, drone_id: str) -> None:
        """Clear one drone's inbox without affecting others.

        Thread-safe. Useful for removing a specific drone from simulation.

        :param drone_id: ID of drone whose inbox to clear
        """
        with self._lock:
            if drone_id in self._inboxes:
                self._inboxes[drone_id].clear()
