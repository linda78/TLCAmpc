from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph


@dataclass
class TrajectoryMessage:
    """Message containing a drone's predicted trajectory.

    Used for trajectory exchange between neighboring drones during
    ADMM iterations in distributed MPC.
    """

    drone_id: str
    trajectory: np.ndarray  # (H, 3) predicted positions
    predicted_velocities: np.ndarray | None  # (H, 3) predicted velocities, or None
    timestamp: int  # Simulation timestep when message was created
    safety_zone_radius: float | np.ndarray | None = None  # Safety zone radius (scalar or per-step array)


@dataclass
class TrajectoryMailbox:
    """Mailbox for storing and routing trajectory messages between drones.

    Provides in-memory message passing for trajectory exchange in
    distributed MPC simulation. Each drone has an inbox where neighbors
    can deposit trajectory messages.
    """

    _inbox: dict[str, dict[str, TrajectoryMessage]] = field(default_factory=dict)

    def broadcast(
        self,
        sender_id: str,
        trajectory: np.ndarray,
        predicted_velocities: np.ndarray | None,
        timestamp: int,
        neighbor_graph: NeighborGraph,
    ) -> None:
        """Broadcast trajectory message to all neighbors.

        :param sender_id: ID of the sending drone
        :param trajectory: Predicted positions (H, 3)
        :param predicted_velocities: Predicted velocities (H, 3), or None
        :param timestamp: Simulation timestep when message was created
        :param neighbor_graph: NeighborGraph for determining recipients
        """
        trajectory = np.asarray(trajectory, dtype=float)
        if predicted_velocities is not None:
            predicted_velocities = np.asarray(predicted_velocities, dtype=float)
        message = TrajectoryMessage(
            drone_id=sender_id,
            trajectory=trajectory,
            predicted_velocities=predicted_velocities,
            timestamp=timestamp,
        )

        for neighbor_id in neighbor_graph.get_neighbors(sender_id):
            self._inbox.setdefault(neighbor_id, {})[sender_id] = message

    def deliver(self, receiver_id: str, message: TrajectoryMessage) -> None:
        """Deliver one message directly into a single drone's inbox, bypassing the NeighborGraph.

        Used by the perception bridge (:mod:`drone_sim.perception.bridge`), where the
        routing decision has already been made by the camera: a neighbor is in an
        observer's estimates precisely because that observer saw it, so re-running the
        geometric comm-radius test on ground truth would only second-guess the sensor.
        Everything else stays identical to broadcast -- the message is filed under
        ``message.drone_id``, and a newer message from the same sender overwrites the
        previous one.

        :param receiver_id: ID of the receiving drone
        :param message: TrajectoryMessage to file under its own ``drone_id``
        """
        self._inbox.setdefault(receiver_id, {})[message.drone_id] = message

    def receive(self, receiver_id: str) -> dict[str, TrajectoryMessage]:
        """Receive all messages for a drone.

        :param receiver_id: ID of the receiving drone
        :return: Dict mapping sender_id to TrajectoryMessage. Returns empty dict if no messages.
        """
        return self._inbox.get(receiver_id, {})

    def clear(self) -> None:
        """Clear all messages from all inboxes.

        Called at start of each timestep to reset for new ADMM iteration.
        """
        self._inbox.clear()

