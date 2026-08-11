"""Distributed MPC coordinator with adaptive-zone conflict detection and evasion.

Wraps :class:`DistributedMPCCoordinator` to add pre-solve conflict detection
via :class:`~drone_sim.simulation.conflict_detection.ConflictDetector`.  When a
pair of adaptive-zone drones are detected on a collision course, *both* drones
deflect independently — there is no negotiation or role assignment.  Each drone
reflects its own z-velocity to diverge from the collision course.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.drone import Drone
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.utils.conflict_detection import ConflictDetector
from drone_sim.simulation.distributed.distributed_coordinator import (
    DistributedMPCCoordinator,
)


@register_coordinator("conflict_evasion_distributed")
@dataclass
class ConflictEvasionDistributedCoordinator(DistributedMPCCoordinator):
    """Distributed MPC coordinator with adaptive-zone conflict detection and evasion.

    Each drone reacts independently using z-velocity reflection — no
    negotiation between drones.  Both drones in a conflicting pair will
    deflect (in opposite z-directions for head-on pairs due to z-reflection).
    """

    # Internal state — not constructor parameters
    _detector: ConflictDetector = field(init=False)
    _saved_waypoints: dict[str, list[np.ndarray]] = field(
        default_factory=dict, init=False
    )
    _saved_idx: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._detector = ConflictDetector()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve_controls(
        self,
        *,
        drones: list[Drone],
        obstacles: list[tuple[np.ndarray, np.ndarray]],
        room_min: np.ndarray | None = None,
        room_max: np.ndarray | None = None,
        lstm_provider: object | None = None,
        perception_mailbox: object | None = None,
    ) -> dict[str, np.ndarray]:
        """Solve with pre-solve conflict detection and evasion waypoint injection.

        1. Update neighbor graph from current positions.
        2. Get warm-start trajectories via init_trajectories().
        3. Run ConflictDetector.
        4. Inject evasion waypoints for all evading drones (both sides deflect).
        5. Call parent solve_controls() — which re-updates the neighbor graph
           internally (idempotent).
        6. Restore original routes.
        7. Return controls.

        :param perception_mailbox: Passed straight through to
           :meth:`DistributedMPCCoordinator.solve_controls`, which switches to single-pass
           perception mode when it is not ``None``. Conflict detection itself stays on true
           positions and true warm-start trajectories — it is a pre-solve safety net, not part
           of the perception loop.
        """
        # 1. Update neighbor graph before calling init_trajectories so that
        #    neighbor pairs are available for conflict detection.
        positions = {
            d.drone_id: np.asarray(d.x, dtype=float)[:3] for d in drones
        }
        self._neighbor_graph.update(positions)

        # 2. Get initial warm-start trajectories for conflict detection
        trajectories, _local_solvers = self.init_trajectories(drones)

        # Add static position fallback for non-optimized drones
        for drone in drones:
            if drone.drone_id not in trajectories:
                pos = np.asarray(drone.x, dtype=float)[:3]
                trajectories[drone.drone_id] = np.tile(pos, (self.horizon, 1))

        # 3. Run conflict detector
        self._detector.update(drones, trajectories, self._neighbor_graph)

        # 4. Inject evasion waypoints — both drones in a pair deflect
        modified_drones: list[Drone] = []
        for drone in drones:
            if not self._detector.is_evading(drone.drone_id):
                continue

            evasion_wp = self._detector.get_evasion_waypoint(drone.drone_id)
            if evasion_wp is None:
                continue

            self._saved_waypoints[drone.drone_id] = drone.route.waypoints.copy()
            self._saved_idx[drone.drone_id] = drone.route.idx
            drone.route.waypoints.insert(0, evasion_wp)
            drone.route.idx = 0
            modified_drones.append(drone)

        # 5. Call parent solver (re-updates neighbor graph; idempotent)
        controls = super().solve_controls(
            drones=drones,
            obstacles=obstacles,
            room_min=room_min,
            room_max=room_max,
            lstm_provider=lstm_provider,
            perception_mailbox=perception_mailbox,
        )

        # 6. Restore modified routes
        for drone in modified_drones:
            did = drone.drone_id
            if did in self._saved_waypoints:
                drone.route.waypoints = self._saved_waypoints.pop(did)
                drone.route.idx = self._saved_idx.pop(did)

        return controls
