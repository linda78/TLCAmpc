"""Central MPC coordinator with adaptive-zone conflict detection and evasion.

Wraps :class:`CentralMPCGlobalCoordinator` to add pre-solve conflict detection
via :class:`~drone_sim.simulation.conflict_detection.ConflictDetector`.  When a
pair of adaptive-zone drones are detected on a collision course, the
*lower* drone_id deflects by temporarily overriding its active route waypoint;
the higher drone_id holds its planned trajectory unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drone_sim.domain.drone import Drone, has_central_cost
from drone_sim.domain.registry import register_coordinator
from drone_sim.simulation.centralized.coordinator import CentralMPCGlobalCoordinator
from drone_sim.simulation.utils.conflict_detection import ConflictDetector
from drone_sim.simulation.distributed.neighbor_graph import NeighborGraph


@register_coordinator("conflict_evasion_central")
@dataclass
class ConflictEvasionCentralCoordinator(CentralMPCGlobalCoordinator):
    """Central MPC coordinator with adaptive-zone conflict detection and evasion.

    Detects when adaptive-zone drones are on a collision course (TTC below
    threshold for N consecutive steps) and injects temporary z-reflection
    evasion waypoints before the MPC solve.

    Centralized role assignment: lower drone_id deflects, higher drone_id
    holds its planned trajectory.
    """

    comm_radius: float | None = None  # For neighbor detection (None = all-to-all)

    # Internal state — not constructor parameters
    _detector: ConflictDetector = field(init=False)
    _neighbor_graph: NeighborGraph = field(init=False)
    _saved_waypoints: dict[str, list[np.ndarray]] = field(
        default_factory=dict, init=False
    )
    _saved_idx: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._detector = ConflictDetector()
        self._neighbor_graph = NeighborGraph(comm_radius=self.comm_radius)

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

        1. Predict warm-start trajectories for conflict detection.
        2. Update neighbor graph and run ConflictDetector.
        3. Inject evasion waypoints for deflecting drones (lower drone_id).
        4. Call parent solve_controls().
        5. Restore original routes.
        6. Return controls.

        :param perception_mailbox: **Accepted and ignored**, forwarded to
           :meth:`CentralMPCGlobalCoordinator.solve_controls`, which logs it at DEBUG and drops it —
           a central solve has no per-drone inbox to feed. Present so the simulator can pass the
           kwarg unconditionally.
        """
        # 1. Get predicted trajectories for conflict detection
        trajectories = self._get_predicted_trajectories(drones)

        # 2. Update neighbor graph and run detector
        positions = {d.drone_id: d.x[:3] for d in drones}
        self._neighbor_graph.update(positions)
        self._detector.update(drones, trajectories, self._neighbor_graph)

        # 3. Inject evasion waypoints for deflecting drones
        modified_drones: list[Drone] = []
        for drone in drones:
            if not self._detector.is_evading(drone.drone_id):
                continue

            # Centralized role: only the drone with the *lower* drone_id deflects
            state = self._detector.get_state(drone.drone_id)
            conflicting = state.conflicting_neighbors
            if not conflicting:
                continue  # safety guard

            # String comparison: lower ID deflects, higher ID holds
            if drone.drone_id >= min(conflicting):
                # This drone holds — do not inject a waypoint
                continue

            # Deflector: save route and inject evasion waypoint
            evasion_wp = self._detector.get_evasion_waypoint(drone.drone_id)
            if evasion_wp is None:
                continue

            self._saved_waypoints[drone.drone_id] = drone.route.waypoints.copy()
            self._saved_idx[drone.drone_id] = drone.route.idx
            drone.route.waypoints.insert(0, evasion_wp)
            drone.route.idx = 0
            modified_drones.append(drone)

        # 4. Call parent solver
        controls = super().solve_controls(
            drones=drones,
            obstacles=obstacles,
            room_min=room_min,
            room_max=room_max,
            lstm_provider=lstm_provider,
            perception_mailbox=perception_mailbox,
        )

        # 5. Restore modified routes
        for drone in modified_drones:
            did = drone.drone_id
            if did in self._saved_waypoints:
                drone.route.waypoints = self._saved_waypoints.pop(did)
                drone.route.idx = self._saved_idx.pop(did)

        return controls

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_predicted_trajectories(
        self, drones: list[Drone]
    ) -> dict[str, np.ndarray]:
        """Build initial-guess trajectories for conflict detection.

        For each drone with the central-cost interface, rolls out
        ``central_initial_guess()`` using the physics model to produce a
        (H, 3) position trajectory.  Falls back to tiling the current
        position when the controller lacks the interface.

        :param drones: All drones in the scene.
        :return: Dict mapping drone_id to (H, 3) position array.
        """
        trajectories: dict[str, np.ndarray] = {}

        for drone in drones:
            if has_central_cost(drone.controller):
                # Roll out the initial-guess control sequence
                u0 = drone.controller.central_initial_guess(drone)  # type: ignore[attr-defined]
                positions = []
                x = np.asarray(drone.x, dtype=float)
                for k in range(self.horizon):
                    if k < len(u0):
                        u = u0[k]
                    else:
                        u = u0[-1]
                    x = drone.physics.step(x, u)
                    positions.append(x[:3].copy())
                trajectories[drone.drone_id] = np.stack(positions, axis=0)
            else:
                # Static fallback: repeat current position
                trajectories[drone.drone_id] = np.tile(
                    drone.x[:3], (self.horizon, 1)
                )

        return trajectories
