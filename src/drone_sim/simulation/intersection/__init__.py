"""Intersection-zone coordinator package.

Detects predicted drone-drone encounters over the MPC horizon and places
a :class:`SphereObstacle` at the prediction meeting point. The conflict
loser is parked via :meth:`GatedDrone.start_waiting` until the priority
drone has passed through.
"""

from drone_sim.simulation.intersection.intersection_central_coordinator import (
   IntersectionCentralCoordinator,
)
from drone_sim.simulation.intersection.intersection_dmpc_coordinator import (
   IntersectionDMPCCoordinator,
)
from drone_sim.simulation.intersection.intersection_mixin import IntersectionMixin
from drone_sim.simulation.intersection.intersection_state import (
   IntersectionStateStore,
   SphereRecord,
)
from drone_sim.simulation.utils.stateless_conflict_detection import (
   IntersectionEvent,
   closest_approach,
   compute_threshold,
   constant_velocity_trajectory,
   detect_intersections,
   is_closing,
)

__all__ = [
   "IntersectionCentralCoordinator",
   "IntersectionDMPCCoordinator",
   "IntersectionEvent",
   "IntersectionMixin",
   "IntersectionStateStore",
   "SphereRecord",
   "closest_approach",
   "compute_threshold",
   "constant_velocity_trajectory",
   "detect_intersections",
   "is_closing",
]
