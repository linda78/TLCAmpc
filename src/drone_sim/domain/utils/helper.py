import numpy as np

from drone_sim.domain.drone import Drone

def all_drones_reached_destination(drones: list[Drone], thresh: float = 1e-3) -> bool:
   reached = [drone.route.target_reached(position=drone.position(), thresh=thresh) for drone in drones]
   return all(reached)

def start_to_dest_ref_path(start: np.ndarray, target: np.ndarray, drone: Drone, dt: float) -> np.ndarray:
   """Build a coarse straight-line reference path from start to target, one
   ``v_max * dt`` segment at a time. Used by the GUI to draw the planned
   trajectory; not consumed by the controller.

   Degenerate inputs are normalised to sane outputs so the GUI never crashes:
   - start ≈ target           → single-point path [start]
   - v_max * dt == 0          → two-point line [start, target] (drone can't
                                step, but we still render the intent)
   - distance < v_max * dt    → two-point line [start, target] (short hop,
                                no room for an intermediate sample)
   """
   direction = target - start
   distance = np.linalg.norm(direction)
   if np.isclose(distance, 0.0):
      return start.reshape(1, 3)

   step = drone.physics.v_max() * dt
   if step == 0:
      return np.vstack([start, target])

   n_steps = int(np.floor(distance / step))
   if n_steps == 0:
      return np.vstack([start, target])

   unit = direction / distance
   distances = np.arange(1, n_steps + 1) * step
   points = start + np.outer(distances, unit)

   # if target is not reached -> append target in last step, also if that is then slightly longer than the other steps
   if not np.allclose(points[-1], target):
      points = np.vstack([points, target])

   return np.vstack([start, points])


def collision_point(drone1: Drone, drone2: Drone) -> np.ndarray:
   """
   Both drones move at constant velocity, so the distance between their centers over time is a quadratic function.
   A collision occurs as soon as the center-to-center distance drops below the sum of the two safety radii.
   The roots of the quadratic equation are the entry and exit times.
   I take the midpoint of the time interval (clipped to the shared flight window) and return the position of drone 1 at that point.

   :param drone1:
   :param drone2:
   :return: None if there is no collision, otherwise the collision point
   """
   dir1 = drone1.route.target - drone1.route.start
   dir2 = drone2.route.target - drone2.route.start
   len1 = np.linalg.norm(dir1)
   len2 = np.linalg.norm(dir2)
   u1 = dir1 / len1 if len1 > 0 else np.zeros(3)
   u2 = dir2 / len2 if len2 > 0 else np.zeros(3)

   v1 = drone1.physics.v_max() * u1
   v2 = drone2.physics.v_max() * u2

   dp = drone1.route.start - drone2.route.start
   dv = v1 - v2
   R = drone1.safety_zone + drone2.safety_zone

   # |dp + dv*t|^2 = R^2  ->  a t^2 + b t + c = 0
   a = dv @ dv
   b = 2 * dp @ dv
   c = dp @ dp - R * R

   t_max = min(len1 / drone1.physics.v_max() if drone1.physics.v_max() > 0 else np.inf, len2 / drone2.physics.v_max() if drone2.physics.v_max() > 0 else np.inf)

   if a == 0:
      if c <= 0:
         # `a == 0` means `dv == 0`: both drones move in parallel (or stand still) and their starts already overlap. If at least one drone is moving,
         # take the position of drone1 at the midpoint of the shared window. If both are stationary, `t_max` is +inf and `v1 * (inf/2) = 0 * inf = nan`,
         # then return the midpoint of the two starts instead.
         if not np.isfinite(t_max):
            return (drone1.route.start + drone2.route.start) / 2
         t_mid = t_max / 2
         return drone1.route.start + v1 * t_mid
      return None

   disc = b * b - 4 * a * c
   if disc < 0:
      return None

   sq = np.sqrt(disc)
   t_enter = (-b - sq) / (2 * a)
   t_exit = (-b + sq) / (2 * a)

   t_lo = max(t_enter, 0.0)
   t_hi = min(t_exit, t_max)

   if t_lo > t_hi:
      return None

   t_mid = (t_lo + t_hi) / 2
   return drone1.route.start + v1 * t_mid