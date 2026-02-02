from __future__ import annotations

import math
import random

import numpy as np

from tools.utility import constants

# Type alias for the public API return type
StartDestPattern = list[tuple[list[float], list[float]]]


# functions for spheres
def _generate_random_point_in_sphere(max_radius: float) -> np.ndarray:
   """Generate a uniformly distributed random point within a sphere."""
   while True:
      point = np.random.uniform(-max_radius, max_radius, 3)
      if np.linalg.norm(point) <= max_radius:
         return point


def _check_valid_position_in_sphere(new_pos: np.ndarray, existing_positions: np.ndarray, r_sphere: float, r_room: float) -> bool:
   """Check if a new position is valid within the room sphere and doesn't collide with existing positions."""
   # Check if sphere fits inside the room sphere (no touching walls)
   if np.linalg.norm(new_pos) + r_sphere >= r_room:
      return False

   # Check minimum distance to all existing positions
   if len(existing_positions) > 0:
      distances = np.linalg.norm(existing_positions - new_pos, axis=1)
      if np.min(distances) < 2 * r_sphere:
         return False

   return True


def _generate_positions_in_sphere(n_drones: int, r_sphere: float, r_room: float, max_attempts: int = 10000, seed: int | None = None) -> StartDestPattern | None:
   """Generate random non-overlapping positions within a spherical room."""
   if seed is not None:
      np.random.seed(seed)

   assert r_sphere > 0 and r_room > 0, "radii must be positive"
   assert r_sphere < r_room, "spheres with r bigger than room are not possible"

   max_center_radius = r_room - r_sphere
   positions = np.empty((0, 3))

   for _ in range(n_drones):
      for _ in range(max_attempts):
         new_pos = _generate_random_point_in_sphere(max_center_radius)
         if _check_valid_position_in_sphere(new_pos, positions, r_sphere, r_room):
            positions = np.vstack([positions, new_pos])
            break

   if len(positions) != n_drones:
      fallback = predefined_patterns_for(n_drones, r_sphere, r_room, in_sphere=True)
      if fallback is None:
         return None
      print("USE BEFORE RANDOMIZED FOUND PREDEFINED PATTERN AS FALLBACK")
      return fallback

   return _create_start_and_destinations(positions)


# functions for a box
def _calc_best_format(room_size: float, r: float) -> tuple[float, np.ndarray]:
   """Calculate optimal hexagonal close-packed sphere positions within a cubic room."""
   d = 2 * r  # minimum distance between sphere centers

   # HCP lattice spacings
   dx = d
   dy = d * math.sqrt(3) / 2
   dz = d * math.sqrt(2 / 3)

   min_coord = r
   max_coord = room_size - r

   if max_coord < min_coord:
      return 0.0, np.empty((0, 3))

   positions = []
   z = min_coord
   stack_idx = 0

   while z <= max_coord:
      offset_x = (stack_idx % 2) * (dx / 2)
      offset_y = (stack_idx % 2) * (dy / 3)

      y = min_coord + offset_y
      row = 0

      while y <= max_coord:
         x_start = min_coord + offset_x + (row % 2) * (dx / 2)
         x = x_start

         while x <= max_coord:
            positions.append([x, y, z])
            x += dx

         y += dy
         row += 1

      z += dz
      stack_idx += 1

   positions_array = np.array(positions)
   min_dist = _compute_min_pairwise_distance(positions_array)
   return min_dist, positions_array


def _generate_positions_in_box(n_drones: int, room_min: np.ndarray, room_max: np.ndarray, min_pair_distance: float = 2.41, wall_margin: float = 1.0,
                               max_attempts: int = 100000) -> StartDestPattern | None:
   """Generate random non-overlapping positions within a box-shaped room."""
   inner_min = room_min + wall_margin
   inner_max = room_max - wall_margin

   positions = np.empty((0, 3))

   for _ in range(max_attempts):
      if len(positions) >= n_drones:
         break

      point = np.random.uniform(inner_min, inner_max)

      # Check distance to all existing positions
      if len(positions) == 0 or np.min(np.linalg.norm(positions - point, axis=1)) >= min_pair_distance:
         positions = np.vstack([positions, point])

   if len(positions) < n_drones:
      # In cubic room, we may have fallback options
      if np.allclose(room_min, room_min[0]) and np.allclose(room_max, room_max[0]):
         r_sphere = min_pair_distance / 2
         room_size = room_max[0] - room_min[0]

         fallback = predefined_patterns_for(n_drones, r_sphere, room_size, in_sphere=False)
         if fallback is not None:
            print("USE PREDEFINED PATTERN AS FALLBACK")
            return fallback

         min_dist, optimized_positions = _calc_best_format(room_size, r_sphere)
         if min_dist >= min_pair_distance:
            print("USE FALLBACK")
            return _create_start_and_destinations(optimized_positions)

      print(f"Failed to place {n_drones} drones with d_min={min_pair_distance} and wall_margin={wall_margin} after {max_attempts} attempts.")
      return None

   return _create_start_and_destinations(positions)


# general functions
def _create_start_and_destinations(positions: np.ndarray) -> StartDestPattern:
   """Convert internal numpy positions to the public API format with mirrored destinations."""
   pattern: StartDestPattern = []
   for pos in positions:
      start = pos.tolist()
      dest = (-pos).tolist()
      pattern.append((start, dest))
   return pattern


def _compute_min_pairwise_distance(positions: np.ndarray) -> float:
   """Compute the minimum pairwise distance between positions."""
   if len(positions) < 2:
      return float("inf")

   min_dist = float("inf")
   for i in range(len(positions)):
      distances = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
      if len(distances) > 0:
         min_dist = min(min_dist, np.min(distances))

   return min_dist


def generate_positions(n_drones: int, room_min: list[float] = list, room_max: list[float] = list, min_pair_distance: float = 2.41, wall_margin: float = 1.0,
                       r_room: float = 0.0, max_attempts: int = 100000) -> StartDestPattern:
   """Generate drone start/destination positions in either a spherical or box-shaped room."""

   if r_room > 0:
      results = _generate_positions_in_sphere(n_drones=n_drones, r_sphere=min_pair_distance / 2, r_room=r_room, max_attempts=max_attempts)
      if results is None:
         print("RUN IN FALLBACK, without having any")
         raise RuntimeError(f"Failed to place {n_drones} drones with d_min={min_pair_distance} and r_room={r_room} after {max_attempts} attempts.")
      return results

   results = _generate_positions_in_box(n_drones, np.array(room_min), np.array(room_max), min_pair_distance, wall_margin, max_attempts)
   if results is None:
      raise RuntimeError(f"Failed to place {n_drones} drones with d_min={min_pair_distance} after {max_attempts} attempts.")
   return results


def predefined_patterns_for(n_drones: int, r_sphere: float, room_size: float, in_sphere: bool = False) -> StartDestPattern | None:
   """Look up predefined patterns for common configurations."""
   if in_sphere:
      if room_size - 3.1 < 0.01 and r_sphere - 1.0 < 0.1:
         return constants.PREDEFINED_POSITIONS_IN_SPHERE_3_1.get(n_drones)
   else:
      if room_size - 8.0 < 0.01 and r_sphere - 1.0 < 0.1:
         return constants.PREDEFINED_PATTERNS_IN_BOX_8.get(n_drones)
      if room_size - 5.0 < 0.01 and r_sphere - 1.0 < 0.1:
         return constants.PREDEFINED_PATTERNS_IN_BOX_5.get(n_drones)
      if room_size - 9.0 < 0.01 and r_sphere - 1.5 < 0.1:
         return constants.PREDEFINED_POSITIONS_IN_BOX_9_R_1_5.get(n_drones)
   return None


# Test functions rudimentary
def main_sphere(n_drones: int, r_room: float, r_sphere: float) -> None:
   print("\nGenerating patterns in sphere...")
   pattern = generate_positions(n_drones=n_drones, r_room=r_room, min_pair_distance=r_sphere*2+0.001, wall_margin=r_sphere+0.001)
   # pattern = _generate_positions_in_sphere(n_drones=n_drones, r_sphere=r_sphere, r_room=r_room, seed=42)
   print(pattern)
   if pattern is not None:
      print(f"Successfully placed all {n_drones} spheres!")
      dict = {}
      for d in range(2, n_drones):
         pattern = generate_positions(n_drones=n_drones, r_room=r_room, min_pair_distance=r_sphere*2+0.001, wall_margin=r_sphere+0.001)
         dict[d] = pattern
      print(dict)


def main_box(n_drones: int, room_size: float, r_sphere: float) -> None:
   random.seed(0)
   np.random.seed(0)

   room_min = [-room_size/2] * 3
   room_max = [room_size/2] * 3
   min_pair_distance = r_sphere * 2 + 0.001
   wall_margin = r_sphere + 0.001

   print("\nGenerating patterns in box...")
   pattern = generate_positions(n_drones=n_drones, room_min=room_min, room_max=room_max, min_pair_distance=min_pair_distance, wall_margin=wall_margin,
                                max_attempts=100000)
   if pattern is not None:
      print(f"Successfully placed all {n_drones} spheres!")
      print(pattern)

   dict = {}
   for d in range(2, n_drones):
      pattern = generate_positions(n_drones=d, room_min=room_min, room_max=room_max, min_pair_distance=min_pair_distance, wall_margin=wall_margin,
                                   max_attempts=100000)
      dict[d] = pattern

   print("\nGenerating patterns in box...")
   print(dict)



if __name__ == "__main__":
   main_box(5, 9, 2)
   # main_sphere(5, 5.58, 2)
