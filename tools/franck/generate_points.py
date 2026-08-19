# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:02:19 2026

@author: Hamekong
"""
# %% imports

import numpy as np

#%% GENERATE UNIFORM POINTS
def drone_jam(cube_side, drone_radius, v_max, u_max):
    """
    Determines the number of drones with dynamics required to cause a deadlock/jam within the bounded airspace
    Corresponds to N_jam in mathematical formulations
    
    Only difference with N_cov is that it considers an extra distance, breaking_dist
        as a distance allowed for the drone to slow down when there is a collision ahead
    
    Args:
        cube_side (float): Side of bounded cubic airspace
        sphere_radius (float): Radius of spheres to be packed
        v_max (float): Maximum permissible drone velocity
        u_max (float): Maximum permissible drone acceleration 
        
    Returns:
        positions (list): List of initial drone positions that will create a deadlock
    """    

    
    # --- Parameters from the scenario ---
    breaking_dist = (v_max ** 2) / u_max # margin for drone to slow down when obstacle is detected
    drone_sep = (2 * drone_radius) + breaking_dist # rho in mathematical calculations
    packing_density = (25 * np.pi) / (24 * np.sqrt(5))  # Sphere Covering: Theoretical BCC density ≈ 1.463503
    tiling_step = packing_density * drone_sep # actual spacing between drones
    

    # --- Obtain theorical number of drones that can fit in the cube in farthest static packing ---
    cube_volume = cube_side ** 3
    sphere_volume = (4/3) * np.pi * drone_sep**3 # for N_cov, r = 2*radius
    theoretical_max = int((cube_volume * packing_density) / sphere_volume)

    
    # --- Calculation of the grid coverage ---
    half_side = cube_side/2
    min_bound = -half_side + drone_radius
    max_bound = half_side - drone_radius
    
    
    # --- Center points spaced using tiling_step ---
    x_centers = np.arange(min_bound, max_bound, tiling_step)
    y_centers = np.arange(min_bound, max_bound , tiling_step)
    z_centers = np.arange(min_bound, max_bound , tiling_step)
    
    
    # --- Keep only points within cube boundaries ---
    valid_x = x_centers[(x_centers >= (-half_side)) & (x_centers <= (half_side))]
    valid_y = y_centers[(y_centers >= (-half_side)) & (y_centers <= (half_side))]
    valid_z = z_centers[(z_centers >= (-half_side)) & (z_centers <= (half_side))]
    
    
    # --- 3D lattice of drone centers ---
    X, Y, Z = np.meshgrid(valid_x, valid_y, valid_z)
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return positions

# %% GENERATE RANDOM POINTS WITHIN THE AIRSPACE
def generate_random_points(drone_radius, cube_side, N: int, d_r, seed, max_attempts: int = 1000) -> np.ndarray:
   """
   Randomly generates a number of points, N, within the bounded airspace with
   a minimum specified distance, d_r, between every point

   Args:
       drone_radius (float): Drone radius
       cube_side (float): Side of cubic airspace
       N (int): Number of points to be generated
       d_r (float): Minimum distance between generated points
       seed (int): Seed used for the reproduceability of the random selection of points
       max_attempts (int): Maximum number of attemps to generating the N number of points
                   before concluding that it is impossible to place this N number of drones,
                   d_r apart within the airspace

   Returns:
       points (list of np.array): Positions of randomly generations points satisfying all specified requirements
   """

   if N <= 0 or d_r <= 0:
      raise ValueError("N and d_r must be positive.\n")

   # --- seed for reproduceability of random points ---
   np.random.seed(seed)

   # --- set airspace bounds ---
   half_side = cube_side / 2
   min_bound = -half_side + drone_radius
   max_bound = half_side - drone_radius

   # --- Calculate the minimum required squared distance ---
   d_r_sq = d_r ** 2

   # --- Obatin array of accepted points ---
   points = np.empty((0, 3))  # returns an empty array of shape (0,3)

   for i in range(N):
      attempts = 0

      while attempts < max_attempts:
         # Generate random point on a unit sphere (using uniform sampling of the sphere)
         new_point = (2 * max_bound * np.random.rand(3)) + min_bound  # np.random.rand(3) generates a (1,3) array with elements btw 0.0 and 1.0, excluding 1.0

         # Check distance constraint
         is_valid = True
         if points.shape[0] > 0:  # check that "points" is not empty
            # Calculate squared Euclidean distances to all existing points
            distances_sq_point = np.sum((points - new_point) ** 2, axis=1)

            # Check if any distance is less than the required minimum
            if np.any(distances_sq_point < d_r_sq):
               is_valid = False

         if is_valid:
            # Append the valid point
            points = np.vstack([points, new_point])
            break  # Move to the next point

         attempts += 1

      if attempts == max_attempts:
         print(f"Warning: Could not place target point {i + 1} after {max_attempts} attempts. Stopping at {points.shape[0]} points.\n")
         break

   # --- Display generated points and their distances apart ---
   # print(f'\nGenerated points: {points}\n')
   # distances = np.linalg.norm(points[:,np.newaxis] - points, axis=2)
   # print(f'Pairwise distances between points: {distances}\n')

   return points


# %% test

if __name__ == '__main__':
   x = True
