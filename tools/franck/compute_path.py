# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:10:22 2026

@author: Hamekong
"""

#%% imports

import numpy as np

# --- Import other python files ---
from tools.franck.generate_points import generate_random_points

# %% SHOW THE DRONES REFERENCE PATHS AS DOTTED LINES FROM INITIAL POSITION TO TARGET POSITION
# Comment: function name says something else. Fine for own usage of course, but for the overall usage there is at least a bit of confusion.
# Because i only saw the naming, i created drone_sim.utils.helper.start_to_dest_ref_path for the reference path. (LM)
def compute_n_drone_collision_paths(positions, cube_side, T=100, collision_time=0.5, post_collision_steps=20):
    """
    Generates reference paths where N drones create a collision scenario.
    
    Args:
        positions (list of np.array): Starting positions of all drones [[x1,y1,z1], [x2,y2,z2], ...]
        cube_side (float): Side of cubic airspace
        T (int): Total timesteps for the paths
        collision_time (float): Fraction of T when collision should occur (0-1)
        post_collision_steps (int): How many steps to continue after collision
        
    Returns:
        list of np.ndarray: Reference paths for all drones
        collision_point (np.ndarray): Point at which collision occurs
    """


    # --- check that there are values in list position ---    
    num_drones = len(positions)
    if num_drones == 0:
        return []
    
    positions = np.array(positions)
    
    # Calculate collision point as centroid of all positions
    collision_point = np.zeros(3)
    
    # Calculate time steps
    t_collision = int(T * collision_time) # number of timesteps before collision
    t_total = T + post_collision_steps # total number of timesteps including post collision stepts
      
    
    # --- Initialize paths for all drones ---
    n_paths = [np.zeros((t_total, 3)) for _ in range(num_drones)]
    
    for i in range(num_drones):
        # Path from start to collision point
        n_paths[i][:t_collision] = np.linspace(positions[i], collision_point, t_collision)
        
        # Path continues beyond collision point; mirror path of before collision
        n_paths[i][t_collision:] = np.linspace(collision_point, -positions[i], post_collision_steps + T//2 + 1)[1:]
           
    return [path[:T] for path in n_paths], collision_point

# %% test

if __name__ == '__main__':
    
    # --- Parameters ---
    cube_side = 6.0
    num_drones = 8
    drone_radius = 1.0
    d_r = 2 * drone_radius
    seed = 40
    
    positions = generate_random_points(drone_radius, cube_side, num_drones, d_r, seed)
    
    paths =compute_n_drone_collision_paths(positions, cube_side)
    
    print(paths)