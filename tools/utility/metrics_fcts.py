import numpy as np
from drone_sim.domain.drone import Drone

def all_drones_reached_destination(drones: list[Drone], thresh: float = 0.1) -> bool:
   reached = [drone.route.target_reached(position=drone.position(), thresh=0.1) for drone in drones]
   return all(reached)

def pairwise_distances(positions: list[np.ndarray]) -> np.ndarray:
   """Compute pairwise distances between all positions."""
   num_positions = len(positions)
   if num_positions <= 1:
      return np.asarray([], dtype=float)

   distances: list[float] = []
   for i in range(num_positions):
      for j in range(i + 1, num_positions):
         distances.append(float(np.linalg.norm(positions[i] - positions[j])))

   return np.asarray(distances, dtype=float)

def compute_jerk_3d_value(positions_by_drone: dict[str, list[np.ndarray]]) -> float:
   jerk_3d_total = 0.0
   for traj in positions_by_drone.values():
      if len(traj) < 2:
         continue
      pts = np.stack(traj, axis=0)
      loss, _, _ = piecewise_linear_loss_3d(pts)
      jerk_3d_total += float(loss)
   return jerk_3d_total

def piecewise_linear_loss_3d(points, penalty=1.0, eps_step=1e-6, angle_threshold_deg=90.0):
   """
   Piecewise-linear fit of a 3D trajectory with penalties for direction turnarounds.

   Args:
       points: array-like of shape (n, 3)
           Sequence of 3D points [x_i, y_i, z_i].
       penalty: float
           Cost added for each turnaround (i.e. each break between line segments).
       eps_step: float
           Threshold to treat very small step vectors as zero (noise).
       angle_threshold_deg: float
           Angle (in degrees) at or above which the change in direction is considered a "turnaround" and causes a new segment to start.
           Default 90°, i.e. direction flips from "mostly one way" to "mostly the opposite way".

   Returns:
       loss: float
           Total loss = sum of squared Euclidean errors to piecewise-linear fit
           + penalty * (# of breaks).
       fitted: ndarray of shape (n, 3)
           Smoothed/fitted 3D points.
       segment_starts: list[int]
           Indices where each segment starts (0 is always included).
   """
   pts = np.asarray(points, dtype=float)
   if pts.ndim != 2 or pts.shape[1] != 3:
      raise ValueError("points must have shape (n, 3)")

   n = pts.shape[0]
   if n < 2:
      # Nothing to fit
      return 0.0, pts.copy(), [0]

   # Parameter along the path (could be time or just index)
   t = np.arange(n, dtype=float)

   # 1. Detect turnarounds based on 3D direction changes
   steps = np.diff(pts, axis=0)  # (n-1, 3)
   # Zero out tiny step components to reduce numerical noise
   steps[np.abs(steps) < eps_step] = 0.0

   # Compute unit direction vectors for non-zero steps
   step_norms = np.linalg.norm(steps, axis=1)
   dirs = np.zeros_like(steps)
   nonzero_mask = step_norms > eps_step
   dirs[nonzero_mask] = steps[nonzero_mask] / step_norms[nonzero_mask, None]

   cos_threshold = np.cos(np.deg2rad(angle_threshold_deg))

   breaks = [0]  # segment start indices

   for i in range(1, len(dirs)):
      d_prev = dirs[i - 1]
      d_cur = dirs[i]

      # Skip if either direction is basically undefined (zero step)
      if (np.linalg.norm(d_prev) < eps_step or np.linalg.norm(d_cur) < eps_step):
         continue

      # cos(theta) = d_prev · d_cur (both unit vectors)
      cos_angle = float(np.dot(d_prev, d_cur))

      # Turnaround if angle >= angle_threshold_deg
      if cos_angle < cos_threshold:
         # New segment starts at index i
         breaks.append(i)

   breaks.append(n)  # sentinel for the last segment end

   # 2. Fit line (3D) on each segment and accumulate squared error
   fitted = np.zeros_like(pts)
   sq_err = 0.0

   for s in range(len(breaks) - 1):
      start = breaks[s]
      end = breaks[s + 1]

      t_seg = t[start:end]  # shape (m,)
      pts_seg = pts[start:end]  # shape (m, 3)

      if end - start == 1:
         # Single point segment: nothing to fit
         fitted[start:end] = pts_seg
         continue

      # Design matrix for linear model: pts ≈ a * t + b
      # A has shape (m, 2)
      A = np.vstack([t_seg, np.ones_like(t_seg)]).T

      # Solve for a and b for all 3 dims at once: shape coeffs = (2, 3)
      coeffs, *_ = np.linalg.lstsq(A, pts_seg, rcond=None)
      a = coeffs[0]  # (3,)
      b = coeffs[1]  # (3,)

      # Fitted points on this segment
      fit_seg = (a[None, :] * t_seg[:, None]) + b[None, :]  # (m, 3)
      fitted[start:end] = fit_seg

      # Squared Euclidean error
      sq_err += np.sum((pts_seg - fit_seg) ** 2)

   num_segments = len(breaks) - 1
   num_breaks = max(0, num_segments - 1)
   loss = sq_err + penalty * num_breaks

   segment_starts = breaks[:-1]

   return loss, fitted, segment_starts