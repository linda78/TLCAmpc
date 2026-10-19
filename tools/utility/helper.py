import numpy as np

from drone_sim.domain.drone import Drone

def pairwise_distances(positions: list[np.ndarray]) -> np.ndarray:
   n = len(positions)
   if n <= 1:
      return np.asarray([], dtype=float)

   dists: list[float] = []
   for i in range(n):
      pi = positions[i]
      for j in range(i + 1, n):
         pj = positions[j]
         dists.append(float(np.linalg.norm(pi - pj)))
   return np.asarray(dists, dtype=float)

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


class DeadlockDetector:
   """Flags a deadlock once every drone has barely moved for a run of consecutive steps.

   Feed it the drone positions after each simulation step (and once with the start positions, so the first step is judged too).
   A drone counts as moving in a step when it travelled at least ``min_movement`` metres since the previous positions; a step is
   quiet when no drone moved. ``window`` consecutive quiet steps are a deadlock. ``window=None`` disables the check, so callers
   can pass the setting straight through without branching.
   """

   def __init__(self, window: int | None, min_movement: float) -> None:
      if window is not None and window < 1:
         raise ValueError(f"window must be >= 1 or None, got {window}")
      if min_movement < 0.0:
         raise ValueError(f"min_movement must be >= 0, got {min_movement}")
      self.window = window
      self.min_movement = min_movement
      self.quiet_steps = 0
      self._previous: dict[str, np.ndarray] | None = None

   def update(self, positions: dict[str, np.ndarray]) -> bool:
      """Record the positions after a step. Returns True once the quiet run has reached ``window`` steps."""
      if self.window is None:
         return False
      current = {drone_id: np.array(p, dtype=float, copy=True) for drone_id, p in positions.items()}
      if self._previous is not None and current.keys() == self._previous.keys() and self._all_quiet(current):
         self.quiet_steps += 1
      else:
         self.quiet_steps = 0
      self._previous = current
      return self.quiet_steps >= self.window

   def _all_quiet(self, current: dict[str, np.ndarray]) -> bool:
      assert self._previous is not None
      return all(float(np.linalg.norm(p - self._previous[drone_id])) < self.min_movement for drone_id, p in current.items())
