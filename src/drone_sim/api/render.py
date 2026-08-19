from __future__ import annotations

import io

import numpy as np
from itertools import combinations
from drone_sim.domain.drone import Drone
from drone_sim.api.utils.render_helper import draw_room_wireframe, draw_obstacles, draw_neighbor_links, draw_sphere_wireframe, draw_trace, \
   draw_ghost_max_sphere, draw_obj_mesh, draw_trace_for_franck
from drone_sim.domain.utils.helper import collision_point, start_to_dest_ref_path


def render_png(*, room_min: np.ndarray, room_max: np.ndarray, drones: list[Drone], drone_traces: dict[str, list[np.ndarray]],
               obstacles: list[tuple[np.ndarray, np.ndarray]], step_count: int, compute_time_s: float, neighbor_links: list[tuple[int, int]] | None = None,
               admm_iteration_count: int | None = None, admm_converged: bool | None = None, safety_alphas: list[float] | None = None, width: int = 900,
               height: int = 700, dpi: int = 120, elev: float = 20.0, azim: float = -60.0, obj_path: str | None = None, obj_scale: float = 0.3,
               draw_sphere_if_obj: bool = False, draw_reference_paths: bool = False, draw_collision_points: bool = False, draw_drone_markers: bool = False) -> bytes:
   """Render a 3D scene to PNG bytes.

   This is intentionally simple (matplotlib wireframe room + points) to keep the REST API self-contained.
   """

   # Import matplotlib lazily so the simulation core remains lightweight.
   from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
   from matplotlib.figure import Figure

   if draw_trace_for_franck:
      from tools.franck.config_plot import configure_plt
      configure_plt()

   fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
   _ = FigureCanvas(fig)

   ax = fig.add_subplot(111, projection="3d")

   draw_room_wireframe(ax, room_min, room_max)

   draw_obstacles(ax, obstacles)

   # Draw neighbor communication links
   draw_neighbor_links(ax, neighbor_links, [d.x for d in drones])

   if draw_collision_points:
      for collision in [cp for d1, d2 in combinations(drones, 2) if (cp := collision_point(d1, d2)) is not None]:
         ax.scatter(collision[0], collision[1], collision[2], color='black', marker='X', s=50, linewidth=3, depthshade=False, zorder=200)

   for drone in drones:
      pos = np.asarray(drone.position(), dtype=float).reshape(3)

      if obj_path is not None:
         draw_obj_mesh(ax, pos, obj_path, scale=drone.radius * obj_scale * 10, color=drone.color, alpha=0.8)
      else:
         r = drone.radius
         if draw_trace_for_franck:
            ax.scatter([pos[0]], [pos[1]], [pos[2]], s=max(60.0, float(r) * 400.0), c=[drone.color], linewidth=1.0, depthshade=True, label=drone.drone_id)
         else:
            ax.scatter([pos[0]], [pos[1]], [pos[2]], s=max(20.0, float(r) * 250.0), c=[drone.color], depthshade=True, label=drone.drone_id)

      if drone.safety_zone is not None and (obj_path is None or draw_sphere_if_obj):
         safety_zone = drone.compute_adaptive_radius(drone.velocity())
         alpha_val = drone.alpha if safety_alphas else 0.8
         draw_sphere_wireframe(ax, pos, radius=safety_zone, color=drone.safety_color, alpha=alpha_val, lw=1.0 if draw_trace_for_franck else 0.6)
         draw_ghost_max_sphere(ax, drone, print_always=False)

      if draw_trace_for_franck:
         draw_trace_for_franck(ax, drone_traces.get(drone.drone_id, []), drone.trace_color)
      else:
         draw_trace(ax, drone_traces.get(drone.drone_id, []), drone.trace_color)

      if draw_drone_markers:
         # Draw start marker
         ax.scatter(drone.route.start[0], drone.route.start[1], drone.route.start[2], color=drone.color, s=50, marker='s', alpha=0.7, depthshade=False)
         # Draw target marker
         ax.scatter(drone.route.start[0], drone.route.start[1], drone.route.start[2], color=drone.color, s=200, marker='X', alpha=0.9, depthshade=False, label=f"{drone.drone_id} target")

      if draw_reference_paths:
         reference_path = start_to_dest_ref_path(drone.route.start, drone.route.target, drone, dt=0.1)  # fix, think its fine enough for a print
         ax.plot(reference_path[:, 0], reference_path[:, 1], reference_path[:, 2], '--', color=drone.color, alpha=1, linewidth=1.5)

   ax.set_xlabel("X [m]")
   ax.set_ylabel("Y [m]")
   ax.set_zlabel("Z [m]")

   ax.set_xlim(float(room_min[0]), float(room_max[0]))
   ax.set_ylim(float(room_min[1]), float(room_max[1]))
   ax.set_zlim(float(room_min[2]), float(room_max[2]))

   # Keep aspect reasonable for non-cubic rooms.
   box = np.asarray(room_max) - np.asarray(room_min)
   if np.all(np.isfinite(box)) and np.all(box > 0):
      ax.set_box_aspect((float(box[0]), float(box[1]), float(box[2])))

   ax.view_init(elev=float(elev), azim=float(azim))
   title = f"t = {int(step_count)} steps | compute = {float(compute_time_s):.2f}s"
   if admm_iteration_count is not None:
      status = "Y" if admm_converged else "!"
      title += f" | ADMM: {admm_iteration_count} iter {status}"
   ax.set_title(title)

   if drones or obstacles:
      ax.legend(loc="upper right")

   buf = io.BytesIO()
   fig.savefig(buf, format="png", bbox_inches="tight")
   return buf.getvalue()
