from __future__ import annotations

import io

import numpy as np
from drone_sim.domain.drone import Drone
from drone_sim.api.utils.render_helper import draw_room_wireframe, draw_obstacles, draw_neighbor_links, draw_sphere_wireframe, draw_trace, draw_ghost_max_sphere, draw_obj_mesh

def render_png(*, room_min: np.ndarray, room_max: np.ndarray, drones: list[Drone], drone_markers = None, drone_traces: dict[str, list[np.ndarray]],
               obstacles: list[tuple[np.ndarray, np.ndarray]], reference_paths = None, collision_point = None, step_count: int, compute_time_s: float, 
               neighbor_links: list[tuple[int, int]] | None = None, admm_iteration_count: int | None = None, admm_converged: bool | None = None,
               safety_alphas: list[float] | None = None, width: int = 900, height: int = 900, dpi: int = 120, elev: float = 20.0, azim: float = -60.0,
               obj_path: str | None = None, obj_scale: float = 0.3, draw_sphere_if_obj: bool = False) -> bytes:
   """Render a 3D scene to PNG bytes.

   This is intentionally simple (matplotlib wireframe room + points) to keep the REST API self-contained.
   """

   # Import matplotlib lazily so the simulation core remains lightweight.
   from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
   from matplotlib.figure import Figure
   import matplotlib.pyplot as plt
   from tools.franck.config_plot import configure_plt

   configure_plt()

   fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
   _ = FigureCanvas(fig)

   ax = fig.add_subplot(111, projection="3d")

   draw_room_wireframe(ax, room_min, room_max)

   draw_obstacles(ax, obstacles)

   # Draw neighbor communication links
   draw_neighbor_links(ax, neighbor_links, [d.x for d in drones])
      
   if reference_paths is not None:
       colors = [d.color for d in drones]
       for path, color in zip(reference_paths, colors):
           ax.plot(path[:,0], path[:,1], path[:,2], '--', color=color, alpha=1, linewidth=1.5)
           
   if collision_point is not None:
       ax.scatter(collision_point[0], collision_point[1], collision_point[2], color='black', marker='X', s=50, linewidth=3, depthshade=False, zorder=200)

   for drone in drones:
      pos = np.asarray(drone.position(), dtype=float).reshape(3)

      if obj_path is not None:
         draw_obj_mesh(ax, pos, obj_path, scale=drone.radius * obj_scale * 10, color=drone.color, alpha=0.8)
      else:
         r = drone.radius
         ax.scatter([pos[0]], [pos[1]], [pos[2]], s=max(60.0, float(r) * 400.0), c=[drone.color], linewidth=1.0, depthshade=True, label=drone.drone_id)

      if drone.safety_zone is not None and (obj_path is None or draw_sphere_if_obj):
         safety_zone = drone.compute_adaptive_radius(drone.velocity())
         alpha_val = drone.alpha if safety_alphas else 0.8
         draw_sphere_wireframe(ax, pos, radius=safety_zone, color=drone.safety_color, alpha=alpha_val, lw=1.0)
         draw_ghost_max_sphere(ax, drone, print_always=False)

      draw_trace(ax, drone_traces.get(drone.drone_id, []), drone.trace_color)
        
   # Plot drone start/target markers
   if drone_markers is not None:
        for marker_data in drone_markers:
            p = np.asarray(marker_data["position"], dtype=float)
            ax.scatter(p[0], p[1], p[2], color=marker_data["color"], marker=marker_data["marker"],
                  s=marker_data.get("size", 300), alpha=1.0, linewidths=2, depthshade=False, zorder=200)


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
