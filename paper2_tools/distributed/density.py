"""Constant-density scaling experiment (todo.md, block B): definition of the campaign cells.

The volume campaigns keep the room at 8 m and let the density grow with N. Here the room grows with N so that the density
stays at the reference N = 10 in the 8 m room, and the cost per drone is measured instead of the cost of the whole fleet.

- room side L(N) = 8 m * (N / 10)^(1/3), so N = 10, 20, 40, 80 give 8.0, 10.1, 12.7, 16.0 m
- every target lies TRAVEL_DISTANCE from its start in a random direction. Targets mirrored at the room centre (the first two runs
  of this experiment) keep the density only at the start: all paths cross the centre, so mid-flight all N drones meet there
  and the way grows with L. With a fixed way the density holds for the whole flight and a drone has the same task at every N
- DMPC neighbourhood r_com = 2 r_s + 2 V_max H dt, the largest distance at which two drones can still touch within the horizon
  (7.04 m for r_s = 1.52 m, V_max = 2.5 m/s, H = 8, dt = 0.1 s)
- DMPC rounds per step k in {1, 3}; the central MPC has no rounds and takes no comm radius
- 10 runs per cell, 12 h timeout per run (the central MPC at N = 40 needs it, DMPC runs take minutes), placement seeds as in the revision campaign (config.placement_seed)

``python -m paper2_tools.distributed.create_queue --experiment density`` puts the tasks into the shared scenario_runs queue as
experiment 'density' with variant 'k1' / 'k3' ('' for the central MPC), so the Hetzner workers of the volume campaign run them
unchanged. Everything a run needs beyond the worker command line (room_size, comm_radius, max_admm_iter, timeout_s, travel_distance)
travels in the task row.
"""

from __future__ import annotations

from typing import Any

from paper2_tools.distributed.config import DENSITY_EXPERIMENT, DT, HORIZON
from paper2_tools.distributed.db import make_task

EXPERIMENT = DENSITY_EXPERIMENT
CONTROLLER = "mpc_agent_adaptive"
CENTRAL = "mpc_central"
DISTRIBUTED = "dmpc_admm"

REF_N = 10
REF_ROOM = 8.0
TRAVEL_DISTANCE = REF_ROOM / 2  # longer ways get cut off by the walls of the 8 m room, whose inner cube has a 6 m side

DMPC_DRONES: tuple[int, ...] = (10, 20, 40, 80)
CENTRAL_DRONES: tuple[int, ...] = (10, 20, 40)
ADMM_ITERS: tuple[int, ...] = (1, 3)
RUNS = 10
TIMEOUT_S = 12 * 3600

# Worker defaults of the campaigns (variables.tf / worker.py); only used for the comm radius here.
STATIC_SAFETY_ZONE = 1.52
V_MAX = 2.5


def room_size_for(n_drones: int, ref_n: int = REF_N, ref_room: float = REF_ROOM) -> float:
   """Room side length that keeps the density of ``ref_n`` drones in a ``ref_room`` cube."""
   return ref_room * (n_drones / ref_n) ** (1.0 / 3.0)


def comm_radius_for(static_safety_zone: float = STATIC_SAFETY_ZONE, v_max: float = V_MAX, horizon: int = HORIZON, dt: float = DT) -> float:
   """Largest centre distance at which two drones can still touch within the horizon: 2 r_s + 2 V_max H dt."""
   return 2.0 * static_safety_zone + 2.0 * v_max * horizon * dt


def variant_for(coordinator: str, max_admm_iter: int | None) -> str:
   """Queue variant label: 'k<rounds>' for the distributed MPC, empty for the central one."""
   return f"k{max_admm_iter}" if coordinator == DISTRIBUTED else ""


def build_tasks(*, runs: int = RUNS, dmpc_drones: tuple[int, ...] = DMPC_DRONES, central_drones: tuple[int, ...] = CENTRAL_DRONES,
                admm_iters: tuple[int, ...] = ADMM_ITERS, timeout_s: int = TIMEOUT_S, comm_radius: float | None = None,
                travel_distance: float = TRAVEL_DISTANCE) -> list[dict[str, Any]]:
   """All queue rows of the experiment (see db.make_task), DMPC cells first, then the central ones."""
   r_com = comm_radius_for() if comm_radius is None else comm_radius
   tasks: list[dict[str, Any]] = []
   for k in admm_iters:
      for n in dmpc_drones:
         for run in range(runs):
            tasks.append(make_task(run, n, DISTRIBUTED, CONTROLLER, experiment=EXPERIMENT, variant=variant_for(DISTRIBUTED, k),
                                   room_size=room_size_for(n), comm_radius=r_com, max_admm_iter=k, timeout_s=timeout_s, travel_distance=travel_distance))
   for n in central_drones:
      for run in range(runs):
         tasks.append(make_task(run, n, CENTRAL, CONTROLLER, experiment=EXPERIMENT, variant=variant_for(CENTRAL, None),
                                room_size=room_size_for(n), comm_radius=None, max_admm_iter=None, timeout_s=timeout_s, travel_distance=travel_distance))
   return tasks


def cells(tasks: list[dict[str, Any]]) -> list[tuple[tuple[str, str, int], int]]:
   """(coordinator, variant, N) of every cell with its run count, in queue order."""
   seen: dict[tuple[str, str, int], int] = {}
   for t in tasks:
      key = (t["coordinator"], t["variant"], t["n_drones"])
      seen[key] = seen.get(key, 0) + 1
   return list(seen.items())
