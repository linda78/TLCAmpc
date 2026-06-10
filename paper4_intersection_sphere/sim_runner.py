"""Shared harness for the Paper 4 intersection-sphere factorial evaluation.

Defines the 8 arms (A1..A8) of the 2x2x2 factorial (MPC mode x sequencing x zone),
builds a ScenarioConfig per (arm, scenario, seed), runs one trial with full metrics
(control energy, makespan, min separation, 4-way outcome, sequencing activity), and
writes results to CSV.

Mirrors paper3_lstm/sim_runner.py in spirit; reuses Simulator.from_config and the
outcome classification from smoke_intersection.py.

Factor encoding:
  M (mode):       central | distributed
  S (sequencing): off (base coordinator) | on (intersection coordinator)
  Z (zone):       fixed (mpc_agent) | adaptive (mpc_agent_adaptive)
"""
from __future__ import annotations

import csv
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from drone_sim.domain.config import (ControllerSpec, DroneConfig, PhysicsSpec, RoomConfig, ScenarioConfig)
from drone_sim.domain.drone import GatedDrone
from drone_sim.domain.utils.helper import all_drones_reached_destination
from drone_sim.simulation.simulator import Simulator

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared physics / MPC parameters — identical across all arms so only the
# (coordinator, controller, zone, drone type) changes between arms.
# Values taken from configs/franck_intersection/4drone_*_intersection.json.
# ---------------------------------------------------------------------------
DT = 0.1
HORIZON = 8
V_MAX = 3.0
U_MAX = 3.0
SAFETY_ZONE = 0.6
RADIUS = 0.2
ROOM_HALF = 5.0
ALPHA = 0.33  # adaptive-zone adaptation parameter (only on adaptive arms)

Q_POS = 3.0
Q_VEL = 1.0  # adaptive controller only
R_U = 0.1

INTERSECTION_SPHERE_RADIUS = 2.0
INTERSECTION_RESOLUTION_MARGIN = 1.1

# DMPC-only solver knobs. max_admm_iter lowered from the franck config's 50 to 20
# (the paper3 default): pilot runs showed the non-sequencing DMPC baseline burns the
# full iteration budget every step on a crossing without converging, which makes 50
# impractical for a multi-seed grid. 20 leaves headroom without runaway wall time.
DMPC_PARAMS = {"rho": 1.0, "max_admm_iter": 20, "primal_tol": 1e-3, "dual_tol": 1e-3}
# Central-only solver knobs.
CENTRAL_PARAMS = {"max_iter": 300, "f_tol": 1e-3}

_COLORS = ["#00FFFF", "#800080", "#DC143C", "#008B8B", "#FF8C00", "#1E90FF",
           "#32CD32", "#FF1493", "#FFD700", "#8B4513"]


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
   """One cell of the factorial."""
   name: str
   mode: str          # "central" | "distributed"
   sequencing: bool
   zone: str          # "fixed" | "adaptive"

   @property
   def controller_type(self) -> str:
      return "mpc_agent_adaptive" if self.zone == "adaptive" else "mpc_agent"

   @property
   def coordinator_type(self) -> str:
      if self.mode == "central":
         return "mpc_central_intersection" if self.sequencing else "mpc_central"
      return "dmpc_admm_intersection" if self.sequencing else "dmpc_admm"

   @property
   def safety_zone_mode(self) -> str:
      return "adaptive" if self.zone == "adaptive" else "fixed"

   @property
   def gated(self) -> bool:
      # Sequencing needs gated drones; without sequencing they stay plain.
      return self.sequencing


def _build_arms() -> dict[str, Arm]:
   arms: dict[str, Arm] = {}
   idx = 1
   for mode in ("central", "distributed"):
      for seq in (False, True):
         for zone in ("fixed", "adaptive"):
            name = f"A{idx}"
            arms[name] = Arm(name=name, mode=mode, sequencing=seq, zone=zone)
            idx += 1
   return arms


ARMS: dict[str, Arm] = _build_arms()


# ---------------------------------------------------------------------------
# Scenarios — symmetric crossings through a shared center (origin).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
   """A crossing geometry parameterized by drone count.

   kind:
     - "sphere"  fibonacci-ring starts, antipodal targets (open crossing)
     - "line"    head-on corridor (2 vs 2)
     - "corners" cube-corner starts, antipodal targets in a (small) room — a
       genuine bottleneck where everyone must pass through the shared center.
   room_half: half-size of the cubic room for this scenario (small ⇒ dense).
   """
   sid: str
   n_drones: int
   kind: str = "sphere"
   start_radius: float = 3.5
   room_half: float = ROOM_HALF


SCENARIOS: dict[str, Scenario] = {
   "S-X3": Scenario("S-X3", 3),
   "S-X4": Scenario("S-X4", 4),
   "S-X6": Scenario("S-X6", 6),
   "S-X8": Scenario("S-X8", 8),
   "S-line": Scenario("S-line", 4, kind="line"),
   # Cube-corner crossings in a small room: dense, bottleneck-like. room_half=3.0
   # is the tightest size that stays feasible for 8 drones (2.5 makes the 8-drone
   # baseline infeasible at the first step — past the packing limit).
   "S-C6": Scenario("S-C6", 6, kind="corners", room_half=3.0),
   "S-C8": Scenario("S-C8", 8, kind="corners", room_half=3.0),
}


def _fibonacci_sphere(n: int, radius: float) -> np.ndarray:
   """n roughly-uniform points on a sphere of given radius (deterministic)."""
   pts = np.zeros((n, 3), dtype=float)
   golden = np.pi * (3.0 - np.sqrt(5.0))
   for i in range(n):
      y = 1.0 - 2.0 * (i + 0.5) / n
      r = np.sqrt(max(0.0, 1.0 - y * y))
      theta = golden * i
      pts[i] = [np.cos(theta) * r, y, np.sin(theta) * r]
   return pts * radius


def _starts_targets(scn: Scenario, seed: int) -> tuple[np.ndarray, np.ndarray]:
   """Start/target pairs for a scenario. Targets are antipodal through origin
   so every route crosses the center. Seeds jitter the start ring slightly."""
   rng = np.random.default_rng(seed)
   if scn.kind == "line":
      # Head-on corridor: half the drones on +x, half on -x, small y/z spread.
      n = scn.n_drones
      half = n // 2
      starts = []
      for i in range(n):
         side = -1.0 if i < half else 1.0
         k = i if i < half else i - half
         offset = (k - (half - 1) / 2.0) * (SAFETY_ZONE * 2.5)
         starts.append([side * scn.start_radius, offset, offset * 0.5])
      starts = np.asarray(starts, dtype=float)
   elif scn.kind == "corners":
      # Cube corners of the (small) room. Each drone flies to the opposite
      # corner, so all 4 space diagonals cross the room center → dense conflict.
      # Inset by the safety zone so each drone's safety sphere starts inside the
      # room (the room constraint is on the safety sphere, not the body).
      c = scn.room_half - SAFETY_ZONE - 0.1
      corners = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                          for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], dtype=float) * c
      starts = corners[:scn.n_drones]
   else:
      starts = _fibonacci_sphere(scn.n_drones, scn.start_radius)

   jitter = rng.normal(scale=SAFETY_ZONE * 0.2, size=starts.shape)
   starts = starts + jitter
   # Keep the safety sphere (not just the body) inside the room.
   lim = scn.room_half - SAFETY_ZONE - 0.05
   starts = np.clip(starts, -lim, lim)
   targets = -starts
   return starts, targets


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------
def _coordinator_spec(arm: Arm, horizon: int) -> ControllerSpec:
   params: dict = {"horizon": horizon}
   if arm.mode == "central":
      params.update(CENTRAL_PARAMS)
   else:
      params.update(DMPC_PARAMS)
   if arm.sequencing:
      params.update({
         "intersection_enabled": True,
         "intersection_sphere_radius": INTERSECTION_SPHERE_RADIUS,
         "intersection_resolution_margin": INTERSECTION_RESOLUTION_MARGIN,
      })
   return ControllerSpec(type=arm.coordinator_type, params=params)


def _controller_spec(arm: Arm, horizon: int) -> ControllerSpec:
   params: dict = {"horizon": horizon, "q_pos": [Q_POS] * 3, "r_u": [R_U] * 3}
   if arm.zone == "adaptive":
      params["q_vel"] = [Q_VEL] * 3
   return ControllerSpec(type=arm.controller_type, params=params)


def build_arm_config(arm: Arm, scn: Scenario, seed: int, *, horizon: int = HORIZON) -> ScenarioConfig:
   """Assemble a ScenarioConfig for one (arm, scenario, seed)."""
   starts, targets = _starts_targets(scn, seed)
   physics = PhysicsSpec(type="linear_kinematics", id="default",
                         params={"v_max": V_MAX, "u_min": [-U_MAX] * 3, "u_max": [U_MAX] * 3})
   controller = _controller_spec(arm, horizon)
   coordinator = _coordinator_spec(arm, horizon)
   room = RoomConfig(min=[-scn.room_half] * 3, max=[scn.room_half] * 3)

   drones = []
   for i in range(scn.n_drones):
      drones.append(DroneConfig(
         drone_id=f"drone-{i + 1}",
         type="gated" if arm.gated else "default",
         start=starts[i].tolist(),
         target=targets[i].tolist(),
         radius=RADIUS,
         safety_zone=SAFETY_ZONE,
         cons_stop=0.0,
         drone_color=_COLORS[i % len(_COLORS)],
         physics="default",
         safety_zone_mode=arm.safety_zone_mode,
         alpha=ALPHA if arm.zone == "adaptive" else None,
      ))

   return ScenarioConfig(dt=DT, physics=physics, controller=controller,
                         coordinator=coordinator, room=room, drones=drones)


# ---------------------------------------------------------------------------
# Trial execution + metrics
# ---------------------------------------------------------------------------
@dataclass
class TrialResult:
   arm: str
   mode: str
   sequencing: bool
   zone: str
   scenario: str
   n_drones: int
   seed: int
   horizon: int = HORIZON
   outcome: str = "TIMEOUT"          # COMPLETED | DEADLOCK | INFEASIBLE | TIMEOUT
   completed: bool = False
   reached: int = 0
   energy: float = 0.0               # sum_i sum_k ||u||^2 * dt
   makespan_steps: int = 0
   makespan_s: float = 0.0
   # Zone margin: dist - (effective-radius_i + effective-radius_j). This is the
   # arm's own separation constraint (S): fixed safety_zone for fixed arms,
   # velocity-dependent radius for adaptive arms.
   min_margin: float = float("inf")
   violation_steps: int = 0          # # steps with any zone margin < 0
   # Body margin: dist - (radius_i + radius_j). A real physical collision is
   # body margin < 0 — this must be 0 for any safe controller.
   min_body_margin: float = float("inf")
   collision_steps: int = 0          # # steps with any body margin < 0 (real collisions)
   peak_spheres: int = 0
   peak_waiting: int = 0
   n_parked: int = 0
   steps: int = 0
   wall_time_s: float = 0.0


def _margins_step(sim: Simulator) -> tuple[float, float]:
   """Smallest pairwise (zone_margin, body_margin) this step.

   - zone_margin = dist - (effective_radius_i + effective_radius_j), the arm's own
     separation constraint (adaptive radius for adaptive arms, fixed for fixed).
   - body_margin = dist - (radius_i + radius_j), the physical-collision margin.
   """
   drones = sim.drones
   zradii = [d.compute_adaptive_radius(d.velocity()) for d in drones]
   bradii = [float(d.radius) for d in drones]
   min_zone = float("inf")
   min_body = float("inf")
   for i in range(len(drones)):
      for j in range(i + 1, len(drones)):
         dist = float(np.linalg.norm(drones[i].position() - drones[j].position()))
         min_zone = min(min_zone, dist - (zradii[i] + zradii[j]))
         min_body = min(min_body, dist - (bradii[i] + bradii[j]))
   return min_zone, min_body


def run_trial(arm: Arm, scn: Scenario, seed: int, *, max_steps: int = 800,
              eps_v: float = 1e-2, stall_window: int = 60, horizon: int = HORIZON) -> TrialResult:
   """Run one simulation trial and collect metrics."""
   t0 = time.perf_counter()
   cfg = build_arm_config(arm, scn, seed, horizon=horizon)
   sim = Simulator.from_config(cfg)
   coord = sim.coordinator
   has_spheres = hasattr(coord, "active_spheres")

   res = TrialResult(arm=arm.name, mode=arm.mode, sequencing=arm.sequencing,
                     zone=arm.zone, scenario=scn.sid, n_drones=scn.n_drones, seed=seed,
                     horizon=horizon)

   # Per-drone first-reached step for makespan.
   reached_step: dict[str, int] = {}
   recent_speeds: list[float] = []
   ever_parked: set[str] = set()

   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      for step in range(max_steps):
         sim.step()
         res.steps = step + 1

         if sim.infeasible:
            res.outcome = "INFEASIBLE"
            break

         # Control energy from the mirrored applied controls.
         for u in sim.last_controls.values():
            res.energy += float(np.dot(u, u)) * DT

         # Safety: zone-constraint margin and physical-body margin.
         m_zone, m_body = _margins_step(sim)
         res.min_margin = min(res.min_margin, m_zone)
         if m_zone < 0.0:
            res.violation_steps += 1
         res.min_body_margin = min(res.min_body_margin, m_body)
         if m_body < 0.0:
            res.collision_steps += 1

         # Sequencing activity.
         if has_spheres:
            res.peak_spheres = max(res.peak_spheres, len(coord.active_spheres()))
         waiting = [d.drone_id for d in sim.drones
                    if isinstance(d, GatedDrone) and d.is_waiting]
         res.peak_waiting = max(res.peak_waiting, len(waiting))
         ever_parked.update(waiting)

         # Makespan bookkeeping (first step each drone is within thresh of goal).
         for d in sim.drones:
            if d.drone_id not in reached_step and d.route.target_reached(position=d.position(), thresh=0.1):
               reached_step[d.drone_id] = step + 1

         if all_drones_reached_destination(sim.drones):
            res.outcome = "COMPLETED"
            res.completed = True
            break

         # Stall -> deadlock.
         fleet_speed = sum(float(np.linalg.norm(d.velocity())) for d in sim.drones)
         recent_speeds.append(fleet_speed)
         if len(recent_speeds) > stall_window:
            recent_speeds.pop(0)
         if len(recent_speeds) == stall_window and max(recent_speeds) < eps_v:
            res.outcome = "DEADLOCK"
            break

   res.reached = sum(d.route.target_reached(position=d.position(), thresh=0.1) for d in sim.drones)
   res.n_parked = len(ever_parked)
   res.makespan_steps = max(reached_step.values()) if len(reached_step) == scn.n_drones else res.steps
   res.makespan_s = res.makespan_steps * DT
   res.wall_time_s = time.perf_counter() - t0
   return res


def run_grid(arm_names: list[str], scenario_ids: list[str], seeds: list[int],
             *, max_steps: int = 800) -> list[TrialResult]:
   """Run the full (arm x scenario x seed) grid."""
   results: list[TrialResult] = []
   total = len(arm_names) * len(scenario_ids) * len(seeds)
   i = 0
   for sid in scenario_ids:
      scn = SCENARIOS[sid]
      for aname in arm_names:
         arm = ARMS[aname]
         for seed in seeds:
            i += 1
            r = run_trial(arm, scn, seed, max_steps=max_steps)
            results.append(r)
            log.info("[%d/%d] %s %s seed=%d -> %s energy=%.2f makespan=%.1fs "
                     "min_margin=%+.3f viol=%d (%.1fs)", i, total, aname, sid, seed,
                     r.outcome, r.energy, r.makespan_s, r.min_margin,
                     r.violation_steps, r.wall_time_s)
   return results


_CSV_FIELDS = ["arm", "mode", "sequencing", "zone", "scenario", "n_drones", "seed",
               "horizon", "outcome", "completed", "reached", "energy", "makespan_steps",
               "makespan_s", "min_margin", "violation_steps", "min_body_margin",
               "collision_steps", "peak_spheres", "peak_waiting", "n_parked",
               "steps", "wall_time_s"]


def results_to_csv(results: list[TrialResult], path: Path) -> None:
   path.parent.mkdir(parents=True, exist_ok=True)
   with path.open("w", newline="", encoding="utf-8") as f:
      w = csv.writer(f)
      w.writerow(_CSV_FIELDS)
      for r in results:
         w.writerow([
            r.arm, r.mode, int(r.sequencing), r.zone, r.scenario, r.n_drones, r.seed,
            r.horizon, r.outcome, int(r.completed), r.reached, f"{r.energy:.4f}", r.makespan_steps,
            f"{r.makespan_s:.2f}",
            "inf" if r.min_margin == float("inf") else f"{r.min_margin:.4f}",
            r.violation_steps,
            "inf" if r.min_body_margin == float("inf") else f"{r.min_body_margin:.4f}",
            r.collision_steps, r.peak_spheres, r.peak_waiting, r.n_parked,
            r.steps, f"{r.wall_time_s:.2f}",
         ])
   log.info("wrote %d rows -> %s", len(results), path)
