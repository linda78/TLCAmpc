
"""T1.1 spike: how to hand a fir tree to the solver.

System A approximates the fir with stacked axis-aligned boxes (trunk plus 1, 3 or 6
crown slices). System B uses the exact shapes (cylinder trunk, cone crown) with two
inequalities per tree. The file contains the distance functions, static checks
(correctness, volume, inequality count, timing) and a scenario with 2 drones and
8 firs in gates of varying width.

Usage, from the repository root:

    python tools/spike_obstacle_shapes.py
        Checks and static measures (a few seconds).
    python tools/spike_obstacle_shapes.py --scenario out.csv
        Same, then the scenario; one CSV row per run (about 5 to 6 minutes on an M4 MacBook).

Results and the decision are in .planning/research/environment_shapes.md.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

from drone_sim.domain import constraints
from drone_sim.domain.config import ScenarioConfig
from drone_sim.domain.constraints import _safety_radii, points_to_box_dist
from drone_sim.domain.utils.helper import all_drones_reached_destination
from drone_sim.simulation.simulator import Simulator

# Fir example of the plan (fir-1): crown radius 1.2, total height 5.0, trunk radius 0.2.
# The trunk height is our assumption (20 % of the total height); the plan leaves it open.
FIR_TOTAL_HEIGHT = 5.0
FIR_CROWN_RADIUS = 1.2
FIR_TRUNK_RADIUS = 0.2
FIR_TRUNK_HEIGHT = 0.2 * FIR_TOTAL_HEIGHT
FIR_CROWN_HEIGHT = FIR_TOTAL_HEIGHT - FIR_TRUNK_HEIGHT


def cylinder_dist(
    points: np.ndarray,
    center_xy: np.ndarray,
    z_base: float,
    radius: float,
    height: float,
    eps: float = 1e-3,
) -> np.ndarray:
    """Smooth point-to-cylinder distance for multiple points.

    The cylinder is upright (axis parallel to z). Like the box in
    ``points_to_box_dist``, the radial and the vertical excess are combined
    with a regularised L2 norm.

    :param points: Query points (N, 3).
    :param center_xy: Cylinder axis position (x, y), shape (2,).
    :param z_base: Height of the bottom face.
    :param radius: Cylinder radius.
    :param height: Full cylinder height (not half).
    :param eps: Regularization constant.
    :return: Smooth distances (N,).
    """
    dxy = points[:, :2] - center_xy
    radial = np.maximum(np.hypot(dxy[:, 0], dxy[:, 1]) - radius, 0.0)
    z_mid = z_base + height / 2.0
    vertical = np.maximum(np.abs(points[:, 2] - z_mid) - height / 2.0, 0.0)
    return np.sqrt(radial**2 + vertical**2 + eps * eps) - eps


def _point_to_segment_dist(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Distance from 2D points to a line segment.

    :param points: Query points (N, 2).
    :param start: First end point of the segment (2,).
    :param end: Second end point of the segment (2,).
    :return: Distances (N,).
    """
    seg = end - start
    t = np.clip((points - start) @ seg / (seg @ seg), 0.0, 1.0)
    closest = start + t[:, None] * seg
    return np.linalg.norm(points - closest, axis=1)


def cone_dist(
    points: np.ndarray,
    center_xy: np.ndarray,
    z_base: float,
    radius: float,
    height: float,
    eps: float = 1e-3,
) -> np.ndarray:
    """Smooth point-to-cone distance for multiple points.

    The cone is upright: a circular base of ``radius`` at ``z_base`` and the
    apex at ``z_base + height``. The distance is computed in the half-plane
    (rho, z), where rho is the horizontal distance to the axis. There the cone
    is the triangle (0, 0), (radius, 0), (0, height). Outside, the distance is
    the one to the base edge or the slanted edge, whichever is nearer; inside
    it is 0.

    :param points: Query points (N, 3).
    :param center_xy: Cone axis position (x, y), shape (2,).
    :param z_base: Height of the base circle.
    :param radius: Radius of the base circle.
    :param height: Height from base to apex.
    :param eps: Regularization constant.
    :return: Smooth distances (N,).
    """
    dxy = points[:, :2] - center_xy
    rho = np.hypot(dxy[:, 0], dxy[:, 1])
    z = points[:, 2] - z_base
    q = np.stack([rho, z], axis=1)

    base_edge = _point_to_segment_dist(q, np.array([0.0, 0.0]), np.array([radius, 0.0]))
    side_edge = _point_to_segment_dist(q, np.array([radius, 0.0]), np.array([0.0, height]))
    d = np.minimum(base_edge, side_edge)

    inside = (z >= 0.0) & (z <= height) & (rho <= radius * (1.0 - z / height))
    d = np.where(inside, 0.0, d)
    return np.sqrt(d**2 + eps * eps) - eps


def _cone_surface_samples(
    radius: float,
    height: float,
    n_angle: int = 360,
    n_side: int = 200,
) -> np.ndarray:
    """Dense points on the surface of an upright cone at the origin (for checks).

    :param radius: Radius of the base circle.
    :param height: Height from base to apex.
    :param n_angle: Number of samples around the axis.
    :param n_side: Number of samples along the slanted side and along the base radius.
    :return: Surface points (n_angle * 2 * n_side, 3).
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False)
    t = np.linspace(0.0, 1.0, n_side)
    rho = np.concatenate([radius * (1.0 - t), radius * t])
    z = np.concatenate([height * t, np.zeros(n_side)])
    x = rho[:, None] * np.cos(phi)[None, :]
    y = rho[:, None] * np.sin(phi)[None, :]
    z_grid = np.broadcast_to(z[:, None], x.shape)
    return np.stack([x.ravel(), y.ravel(), z_grid.ravel()], axis=1)


def fir_true_dist(
    points: np.ndarray,
    center_xy: np.ndarray,
    z_base: float = 0.0,
) -> np.ndarray:
    """Exact distance from points to a fir tree (cylinder trunk plus cone crown).

    The distance to a union is the smaller of the distances to its parts.
    No smoothing is applied, so this serves as ground truth for the
    approximations.

    :param points: Query points (N, 3).
    :param center_xy: Tree axis position (x, y), shape (2,).
    :param z_base: Height of the trunk's bottom.
    :return: Distances (N,); 0 inside the tree.
    """
    trunk = cylinder_dist(
        points, center_xy, z_base, FIR_TRUNK_RADIUS, FIR_TRUNK_HEIGHT, eps=0.0
    )
    crown = cone_dist(
        points,
        center_xy,
        z_base + FIR_TRUNK_HEIGHT,
        FIR_CROWN_RADIUS,
        FIR_CROWN_HEIGHT,
        eps=0.0,
    )
    return np.minimum(trunk, crown)


def fir_boxes(
    center_xy: np.ndarray,
    z_base: float,
    n_slices: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Approximate a fir with axis-aligned boxes (System A).

    The trunk is one box. The crown is split into ``n_slices`` stacked boxes,
    each as wide as the cone at the slice's lower edge, so the staircase
    encloses the cone.

    :param center_xy: Tree axis position (x, y), shape (2,).
    :param z_base: Height of the trunk's bottom.
    :param n_slices: Number of crown slices.
    :return: List of (center, half_extents) pairs; the trunk comes first.
    """
    cx, cy = center_xy
    trunk_half = np.array([FIR_TRUNK_RADIUS, FIR_TRUNK_RADIUS, FIR_TRUNK_HEIGHT / 2.0])
    boxes = [(np.array([cx, cy, z_base + FIR_TRUNK_HEIGHT / 2.0]), trunk_half)]
    slice_height = FIR_CROWN_HEIGHT / n_slices
    z_crown = z_base + FIR_TRUNK_HEIGHT
    for k in range(n_slices):
        r_k = FIR_CROWN_RADIUS * (1.0 - k / n_slices)
        center = np.array([cx, cy, z_crown + (k + 0.5) * slice_height])
        boxes.append((center, np.array([r_k, r_k, slice_height / 2.0])))
    return boxes


def fir_boxes_dist(
    points: np.ndarray,
    center_xy: np.ndarray,
    z_base: float,
    n_slices: int,
    eps: float = 1e-3,
) -> np.ndarray:
    """Distances from points to each box of the System A fir.

    :param points: Query points (N, 3).
    :param center_xy: Tree axis position (x, y), shape (2,).
    :param z_base: Height of the trunk's bottom.
    :param n_slices: Number of crown slices.
    :param eps: Regularization constant.
    :return: Distances (N, n_slices + 1), one column per box (one inequality each).
    """
    boxes = fir_boxes(center_xy, z_base, n_slices)
    return np.stack([points_to_box_dist(points, c, h, eps) for c, h in boxes], axis=1)


def fir_round_dist(
    points: np.ndarray,
    center_xy: np.ndarray,
    z_base: float = 0.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """Distances from points to the two parts of the System B fir.

    The trunk (cylinder) and the crown (cone) are kept apart, so the solver
    receives two inequalities per tree instead of one non-smooth minimum.

    :param points: Query points (N, 3).
    :param center_xy: Tree axis position (x, y), shape (2,).
    :param z_base: Height of the trunk's bottom.
    :param eps: Regularization constant.
    :return: Distances (N, 2); column 0 is the trunk, column 1 the crown.
    """
    trunk = cylinder_dist(
        points, center_xy, z_base, FIR_TRUNK_RADIUS, FIR_TRUNK_HEIGHT, eps
    )
    crown = cone_dist(
        points,
        center_xy,
        z_base + FIR_TRUNK_HEIGHT,
        FIR_CROWN_RADIUS,
        FIR_CROWN_HEIGHT,
        eps,
    )
    return np.stack([trunk, crown], axis=1)


def fir_true_volume() -> float:
    """Volume of the fir: cylinder trunk plus cone crown."""
    trunk = np.pi * FIR_TRUNK_RADIUS**2 * FIR_TRUNK_HEIGHT
    crown = np.pi * FIR_CROWN_RADIUS**2 * FIR_CROWN_HEIGHT / 3.0
    return float(trunk + crown)


def fir_boxes_volume(n_slices: int) -> float:
    """Total volume of the System A boxes of one fir (the boxes do not overlap).

    :param n_slices: Number of crown slices.
    :return: Sum of the box volumes.
    """
    boxes = fir_boxes(np.zeros(2), 0.0, n_slices)
    return float(sum(np.prod(2.0 * half) for _, half in boxes))


def make_system_a_call(
    centres: np.ndarray,
    points: np.ndarray,
    n_slices: int,
) -> Callable[[], list[np.ndarray]]:
    """Return a function that evaluates all System A box distances for the trees.

    The boxes are built once, outside the timed call.

    :param centres: Tree axis positions (T, 2).
    :param points: Query points (N, 3).
    :param n_slices: Number of crown slices.
    :return: A function without arguments that returns the distance arrays.
    """
    all_boxes = [fir_boxes(c, 0.0, n_slices) for c in centres]

    def call() -> list[np.ndarray]:
        return [
            points_to_box_dist(points, center, half, 1e-3)
            for boxes in all_boxes
            for center, half in boxes
        ]

    return call


def make_system_b_call(
    centres: np.ndarray,
    points: np.ndarray,
) -> Callable[[], list[np.ndarray]]:
    """Return a function that evaluates all System B distances for the trees.

    :param centres: Tree axis positions (T, 2).
    :param points: Query points (N, 3).
    :return: A function without arguments that returns the distance arrays.
    """

    def call() -> list[np.ndarray]:
        return [fir_round_dist(points, c, 0.0, 1e-3) for c in centres]

    return call


def time_call(fn: Callable[[], object], n_repeat: int = 2000) -> float:
    """Median wall time of ``fn()`` in microseconds, after a short warm-up.

    :param fn: Function without arguments.
    :param n_repeat: Number of timed calls.
    :return: Median time per call in microseconds.
    """
    for _ in range(50):
        fn()
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1e6


def max_slope_jump(
    dist_fn: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    n_points: int,
    h: float = 1e-6,
) -> float:
    """Largest change of the numerical slope between neighbouring samples on a line.

    The slope of ``dist_fn`` along the line is estimated with central
    differences. A smooth function gives jumps that shrink when ``n_points``
    grows; a kink gives a jump that stays.

    :param dist_fn: Maps points (N, 3) to distances (N,).
    :param start: Line start point (3,).
    :param end: Line end point (3,).
    :param n_points: Number of equally spaced sample points on the line.
    :param h: Step of the central difference.
    :return: Largest absolute difference between neighbouring slopes.
    """
    direction = (end - start) / np.linalg.norm(end - start)
    s = np.linspace(0.0, 1.0, n_points)[:, None]
    pts = start + s * (end - start)
    slope = (dist_fn(pts + h * direction) - dist_fn(pts - h * direction)) / (2.0 * h)
    return float(np.max(np.abs(np.diff(slope))))


BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs/narrow_gap/NarrowGapBaseline.json"
LANE_Y = 3.0
COLUMN_X = (-2.0, 3.0)
FLIGHT_Z = 3.0
START_X = -8.0
TARGET_X = 8.0
VARIANTS = (1, 3, 6, None)
SEEDS_PER_GAP = {2.6: (0,), 2.4: (0,), 2.2: (0, 1, 2), 2.0: (0, 1, 2)}


@dataclass
class RunResult:
    label: str
    half_gap: float
    seed: int
    reached_step: int | None
    min_true_dist: float
    max_lane_offset: float
    max_height_offset: float
    cost_evals: int
    cpu_constraints_s: float


def gate_layout(half_gap: float) -> np.ndarray:
    """Return the xy positions of the eight tree axes.

    The half gap g is the distance from a lane's centre line (y = +-LANE_Y) to the axis of the
    tree on either side of it, so the two trees of a gate are 2 g apart. Each lane has one gate
    in each column of COLUMN_X.

    :param half_gap: g in metres.
    :return: Array of shape (8, 2) with one row [x, y] per tree.
    """
    ys = [-LANE_Y + half_gap, -LANE_Y - half_gap, LANE_Y - half_gap, LANE_Y + half_gap]
    return np.array([[x, y] for x in COLUMN_X for y in ys], dtype=float)


def build_scenario_config() -> dict:
    """Return NarrowGapBaseline without walls, in a larger room, with one lane per drone."""
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg["room"] = {"min": [-10, -10, 0], "max": [10, 10, 10]}
    cfg["obstacles"] = []
    for drone, lane in zip(cfg["drones"][:2], (-LANE_Y, LANE_Y)):
        drone["start"] = [START_X, lane, FLIGHT_Z]
        drone["target"] = [TARGET_X, lane, FLIGHT_Z]
    return cfg


def make_patched_evaluate(
    original: Callable[..., np.ndarray], firs: np.ndarray, n_slices: int | None
) -> Callable[..., np.ndarray]:
    """Build a replacement for ObstacleAvoidanceConstraints._evaluate that adds the fir rows.

    :param original: The unpatched method, still used for obstacles from the config.
    :param firs: Array of shape (N, 2) with the xy position of each tree axis.
    :param n_slices: Slices per crown for System A, or None for System B.
    :return: The patched method.
    """
    boxes = None if n_slices is None else [fir_boxes(c, 0.0, n_slices) for c in firs]

    def patched(self, drone, pred_pos, obstacles, pred_vel=None):
        radii = _safety_radii(drone, pred_vel, self._horizon)
        parts = []
        if obstacles:
            parts.append(original(self, drone, pred_pos, obstacles, pred_vel=pred_vel))
        if boxes is None:
            for center in firs:
                dist = fir_round_dist(pred_pos, center)
                parts.extend([dist[:, 0] - radii, dist[:, 1] - radii])
        else:
            for fir in boxes:
                for center, half_extents in fir:
                    parts.append(points_to_box_dist(pred_pos, center, half_extents) - radii)
        return np.concatenate(parts)

    return patched


def run_scenario(
    n_slices: int | None, half_gap: float, seed: int, max_steps: int = 300
) -> RunResult:
    """Fly both drones through the gates and measure the run.

    :param n_slices: Slices per crown for System A, or None for System B.
    :param half_gap: g from gate_layout, in metres.
    :param seed: Seed for random and numpy before the simulator is built.
    :param max_steps: Stop after this many steps if the targets are not reached.
    :return: The measurements of the run.
    """
    firs = gate_layout(half_gap)
    random.seed(seed)
    np.random.seed(seed)
    sim = Simulator.from_config(ScenarioConfig.model_validate(build_scenario_config()))
    original = constraints.ObstacleAvoidanceConstraints._evaluate
    constraints.ObstacleAvoidanceConstraints._evaluate = make_patched_evaluate(
        original, firs, n_slices
    )
    reached_step = None
    min_true = np.inf
    max_lane_offset = 0.0
    max_height_offset = 0.0
    try:
        for step in range(1, max_steps + 1):
            sim.step()
            for drone in sim.drones:
                pos = drone.position()
                min_true = min(min_true, min(float(fir_true_dist(pos[None, :], c)[0]) for c in firs))
                max_lane_offset = max(max_lane_offset, abs(abs(pos[1]) - LANE_Y))
                max_height_offset = max(max_height_offset, abs(pos[2] - FLIGHT_Z))
            if all_drones_reached_destination(sim.drones):
                reached_step = step
                break
    finally:
        constraints.ObstacleAvoidanceConstraints._evaluate = original
    label = "B" if n_slices is None else f"A{n_slices}"
    return RunResult(
        label,
        half_gap,
        seed,
        reached_step,
        float(min_true),
        float(max_lane_offset),
        float(max_height_offset),
        sim.costs.counts.get("cost", 0),
        sim.costs.seconds.get("constraints", 0.0),
    )


def run_table() -> list[RunResult]:
    """Run every variant at every gap and seed, printing one line per run."""
    results = []
    for half_gap, seeds in SEEDS_PER_GAP.items():
        for n_slices in VARIANTS:
            for seed in seeds:
                r = run_scenario(n_slices, half_gap, seed)
                results.append(r)
                print(
                    f"g={r.half_gap} {r.label:>2} seed={r.seed} reached={r.reached_step} "
                    f"min_true={r.min_true_dist:.3f} lane_off={r.max_lane_offset:.2f} "
                    f"height_off={r.max_height_offset:.2f} cost_evals={r.cost_evals} "
                    f"cpu_con={r.cpu_constraints_s:.1f}s",
                    flush=True,
                )
    return results


def write_results_csv(results: list[RunResult], path: Path) -> None:
    """Write one CSV row per run.

    :param results: Runs returned by run_table.
    :param path: Output file, overwritten if it exists.
    """
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


if __name__ == "__main__":
    scenario_csv = None
    if "--scenario" in sys.argv:
        position = sys.argv.index("--scenario") + 1
        if position >= len(sys.argv):
            sys.exit("usage: python tools/spike_obstacle_shapes.py [--scenario OUT.csv]")
        scenario_csv = Path(sys.argv[position])
    centre = np.array([0.0, 0.0])

    cyl_pts = np.array([[3.0, 0.0, 0.5], [3.0, 0.0, 6.0], [0.0, 0.0, 2.0]])
    for eps in (0.0, 1e-3):
        print("cylinder", eps, cylinder_dist(cyl_pts, centre, 0.0, 1.0, 4.0, eps))

    cone_pts = np.array(
        [[3.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, 0.0, -1.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]
    )
    print("cone", cone_dist(cone_pts, centre, 0.0, 1.0, 4.0, eps=0.0))

    rng = np.random.default_rng(0)
    rand_pts = rng.uniform([-3.0, -3.0, -2.0], [3.0, 3.0, 7.0], size=(200, 3))
    surface = _cone_surface_samples(1.0, 4.0)
    brute = np.array([np.min(np.linalg.norm(surface - p, axis=1)) for p in rand_pts])
    fast = cone_dist(rand_pts, centre, 0.0, 1.0, 4.0, eps=0.0)
    outside = fast > 0.0
    print("cone vs brute force, max error (outside points):", np.max(np.abs(fast - brute)[outside]))
    lines = {
        "over apex (0.05 above)": ((-1.0, 0.0, 4.05), (1.0, 0.0, 4.05)),
        "under rim (0.05 below)": ((0.0, 0.0, -0.05), (2.0, 0.0, -0.05)),
        "base-to-corner switch": ((0.0, 0.0, -0.3), (2.0, 0.0, -0.3)),
        "side-to-apex switch": ((1.5, 0.0, 3.0), (-1.5, 0.0, 6.0)),
        "through the cone": ((-2.003, 0.0, 2.0), (2.0, 0.0, 2.0)),
    }
    for eps in (0.0, 1e-3):
        dist_fn = partial(
            cone_dist, center_xy=centre, z_base=0.0, radius=1.0, height=4.0, eps=eps
        )
        print("smoothness, eps =", eps)
        for name, (start, end) in lines.items():
            j1 = max_slope_jump(dist_fn, np.array(start), np.array(end), 2001)
            j2 = max_slope_jump(dist_fn, np.array(start), np.array(end), 4001)
            print(f"  {name:26s} {j1:.4f} -> {j2:.4f}  ratio {j2 / j1:.2f}")
    fir_pts = np.array(
        [[3.0, 0.0, 0.5], [0.0, 0.0, 5.5], [0.0, 0.0, 0.5], [1.0, 0.0, 0.5]]
    )
    print("fir", fir_true_dist(fir_pts, centre))
    cone_vol = np.pi * FIR_CROWN_RADIUS**2 * FIR_CROWN_HEIGHT / 3.0
    crown_surface = _cone_surface_samples(FIR_CROWN_RADIUS, FIR_CROWN_HEIGHT)
    crown_surface = crown_surface + np.array([0.0, 0.0, FIR_TRUNK_HEIGHT])
    for n_slices in (1, 3, 6):
        boxes = fir_boxes(centre, 0.0, n_slices)
        gap = fir_boxes_dist(crown_surface, centre, 0.0, n_slices, eps=0.0).min(axis=1)
        crown_vol = sum(float(np.prod(2.0 * half)) for _, half in boxes[1:])
        print(
            f"System A n={n_slices}: {len(boxes)} boxes, "
            f"max gap {gap.max():.1e}, crown volume ratio {crown_vol / cone_vol:.3f}"
        )
    sample_pts = rng.uniform([-3.0, -3.0, -1.0], [3.0, 3.0, 7.0], size=(5000, 3))
    true_dist = fir_true_dist(sample_pts, centre)
    for eps in (0.0, 1e-3):
        round_dist = fir_round_dist(sample_pts, centre, 0.0, eps)
        diff = true_dist - round_dist.min(axis=1)
        print(
            f"System B eps={eps}: {round_dist.shape[1]} columns, "
            f"true minus B: min {diff.min():.1e}, max {diff.max():.1e}"
        )
    true_vol = fir_true_volume()
    print(f"true fir volume {true_vol:.3f} m^3")
    for n_slices in (1, 3, 6):
        box_vol = fir_boxes_volume(n_slices)
        print(
            f"System A n={n_slices}: {n_slices + 1} inequalities per fir, "
            f"volume {box_vol:.3f} m^3, ratio {box_vol / true_vol:.3f}"
        )
    print("System B: 2 inequalities per fir, exact shapes, ratio 1.000")
    firs_xy = np.array(
        [
            [-3.0, -3.0], [-1.0, -3.0], [1.0, -3.0], [3.0, -3.0],
            [-3.0, 3.0], [-1.0, 3.0], [1.0, 3.0], [3.0, 3.0],
        ]
    )
    drone_pts = rng.uniform(-4.0, 4.0, size=(16, 3))
    for n_slices in (1, 3, 6):
        t_us = time_call(make_system_a_call(firs_xy, drone_pts, n_slices))
        print(f"timing System A n={n_slices}: {t_us:.0f} us per call")
    t_us = time_call(make_system_b_call(firs_xy, drone_pts))
    print(f"timing System B: {t_us:.0f} us per call")

    if scenario_csv is not None:
        write_results_csv(run_table(), scenario_csv)
