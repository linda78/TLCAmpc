# Environment shapes: how the solver receives fir trees (T1.1)

Author: Pratham · Date: 23 Nov 2026 · Base: `main` at 360c51f · Branch: `pratham/env-t1.1`
Status: for sign-off by Linda before T1.2

## 1. Result in short

- Two ways to hand a fir to the solver were prototyped in numpy and compared: **System A** (stacked
  boxes: trunk plus n crown slices, n = 1, 3, 6) and **System B** (exact shapes: cylinder trunk plus
  cone crown, two inequalities per tree).
- In a scenario with 2 drones and 8 firs in gates of varying width, A with 1 slice blocked every gap
  tested, A with 3 slices blocked from g = 2.2, A with 6 slices passed all gaps but needed a larger
  detour in the narrow ones. B passed all gaps and came as close as the safety zone allows.
- Time per constraint evaluation is about the same for A with 6 slices and B (unoptimised code).
  B needs 2 inequalities per fir, A with 6 slices needs 7.
- Proposal: **hybrid**. Houses, blocks and walls stay boxes; round objects (fir, tower, sphere)
  use B; A with 6 slices is the fallback that needs no solver change. Sign-off needed (section 8).

## 2. What was compared

- **Fir.** Crown radius 1.2 m, total height 5.0 m, trunk radius 0.2 m (the plan's `fir-1`). Assumed:
  trunk height 1.0 m (20 %), crown 4.0 m on top, `radius` = crown radius, `height` = total height.
- **System A.** Trunk = one box. Crown = n stacked square slices, each as wide as the cone at its
  lower edge (the staircase encloses the cone). Uses the existing `points_to_box_dist`.
- **System B.** `cylinder_dist` and `cone_dist` in numpy, smoothed like `points_to_box_dist`
  (`sqrt(d^2 + eps^2) - eps`, eps = 1e-3). The cone formula is our own derivation in the (rho, z)
  half-plane; against 144,000 sampled surface points the maximum error is 3.4e-4 m. The solver gets
  two rows per fir (trunk, crown), not one `min()` (see section 6).
- **Ground truth.** Exact distance, no smoothing: the smaller of trunk and crown distance.
- **Plug.** The spike replaces `ObstacleAvoidanceConstraints._evaluate` during a run and restores
  it afterwards; no repository file changes. The true distance to the tree is computed from the drone
  positions after every step.
- **Libraries.** Quilez (closed-form distances) was the reference. python-fcl, coal and
  `trimesh.proximity` were not adopted (only their documentation was read): new dependencies,
  per-object-pair queries, and our formulas are short.

## 3. Static comparison (one fir)

| Variant | Inequalities per fir | Volume (m^3) | Volume / true |
|---|---|---|---|
| A, n = 1 | 2 | 23.200 | 3.768 |
| A, n = 3 | 4 | 12.107 | 1.966 |
| A, n = 6 | 7 | 9.867 | 1.602 |
| B | 2 | 6.158 | 1.000 (by construction) |

Square boxes cannot go below 4/pi = 1.27 for the crown, however many slices. All A boxes enclose the
cone (gap between sampled surface points and boxes at most 6e-16 m).

## 4. Scenario with 2 drones and 8 firs

- **Layout.** Room 20 x 20 x 10 m, no walls. Drone 1 flies (-8, -3, 3) to (8, -3, 3), drone 2
  (-8, 3, 3) to (8, 3, 3). Eight firs on the floor in two columns, x = -2 and x = 3; each lane has a
  gate of two firs per column. **g** is the distance from the lane's centre line to the axis of each
  of its two trees (gap between axes = 2g).
- **Not a plain copy of NarrowGapBaseline:** its wall boxes make its targets unreachable, so they are
  removed. Otherwise as the baseline (`mpc_central`, dt 0.1, horizon 8, safety zone 1.52 m, body
  radius 0.4 m). Close to the plan's `fir_forest` example.
- **Runs.** Seed 0 for g = 2.6, 2.4; seeds 0, 1, 2 for g = 2.2, 2.0; at most 300 steps; reached =
  both drones within 0.1 m of their targets.

Seed 0. Each cell: step at which both drones reached their targets, climb (largest change from the
flight height of 3 m), closest approach to the true fir in m. "blocked" = not reached in 300 steps;
the value is then where the drones stopped, not a clearance.

| g (m) | A, n = 1 | A, n = 3 | A, n = 6 | B |
|---|---|---|---|---|
| 2.6 | blocked (2.45) | 79, 0.00 (1.915) | 79, 0.00 (1.915) | 79, 0.00 (1.915) |
| 2.4 | blocked (2.50) | 80, 0.34 (1.802) | 79, 0.00 (1.724) | 79, 0.00 (1.724) |
| 2.2 | blocked (2.50) | blocked (2.05) | 81, 0.75 (1.706) | 79, 0.00 (1.532) |
| 2.0 | blocked (2.49) | blocked (2.11) | 84, 1.54 (1.705) | 80, 0.78 (1.521) |

- The three seeds agree in every cell (closest approach within 0.001 m); the layout is symmetric,
  so seeds do not test other layouts. A full rerun was identical except the measured seconds.
- Nobody swerves sideways (below 0.001 m); the only way through a narrow gate is to climb.
- B stays at or just above the safety zone (1.521 vs 1.52 m). A with 6 slices keeps about 0.18 m
  more distance than needed, climbs twice as high at g = 2.0 and arrives 4 steps (5 %) later.

## 5. Time

- One call (8 firs x 16 points, median of 2000): A n = 1 51 us, n = 3 102 us, n = 6 181 us, B 186 us;
  differences below about 5 % are noise.
- Whole run, `cpu_constraints_s` at g = 2.6 (same trajectory for A3, A6, B): A3 4.3 s, A6 6.8 s,
  B 6.8 s. The cone code is unoptimised (small arrays per call, not vectorised over firs); a speed-up
  was not tried, so B has no faster number yet.

## 6. Statements in the plan the measurements did not support

- **"The cone apex is a non-smooth point."** Along four lines outside the cone (including 5 cm over
  the apex) the distance is smooth (slope-jump ratio 0.50 when the spacing is halved). The real kink
  for a whole fir is a ridge where trunk and crown are equally near; hence two rows, not one `min()`.
- **OA-MPC "vertex representation solves faster than half-spaces."** Not in the text of arXiv
  2211.09156 (v2); it uses capsules and half-spaces without that comparison. "Smoother distances for
  SLSQP" was not measured either. (Capsule distance is analytic: supported by COSMIK-MPPI, III-B.)
- **`cpu_constraints_s`, `cost_evals` "exist in the metrics export":** they exist on `main`
  (`CpuCostLedger`, `sim.costs`, exported in `tools/utility/printer.py`).

## 7. Limits

- One scenario, one flight height (z = 3, where the cone is only 0.6 m wide), symmetric layout,
  central MPC only; B below g = 2.0 not tried; solver convergence not examined.
- Fir proportions are assumptions. B's ground truth uses the same functions as B, so "B stays outside
  the safety zone" is partly circular; the independent check is the brute-force cone test.
- Timings: unoptimised code, M4 MacBook, Python 3.12.16, numpy 2.5.5, scipy 1.18.1.

## 8. Decision (proposal) and what Linda needs to confirm

**Hybrid.** Box objects (house, block, wall) are boxes for the solver. Round objects use B
(`points_to_cylinder_dist`, `points_to_cone_dist`, `points_to_sphere_dist`), as the plan's T1.4
describes (a `distance(points)` method on the obstacle elements; old `(center, half_extents)` tuples
stay valid). T1.2 builds `solver_primitives(obj, system, slices)` for both systems; A with 6 slices
works with the existing solver and is the fallback if B takes longer than planned.

**Reasons.** A with 3 or fewer slices blocks gaps that are really open, A with 6 slices still costs a
larger detour in the narrowest gate, B passes everywhere at the same constraint time and with fewer
inequalities. Against B: more code to change (`constraints.py`, `Simulator._compute_collisions`,
`from_config`, `/state`, `local_mpc.py`) and no measured effect on convergence.

**To confirm:** (1) the hybrid decision; (2) trunk height 1.0 m of 5.0 m, `radius` = crown radius,
`height` = total height; (3) whether a second scenario is wanted now or later with real environment
files (T1.4).

## 9. Reproduce

```
python tools/spike_obstacle_shapes.py                      # checks, about 2 s
python tools/spike_obstacle_shapes.py --scenario out.csv   # plus scenario, about 5.5 min
```

CSV columns: label, half_gap, seed, reached_step (empty = blocked), min_true_dist, max_lane_offset,
max_height_offset, cost_evals, cpu_constraints_s. Also from T1.1: `tools/regression_positions.py`
(record positions for N steps with a seed; compare two CSVs).
