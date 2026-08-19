# Paper 4 — Intersection-Sphere Sequencing: Evaluation & Test Plan

Plan for **Block 2 — Evaluation** of `tex/paper/intersection_spheres/06_evaluation.tex`.
Goal: turn Section VI from a bullet list of intentions into real results that back
the abstract's claims, and feed the empirical limits back to the theory (§IV).

**Central thesis of the evaluation:** *under conflict-dense conditions, adding
intersection-sphere sequencing on top of the same MPC lowers control energy (and
removes deadlock) — on both the central and the distributed MPC, and with both
fixed and adaptive safety zones.* The experiment is a clean factorial that isolates
exactly the sequencing effect.

> **Directory note:** an empty, mis-spelled `paper4_interswection_sphere/` exists at
> the repo root. This plan lives in the correctly-spelled `paper4_intersection_sphere/`.
> Delete the typo'd directory once this is confirmed.

---

## 0. What the paper claims → what we must measure

Every experiment exists to support one sentence in the abstract / methodology.
We design backward from the claims so nothing is run that no claim needs.

| # | Claim (source) | Metric that proves it | Experiment |
|---|----------------|-----------------------|------------|
| C1 | "removes the deadlocks of fixed-zone MPC and DMPC" (abstract; req. **(L)**) | Deadlock rate, mission-completion rate | E1 factorial |
| C2 | "completes the missions with lower control energy" — **under conflict-dense conditions** (abstract; req. **(E)**) | Control energy `∫‖u‖²dt`, per sequencing-on/off pair | E1 factorial + E3 density |
| C3 | Safety holds throughout (req. **(S)**) | Min pairwise separation, # safety violations (target = 0) | all experiments |
| C4 | Feasibility needs `H ≥ H_min` (§IV-B, Prop. 4.1, eq. budget `κ_a`) | Feasibility/completion vs horizon | E2 horizon ablation |
| C5 | Capacity limit `N_c^v` / `κ_a` are real (§IV-A, "validation of theoretical limits") | Deadlock onset vs density & conflict degree | E3 density sweep |

The "**under certain circumstances**" qualifier the paper makes is itself a result:
sequencing wins when conflicts are dense; in sparse scenes it should be roughly
cost-neutral (the rule places no sphere when nothing conflicts). E1 (across
densities) and E3 (density sweep) together map *where* the crossover is.

---

## 1. Infrastructure already in place (reuse, don't rebuild)

- **Four coordinators, all registered and tested:**
  - `mpc_central` — central MPC, no sequencing (`centralized/coordinator.py`).
  - `dmpc_admm` — distributed ADMM MPC, no sequencing (`distributed/distributed_coordinator.py`).
  - `mpc_central_intersection` — central MPC + sequencing (`intersection/intersection_central_coordinator.py`).
  - `dmpc_admm_intersection` — distributed MPC + sequencing (`intersection/intersection_dmpc_coordinator.py`).

  The two intersection coordinators share one `IntersectionMixin`
  (`intersection/intersection_mixin.py`); the base coordinators are untouched, so
  *sequencing-off vs sequencing-on is the only difference* within an MPC mode.
- **The wait-controller swap is the coordinator's job, not the config's.** Every
  drone keeps its *base* controller (`mpc_agent` or `mpc_agent_adaptive`) in the
  config. When sequencing parks a yielding drone, the intersection coordinator (via
  the state store) swaps that drone to `BRAKING_CONTROLLER` → `WAIT_CONTROLLER` and
  back automatically (`drone.py:GatedDrone`, `intersection_state.py`). Sequencing-on
  arms therefore only need `type:"gated"` drones; the controller column always names
  the base controller.
- **State machine:** `IntersectionStateStore` / `compute_priorities` /
  `apply_deadlock_dominance` (TTC right-of-way, wait-veto, boost, Tarjan-SCC cycle
  dissolution — matches §V-B and the liveness lemmas).
- **Working configs:**
  `configs/franck_intersection/4drone_admm_intersection.json` (distributed) and
  `configs/franck_intersection/4drone_central_intersection.json` (central).
- **Smoke runner already written:** `paper4_intersection_sphere/smoke_intersection.py`
  — outcome classification + safety + sequencing activity for one config. The full
  multi-arm runner (T1) generalizes it.
- **Runner pattern to copy:** `paper3_lstm/sim_runner.py` — `Simulator.from_config`,
  `sim.step()`, `sim.infeasible`, `all_drones_reached_destination`, CSV writers,
  multi-seed `run_trials`. Mirror this; do **not** invent a new harness.
- **Verify-script pattern:** `tools/utility/verify_dmpc.py` /
  `verify_adaptive_spheres_paper.py` — per-step margin computation reused for C3.

### 1.1 One small code change needed first (Task T0)

Control energy `∫‖u‖²dt` (C2) needs the **applied** acceleration command. The
simulator computes `us` per step (`simulator.py:299–317`) but discards it. Add a
one-line mirror, analogous to `last_collisions`:

```python
self.last_controls = {d.drone_id: u for d, u in zip(self.drones, us, strict=True)}
```

set just before `self.last_collisions = self._compute_collisions()`. Additive and
harmless. **Decided:** explicit mirror (exact). The finite-difference proxy
`u_k ≈ (v_{k+1} − v_k)/dt` is rejected — the wall-clamp (`simulator.py:321–330`)
corrupts it.

---

## 2. Scenarios (fix and freeze these)

Conflict density is the independent variable that drives the thesis, so we
parameterize by it. All scenarios are **symmetric crossings through a shared
center** — the canonical deadlock generator and the geometry already used in the
franck config (targets antipodal through the origin).

| ID | Drones | Geometry | Purpose |
|----|--------|----------|---------|
| **S-X3** | 3 | 3 routes crossing one center (120° apart) | mild conflict; sequencing ≈ cost-neutral |
| **S-X4** | 4 | franck geometry (antipodal pairs through origin) | primary conflict-dense case |
| **S-X6** | 6 | 3 antipodal pairs, staggered planes | high conflict degree |
| **S-X8** | 8 | 4 antipodal pairs | stress test → expect baseline deadlock/infeasible |
| **S-line** | 4 | head-on single corridor (2 vs 2) | clean queueing / slot-count check |

Design rules:
- Reuse the two franck configs as **templates**; scenarios differ only in `drones[]`
  start/target and count. Keep `dt`, `room`, `physics`, `safety_zone=0.6`,
  `radius=0.2` **identical across all arms** so only the coordinator + base
  controller change.
- Generate start/target rings programmatically (`tools/utility/generate_sphere_positions.py`,
  `generate_positions`) so density scales cleanly and seeds are reproducible.
- The C7 smoke run already showed the 4-drone franck geometry does **not**
  hard-deadlock the central baseline (it grazes the safety boundary, margin ≈ 0).
  So **S-X6 / S-X8 are the scenarios that must produce real deadlock** for the
  sequencing-off arms — verify in T2; tighten room / raise N if needed.

Freeze the final geometries as JSON under `paper4_intersection_sphere/scenarios/`.

---

## 3. The factorial: arms (name them, pin their configs)

Three binary factors, fully crossed → **8 arms**. The only thing that changes
between arms is (coordinator, base controller, `safety_zone_mode`, drone `type`).

- **Factor M — MPC mode:** `central` vs `distributed`.
- **Factor S — Sequencing:** `off` (base coordinator) vs `on` (intersection coordinator).
- **Factor Z — Safety zone / controller:** `fixed` (`mpc_agent`) vs `adaptive` (`mpc_agent_adaptive`).

| Arm | Mode | Coordinator | Seq. | Base controller | Zone (`safety_zone_mode`) | Drone `type` |
|-----|------|-------------|------|-----------------|---------------------------|--------------|
| **A1** | central | `mpc_central` | off | `mpc_agent` | fixed | default |
| **A2** | central | `mpc_central` | off | `mpc_agent_adaptive` | adaptive | default |
| **A3** | central | `mpc_central_intersection` | on | `mpc_agent` | fixed | gated |
| **A4** | central | `mpc_central_intersection` | on | `mpc_agent_adaptive` | adaptive | gated |
| **A5** | distributed | `dmpc_admm` | off | `mpc_agent` | fixed | default |
| **A6** | distributed | `dmpc_admm` | off | `mpc_agent_adaptive` | adaptive | default |
| **A7** | distributed | `dmpc_admm_intersection` | on | `mpc_agent` | fixed | gated |
| **A8** | distributed | `dmpc_admm_intersection` | on | `mpc_agent_adaptive` | adaptive | gated |

### 3.1 The comparisons the factorial buys us

- **Sequencing effect (the headline) — 4 matched pairs**, each holding mode + zone
  fixed and toggling only sequencing:
  - central, fixed:    **A3 vs A1**
  - central, adaptive: **A4 vs A2**
  - distributed, fixed:    **A7 vs A5**
  - distributed, adaptive: **A8 vs A6**
  These four deltas are the C1/C2 evidence: lower energy + deadlock removal from
  sequencing, on each substrate, with each zone type.
- **Central vs distributed:** A1↔A5, A3↔A7, A2↔A6, A4↔A8 — shows the rule behaves
  the same on both MPC substrates (the paper claims it applies to both unchanged).
- **Fixed vs adaptive zone:** A1↔A2, A3↔A4, A5↔A6, A7↔A8 — separates the
  adaptive-zone effect (published `muemken2026dmpc`) from the sequencing effect
  (this paper's contribution). Adaptive zone + sequencing (A4/A8) should be best.

### 3.2 Decisions baked in

- **Z = analytic adaptive radius.** Adaptive arms use `safety_zone_mode:"adaptive"`
  (`mpc_agent_adaptive`), i.e. the analytic `r_i = r_min + α·‖v‖²/(2 U_max)` of §IV.
  The LSTM-predicted radius (`"lstm"`) is **not** used — it would couple Paper 4's
  results to the Paper 3 model.
- **Sequencing-off uses the dedicated base coordinators** (`mpc_central` / `dmpc_admm`)
  with plain drones — not the intersection coordinator with `intersection_enabled:false`.
  Both are behaviorally identical (the mixin adds nothing when disabled), but naming
  the real baseline coordinator is cleaner for the paper. The runner may instead
  toggle `intersection_enabled` on the intersection coordinator if that simplifies
  config generation — note the equivalence in the runner.
- **Fixed+sequencing is now a first-class arm (A3/A7).** (It was a deferred "5th arm"
  in the earlier 4-arm draft; the factorial subsumes it and it directly isolates
  sequencing from the adaptive zone.)

---

## 4. Metrics (definitions frozen here)

Computed each step on **real** drone state (not MPC-predicted), mirroring
`sim_runner.py`'s near-miss logic and `verify_dmpc.py`'s margin logic. One CSV row
per (arm, scenario, seed).

1. **Mission completion** — `all_drones_reached_destination(sim.drones, thresh=0.1)`
   within `max_steps`. Per-arm **completion rate** over seeds.
2. **Outcome classification (4-way)** — the core C1 evidence:
   - `COMPLETED`, `INFEASIBLE` (`sim.infeasible`/solver raised),
   - `DEADLOCK` (not completed, fleet speed `Σ‖v_i‖ < ε_v` sustained `W` steps),
   - `TIMEOUT` (hit `max_steps`, still moving). (`smoke_intersection.py` already
     implements this classification — lift it into the runner.)
3. **Control energy** (C2) — `E = Σ_i Σ_k ‖u_{i,k}‖² · dt` using `last_controls`
   (Task T0). Mean ± std over seeds. **Report it per matched pair** (A3−A1, A4−A2,
   A7−A5, A8−A6) so the sequencing delta is explicit. **Define the formula in §VI.**
4. **Travel time / makespan** — per drone `dt·(first step target_reached)`; fleet
   makespan = max over drones. (Sequencing trades makespan for energy/safety — report
   both so the trade-off is honest.)
5. **Min separation & safety violations** (C3) — per step, per pair
   `margin = ‖p_i−p_j‖ − (r_i + r_j)`, using adaptive radii for adaptive arms via
   the simulator's radius helper (as `verify_adaptive_spheres_paper.py` does).
   Report **min margin over the run** and **# steps margin < 0** (must be 0 for (S)).
6. **(diagnostic) sequencing activity** — peak active spheres, peak waiting drones,
   total parks, # deadlock-cycles dissolved (from `coordinator.active_spheres()` /
   store logs). Explains *why* a sequencing arm wins; `smoke_intersection.py` already
   collects peak spheres / peak waiting / parked set.

---

## 5. Experiments

### E1 — Factorial comparison (C1, C2, C3) ← the core result
- Grid: **8 arms (A1–A8) × scenarios {S-X3,S-X4,S-X6,S-X8,S-line} × seeds {0..29}**.
  Seeds jitter start positions (ring jitter), like `sim_runner`'s `seed = trial*1000 + salt`.
- `max_steps`: pick so the slowest completing arm finishes with margin (start 800 at
  `dt=0.1`; calibrate in T2).
- **Primary readout:** for each (mode, zone) cell and each scenario, the
  sequencing-on − sequencing-off delta in (a) completion/deadlock and (b) control
  energy. Expect: near-zero delta on S-X3 (sparse → sequencing cost-neutral),
  growing energy savings + deadlock removal as density rises (S-X6/S-X8).
- Deliverables: completion-rate table (8×5), outcome breakdown, **energy table with
  per-pair deltas**, makespan table, safety table (all-zero violations expected).

### E2 — Horizon ablation (C4) ← validates `H_min` / `κ_a`
- Fix scenario **S-X6** (conflict-dense). Run the four sequencing-on arms
  (A3,A4,A7,A8) plus their off baselines for contrast.
- Sweep `H ∈ {2,3,4,5,6,8,10}` (the "5–10 horizons" promised in VI-A-2).
- Per H: completion, deadlock/infeasible, energy, makespan over seeds.
- Overlay theoretical `H_min = ⌈(1/Δt)·√(2 r_min α / U_max)⌉` (§IV; `U_max=3.0`,
  `Δt=0.1`, `r_min`, `α` from config). Expect feasibility to collapse for `H < H_min`.
- Conflict budget `κ_a = ⌊H·Δt / τ_r^a⌋` — estimate `τ_r^a` from observed mean
  resolution-slot duration (store logs); check runs needing `> κ_a(H)` conflicts per
  horizon are exactly those that fail.

### E3 — Density sweep (C5 + the "under certain circumstances" crossover) ← key
- Run, per N, the four matched pairs (A1/A3, A2/A4, A5/A7, A6/A8) on the
  shared-center geometry. Sweep N ∈ {3,4,6,8,10,…} until the sequencing-on arms
  themselves start to fail.
- Plot, per (mode, zone): completion vs N and **energy vs N for on/off**, so the
  density at which sequencing starts to pay off (the crossover) is explicit.
- Mark: empirical baseline-deadlock onset (early), sequencing-arm failure onset
  (late), theoretical `N_c^v` from packing (§IV). Claim: sequencing failure tracks
  `N_c^v`/`κ_a`; baselines fail well below. (Reuse `tools/utility/calcN.py` /
  `crit_values_calculator.py` if they already compute these numbers.)

---

## 6. Plots & tables (what goes in §VI)

Produce with a `plot_utils.py` mirroring `paper3_lstm/plot_utils.py` (matplotlib,
IEEE-column width). PNG/PDF to `paper4_intersection_sphere/results/figures/`, then
copy chosen ones into `tex/paper/intersection_spheres/figures/`.

1. **Cost-over-time** (cumulative energy per step) for a matched pair on S-X6 —
   baseline diverging/stalling vs sequencing converging. (VI "Kosten-über-Zeit".)
2. **Factorial energy bar chart** — grouped by (mode, zone), on/off side by side, so
   the four sequencing deltas read at a glance.
3. **Energy-vs-density** (E3) — on/off curves per cell, crossover marked.
4. **Trajectory plot of deadlock resolution** — top-down XY on S-X6: a
   sequencing-off arm (deadlocked) vs its sequencing-on partner (sequenced through),
   active sphere drawn. **Most persuasive single figure** — prioritize. Reuse the GUI
   sphere rendering / `screenshots/4drone_admm_intersection_*.json`.
5. **Horizon ablation plot** (E2) with `H_min` line.
6. **Density/completion plot** (E3) with `N_c^v` line.
7. **Overview figure** for §V (`figures/sequencing_overview.png`) — referenced but
   missing (`05_methodology.tex:33`). Schematic, drawn separately.

---

## 7. Writing §VI (rewrite from intentions → observations)

`06_evaluation.tex` is a stub of intentions. After E1–E3, rewrite each subsection as
observations + interpretation around the factorial:
- **VI-A Validation of theoretical limits** ← E2 + E3 (H_min, κ_a, N_c^v).
- **VI-B Adaptive vs fixed zone** ← the Z comparison (A*even vs A*odd) from E1.
- **VI-C Sequencing** ← the four matched on/off pairs from E1 (the contribution).
- **VI-D Central vs distributed** ← the M comparison; same conclusion both substrates.
- **VI-E Combined** ← adaptive + sequencing (A4/A8): best energy + deadlock-free.
Fill the experimental-setup boilerplate (commented block at file bottom):
environment = this simulator, parameter table, scenario + factor descriptions.

**Theory feedback loop:** once E2/E3 exist, state whether empirical
deadlock/infeasibility onsets match `κ_a` and `N_c^v`; divergence is itself a result
(and possibly a §IV constant to re-tune).

---

## 8. Decisions

**Resolved:**
- **D1 Energy source:** `last_controls` mirror in the simulator (exact). ✓
- **D2 Adaptive zone:** analytic `"adaptive"` radius, not LSTM. ✓
- **D3 Design:** full 2×2×2 factorial (8 arms) over the four named coordinators and
  the two base controllers; fixed+sequencing is now first-class (A3/A7). ✓

**Still to calibrate (in T2, not blocking):**
- **Seeds:** start 10 for calibration, scale to 30 (paper3 default). The 6/8-drone
  DMPC runs are slow and hard to converge (project memory) — budget wall time.
- **`max_steps`, deadlock window `W`, `ε_v`:** set empirically (smoke defaults:
  `max_steps=800`, `W=60`, `ε_v=1e-2` — already in `smoke_intersection.py`).
- **Adaptive params (`α`, `q_vel`):** pick analytic-adaptive params in a **fresh**
  template config — do not edit the franck-tuned configs in place (project memory).

---

## 9. Task breakdown (suggested order)

```
T0  Add Simulator.last_controls mirror (+ unit test).                  [code, small]
T1  paper4_intersection_sphere/sim_runner.py                            [harness]
       - ARMS table A1..A8: (coordinator, base controller, zone, type)
       - build_arm_config(arm, scenario, seed) -> ScenarioConfig
       - run_trial(cfg) -> TrialResult (energy, makespan, separation,
         4-way outcome) — lift classification from smoke_intersection.py
       - run_grid(...) + results_to_csv()  (mirror paper3 sim_runner)
T2  Calibrate: confirm S-X6/S-X8 deadlock the sequencing-off arms,
       sequencing-on arms pass; set max_steps,W,ε_v. Freeze scenarios/*.json.
T3  E1 factorial grid (A1..A8) -> results/e1_*.csv                      [runs]
T4  E2 horizon ablation -> results/e2_horizon.csv                       [runs]
T5  E3 density sweep (matched pairs) -> results/e3_density.csv          [runs]
T6  plot_utils.py + scripts -> results/figures/*.{png,pdf}             [plots]
T7  Rewrite 06_evaluation.tex from results; add figures to paper.      [writing]
T8  Theory-feedback paragraph: empirical vs κ_a / N_c^v / H_min.       [writing]
```

Directory layout:
```
paper4_intersection_sphere/
  TEST_PLAN.md             <- this file
  CENTRAL_MPC_PLAN.md      <- central coordinator implementation plan (done)
  smoke_intersection.py    <- C7 single-config smoke (done)
  sim_runner.py            (T1)
  plot_utils.py            (T6)
  run_e1_factorial.py      (T3)
  run_e2_horizon.py        (T4)
  run_e3_density.py        (T5)
  scenarios/*.json         (T2, frozen geometries)
  results/*.csv
  results/figures/*.{png,pdf}
```

---

## 10. Risks / known gotchas

- **Sequencing-on arms need `type:"gated"`.** A sequencing-on config with
  `type:"default"` silently disables parking (`_park_loser` no-ops on plain Drone).
  Assert `type=="gated"` for A3/A4/A7/A8 in the runner.
- **The wait-controller swap is automatic** — do **not** put a wait/brake controller
  in the config; the coordinator does it. Configs only name the base controller
  (`mpc_agent` / `mpc_agent_adaptive`).
- **DMPC convergence on symmetric crossings is hard** (project memory: 4-drone
  head-on). Tune `max_admm_iter` / `rho`; don't write brittle convergence asserts.
- **GlobalMPCSolver assumes an all-opt drone list at solve time.** The intersection
  flow never violates this (losers are FREE or STOPPING at solve time, both carry
  central_cost — see `CENTRAL_MPC_PLAN.md` §7). Don't construct configs/tests that
  leave a drone WAITING across a central solve.
- **Adaptive radius in the safety metric** must use the simulator's radius helper,
  not the static `safety_zone`, or adaptive-arm margins are wrong.
- **Wall-clamp corrupts velocity-difference energy** — use the `last_controls` mirror.
- **Don't edit the franck-tuned configs in place** — copy them as templates
  (project memory).
```
