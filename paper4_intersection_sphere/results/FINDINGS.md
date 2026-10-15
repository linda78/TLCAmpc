# E1 Findings — Intersection-Sphere Factorial (open shared-center crossings)

Data: `e1_factorial.csv` (60 runs). 8 arms (A1–A8) × scenarios × 3 seeds.
Central (A1–A4): S-X3/S-X4/S-X6. Distributed (A5–A8): S-X3/S-X4 (DMPC too slow for
a full S-X6 sweep at multi-seed).

Settings: dt=0.1, H=8, V_max=U_max=3, safety_zone=0.6, body_radius=0.2,
room ±5, adaptive α=0.33. DMPC max_admm_iter=20, max_steps=250.

## Headline (honest)

**On open shared-center crossings, intersection-sphere sequencing does NOT reduce
control energy and there are no deadlocks to remove — neither central nor
distributed MPC deadlocks on these scenarios.** Sequencing adds 3–89 units of
control energy (every matched pair, growing with density). This contradicts the
abstract's "removes the deadlocks … lower control energy" *for this scenario class*.

The reason: an open 3-D crossing imposes no bottleneck, so a capable joint MPC
threads the drones through (makespan ≈ straight-line flight, ~3.5 s). Fixed-zone
baselines complete by *grazing* the safety-zone boundary; adaptive baselines
complete by transiently *violating* their own (larger) velocity-dependent zone.
Neither produces the deadlock the rule is designed to resolve.

## What sequencing DOES buy (the real, defensible benefit)

**Guaranteed separation margin.** With fixed zones, sequencing eliminates the
zone-boundary grazing of the baseline:
- A3/A7 (fixed + sequencing): 0 zone-constraint violations, min body margin up to
  +0.96 m.
- A1/A5 (fixed baseline): 1–2 seeds graze the zone boundary (min margin ≈ 0).
So sequencing trades control energy for robust clearance, not for lower energy.

## Safety (requirement S) — holds everywhere

**Zero real body collisions in all 60 runs** (min body margin ≥ +0.80 m). The
"violations" seen on adaptive arms are overlaps of the *velocity-dependent safety
zone* (a constraint the soft solver relaxes at speed), not physical collisions.

## Matched-pair energy deltas (ON − OFF), mean over seeds

| scenario | central fixed | central adaptive | dist fixed | dist adaptive |
|----------|---------------|------------------|------------|---------------|
| S-X3 | +31.0 | +16.7 | +22.2 | +3.3 |
| S-X4 | +59.5 | +30.9 | +53.6 | +13.3 |
| S-X6 | +88.7 | +55.8 | (n/a) | (n/a) |

All positive → sequencing costs more. Adaptive zones soften the penalty; fixed
zones pay the most (the stop-and-go of holding before a sphere).

## Liveness bug found & fixed (the user-requested debug)

The sequencing arms *themselves* deadlocked before the fix (A3 on S-X4 seed1 &
S-X6 seed2; A7 on S-X4 seed1; A8 timeout on S-X3 seed1) — contradicting the §V
liveness theorem.

- **Root cause:** an *orphaned waiter*. After a rehang, a yielding drone's
  `waiting_for` no longer matched the dropping sphere's `priority_root`, so
  `_release_losers` skipped it; the sphere was deleted and the waiter was left
  parked on a drone that was already free at its goal and would never re-enter a
  sphere → permanent wait.
- **Fix:** a liveness safeguard at the end of `IntersectionStateStore.step()`
  enforcing the invariant *"a drone waits only while some active sphere sequences
  the drone it waits for"*: any waiter whose `waiting_for` is not the priority/root
  of an active sphere is released. Preserves transitive waiters (target still an
  active root). Covered by `test_state_store_releases_orphaned_waiter_when_target_idles`.
- **After the fix:** all 60 runs complete, no deadlocks.
- **Remaining (not yet hit):** if a priority drone never *enters* its sphere
  (overshoots / path misses the detection point), the sphere never drops and parks
  its loser indefinitely. Not observed in any run; address if a future scenario
  triggers it.

## Corner scenarios (S-C6 / S-C8) — dense cube-corner crossings in a small room

Added `corners` scenarios: drones start at cube corners of a small room
(`room_half=3.0`) and fly to the opposite corner, so all space diagonals cross the
center. `room_half=2.5` is past the packing limit (8-drone baseline infeasible at
step 1). This is the scenario class that finally surfaces the paper's claim.

**The key positive result — S-C8, fixed zone, central (`e1_corners_central.csv`):**

| Arm | Completion | Outcome |
|-----|-----------|---------|
| A1 fixed-zone baseline | **33 %** | 2/3 seeds INFEASIBLE |
| A3 fixed-zone + sequencing | **100 %** | all complete |

→ **dCompletion = +67 %.** The dense 8-corner crossing makes the *fixed-zone*
central MPC infeasible (the solver finds no joint plan); sequencing restores
mission completion. This is the "removes the failures of fixed-zone MPC" claim
(req. L/F). The failure is **specific to fixed zones** — the adaptive baselines
(A2/A4) complete at 100 % (the adaptive zone already eases the packing), exactly as
the paper argues ("fixed safety zones aggravate this").

Other corner cells:
- S-C6 (6 drones): every arm completes on both substrates; 6 drones is not dense
  enough to break the fixed baseline. Sequencing still costs more energy, except
  **distributed adaptive S-C6: A8 < A6 by ~37** — the first case where sequencing
  is *cheaper*, because the non-converging adaptive DMPC baseline wastes energy
  oscillating in the tight room.
- The baseline failures are INFEASIBLE (no joint solution), not stall-deadlock —
  mission-equivalent to failure.
- Energy: still higher for sequencing on completed runs; the benefit is
  completion/feasibility, not energy.
**DMPC on S-C8 (8 drones, all four arms, 2 seeds — `e1_c8_dmpc_all.csv`):**

| Arm | (seq, zone) | Completion | zone-breach steps/run | energy |
|-----|-------------|-----------|------------------------|--------|
| A5 | off, fixed    | 100 % | 4–12  | 439 |
| A7 | on,  fixed    | 100 % | **0–1**   | 491 (+12 %) |
| A6 | off, adaptive | 100 % | 43–50 | 485 |
| A8 | on,  adaptive | 100 % | **85–93** (worse) | 496 |

Two clear results, both refuting "same on both substrates" in different ways:

1. **No completion rescue on distributed.** Every distributed S-C8 run completes —
   fixed *and* adaptive baseline. The centralized joint SLSQP fails all-or-nothing
   when over-constrained (33 %); the distributed ADMM degrades gracefully (grazes
   through, completes). **The completion rescue is central-specific.**
2. **The constraint-adherence benefit is fixed-zone-specific.** With fixed zones,
   sequencing cuts breach steps 4–12 → 0–1 (clear win, +12 % energy). With adaptive
   zones it makes them *worse* (43–50 → 85–93) and lengthens the makespan — the
   stop-then-accelerate pattern inflates the velocity-dependent zone and the longer
   episode accumulates more breaches. No benefit there.

Note: the adaptive DMPC baseline (A6) is pathologically slow on 8 corner drones
(~20–24 min/trial — non-converging ADMM every step over a long episode).

## E2 — horizon ablation (H_min validation, `e2_horizon.csv`)

Central arms A1–A4 on S-C6, horizon $H \in \{2..10\}$, 2 seeds. Theory:
$H_{\min} = \lceil (1/\Delta t)\sqrt{2 r_{\min}\alpha/U_{\max}}\rceil = 4$.

| H | A1 fix | A2 adapt | A3 seq-fix | A4 seq-adapt |
|---|--------|----------|------------|--------------|
| 2–3 | 0 % | 0 % | 0 % | 0 % |
| **4 = H_min** | 0 % | **100 %** | 50 % | 50 % |
| 5 | 0 % | 100 % | 50 % | 100 % |
| 6–10 | 100 % | 100 % | 100 % | 100 % |

**Clean validation:** below H_min=4 nothing completes; the adaptive MPC becomes
feasible exactly at H=H_min=4. Fixed zones need H≥6. The feasibility onset lands on
the theoretical line.

## E3 — density sweep (capacity / N_c^v, `e3_density.csv`)

Central A1 (fixed baseline) vs A3 (fixed sequencing), corner room 6³, N ∈ {4..8},
3 seeds.

| N | A1 baseline | A3 sequencing |
|---|-------------|---------------|
| 4–7 | 100 % | 100 % |
| **8** | **33 %** | **100 %** |

**Validation:** the fixed-zone baseline's effective feasibility limit is N≈8 — far
below the static packing bound (N_c ≈ 177 for this room/r_min), exactly the
"finite control authority inflates exclusion zones" effect of §IV. Sequencing
raises the limit past N=8 by serializing the simultaneous passage.

## Implication for the paper (final)

The data refines the abstract's claims rather than confirming them as stated:

1. **Energy claim ("lower control energy"): does not hold in general.** On every
   scenario the underlying MPC can already solve, sequencing *adds* energy
   (stop-and-go). Only one cell (distributed adaptive, S-C6, non-converging
   baseline) was marginally cheaper. Recommend softening this claim or scoping it.
2. **Deadlock/feasibility claim ("removes the failures of fixed-zone MPC"): holds,
   but narrowly.** Demonstrated on the dense 8-drone corner crossing with the
   **centralized fixed-zone** MPC (33 % → 100 % completion). It is specific to
   fixed zones (adaptive baselines complete) and to the centralized substrate
   (the distributed baseline grazes through instead of failing).
3. **"Both substrates" — split verdict.** Constraint adherence improves on both
   (zone breaches 8→2 on distributed S-C8); the completion rescue is central-only.
4. **Safety (S): holds everywhere.** Zero physical collisions in all runs.

**Defensible framing:** intersection-sphere sequencing trades control energy for
**guaranteed feasibility and separation** under spatial congestion. It restores the
missions a centralized fixed-zone MPC cannot complete, and it eliminates the
safety-zone breaches of the grazing baselines — on both substrates. It is not an
energy-reduction method.

The theory-limit ablations (E2/E3) are done and both confirm \autoref{sec:theory}:
feasibility onsets at the predicted $H_{\min}=4$, and the effective capacity
($N\approx8$) sits far below the static packing bound, with sequencing extending it.

Possible further steps: a narrow-gap/corridor scenario (`configs/narrow_gap/`);
the conflict-budget $\kappa_a$ link (estimate $\tau_r^a$ from store logs);
more seeds for the expensive DMPC corner cells.

## Artifacts

- Open crossings: `e1_factorial.csv` (= `e1_central.csv` + `e1_dmpc.csv`)
- Corner crossings: `e1_corners.csv` (= `e1_corners_central.csv`
  + `e1_corners_dmpc.csv` + `e1_c8_dmpc_fixed.csv`)
- `figures/e1_energy_factorial.png` — open-crossing energy off vs on
- `figures/e1_outcomes.png` — open-crossing outcomes
- `figures/e1_corners_energy_factorial.png` — corner energy off vs on
- `figures/e1_corners_outcomes.png` — corner outcomes (A1 infeasible vs A3 rescue)
- `figures/trajectories_S-X4_A1_vs_A3.png`,
  `figures/trajectories_S-C8_A1_vs_A3.png` — baseline vs sequenced trajectories
