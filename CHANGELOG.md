# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).


## [v2.2.2] — 2026-11-04 (IEEE T-ASE Revision)

Changes made in response to the reviews: a braking-distance terminal constraint for the MPC horizon,
CPU-cost measurement, deadlock detection, and the tooling for the revision campaigns (volume and
constant-density scaling) on Hetzner.

### Added
- Terminal radius `Drone.compute_terminal_radius(v)` = `safety_zone + ||v||²/(2·U_max) + 0.5·dt·||v||`.
  Braking to a standstill is always a feasible continuation of the plan, so the next MPC problem stays feasible.
- Terminal constraint rows at the last horizon step, switchable per coordinator
  (`mpc_central`, `dmpc_admm`, threaded DMPC) and in `GlobalMPCSolver`/`LocalMPCSolver`:
  - `terminal_constraint` (default **on**): room walls (`RoomConstraints`, box and sphere rooms)
  - `terminal_constraint_pairs` (default off): drone pairs (`MovingObstacleAvoidanceConstraints`)
  - `terminal_constraint_obstacles` (default off): static obstacle boxes (`ObstacleAvoidanceConstraints`)
- `CpuCostLedger` (`simulation/utils/cpu_costs.py`): exclusive CPU seconds per phase (setup, cost,
  constraints, solver, validate, exchange, admm_update, physics, metrics, deadlock), evaluation counts,
  and DMPC per-drone timings (`local_solve`, `parallel_step`, `max_admm_iter_step`).
  Shared by the simulator, the coordinators and the solvers.
- `DeadlockDetector`: a run ends with the new status `deadlock` once no drone has moved ≥ 0.01 m per step
  for 100 consecutive steps (configurable, `--deadlock-steps 0` turns it off).
- `run_single_scenario` returns a cost row as an 8th element, and metrics CSV/DB rows carry the CPU-cost columns.
- Distributed campaign tooling (`paper2_tools/distributed/`):
  - `create_queue --experiment volume|density`, `--dry-run`, `--sql [FILE]` (INSERT script for the DB console)
  - per-task overrides for `room_size`, `comm_radius`, `max_admm_iter`, `timeout_s`; new columns `experiment`/`variant`
  - package-based claiming: a host keeps an (experiment, variant, N, coordinator, controller) package until it is done
  - `queue_waves` (breadth-first processing over N), `expected_results` (synthetic rows for known
    no-placement/timeout cells, with `undo`), `db_util` (status, dedup, packages), `import_existing`
  - `export_csv --experiment`
  - idempotent schema migration through `ensure_schema()`, also called by the worker at startup
- Reproducible placement: `placement_seed(run, N)` gives every coordinator/controller combination of a
  (run, N) pair identical start and target positions (worker and local CSV runner, which now has a `run` column).
- Paper figures in `plot_scenario_1_2.py`: deadlock rate, time per unit, scaling curves, combined
  four-panel deadlock plot, IEEE sizing, `--n-max`, `--font-scale`, `--legacy`.
- Drone colours 41–100 for larger fleets.
- Tests: constraints (terminal rows), drone terminal radius, coordinators, CPU costs, create_queue,
  density scaling, placement seed, worker settings, live view, deadlock helper, scenario creator.

### Changed
- Campaign defaults (`paper2_tools/distributed/config.py`): `HORIZON` 4 → 8, `ALPHA` 1.5 → 0.5,
  `MAX_ADMM_ITER = 20`, campaigns run with `TERMINAL_CONSTRAINT = False` (same as runs before 22.10.2026).
- `create_scenario` accepts `comm_radius` (DMPC only), `max_admm_iter`, `terminal_constraint`.
- The constraint breakdown in `GlobalMPCSolver` uses `RoomConstraints` directly, so the log matches the solver.
- Hetzner workers run with `OPENBLAS_NUM_THREADS=1` (and OMP/MKL); the default worker type is `cpx42`.

### Fixed
- `predefined_patterns_for` checked the room size one-sided only, so the 8 m room with r = 1.52
  picked up the 9 m pattern, whose positions lie in the wall margin.

### Breaking Changes
- `run_single_scenario` returns 8 values instead of 7.
- Local queue CSVs without a `run` column are rejected, so old queues have to be recreated.
- The `scenario_runs` unique key now includes `experiment` and `variant` (migrated automatically).

## [v2.2.1] — 2026-04-02 (IEEE T-ASE Submission)

### Added
- GUI with PySide6 for interactive simulation control
- `ConflictDetector`, `ConflictEvasionCentralCoordinator` and `ConflictEvasionDistributedCoordinator` for MPC-based conflict evasion
- Drone model rendering (instead of sphere-only) in GUI and API
- Opaque obstacle rendering
- Obstacle logging
- Terraform integration for running large-scale scenarios on Hetzner
- Infeasibility detection when drones cannot be placed due to insufficient space
- Plot tools for paper figures (EPS export, scenario plots)
- Joblib-based parallelization (replaces manual threading)

### Changed
- DMPC performance improvements (timeout tuning, reduced max iterations)
- Drone rendering available in API-render, not only in GUI

### Fixed
- Obstacle avoidance constraint fix
- Logging bug in distributed coordinator
- Controller issue in live view

## [v2.2.0] — Threaded MPC

### Added
- `ThreadedMPCCoordinator` for asynchronous distributed MPC
- Thread-safe mailbox system for inter-drone communication
- Tests and verification for threaded MPC milestone

### Fixed
- Warm-start issue in threaded MPC

## [v2.1.0] — Adaptive Safety Zones

### Added
- `AdaptiveMPCAgent` controller class with velocity-dependent adaptive safety spheres
- `compute_adaptive_radius` with `safety_zone` as minimum floor
- Per-step safety radii in distributed MPC data structures
- Adaptive safety radius exposed in API response
- Adaptive spheres configurable via scenario config
- Parameter sweep script with CSV output

### Changed
- Refactoring: extracted `GlobalMPCSolver` from `CentralMPCGlobalCoordinator`

## [v2.0.0] — Distributed MPC (ADMM)

### Added
- `DistributedMPCCoordinator` with ADMM-based coordination
- `LocalMPCSolver` as per-drone local optimization solver
- `TrajectoryMailbox` and `TrajectoryMessage` for trajectory communication
- `NeighbourGraph` for neighborhood-based communication structure
- `ADMMState` class for convergence tracking
- Gauss-Seidel style updates with symmetry breaking
- ADMM statistics tracking
- Constraint classes for modular constraint handling
- Live view support for distributed simulations

### Changed
- Coordinator refactored to use Drone objects instead of raw lists
- Docstrings converted to Sphinx format

## [v1.1.0]

### Added
- Room wall tolerance configurable via scenario config
- Horizon feasibility test
- Live view without REST server
- Scripts for CSV/GIF generation and horizon discussion plots

### Fixed
- Room constraint bug with invalid start positions
- Velocity constraint to match paper description

## [v1.0] — 2026-01-15 (Initial Release)

Initial version used to generate simulation results for the first paper.

### Features
- Centralized MPC with fixed safety zones
- REST-based live view
- Basic scenario configuration
- Room and obstacle constraints
- Linear kinematics physics model