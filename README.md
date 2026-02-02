# Geometric and Control-Theoretic Limits – Simulation Guide

This document describes how to reproduce, verify, and visualize the simulation scenarios used in the paper *"Geometric and Control-Theoretic Limits on Drone Density in Bounded Airspace"* with this repository.
The configuration files under `configs/basic_paper` implement the MPC framework described in the paper. The overall software architecture is intentionally kept relatively modular and complex in order to make it straightforward to extend the implementation with additional controllers, physics models, and scenarios beyond those used in the manuscript.

## 1. Setup

Requirement: Python 3.11+ (or a compatible version).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Fast Usage

This section provides a quick overview of the main tools for running simulations.

### 2.1 Parameter Sweep (Batch Simulations)

The `simulation_parameter_sweep` tool runs multiple simulation scenarios in parallel, sweeping over parameters like number of drones, horizon, velocity, and acceleration limits. Results include metrics CSVs and optional GIF animations.

```bash
python -m tools.simulation_parameter_sweep
```

The sweep configurations are defined directly in the script. Key parameters include:
- `n_drones`: Range of drone counts to test
- `horizon`: MPC prediction horizon
- `safety_zone`: Safety radius around each drone
- `v_max`: Maximum velocity
- `u`: Maximum acceleration
- `room_size`: Cubic room side length (or `r_room` for spherical rooms)

Results are saved to timestamped folders under `param_swep_result/` with:
- `metrics.csv`: Performance metrics for each scenario
- `*.gif`: Animation of each simulation (if enabled)
- Heatmaps and combined analysis files

### 2.2 Live View (Single Simulation with Visualization)

The `live_view` tool runs a single simulation with real-time 3D visualization and optional GIF export. This is useful for debugging and visualizing specific configurations.

```bash
python -m tools.live_view \
  --config configs/2DronesHorizon2.json \
  --steps 200 \
  --trace-len 100 \
  --gif results/2DronesHorizon2.gif \
  --gif-fps 20
```

Key options:
- `--config`: Path to a JSON configuration file
- `--steps`: Number of simulation steps to run
- `--trace-len`: Length of trajectory trails to display
- `--gif`: Output path for animated GIF
- `--gif-fps`: Frame rate for the GIF
- `--param KEY=VALUE`: Override configuration parameters (can be repeated)

### 2.3 REST API (Programmatic Access)

For programmatic control or integration with other tools, use the REST API.

**Starting the server:**

```bash
uvicorn drone_sim.api.app:app --reload
```

The server will listen on `http://127.0.0.1:8000`.

**Loading and stepping a simulation:**

```bash
# Load a configuration
curl -s -X POST http://127.0.0.1:8000/config \
  -H "Content-Type: application/json" \
  --data-binary @configs/2DronesHorizon2.json

# Step the simulation
curl -s -X POST "http://127.0.0.1:8000/step?n=10"

# Get current state
curl -s http://127.0.0.1:8000/state
```

## 3. Central MPC Architecture in the Codebase

All paper experiments use the same centralized MPC architecture:

- **Per-drone controller**
  `controller.type = "mpc_agent"`
  Implementation: `src/drone_sim/controllers/central_cost.py` (`CentralMPCAgent`)

- **Central coordinator**
  `coordinator.type = "mpc_central"`
  Implementation: `src/drone_sim/simulation/coordinator.py` (`CentralMPCGlobalCoordinator`)

- **Simulator**
  Implementation: `src/drone_sim/simulation/simulator.py` (`Simulator`)
  - Constructs from `ScenarioConfig`:
    - the physics model (`linear_kinematics`),
    - all `Drone` objects,
    - obstacles and room bounds.
  - For each time step it:
    1. Evaluates the local controller of each drone (used for non-optimized drones and as fallback),
    2. Invokes the coordinator (`solve_controls`) to perform the global SLSQP solve,
    3. Applies the physics update and collision detection.


<p align="center">
  <img src="./results/2DronesHorizon2.gif" width="250" height="250" alt="2 Drones small horizon" style="margin-right: 10px;">
</p>

<details>
<summary><strong>Json Configuration for this scenario:</strong></summary>

All paper scenarios are defined in `configs/basic_paper/*.json` and follow the pattern:

```json
{
  "dt": 0.1,
  "room": { "min": [...], "max": [...] },
  "physics": { "type": "linear_kinematics", "params": {} },

  "controller": {
    "type": "mpc_agent",
    "params": {
      "horizon": H
      "u_min": [-3.0, -3.0, -3.0],
      "u_max": [ 3.0,  3.0,  3.0]
    }
  },

  "coordinator": {
    "type": "mpc_central",
    "params": {
      "horizon": H,
      "room_wall_tolerance": 0.5,
      [...]
    }
  },

  "drones": [...],
  "obstacles": [...]
}
```

</details>

<p align="center">
  <img src="./results/4DronesHorizon1.gif" width="250" height="250" alt="4 Drones Horizon 1" style="margin-right: 10px;">
  <img src="./results/4DronesHorizon2.gif" width="250" height="250" alt="4 Drones Horizon 2" style="margin-right: 10px;">
</p>

<p align="center">
  <img src="./results/6DronesHorizon4.gif" width="250" height="250" alt="6 Drones Horizon 4" style="margin-right: 10px;">
  <img src="./results/6DronesHorizon10.gif" width="250" height="250" alt="6 Drones Horizon 10" style="margin-right: 10px;">
</p>
Four-drone scenarios are easily solvable, but the chosen horizon should be neither too small nor too large.
Six-drone scenarios are solvable, a small horizon will result in many calculation steps, a large horizon will slow down the calculation.

## 4. MPC Model (Brief Description)

### 4.1 Dynamics

Each drone is modeled as a discrete-time double integrator in three dimensions:

- State  $x_k = [p_{x,k}, p_{y,k}, p_{z,k}, v_{x,k}, v_{y,k}, v_{z,k}]^\top \in \mathbb{R}^6$
- Input (acceleration) $u_k = [a_{x,k}, a_{y,k}, a_{z,k}]^\top \in \mathbb{R}^3$
- Sampling time $\Delta t = 0.1 \,\text{s}$.
- Discrete-time dynamics $x_{k+1} = A x_k + B u_k$, with
  $$
  A = \begin{bmatrix}
      I_3 & \Delta t\, I_3 \\
      0_3 & I_3
  \end{bmatrix},
  \quad
  B = \begin{bmatrix}
      \tfrac{1}{2}\Delta t^2 I_3 \\
      \Delta t I_3
  \end{bmatrix}.
  $$

The implementation is provided by `LinearKinematicsPhysics` in `src/drone_sim/physics/linear_kinematics.py`.

### 4.2 Cost Function

For each drone \(k\) with reference position \(\bar p_k\) and prediction horizon \(H\), the stage cost is

  $$
  J_k = \sum_{h=0}^{H-1}
  \left(
    (p_k(h) - \bar p_k)^\top Q_p (p_k(h) - \bar p_k)
    + v_k(h)^\top Q_v v_k(h)
    + u_k(h)^\top R u_k(h)
  \right),
  $$


The central coordinator minimizes the aggregate cost over all drones:

$$
J = \sum_{k=1}^N J_k.
$$

### 4.3 Constraints

The main constraints (implemented in the `mpc_central` coordinator) are:

1. **Inter-drone distance**
   For all drone pairs \(i \neq j\) and all prediction steps \(h\):
   $$
   \|p_i(h) - p_j(h)\|_2
   \;\ge\;
   \max\bigl(s_i + r_j,\; s_j + r_i\bigr) + \text{safety\_buffer}.
   $$
   Here \(s_i\) is the safety zone radius of drone \(i\), and \(r_i\) is its physical radius.

2. **Input (acceleration) bounds**
   Component-wise,
   $$
     u_{\min} \le u_k(h) \le u_{\max}
   $$
   using `u_min` and `u_max` from the configuration.

3. **Room (workspace) constraints**
   With room \(\Omega = [\text{room\_min}, \text{room\_max}]\) and physical radius \(r_k\),
   $$
     B_{r_k}(p_k(h)) \subset \Omega
     \quad\Leftrightarrow\quad
     \text{room\_min}_d \le p_{k,d}(h) - r_k,\;
     p_{k,d}(h) + r_k \le \text{room\_max}_d
     \;\;\forall d \in \{x,y,z\}.
   $$

These constraints are enforced within the centralized SLSQP optimization and are additionally checked at the simulation level (via `Simulator._compute_collisions` and the verification script).


## 5. Citation
If you use this code or build upon our work, please cite our paper:


```bibtex
@article{dronesxxx,
  title={Geometric and Control-Theoretic Limits on Drone Density in
Bounded Airspace},
  author={Altinses  Muemken, Lier, and Schwung},
  journal={Drones}
}
```