from drone_sim.domain.config import ControllerSpec, PhysicsSpec, RoomConfig, ScenarioConfig, DroneConfig, ObstacleConfig
from tools.utility import COLOR_BY_DRONE_INDEX
from tools.utility.generate_sphere_positions import generate_positions


def create_scenario(dt: float = 0.1, # simulation & coordinator
                    physics_v: float = 3.0, physics_u: float = 3.0, # physics TODO be able to use more than one
                    horizon: int = 4, controller_type: str = "mpc_agent", alpha: float | None = None, # controller TODO add params, be able to add more than one
                    coordinator_type: str = "dmpc_admm", # coordinator TODO add params
                    room_size: float = 5.0, # room TODO add sphere
                    n_drones: int = 4, drones_radius: float = 0.2, safety_zone: float = 1.5, waypoints:list[list[list[float]]]|None=None, # drones TODO, be able to use different controllers and physics
                    obstacles: list[dict[str, list[float]]]|None=None,
                    comm_radius: float | None = None, # DMPC neighbourhood: None = every drone is a neighbour of every other
                    max_admm_iter: int = 20, # DMPC rounds per step
                    terminal_constraint: bool = True, # braking-distance terminal radius at the last horizon step (room walls); False = behaviour of runs before 22.10.2026
                    ) -> ScenarioConfig:
   physics = create_physics(v_max=physics_v, u_max=physics_u)
   if controller_type == "mpc_agent":
      controller = ControllerSpec(type="mpc_agent", params={"horizon": horizon,"q_pos": [3.0, 3.0, 3.0],"r_u": [0.5, 0.5, 0.5]})
      alpha_set = None
   elif controller_type == "mpc_agent_adaptive":
      controller = ControllerSpec(type="mpc_agent_adaptive", params={"horizon": horizon, "q_pos": [2.0, 2.0, 2.0], "q_vel": [0.2, 0.2, 0.2], "r_u": [0.5, 0.5, 0.5]})
      alpha_set = alpha if alpha is not None else 0.3
   else:
      raise ValueError("Unknown controller type: %s", controller_type)
   if coordinator_type == "dmpc_admm":
      coordinator_spec = ControllerSpec(type="dmpc_admm", params={"horizon": horizon, "rho": 1.0, "max_admm_iter": max_admm_iter, "primal_tol": 1e-2, "dual_tol": 1e-2,
                                                                     "terminal_constraint": terminal_constraint})
   elif coordinator_type == "mpc_central":
      coordinator_spec = ControllerSpec(type="mpc_central", params={"horizon": horizon, "terminal_constraint": terminal_constraint})
   else:
      raise ValueError("Unknown coordinator type: %s", coordinator_type)

   room = RoomConfig(min=[-room_size/2, -room_size/2, -room_size/2], max=[room_size/2, room_size/2, room_size/2])
   if waypoints is None:
      waypoints = generate_positions(n_drones=n_drones, room_min=room.min, room_max=room.max, min_pair_distance=safety_zone*2, wall_margin=safety_zone)
   drones = [DroneConfig(drone_id=f"drone-{i}", start=waypoints[i][0], target=waypoints[i][1], controller=controller, physics="default", alpha=alpha_set,
                         radius=drones_radius, safety_zone=safety_zone, cons_stop=0.0, drone_color=COLOR_BY_DRONE_INDEX[i + 1]) for i in range(n_drones)]

   if obstacles is not None:
      obstacles = [ObstacleConfig(center=obstacle["center"], half_extents=obstacle["half_extents"]) for obstacle in obstacles]
   # The central coordinator takes no comm_radius (all pairs are coupled); only the distributed one restricts the neighbourhood.
   return build_scenario(physics=physics, controller=controller, coordinator=coordinator_spec, comm_radius=comm_radius if coordinator_type == "dmpc_admm" else None,
                         obstacles=obstacles, room=room, drones=drones, dt=dt)

def build_scenario(physics: PhysicsSpec, controller: ControllerSpec, coordinator: ControllerSpec|None, room: RoomConfig, drones:list[DroneConfig], dt:float=0.1, comm_radius:float|None=None, obstacles:list[ObstacleConfig]|None=None) -> ScenarioConfig:
   return ScenarioConfig(dt=dt, physics=physics, controller=controller, coordinator=coordinator, room=room, drones=drones, comm_radius=comm_radius, obstacles=obstacles if obstacles is not None else [])

def create_physics(type:str="linear_kinematics", id:str="default", v_max:float=3.0, u_max:float=3.0) -> PhysicsSpec:
   return PhysicsSpec(type=type, id=id, params={"v_max": v_max, "u_min": [-u_max, -u_max, -u_max], "u_max": [u_max, u_max, u_max]})

# mpc_agent_adaptive   -> AdaptiveMPCAgent              (central_cost)
# mpc_central          -> CentralMPCGlobalCoordinator   (coordinator)
# mpc_agent            -> CentralMPCAgent               (central_cost)
def create_controller(type:str="mpc_agent", horizon:int=4, q_pos:float=2.0, q_vel:float=0.2, r_u:float=0.5) -> ControllerSpec:
   ControllerSpec(type=type, params={"horizon": horizon, "q_pos": [q_pos, q_pos, q_pos], "q_vel": [q_vel, q_vel, q_vel], "r_u": [r_u, r_u, r_u]})

   # @register_coordinator("mpc_central")
   # @dataclass
   # class CentralMPCGlobalCoordinator:
   #    dt: float
   #    horizon: int = 5
   #    room_wall_tolerance: float = 0.0
   #    symmetry_break_accel: float = 0.05
   #    max_iter: int = 120
   #    f_tol: float = 1e-3

   # @register_controller("mpc_agent")
   # @dataclass
   # class CentralMPCAgent(Controller):
   #    dt: float
   #    horizon: int = 12
   #    q_pos: list[float] = (8.0, 8.0, 8.0)
   #    q_vel: list[float] = (0.5, 0.5, 0.5)
   #    r_u: list[float] = (0.2, 0.2, 0.2)

   # @register_controller("mpc_agent_adaptive")
   # @dataclass
   # class AdaptiveMPCAgent(CentralMPCAgent):
   #    lambda_vel: float = 1.0