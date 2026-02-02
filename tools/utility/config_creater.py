from drone_sim.domain.config import ControllerSpec, DroneConfig, PhysicsConfig, RoomConfig, ScenarioConfig
from tools.utility.constants import _COLOR_BY_DRONE_INDEX
from tools.utility.generate_sphere_positions import generate_positions, predefined_patterns_for

def create_default_config(n_drones: int, horizon: int):
   positions = predefined_patterns_for(n_drones=n_drones, r_sphere=1.0, room_size=5.0, in_sphere=False)
   return create_config(n_drones=n_drones, horizon=horizon, safety_zone=1.0, v_max=2.5, u=3.0, room_size=5.0,
                        r_room=0.0, extra_space=0.0, positions=positions)

def create_config(n_drones: int, horizon: int, safety_zone: float, v_max: float, u: float, room_size: float, r_room: float = 0.0,
                  extra_space: float = 0.2, positions: list[tuple[list[float], list[float]]] = None):
   try:
      if r_room > 0 and room_size > 0:
         raise ValueError("cannot provide both room_size and r_room ")

      if r_room > 0:
         room = RoomConfig(r=r_room)
      else:
         room = RoomConfig(min=[-room_size / 2] * 3, max=[room_size / 2] * 3)

      physics = [PhysicsConfig(id="standard", type="linear_kinematics", params={"v_max": v_max, "u_min": [-u]*3, "u_max": [u]*3})]
      controller = ControllerSpec(type="mpc_agent", params={"horizon": horizon, "q_pos": [1.0]*3, "r_u": [1.0]*3, "q_vel": [1.0]*3})
      coordinator = ControllerSpec(type="mpc_central", params={"horizon": horizon})

      if positions is None:
         positions = generate_positions(n_drones=n_drones, room_min=[-room_size / 2] * 3, room_max=[room_size / 2] * 3, max_attempts=100000,
                                     min_pair_distance=2 * safety_zone + extra_space, wall_margin=safety_zone, r_room=r_room)
      drones: list[DroneConfig] = []

      for position, i in zip(positions, range(len(positions))):
         drones.append(DroneConfig(drone_id=f"drone-{i + 1}", start=position[0], waypoints=[], target=position[1],
                                   physics="standard", radius=0.2, safety_zone=safety_zone, drone_color=_COLOR_BY_DRONE_INDEX[i + 1]))

      return ScenarioConfig(dt=0.1, physics=physics, controller=controller, coordinator=coordinator, drones=drones, obstacles=[], room=room)
   except Exception as e:
      print(f"Error creating config: {e}")
      return None
