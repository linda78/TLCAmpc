import math

KEPLER_DENSITY = math.pi / (3 * math.sqrt(2))  # ≈ 0.7405

def sphere_vol(r: float) -> float:
   return 4/3 * math.pi * r * r * r

def dice_vol(l: float) -> float:
   return l * l * l

def theoretical_n(l: float, r: float) -> float:
   return dice_vol(l) * KEPLER_DENSITY / sphere_vol(r)

def stop_distance(r: float, u_max: float, v_max: float) -> float:
   return 2 * r + (v_max * v_max / 2 / u_max)

def rho_stop(r: float, u_max: float, v_max: float) -> float:
   return stop_distance(r, u_max, v_max)

def rho_jam(r: float, u_max: float, v_max: float) -> float:
   return stop_distance(r, u_max, v_max) / 2

def theo_stop_jam(l: float, r: float, u_max: float, v_max: float) -> float:
   return KEPLER_DENSITY * dice_vol(l) /  sphere_vol(rho_stop(r, u_max, v_max))

def theo_jam_n(l: float, r: float, u_max: float, v_max: float) -> float:
   return KEPLER_DENSITY * dice_vol(l) /  sphere_vol(rho_jam(r, u_max, v_max))

def print_dict(result: dict):
   key_list = result.keys()

   print('=' * 32 * len(key_list))
   print(''.join([f"{k} -> N: {str(int(result[k][0]['N']))}\t|" for k in key_list]))
   print('-' * 32 * len(key_list))
   print(f" u_max v_max| rho\t  N_j\t\t|" * len(key_list))
   print('-' * 32 * len(key_list))

   for i in range(len(result['side 5.0 - radius 1.0'])):
      line_str = ''
      for key in key_list:
         line = result[key][i]
         line_str += f" {line['u']:>3.1f}\t{line['v']:>3.1f}\t | {line['rho']:>5.3f} {line['N_j']:>5.1f}\t\t|"
      print(line_str)

   print('=' * 32 * len(key_list))

def run_sweep() -> dict:
   result = {}
   for l in [5, 7.5, 8, 9]:
      for r in [1, 1.5, 2]:
         for u_max in [1.0, 2.5, 3.0, 5.0]:
            for v_max in [1.0, 2.5, 3.0, 3.5, 5.0]:
               key = f"side {l:>3.1f} - radius {r:>3.1f}"
               if not result.keys().__contains__(key):
                  result[key] = []
               result[key].append({'u': u_max, 'v': v_max, 'N': theoretical_n(l, r), 'rho': rho_jam(r, u_max, v_max), 'N_j': theo_jam_n(l, r, u_max, v_max)})
   return result

if __name__ == "__main__":
   result = run_sweep()
   print_dict(result)

   print(result)

