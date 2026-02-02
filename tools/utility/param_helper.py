import json
import re
from typing import Any
from pathlib import Path
import ast

def _parse_kv_params(items: list[str]) -> dict[str, str]:
   out: dict[str, str] = {}
   for item in items:
      if "=" not in item:
         raise ValueError(f"Bad --param '{item}'. Expected KEY=VALUE")
      k, v = item.split("=", 1)
      out[k] = v
   return out

def set_nested(data: dict, path: list[str], value: Any) -> None:
   for key in path[:-1]:
      data = data[key]
   data[path[-1]] = value

def apply_to_all_items(items: list[dict], key: str, value: Any) -> None:
   for item in items:
      if key in item:
         item[key] = value

def typed_value(value):
   try:
      value = ast.literal_eval(value)
   except ValueError:
      return value
   if isinstance(value, float):
      return float(value)
   if isinstance(value, bool):
      return bool(value)
   if isinstance(value, int):
      return int(value)
   if isinstance(value, list):
      return list(value)

   print(f'value {value} could not be evaluated')
   return value

def apply_param(config: dict, param_path: str, value: Any) -> None:
   parts = param_path.split('.')

   index_match = re.match(r'(\w+)\[(\d+)\]', parts[0])

   if index_match:
      array_name = index_match.group(1)
      index = int(index_match.group(2))
      if array_name in config and isinstance(config[array_name], list):
         target = config[array_name][index]
         if len(parts) > 1:
            set_nested(target, parts[1:], value)
         else:
            config[array_name][index] = value

   elif parts[0] in config:
      if len(parts) == 1:
         config[parts[0]] = value
      elif isinstance(config[parts[0]], list):
         apply_to_all_items(config[parts[0]], parts[1], value)
      elif isinstance(config[parts[0]], dict):
         set_nested(config, parts, value)

   else:
      for section_name, section in config.items():
         if isinstance(section, list) and section:
            if isinstance(section[0], dict) and param_path in section[0]:
               apply_to_all_items(section, param_path, value)
               return

      raise KeyError(f"Parameter {param_path} not found")


def override_config(config: dict, params: dict[str, Any]) -> dict:
   config = json.loads(json.dumps(config))  # Deep copy

   for param in params:
      apply_param(config, param, typed_value(params[param]))

   print(json.dumps(config, indent=2))
   return config

def load_parametrized_json(path: str | Path, params: dict[str, str] | None = None) -> dict[str, Any]:
   with open(path, 'r') as f:
      config = json.load(f)

   # override params:
   if params:
      return override_config(config, params)

   return config

if __name__ == "__main__":
   config = {"dt": 0.1, "room": {"min": [-2.5, -2.5, -2.5], "max": [2.5, 2.5, 2.5]},
         "controller": {"type": "mpc_agent", "params": {"horizon": 4, "q_pos": [3.0, 3.0, 3.0]}},
         "drones": [{"drone_id": "drone-1", "safety_zone": 1.0, "radius": 0.2}, {"drone_id": "drone-2", "safety_zone": 1.0, "radius": 0.2}]}

   # example in live_view.py: --param "safety_zone"=1.5 --param "drones.radius"=0.3 --param "controller.params.horizon"=8 --param "drones[0].drone_id"=leader --param "dt"=0.05
   params = {'safety_zone': '1.5', 'drones.radius': '0.3', 'controller.params.horizon': '8', 'drones[0].drone_id': 'leader', 'dt': '0.05'}

   new_config = override_config(config, params)
   print(json.dumps(new_config, indent=2))
