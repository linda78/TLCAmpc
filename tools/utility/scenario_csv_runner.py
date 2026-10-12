from enum import StrEnum
from pathlib import Path
import filelock
import csv
import os

class QueueStatus(StrEnum):
   TO_BE_DONE = "toBeDone"
   IN_PROCESS = "inProcess"
   DONE = "done"
   ERROR = "error"

_SCENARIO_QUEUE_FIELDNAMES = ['id', 'status', 'run', 'n_drones', 'coordinator', 'controller']

_COORDINATORS = ['mpc_central', 'dmpc_admm', 'dmpc_threaded']
_CONTROLLERS = ['mpc_agent', 'mpc_agent_adaptive']

def create_queue(queue_path: Path, runs: range, drones: range, coordinators: list[str]|None = None, controllers: list[str]|None = None):
   """
   create queue csv. If its already exists, clean it, while set all 'in progress' states to 'to be done'
   :param queue_path: path to the queue_csv
   :param runs:
   :param drones:
   :param coordinators:
   :param controllers:
   """
   queue_path.parent.mkdir(parents=True, exist_ok=True)
   if os.path.exists(queue_path):
      changed = 0
      to_be_done = 0
      rows = []
      with queue_path.open('r', newline='', encoding='utf-8') as f:
         reader = csv.DictReader(f)
         rows = list(reader)
         for row in rows:
            if row['status'] == QueueStatus.IN_PROCESS:
               row['status'] = QueueStatus.TO_BE_DONE
               changed += 1
            if row['status'] == QueueStatus.TO_BE_DONE:
               to_be_done += 1
      with queue_path.open('w', newline='', encoding='utf-8') as f:
         writer = csv.DictWriter(f, fieldnames=_SCENARIO_QUEUE_FIELDNAMES)
         writer.writeheader()
         writer.writerows(rows)

      if changed > 0:
         print(f'Fixed already existing scenario CSV with {len(rows)} entries at {queue_path}, repaired {changed} lines')
      else:
         print(f'Found already existing scenario CSV with {len(rows)} entries at {queue_path}, nothing done')
      print(f"still to be done: {to_be_done}")

   else:
      rows = []
      index = 0
      coordinators = coordinators if coordinators is not None else _COORDINATORS
      controllers = controllers if controllers is not None else _CONTROLLERS
      for run in runs:
         for n in drones:
            for coord in coordinators:
               for controller in controllers:
                  rows.append({'id': index, 'status': QueueStatus.TO_BE_DONE, 'run': run, 'n_drones': n, 'coordinator': coord, 'controller': controller})
                  index += 1

      with queue_path.open('w', newline='', encoding='utf-8') as f:
         writer = csv.DictWriter(f, fieldnames=_SCENARIO_QUEUE_FIELDNAMES)
         writer.writeheader()
         writer.writerows(rows)

      print(f'Created scenario CSV with {len(rows)} entries at {queue_path}')

def claim_row(queue_path: Path, lock_path: Path):
   """
   :param queue_path: path to the queue_csv
   :param lock_path: path to the lock file
   :return: claimed row
   """
   with filelock.FileLock(lock_path):
      with queue_path.open('r', newline='', encoding='utf-8') as f:
         reader = csv.DictReader(f)
         rows = list(reader)

      rows.sort(key=lambda x: int(x['n_drones']))

      claimed = None
      for row in rows:
         if row['status'] == QueueStatus.TO_BE_DONE:
            row['status'] = QueueStatus.IN_PROCESS
            claimed = row.copy()
            break

      if claimed is None:
         return None

      with queue_path.open('w', newline='', encoding='utf-8') as f:
         writer = csv.DictWriter(f, fieldnames=_SCENARIO_QUEUE_FIELDNAMES)
         writer.writeheader()
         writer.writerows(rows)

      return claimed

def update_row(queue_path: Path, lock_path: Path, index: int) -> None:
   """
   :param queue_path:  path to the queue_csv
   :param lock_path: path to the lock file
   :param index: index, which has to be updated
   """
   with filelock.FileLock(lock_path):
      with queue_path.open('r', newline='', encoding='utf-8') as f:
         reader = csv.DictReader(f)
         rows = list(reader)
         for row in rows:
            if row['id'] == index:
               if row['status'] == QueueStatus.TO_BE_DONE:
                  row['status'] = QueueStatus.IN_PROCESS
                  break
               elif row['status'] == QueueStatus.IN_PROCESS:
                  row['status'] = QueueStatus.DONE
                  break
               else:
                  print(f"Warning: strange status for: {row}")
                  row['status'] = QueueStatus.ERROR
      with queue_path.open('w', newline='', encoding='utf-8') as f:
         writer = csv.DictWriter(f, fieldnames=_SCENARIO_QUEUE_FIELDNAMES)
         writer.writeheader()
         writer.writerows(rows)

def count_remaining_scenarios(queue_path: Path, lock_path: Path) -> tuple[int, int]:
   """
   :param queue_path: path to the queue_csv
   :param lock_path: path to the lock file
   :return: tuple of numbers 'to be done' and 'in process'
   """
   with filelock.FileLock(lock_path):
      if not queue_path.exists():
         return 0, 0
      with queue_path.open('r', newline='', encoding='utf-8') as f:
         reader = csv.DictReader(f)
         return sum(1 for row in reader if row['status'] in QueueStatus.TO_BE_DONE), sum(1 for row in reader if row['status'] in QueueStatus.IN_PROCESS)

