"""CPU-time bookkeeping per simulation phase.

The simulator, the coordinators and the MPC solvers share one ``CpuCostLedger`` and wrap their work in named phases. Each phase
books the CPU seconds (``time.process_time``, user plus system time of the whole process) that were spent in it *exclusively*:
time spent in a nested phase is booked to the nested phase only, so the sum over all phases never counts a second twice. That is
what makes ``solver`` the SLSQP core alone, although the ``cost`` and ``constraints`` callbacks run inside the ``minimize`` call.

Phases used in the simulation:

- ``setup``: per-step problem setup (initial guess, warm start, neighbour graph, ADMM init, bounds, ...)
- ``cost`` / ``constraints``: objective and constraint evaluations requested by the optimizer (and by the checks around it)
- ``solver``: the optimizer itself, without the evaluations it requests
- ``validate``: post-solve checks, constraint breakdown logging and infeasibility fallbacks
- ``exchange`` / ``admm_update``: DMPC trajectory exchange via the mailbox, and ADMM consensus, stagnation and convergence updates
- ``physics``: everything in ``Simulator.step`` outside the coordinator (fallback controls, integration, wall clamping, traces, collisions)
- ``metrics`` / ``deadlock``: the scenario runner's distance metrics, target check and deadlock check

The ledger is single-threaded by design: both coordinators used in the campaigns run in the calling thread.
"""

from __future__ import annotations

from time import process_time


class _Phase:
   """Reusable context manager for one phase name; the ledger hands out one per name."""

   __slots__ = ("_ledger", "name")

   def __init__(self, ledger: CpuCostLedger, name: str) -> None:
      self._ledger = ledger
      self.name = name

   def __enter__(self) -> None:
      self._ledger._enter(self.name)

   def __exit__(self, *exc: object) -> bool:
      self._ledger._exit()
      return False


class CpuCostLedger:
   """Exclusive CPU seconds and entry counts per phase."""

   def __init__(self) -> None:
      self.seconds: dict[str, float] = {}
      self.counts: dict[str, int] = {}
      self._stack: list[list] = []  # frames [name, t0, seconds booked to nested phases]
      self._phases: dict[str, _Phase] = {}

   def phase(self, name: str) -> _Phase:
      """Context manager that books the CPU time of its block to ``name``."""
      phase = self._phases.get(name)
      if phase is None:
         phase = self._phases[name] = _Phase(self, name)
      return phase

   def count(self, name: str, n: int = 1) -> None:
      """Add to a plain counter (for example ADMM iterations) without booking time."""
      self.counts[name] = self.counts.get(name, 0) + n

   def total(self) -> float:
      """CPU seconds booked across all phases."""
      return sum(self.seconds.values())

   def _enter(self, name: str) -> None:
      self._stack.append([name, process_time(), 0.0])

   def _exit(self) -> None:
      name, t0, nested = self._stack.pop()
      elapsed = process_time() - t0
      self.seconds[name] = self.seconds.get(name, 0.0) + elapsed - nested
      self.counts[name] = self.counts.get(name, 0) + 1
      if self._stack:
         self._stack[-1][2] += elapsed
