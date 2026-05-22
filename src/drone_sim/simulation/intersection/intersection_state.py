"""Per-pair sphere state + wait-state lifecycle for the intersection layer.

Each active sphere has one priority drone (allowed to pass through) and
parks every other drone in the pair via :meth:`GatedDrone.start_waiting`.
:class:`SphereRecord` remembers the wait-chain root so a transitively
rehung waiter is not released by the wrong sphere drop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from drone_sim.domain.drone import Drone, GatedDrone, has_waiters
from drone_sim.domain.sphere_obstacle import SphereObstacle
from drone_sim.simulation.utils.stateless_conflict_detection import (IntersectionEvent)

_log = logging.getLogger(__name__)


@dataclass
class SphereRecord:
   """One active intersection sphere for a conflict pair.

   :param pair: Frozenset of the two drone IDs.
   :param sphere: :class:`SphereObstacle` placed at the detection point.
   :param priority: drone_id allowed to pass. ``None`` for deadlocked pairs.
   :param priority_root: wait-chain root at insert time. Drop releases
          only losers whose ``waiting_for`` still equals this root.
   :param priority_entered: ``True`` once the priority drone is inside
          the sphere. Drop fires only after entry + exit.
   :param created_step: step the sphere was inserted.
   :param last_seen_step: most recent step the conflict was predicted.
   """

   pair: frozenset[str]
   sphere: SphereObstacle
   priority: str | None
   priority_root: str | None
   created_step: int
   last_seen_step: int
   priority_entered: bool = False


@dataclass
class IntersectionStateStore:
   """Persistent store of active intersection spheres across simulation steps.

   :param radius: Sphere radius applied when a new conflict is recorded.
   :param margin: Anti-flapping factor for the dissolution check: the
          priority drone must be at least ``radius * margin`` away from the
          sphere center before the sphere is dropped.
   """

   radius: float
   margin: float = 1.1
   _active: dict[frozenset[str], SphereRecord] = field(default_factory=dict)
   _step: int = 0

   def active_pairs(self) -> dict[frozenset[str], SphereRecord]:
      """Read-only view of the currently active spheres."""
      return dict(self._active)

   def step(self, events: Sequence[IntersectionEvent], drones: Sequence[Drone], threshold: float) -> dict[frozenset[str], SphereRecord]:
      """Advance one simulation step.

      Inserts/refreshes spheres for active conflicts (parks gated losers,
      rehangs transitive waiters), drops spheres whose priority has
      passed through the zone (releases eligible waiters).

      :param events: Conflicts detected this step.
      :param drones: Drones in the scenario.
      :param threshold: Detection threshold (kept for API symmetry).
      :return: Snapshot of the active sphere map after the update.
      """
      self._step += 1
      events_by_pair = {e.pair: e for e in events}
      drones_by_id = {d.drone_id: d for d in drones}
      pos_by_id = {d.drone_id: np.asarray(d.x, dtype=float)[:3] for d in drones}

      # Insert / refresh spheres for active conflicts.
      ttc_priorities = compute_priorities(events, drones)
      dominance = apply_deadlock_dominance(ttc_priorities)
      if events_by_pair:
         _log.debug("IntersectionStateStore.step #%d: %d events, ttc_priorities=%s dominance=%s", self._step, len(events_by_pair),
                    {tuple(sorted(p)): w for p, w in ttc_priorities.items()}, {tuple(sorted(p)): w for p, w in dominance.items()})
      for pair, event in events_by_pair.items():
         priority = dominance.get(pair, None)
         # Deadlock-dominance dissolved this pair — no sphere, no wait.
         if priority is None:
            existing = self._active.pop(pair, None)
            if existing is not None and existing.priority_root is not None:
               _log.info("IntersectionStateStore: DEADLOCK dissolve pair=%s, releasing losers", sorted(pair))
               self._release_losers(existing, drones, drones_by_id)
            continue
         losers = pair - {priority}
         existing = self._active.get(pair)
         if existing is None:
            priority_root = self._resolve_target(priority, drones_by_id)
            _log.info("IntersectionStateStore: INSERT sphere pair=%s priority=%s priority_root=%s losers=%s "
                      "detection_point=%s", sorted(pair), priority, priority_root, sorted(losers), np.round(event.detection_point, 3).tolist())
            for loser_id in losers:
               self._park_loser(loser_id, priority_root, drones, drones_by_id)
            sphere = SphereObstacle(center=event.detection_point.copy(), radius=self.radius,
                                    meta={"source": "intersection", "pair": tuple(sorted(pair)), "priority": priority, "priority_root": priority_root,
                                          "created_step": self._step, })
            self._active[pair] = SphereRecord(pair=pair, sphere=sphere, priority=priority, priority_root=priority_root, created_step=self._step,
                                              last_seen_step=self._step)
         else:
            if existing.priority != priority:
               _log.info("IntersectionStateStore: REFRESH pair=%s priority %s -> %s", sorted(pair), existing.priority, priority)
            existing.priority = priority
            existing.sphere.meta["priority"] = priority
            existing.last_seen_step = self._step

      # Drop spheres whose priority drone has entered then exited the zone.
      for pair in list(self._active.keys()):
         record = self._active[pair]
         priority = record.priority
         if priority is None:
            del self._active[pair]
            continue
         pos = pos_by_id.get(priority)
         if pos is None:
            _log.info("IntersectionStateStore: DROP pair=%s (priority %s no longer present)", sorted(pair), priority)
            self._release_losers(record, drones, drones_by_id)
            del self._active[pair]
            continue
         dist = float(np.linalg.norm(pos - record.sphere.center))
         if dist <= record.sphere.radius and not record.priority_entered:
            _log.info("IntersectionStateStore: priority %s ENTERED sphere pair=%s (dist=%.3f <= r=%.3f)", priority, sorted(pair), dist, record.sphere.radius)
            record.priority_entered = True
         elif dist <= record.sphere.radius:
            record.priority_entered = True
         if record.priority_entered and dist > record.sphere.radius * self.margin:
            _log.info("IntersectionStateStore: DROP pair=%s priority=%s exited (dist=%.3f > r*margin=%.3f)", sorted(pair), priority, dist,
                      record.sphere.radius * self.margin)
            self._release_losers(record, drones, drones_by_id)
            del self._active[pair]

      return self.active_pairs()

   def _park_loser(self, loser_id: str, priority_root: str, drones: Sequence[Drone], drones_by_id: dict[str, Drone]) -> None:
      """Park ``loser_id`` behind ``priority_root`` and rehang its waiters.

      No-op for a plain :class:`Drone`. For a fresh :class:`GatedDrone`,
      anyone waiting on this loser is rehung onto ``priority_root`` so
      the chain stays flat. Idempotent if the loser is already waiting.
      """
      loser = drones_by_id.get(loser_id)
      if not isinstance(loser, GatedDrone):
         _log.debug("IntersectionStateStore._park_loser: %s is not GatedDrone — skipping", loser_id)
         return
      if not loser.is_waiting:
         rehung: list[str] = []
         for other in drones:
            if isinstance(other, GatedDrone) and other.waiting_for == loser_id:
               other.waiting_for = priority_root
               rehung.append(other.drone_id)
         if rehung:
            _log.info("IntersectionStateStore: REHANG waiters of %s onto %s: %s", loser_id, priority_root, rehung)
      loser.start_waiting(priority_root)

   def _release_losers(self, record: SphereRecord, drones: Sequence[Drone], drones_by_id: dict[str, Drone]) -> None:
      """Release losers when the sphere drops.

      Stopping losers transition through ``waiting`` to ``free`` in one
      step. Losers whose ``waiting_for`` has been rehung onto a
      different root are left alone — the new sphere's drop will
      release them.
      """
      priority = record.priority
      if priority is None:
         return
      for loser_id in record.pair - {priority}:
         loser = drones_by_id.get(loser_id)
         if (isinstance(loser, GatedDrone) and loser.is_waiting and loser.waiting_for == record.priority_root):
            if loser.is_stopping:
               loser.transition_to_parked()
            loser.stop_waiting()

   def _resolve_target(self, target_id: str, drones_by_id: dict[str, Drone]) -> str:
      """Walk the wait-chain from ``target_id`` to its non-waiting root.

      Cycle guard breaks if a drone is revisited — defensive only.
      """
      target = drones_by_id.get(target_id)
      seen: set[str] = {target_id}
      while isinstance(target, GatedDrone) and target.is_waiting:
         next_id = target.waiting_for
         if next_id is None or next_id in seen:
            break
         seen.add(next_id)
         next_drone = drones_by_id.get(next_id)
         if next_drone is None:
            break
         target = next_drone
      if target is None:
         return target_id
      return target.drone_id


def compute_priorities(events: Iterable[IntersectionEvent], drones: Sequence[Drone] | None = None) -> dict[frozenset[str], str]:
   """Per-pair priority drone after veto stages over TTC.

   Order: ``Type-Veto`` → ``Wait-Veto`` → ``Boost`` → ``TTC``. First
   stage that produces an unambiguous winner wins.

   - **Type-Veto**: plain :class:`Drone` beats :class:`GatedDrone`.
   - **Wait-Veto**: non-waiting beats waiting.
   - **Boost**: drone with waiters beats one without.
   - **TTC**: smallest TTC (sorted-id tie-break).

   :param events: Detected intersection events.
   :param drones: Optional. When omitted, only the TTC stage runs.
   """
   drone_list = list(drones) if drones is not None else []
   drones_by_id = {d.drone_id: d for d in drone_list}
   priorities: dict[frozenset[str], str] = {}
   for event in events:
      pair_ids = sorted(event.pair)
      a_id, b_id = pair_ids
      a = drones_by_id.get(a_id)
      b = drones_by_id.get(b_id)

      a_gated = isinstance(a, GatedDrone)
      b_gated = isinstance(b, GatedDrone)

      # 1. Type-Veto
      if a_gated != b_gated:
         priorities[event.pair] = b_id if a_gated else a_id
         continue

      # 2. Wait-Veto
      a_wait = a_gated and a.is_waiting  # type: ignore[union-attr]
      b_wait = b_gated and b.is_waiting  # type: ignore[union-attr]
      if a_wait and not b_wait:
         priorities[event.pair] = b_id
         continue
      if b_wait and not a_wait:
         priorities[event.pair] = a_id
         continue

      # 3. Boost
      a_has = has_waiters(drone_list, a_id)
      b_has = has_waiters(drone_list, b_id)
      if a_has and not b_has:
         priorities[event.pair] = a_id
         continue
      if b_has and not a_has:
         priorities[event.pair] = b_id
         continue

      # 4. TTC default — smallest TTC wins, sorted-id tie-break.
      ttc_a = event.ttc.get(a_id, float("inf"))
      ttc_b = event.ttc.get(b_id, float("inf"))
      priorities[event.pair] = a_id if ttc_a <= ttc_b else b_id
   return priorities


def apply_deadlock_dominance(priorities: dict[frozenset[str], str]) -> dict[frozenset[str], str | None]:
   """Propagate dominance across "waits-for" cycles.

   Each pair contributes an edge ``loser -> winner``. Drones inside an
   SCC of size ≥ 2 are flagged dominant in *all* their pairs. A pair
   with two dominant drones gets priority ``None`` — no sphere is placed.
   """
   if not priorities:
      return {}

   adj: dict[str, set[str]] = {}
   for pair, winner in priorities.items():
      a, b = tuple(pair)
      loser = b if winner == a else a
      adj.setdefault(loser, set()).add(winner)
      adj.setdefault(winner, set())

   sccs = _tarjan_scc(adj)
   dominant: set[str] = set()
   for component in sccs:
      if len(component) >= 2:
         dominant.update(component)

   resolved: dict[frozenset[str], str | None] = {}
   for pair, winner in priorities.items():
      a, b = tuple(pair)
      a_dom = a in dominant
      b_dom = b in dominant
      if a_dom and b_dom:
         resolved[pair] = None
      elif a_dom:
         resolved[pair] = a
      elif b_dom:
         resolved[pair] = b
      else:
         resolved[pair] = winner
   return resolved


def _tarjan_scc(adj: dict[str, set[str]]) -> list[list[str]]:
   """Tarjan's strongly connected components on a string-keyed digraph."""
   index_counter = [0]
   stack: list[str] = []
   on_stack: set[str] = set()
   indices: dict[str, int] = {}
   lowlink: dict[str, int] = {}
   result: list[list[str]] = []

   def strongconnect(node: str) -> None:
      indices[node] = index_counter[0]
      lowlink[node] = index_counter[0]
      index_counter[0] += 1
      stack.append(node)
      on_stack.add(node)
      for neighbor in adj.get(node, ()):
         if neighbor not in indices:
            strongconnect(neighbor)
            lowlink[node] = min(lowlink[node], lowlink[neighbor])
         elif neighbor in on_stack:
            lowlink[node] = min(lowlink[node], indices[neighbor])
      if lowlink[node] == indices[node]:
         component: list[str] = []
         while True:
            w = stack.pop()
            on_stack.discard(w)
            component.append(w)
            if w == node:
               break
         result.append(component)

   for node in list(adj.keys()):
      if node not in indices:
         strongconnect(node)
   return result


__all__ = ("IntersectionStateStore", "SphereRecord", "apply_deadlock_dominance", "compute_priorities",)
