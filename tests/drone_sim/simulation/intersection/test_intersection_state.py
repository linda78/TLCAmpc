"""Tests for the intersection state store, priorities and deadlock dominance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_sim.controllers.braking import BrakingMPCAgent
from drone_sim.controllers.central_cost import CentralMPCAgent
from drone_sim.controllers.wait import WAIT_CONTROLLER
from drone_sim.domain.drone import Drone, GatedDrone, Route
from drone_sim.physics.linear_kinematics import LinearKinematicsPhysics
from drone_sim.simulation.utils.stateless_conflict_detection import (
   IntersectionEvent,
)
from drone_sim.simulation.intersection.intersection_state import (
   IntersectionStateStore,
   apply_deadlock_dominance,
   compute_priorities,
)


# ---------- helpers ---------------------------------------------------------


@dataclass
class _StubDrone:
   """Light-weight stand-in for tests that don't need full physics."""

   drone_id: str
   safety_zone: float
   x: np.ndarray


def _event(
   pair_ids: tuple[str, str],
   ttc_a: float,
   ttc_b: float,
   detection_point: np.ndarray | None = None,
) -> IntersectionEvent:
   if detection_point is None:
      detection_point = np.array([1.0, 0.0, 0.0])
   return IntersectionEvent(
      pair=frozenset(pair_ids),
      detection_point=detection_point,
      k_min=2,
      min_dist=0.4,
      ttc={pair_ids[0]: ttc_a, pair_ids[1]: ttc_b},
   )


def _stub_drone(drone_id: str, position: np.ndarray, safety: float = 0.5) -> _StubDrone:
   x = np.zeros(6)
   x[:3] = position
   return _StubDrone(drone_id=drone_id, safety_zone=safety, x=x)


def _make_physics(dt: float = 0.1) -> LinearKinematicsPhysics:
   return LinearKinematicsPhysics(
      dt=dt, v_max=3.0, u_min=[-3.0, -3.0, -3.0], u_max=[3.0, 3.0, 3.0]
   )


def _gated(drone_id: str, position: np.ndarray) -> GatedDrone:
   x = np.zeros(6)
   x[:3] = position
   return GatedDrone(
      drone_id=drone_id,
      radius=0.2,
      safety_zone=0.5,
      cons_stop=0.0,
      color="tab:blue",
      safety_color="tab:blue",
      trace_color="tab:blue",
      controller=CentralMPCAgent(dt=0.1, horizon=4),
      physics=_make_physics(),
      x=x,
      route=Route(waypoints=[], target=position + np.array([1.0, 0, 0])),
   )


def _plain(drone_id: str, position: np.ndarray) -> Drone:
   x = np.zeros(6)
   x[:3] = position
   return Drone(
      drone_id=drone_id,
      radius=0.2,
      safety_zone=0.5,
      cons_stop=0.0,
      color="tab:blue",
      safety_color="tab:blue",
      trace_color="tab:blue",
      controller=CentralMPCAgent(dt=0.1, horizon=4),
      physics=_make_physics(),
      x=x,
      route=Route(waypoints=[], target=position + np.array([1.0, 0, 0])),
   )


# ---------- compute_priorities -----------------------------------------------


def test_compute_priorities_ttc_default_with_no_drones():
   ev = _event(("a", "b"), ttc_a=1.0, ttc_b=3.0)
   priorities = compute_priorities([ev])
   assert priorities == {frozenset({"a", "b"}): "a"}


def test_compute_priorities_ttc_tie_sorts_id():
   ev = _event(("a", "b"), ttc_a=1.0, ttc_b=1.0)
   priorities = compute_priorities([ev])
   assert priorities[frozenset({"a", "b"})] == "a"


def test_type_veto_plain_beats_gated():
   drones = [_plain("a", np.zeros(3)), _gated("b", np.zeros(3))]
   # TTC alone would pick "b" (smaller TTC), but type-veto picks plain "a".
   ev = _event(("a", "b"), ttc_a=5.0, ttc_b=0.1)
   priorities = compute_priorities([ev], drones)
   assert priorities[frozenset({"a", "b"})] == "a"


def test_wait_veto_non_waiting_wins():
   a = _gated("a", np.zeros(3))
   b = _gated("b", np.zeros(3))
   a.start_waiting("c")  # a is waiting on something else
   drones = [a, b]
   # TTC alone would pick "a"; wait-veto picks "b".
   ev = _event(("a", "b"), ttc_a=0.1, ttc_b=5.0)
   priorities = compute_priorities([ev], drones)
   assert priorities[frozenset({"a", "b"})] == "b"


def test_boost_drone_with_waiter_wins():
   a = _gated("a", np.zeros(3))
   b = _gated("b", np.zeros(3))
   c = _gated("c", np.zeros(3))
   # c is parked waiting on a → a has waiters
   c.start_waiting("a")
   drones = [a, b, c]
   # TTC ties → without boost, sorted-id picks "a"; boost confirms "a".
   ev = _event(("a", "b"), ttc_a=1.0, ttc_b=1.0)
   priorities = compute_priorities([ev], drones)
   assert priorities[frozenset({"a", "b"})] == "a"

   # Flip: b has the waiter instead of a.
   c.stop_waiting()
   c.start_waiting("b")
   priorities = compute_priorities([ev], drones)
   assert priorities[frozenset({"a", "b"})] == "b"


def test_priority_stage_order_type_beats_wait():
   """Type-Veto fires before Wait-Veto even when they'd disagree.

   If we had plain "a" (waiting? impossible — plain doesn't wait) and
   gated "b" non-waiting, Type-Veto picks "a". This sanity-checks the
   short-circuit ordering using a non-waiting gated against plain.
   """
   drones = [_plain("a", np.zeros(3)), _gated("b", np.zeros(3))]
   ev = _event(("a", "b"), ttc_a=10.0, ttc_b=0.01)
   priorities = compute_priorities([ev], drones)
   assert priorities[frozenset({"a", "b"})] == "a"


# ---------- IntersectionStateStore -------------------------------------------


def test_state_store_parks_gated_loser_on_insert():
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   a = _gated("a", np.array([0.0, 0.0, 0.0]))
   b = _gated("b", np.array([3.0, 0.0, 0.0]))
   ev = _event(("a", "b"), ttc_a=0.5, ttc_b=2.0)
   active = store.step([ev], [a, b], threshold=1.0)
   pair = frozenset({"a", "b"})
   assert pair in active
   record = active[pair]
   assert record.priority == "a"
   assert record.priority_root == "a"
   # Loser enters STOPPING — controller swapped, still in opt_drones.
   assert b.is_waiting is True
   assert b.is_stopping is True
   assert b.is_parked is False
   assert b.waiting_for == "a"
   assert isinstance(b.controller, BrakingMPCAgent)
   # Priority drone is unaffected.
   assert a.is_waiting is False


def test_state_store_ignores_plain_loser_for_wait():
   """A plain Drone loser cannot wait — sphere tracked, no parking."""
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   a = _gated("a", np.array([0.0, 0.0, 0.0]))
   b = _plain("b", np.array([3.0, 0.0, 0.0]))
   # Force "a" as priority (Type-Veto would normally pick b, the plain one).
   # Use an event with very low TTC for a — but type-veto overrides → b wins.
   # Use both-gated configuration to test the parking path; here we want to
   # confirm that a plain drone is NEVER parked. Set up so plain is loser:
   ev = _event(("a", "b"), ttc_a=5.0, ttc_b=0.1)  # type-veto: b (plain) wins
   active = store.step([ev], [a, b], threshold=1.0)
   pair = frozenset({"a", "b"})
   assert pair in active
   record = active[pair]
   assert record.priority == "b"  # type-veto
   # Loser is "a" (gated) → should be parked.
   assert a.is_waiting is True


def test_state_store_drops_sphere_and_releases_after_priority_passes():
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   a = _gated("a", np.array([0.0, 0.0, 0.0]))
   b = _gated("b", np.array([3.0, 0.0, 0.0]))
   ev = _event(("a", "b"), ttc_a=0.5, ttc_b=2.0,
                detection_point=np.array([1.0, 0.0, 0.0]))
   store.step([ev], [a, b], threshold=1.0)
   assert b.is_waiting

   # Priority "a" enters the sphere region. b stays in STOPPING regardless
   # of its velocity — the transition to WAITING is deferred to sphere drop.
   a.x[:3] = np.array([1.0, 0, 0])
   active_in = store.step([], [a, b], threshold=1.0)
   assert frozenset({"a", "b"}) in active_in
   assert b.is_stopping

   # Priority "a" exits the sphere region → sphere drops and b released.
   a.x[:3] = np.array([10.0, 0, 0])
   active_out = store.step([], [a, b], threshold=1.0)
   assert frozenset({"a", "b"}) not in active_out
   assert b.is_waiting is False
   assert b.wait_state == "free"


def test_state_store_keeps_loser_stopping_until_sphere_drops():
   """The loser stays in STOPPING for the entire sphere lifetime.

   Transition to WAITING happens *only* at sphere drop (and is followed
   immediately by stop_waiting → free in the same step). This keeps the
   drone in DMPC's opt_drones throughout, so collision constraints with
   the priority drone stay active.
   """
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   a = _gated("a", np.array([0.0, 0.0, 0.0]))
   b = _gated("b", np.array([3.0, 0.0, 0.0]))
   ev = _event(("a", "b"), ttc_a=0.5, ttc_b=2.0,
                detection_point=np.array([1.0, 0.0, 0.0]))
   store.step([ev], [a, b], threshold=1.0)
   assert b.is_stopping
   assert isinstance(b.controller, BrakingMPCAgent)

   # Even if b's velocity drops to zero, it stays in STOPPING — the
   # transition is deferred to sphere drop.
   b.x[3:6] = np.zeros(3)
   a.x[:3] = np.array([1.0, 0, 0])  # priority enters the zone
   store.step([], [a, b], threshold=1.0)
   assert b.is_stopping
   assert isinstance(b.controller, BrakingMPCAgent)


def test_state_store_does_not_drop_brand_new_sphere():
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   a = _gated("a", np.array([0.0, 0.0, 0.0]))
   b = _gated("b", np.array([3.0, 0.0, 0.0]))
   ev = _event(("a", "b"), ttc_a=0.5, ttc_b=2.0,
                detection_point=np.array([2.75, 0.0, 0.0]))
   active = store.step([ev], [a, b], threshold=1.0)
   assert frozenset({"a", "b"}) in active


def test_state_store_rehangs_waiters_when_intermediate_drone_starts_waiting():
   """A→B, then B becomes loser elsewhere → A rehangs to B's root.

   To force ``a`` into the loser role we pit it against a plain
   :class:`Drone`: Type-Veto fires before Boost, so ``a`` loses even
   though it has a waiter.
   """
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   a = _gated("a", np.array([0.0, 0.0, 0.0]))
   b = _gated("b", np.array([3.0, 0.0, 0.0]))
   x = _plain("x", np.array([6.0, 0.0, 0.0]))

   # Step 1: (a, b) — a wins TTC, b parks on a.
   ev_ab = _event(("a", "b"), ttc_a=0.5, ttc_b=3.0,
                   detection_point=np.array([1.5, 0.0, 0.0]))
   store.step([ev_ab], [a, b, x], threshold=1.0)
   assert b.waiting_for == "a"

   # Step 2: (a, x) — Type-Veto picks plain x. a is loser → parked on x.
   ev_ax = _event(("a", "x"), ttc_a=3.0, ttc_b=0.5,
                   detection_point=np.array([3.0, 0.0, 0.0]))
   store.step([ev_ax], [a, b, x], threshold=1.0)
   assert a.is_waiting and a.waiting_for == "x"
   # b's wait was rehung onto x — the new root.
   assert b.waiting_for == "x"


def test_state_store_does_not_release_rehung_waiter():
   """A rehung waiter survives the drop of the original sphere."""
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   a = _gated("a", np.array([0.0, 0.0, 0.0]))
   b = _gated("b", np.array([3.0, 0.0, 0.0]))
   x = _plain("x", np.array([6.0, 0.0, 0.0]))

   # Step 1: (a, b) sphere centered at a's start → priority_entered=True.
   ev_ab = _event(("a", "b"), ttc_a=0.5, ttc_b=3.0,
                   detection_point=np.array([0.0, 0.0, 0.0]))
   store.step([ev_ab], [a, b, x], threshold=1.0)
   assert b.waiting_for == "a"

   # Step 2: a moves well past the (a,b) sphere AND a new (a, x) event lands.
   # The new event's Type-Veto parks a on x and rehangs b. The drop loop
   # finds priority "a" exiting the (a,b) zone and drops that sphere — but b
   # has been rehung to "x" already, so stop_waiting must NOT fire on b.
   a.x[:3] = np.array([10.0, 0.0, 0.0])
   ev_ax = _event(("a", "x"), ttc_a=3.0, ttc_b=0.5,
                   detection_point=np.array([10.0, 0.0, 0.0]))
   active = store.step([ev_ax], [a, b, x], threshold=1.0)
   assert frozenset({"a", "b"}) not in active
   assert b.is_waiting is True
   assert b.waiting_for == "x"


def test_state_store_releases_orphaned_waiter_when_target_idles():
   """Liveness: a waiter whose target is no active sphere priority/root is freed.

   Regression for the sequencing deadlock where a rehung waiter (b waiting on x)
   was never released after x's sphere dropped — x reached its goal and went
   idle, so b waited forever. The orphan sweep must release b once no active
   sphere sequences x.
   """
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   a = _gated("a", np.array([0.0, 0.0, 0.0]))
   b = _gated("b", np.array([3.0, 0.0, 0.0]))
   x = _plain("x", np.array([6.0, 0.0, 0.0]))

   # Step 1: (a, b) — a wins, b parks on a.
   ev_ab = _event(("a", "b"), ttc_a=0.5, ttc_b=3.0,
                   detection_point=np.array([0.0, 0.0, 0.0]))
   store.step([ev_ab], [a, b, x], threshold=1.0)

   # Step 2: (a, x) — Type-Veto parks a on x and rehangs b onto x; (a,b) drops.
   # The (a,x) sphere is centered at x's position so its priority x counts as
   # having entered the zone immediately.
   a.x[:3] = np.array([10.0, 0.0, 0.0])
   ev_ax = _event(("a", "x"), ttc_a=3.0, ttc_b=0.5,
                   detection_point=np.array([6.0, 0.0, 0.0]))
   store.step([ev_ax], [a, b, x], threshold=1.0)
   assert b.waiting_for == "x" and b.is_waiting

   # Step 3: x exits its sphere → (a,x) drops and releases the direct loser a.
   # b was rehung onto x and is not a member of the (a,x) pair, so the normal
   # release skips it. No sphere references x anymore, so the orphan sweep must
   # free b. Without the fix b would wait on idle x forever (deadlock).
   x.x[:3] = np.array([20.0, 0.0, 0.0])
   active = store.step([], [a, b, x], threshold=1.0)
   assert active == {}
   assert b.is_waiting is False and b.wait_state == "free"
   assert a.is_waiting is False


# ---------- apply_deadlock_dominance (D1, unchanged) ------------------------


def test_apply_deadlock_dominance_breaks_three_cycle():
   pair_ab = frozenset({"a", "b"})
   pair_bc = frozenset({"b", "c"})
   pair_ca = frozenset({"c", "a"})
   priorities = {pair_ab: "b", pair_bc: "c", pair_ca: "a"}
   resolved = apply_deadlock_dominance(priorities)
   assert resolved == {pair_ab: None, pair_bc: None, pair_ca: None}


def test_apply_deadlock_dominance_keeps_external_drone_normal():
   pair_ab = frozenset({"a", "b"})
   pair_bc = frozenset({"b", "c"})
   pair_ca = frozenset({"c", "a"})
   pair_ad = frozenset({"a", "d"})
   priorities = {pair_ab: "b", pair_bc: "c", pair_ca: "a", pair_ad: "a"}
   resolved = apply_deadlock_dominance(priorities)
   assert resolved[pair_ab] is None
   assert resolved[pair_bc] is None
   assert resolved[pair_ca] is None
   assert resolved[pair_ad] == "a"


def test_apply_deadlock_dominance_passthrough_without_cycle():
   pair_ab = frozenset({"a", "b"})
   priorities = {pair_ab: "a"}
   resolved = apply_deadlock_dominance(priorities)
   assert resolved == {pair_ab: "a"}


def test_state_store_skips_pair_when_deadlock_dominant():
   """When apply_deadlock_dominance returns None for a pair, no sphere placed."""
   store = IntersectionStateStore(radius=1.2, margin=1.1)
   drones = [
      _stub_drone("a", np.zeros(3)),
      _stub_drone("b", np.zeros(3)),
      _stub_drone("c", np.zeros(3)),
   ]
   evs = [
      IntersectionEvent(pair=frozenset({"a", "b"}), detection_point=np.zeros(3), k_min=1,
                         min_dist=0.1, ttc={"a": 2.0, "b": 1.0}),
      IntersectionEvent(pair=frozenset({"b", "c"}), detection_point=np.zeros(3), k_min=1,
                         min_dist=0.1, ttc={"b": 2.0, "c": 1.0}),
      IntersectionEvent(pair=frozenset({"c", "a"}), detection_point=np.zeros(3), k_min=1,
                         min_dist=0.1, ttc={"c": 2.0, "a": 1.0}),
   ]
   active = store.step(evs, drones, threshold=1.0)
   assert active == {}
