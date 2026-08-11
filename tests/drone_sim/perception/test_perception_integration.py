"""End-to-end tests for the camera perception pipeline (step 12 of the perception plan).

Every other file in this package pins one link of the chain in isolation — the visibility cone, the mailbox, the worker, the bridge. This one runs
the whole chain through a real :class:`~drone_sim.simulation.simulator.Simulator` built from a scenario config:

    step() -> CameraModel.capture -> PerceptionWorker -> StubPerceptionAdapter -> PerceptionMailbox
           -> feed_trajectory_mailbox -> Trajectory-/ThreadSafeMailbox -> LocalMPCSolver -> new positions

Three questions, and deliberately nothing else:

1. **Does the system still fly when the DMPC inboxes come from cameras instead of broadcasts?** Once per DMPC coordinator, because the two feed
   their mailboxes at completely different places (in the ADMM loop vs. before the solver threads are spawned).
2. **Does merely observing leave the control loop untouched?** A scenario with ``camera_feeds_dmpc=false`` must produce the same positions as one
   without a camera at all — bit for bit, not approximately.
3. **Does a run give its worker thread back?** Including through the GUI backend, which replaces simulations under a long-lived process.

What is *not* tested here is the image perception itself. The detector is developed in parallel and is not part of this feature; these tests run
against the linear stand-in ``StubPerceptionAdapter``, which derives the estimate from geometry instead of from pixels. The visibility half stays
real either way — FOV, range and self-exclusion come from ``CameraModel.capture``, so a neighbor outside the cone is missing from the solve here
exactly as it would be with a real camera. See the "Verifikation" section of ``pratham_plan.md``.

Determinism (risk R5): the inline scenarios use ``camera_async=False``, so the estimates of step k are in the mailbox before the solve of step k, and
``camera_noise_sigma=0``, which makes the stub the identity on the visible neighbors. That is enough for everything except the exact-equality
regression, which additionally has to pin the *simulation's own* randomness — see :func:`_run`. The two shipped example scenarios are used as they
ship, i.e. asynchronously, and are therefore polled rather than read straight after a step.
"""

from __future__ import annotations

import json
import random
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from drone_sim.domain.config import ControllerSpec, DroneConfig, PhysicsSpec, RoomConfig, ScenarioConfig
from drone_sim.gui.direct_backend import DirectBackend
from drone_sim.simulation.simulator import Simulator

# Repo root, for the shipped example scenarios of step 10.
_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "perception"

# Jacobi rather than Gauss-Seidel: the Gauss-Seidel ordering shuffles the drones through the global `random`
# module (distributed_coordinator.py), and a run that reorders its solves is not the run it was before.
_ADMM = ControllerSpec(type="dmpc_admm", params={"horizon": 4, "max_admm_iter": 20, "gauss_seidel": False})
# A tight timeout on purpose: if perception mode ever stops solving (risk R6), the run must fail in seconds
# instead of parking the suite on the 30 s default for every single step.
_THREADED = ControllerSpec(type="dmpc_threaded", params={"horizon": 4, "convergence_timeout_sec": 2.0})


def _drones() -> list[DroneConfig]:
   """Three drones crossing the same piece of room, so evasion is actually required.

   Not a symmetric star: the offsets keep the pairwise geometry from being degenerate, which is what makes an
   MPC solve pick a side instead of stalling between two equally good ones.
   """
   return [
      DroneConfig(drone_id="d1", start=[-2.0, 0.0, 0.0], target=[2.0, 0.0, 0.0], radius=0.2, safety_zone=0.6),
      DroneConfig(drone_id="d2", start=[2.0, 0.3, 0.0], target=[-2.0, -0.3, 0.0], radius=0.2, safety_zone=0.6),
      DroneConfig(drone_id="d3", start=[0.0, -2.0, 0.2], target=[0.0, 2.0, -0.2], radius=0.2, safety_zone=0.6),
   ]


def _cfg(coordinator: ControllerSpec, *, drones: list[DroneConfig] | None = None, **camera) -> ScenarioConfig:
   """The scenario above plus whatever camera flags the test needs; no camera key means the feature is off."""
   return ScenarioConfig(
      dt=0.1,
      physics=PhysicsSpec(id="default", type="linear_kinematics", params={"v_max": 3.0, "u_min": [-3.0] * 3, "u_max": [3.0] * 3}),
      controller=ControllerSpec(type="mpc_agent", params={"horizon": 4, "q_pos": [3.0] * 3, "r_u": [0.1] * 3}),
      coordinator=coordinator,
      drones=drones if drones is not None else _drones(),
      room=RoomConfig(min=[-3.0, -3.0, -3.0], max=[3.0, 3.0, 3.0]),
      **camera,
   )


def _perceiving(coordinator: ControllerSpec, **overrides) -> ScenarioConfig:
   """A scenario whose DMPC mailboxes are fed by the cameras — the configuration this whole feature is about."""
   camera = {"camera_enabled": True, "camera_async": False, "camera_feeds_dmpc": True,
             "camera_noise_sigma": 0.0, "camera_fov_deg": 180.0, "camera_range": 10.0}
   return _cfg(coordinator, **{**camera, **overrides})


def _positions(sim: Simulator) -> dict[str, np.ndarray]:
   return {d.drone_id: d.position().copy() for d in sim.drones}


def _run(cfg: ScenarioConfig, steps: int) -> list[dict[str, np.ndarray]]:
   """Step a fresh simulation from a seeded start and return the position of every drone after every step.

   Both global generators are seeded because a DMPC run draws from both: :mod:`random` for the Gauss-Seidel
   ordering, and ``np.random`` for the symmetry-breaking noise every local solve adds to its initial guess
   (``local_mpc.py``). Without the seed two identical scenarios differ by ~1e-5 after a few steps, which is
   the solver being nudged off a collinear deadlock — not a defect, but it makes exact comparison meaningless.
   Neither generator is touched by the perception path (the stub adapter owns a private ``Generator`` and
   leaves it alone at ``noise_sigma=0``), which is precisely why seeding here is enough to isolate the camera.
   """
   random.seed(20260811)
   np.random.seed(20260811)

   sim = Simulator.from_config(cfg)
   try:
      trace = []
      for _ in range(steps):
         sim.step()
         trace.append(_positions(sim))
      return trace
   finally:
      sim.close()


def _worker_threads() -> list[threading.Thread]:
   return [t for t in threading.enumerate() if t.name == "PerceptionWorker"]


def _wait_until(predicate, timeout: float = 5.0) -> bool:
   """Poll ``predicate`` until it holds or ``timeout`` passes — for the asynchronous example scenarios.

   With ``camera_async=true`` a capture is handed to a worker thread and the simulation carries on, so nothing
   guarantees that the estimates of a step are in the mailbox when ``step()`` returns. A healthy run satisfies
   the predicate almost immediately; the deadline only exists so a broken one fails instead of hanging.
   """
   deadline = time.monotonic() + timeout
   while time.monotonic() < deadline:
      if predicate():
         return True
      time.sleep(0.01)
   return predicate()


class TestAdmmPerceptionRun:
   """The ADMM coordinator driven entirely by camera estimates."""

   def test_ten_steps_run_and_the_drones_move(self):
      """Test the headline promise: the swarm flies with its inboxes filled from cameras instead of broadcasts.

      A ``RuntimeWarning`` is turned into an error for the duration of the run. ADMM raises one when it leaves
      its loop unconverged, and under perception it always would — a single pass has nothing to converge to.
      Forcing ``converged=True`` there (risk R4) is what keeps that warning from firing on every simulation
      step, and this is the test that would notice if it came back.
      """
      sim = Simulator.from_config(_perceiving(_ADMM))

      try:
         start = _positions(sim)
         with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for _ in range(10):
               sim.step()

         for drone_id, position in _positions(sim).items():
            assert np.linalg.norm(position - start[drone_id]) > 0.1, f"{drone_id} never moved"
      finally:
         sim.close()

   def test_every_drone_perceives_the_other_two(self):
      """Test the mailbox is the thing that carries the run: filled, per observer, without the observer itself."""
      sim = Simulator.from_config(_perceiving(_ADMM))

      try:
         for _ in range(10):
            sim.step()

         for drone in sim.drones:
            observed = sim._perception_mailbox.latest(drone.drone_id)
            assert sorted(observed) == sorted(d.drone_id for d in sim.drones if d.drone_id != drone.drone_id)
      finally:
         sim.close()

   def test_the_run_stays_single_pass(self):
      """Test the semantics change of risk R4 survives a whole run, not just the first step.

      Estimates are a fixed snapshot; a second ADMM iteration would re-solve against identical neighbor data.
      One iteration per step is therefore the contract, and it is also what keeps a perception run affordable.
      """
      sim = Simulator.from_config(_perceiving(_ADMM))

      try:
         for _ in range(10):
            sim.step()
            assert sim.coordinator.get_last_iteration_count() == 1
            assert sim.coordinator.get_last_converged() is True
      finally:
         sim.close()

   def test_drones_keep_their_distance(self):
      """Test the point of the whole exercise: they evade each other on estimates alone.

      With ``camera_noise_sigma=0`` the estimates equal the true positions, so this asks whether the *plumbing*
      preserves collision avoidance — not whether a noisy detector would. Anything closer than the drone radii
      would be a real collision.
      """
      sim = Simulator.from_config(_perceiving(_ADMM))

      try:
         for _ in range(20):
            sim.step()
            positions = _positions(sim)
            for i, a in enumerate(sim.drones):
               for b in sim.drones[i + 1:]:
                  distance = np.linalg.norm(positions[a.drone_id] - positions[b.drone_id])
                  assert distance > a.radius + b.radius, f"{a.drone_id}/{b.drone_id} touched at {distance:.3f} m"
      finally:
         sim.close()

   def test_a_blind_drone_still_gets_a_control(self):
      """Test the empty field of view: an unseen neighbor is missing from the solve, and that is not an error.

      The narrow cone leaves at least one drone observing nothing at the start. It must still fly its own
      route — realistic camera blindness, not a stalled solver.
      """
      sim = Simulator.from_config(_perceiving(_ADMM, camera_fov_deg=30.0, camera_range=1.5))

      try:
         start = _positions(sim)
         sim.step()
         # After the first capture, not before — the mailbox is empty for everyone until then, so checking
         # earlier would pass on any scenario at all.
         assert any(sim._perception_mailbox.latest(d.drone_id) == {} for d in sim.drones), "scenario no longer has a blind drone"

         for _ in range(9):
            sim.step()

         for drone_id, position in _positions(sim).items():
            assert np.linalg.norm(position - start[drone_id]) > 0.1, f"{drone_id} never moved"
      finally:
         sim.close()


class TestThreadedPerceptionRun:
   """The threaded coordinator on the same scene — the second, quite different mailbox seam.

   Not marked ``slow``, on purpose. The default pytest run excludes that marker, and the one thing these tests
   exist for — the 30 s-per-step hang of risk R6 — would then never be checked in a normal run. Perception mode
   is single-solve, so the whole class costs a fraction of a second when it is healthy.
   """

   def test_steps_complete_promptly(self):
      """Test the symptom of risk R6 directly: a perception step never waits out the convergence timeout.

      Two things keep it from happening, and this test only fires when *both* are gone: a drone with an empty
      field of view still solves, and the convergence poll is skipped because there is nothing to converge to.
      Either one alone still finishes promptly (verified by removing each in turn), so the sharp guard for the
      empty-inbox half is ``test_a_blind_drone_still_flies`` below — but with both removed the run takes 2 s
      per step, and that is exactly the pre-fix state the plan describes. The wall clock is the assertion
      because it is the symptom; the bound is the coordinator's own 2 s timeout, so one timed-out step fails.
      """
      sim = Simulator.from_config(_perceiving(_THREADED, camera_fov_deg=60.0))

      try:
         t0 = time.perf_counter()
         for _ in range(5):
            sim.step()
         elapsed = time.perf_counter() - t0

         assert elapsed < 2.0, f"5 perception steps took {elapsed:.1f}s — a solver is waiting out its timeout"
      finally:
         sim.close()

   def test_a_blind_drone_still_flies(self):
      """Test the second half of risk R6: an empty field of view is a state to plan in, not a reason not to plan.

      Both drones fly ``+x`` in single file, so the leader never has the follower in its cone and its inbox
      stays empty for the whole run. In negotiation mode an empty inbox means "the neighbors have not spoken
      yet" and the solve is skipped; keeping that rule under perception would leave the leader's ``u_prev`` at
      ``None``, which the coordinator answers with a zero control — a drone that simply sits there. It moving
      is what makes ``perception_mode`` on the solver observable from the outside.
      """
      single_file = [
         DroneConfig(drone_id="follower", start=[-2.0, 0.0, 0.0], target=[2.0, 0.0, 0.0], radius=0.2, safety_zone=0.6),
         DroneConfig(drone_id="leader", start=[-0.5, 0.0, 0.0], target=[3.5, 0.0, 0.0], radius=0.2, safety_zone=0.6),
      ]
      sim = Simulator.from_config(_perceiving(_THREADED, camera_fov_deg=60.0, drones=single_file))

      try:
         start = _positions(sim)
         for _ in range(5):
            sim.step()
            assert sim._perception_mailbox.latest("leader") == {}, "scenario no longer has a blind drone"

         assert np.linalg.norm(_positions(sim)["leader"] - start["leader"]) > 0.1, "the blind drone never solved"
         assert sim._perception_mailbox.latest("follower") != {}, "the seeing drone stopped seeing"
      finally:
         sim.close()

   def test_five_steps_run_and_the_drones_move(self):
      sim = Simulator.from_config(_perceiving(_THREADED))

      try:
         start = _positions(sim)
         for _ in range(5):
            sim.step()

         for drone_id, position in _positions(sim).items():
            assert np.linalg.norm(position - start[drone_id]) > 0.1, f"{drone_id} never moved"

         for drone in sim.drones:
            assert sim._perception_mailbox.latest(drone.drone_id) != {}
      finally:
         sim.close()

   def test_no_solver_threads_survive_the_run(self):
      """Test the coordinator joins what it spawned; a leaked solver thread would keep writing into a dead mailbox."""
      before = threading.active_count()
      sim = Simulator.from_config(_perceiving(_THREADED))

      try:
         for _ in range(3):
            sim.step()
      finally:
         sim.close()

      assert threading.active_count() == before


class TestPassiveObservationIsFree:
   """Turning the camera on without ``camera_feeds_dmpc`` must not cost the existing scenarios anything.

   This is the regression that protects every config in the repository: perception is opt-in twice over, and
   the first opt-in — observing — is not allowed to touch the control loop. Equality is exact, not approximate,
   which is what makes it sharp: both runs start from the same seed, so a capture that drew from a generator
   the solver also uses would shift every later draw, and a bridge that wrote into the trajectory mailbox
   regardless of the flag would change the solve outright. Either shows up as a drift in the last digits long
   before it shows up as different behaviour.
   """

   def _observed_and_blind(self, coordinator: ControllerSpec, steps: int):
      observing = _cfg(coordinator, camera_enabled=True, camera_async=False, camera_feeds_dmpc=False,
                       camera_noise_sigma=0.0, camera_fov_deg=180.0, camera_range=10.0)
      return _run(observing, steps), _run(_cfg(coordinator), steps)

   def test_admm_positions_are_identical(self):
      observed, blind = self._observed_and_blind(_ADMM, steps=8)

      for step, (with_camera, without) in enumerate(zip(observed, blind, strict=True)):
         for drone_id, position in with_camera.items():
            assert_array_equal(position, without[drone_id], err_msg=f"{drone_id} diverged at step {step}")

   def test_the_camera_really_was_running(self):
      """Test the guard for the test above: identical sequences prove nothing if the camera never observed."""
      sim = Simulator.from_config(_cfg(_ADMM, camera_enabled=True, camera_async=False, camera_feeds_dmpc=False,
                                       camera_noise_sigma=0.0, camera_fov_deg=180.0, camera_range=10.0))

      try:
         sim.step()

         assert sim._perception_mailbox.latest("d1") != {}
         assert sim._camera_feeds_dmpc is False
      finally:
         sim.close()


class TestRunLifecycle:
   """What a finished run leaves behind — checked on the shipped example scenarios of step 10.

   ``test_direct_backend.py`` already covers the same lifecycle against inline configs. Going through
   ``configs/perception/*.json`` here adds the part those cannot: that the two scenarios the plan ships are
   loadable, complete, and run the pipeline they advertise.
   """

   @pytest.fixture(params=["4DronesDMPCPerception.json", "2DronesThreadedPerception.json"])
   def example_config(self, request) -> Path:
      path = _CONFIGS / request.param
      assert path.is_file(), f"shipped example scenario is missing: {path}"
      return path

   def test_example_scenario_runs_with_a_live_camera(self, example_config: Path):
      cfg = ScenarioConfig.model_validate(json.loads(example_config.read_text(encoding="utf-8")))
      assert cfg.camera_enabled and cfg.camera_feeds_dmpc, f"{example_config.name} no longer feeds the DMPC"

      sim = Simulator.from_config(cfg)
      try:
         for _ in range(5):
            sim.step()

         assert _wait_until(lambda: all(sim._perception_mailbox.latest(d.drone_id) for d in sim.drones)), \
            f"{example_config.name} ran 5 steps without every drone observing a neighbor"
      finally:
         sim.close()

   def test_close_is_idempotent_and_leaves_no_worker(self, example_config: Path):
      cfg = ScenarioConfig.model_validate(json.loads(example_config.read_text(encoding="utf-8")))
      sim = Simulator.from_config(cfg)
      sim.step()

      sim.close()
      sim.close()  # must not raise

      assert _worker_threads() == []

   def test_reloading_through_the_gui_backend_accumulates_no_workers(self, example_config: Path):
      """Test risk R2 on the path that actually suffers from it: a GUI process outlives the scenarios it loads."""
      backend = DirectBackend()

      try:
         backend.load_config(example_config)
         backend.step()
         backend.load_config(example_config)
         backend.step()

         assert len(_worker_threads()) == 1
      finally:
         backend.close()

      assert _worker_threads() == []
