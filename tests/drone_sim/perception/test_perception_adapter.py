"""Tests for drone_sim.perception.adapter module.

Covers the seam between CameraView and PositionEstimate:
- StubPerceptionAdapter: id pass-through, the noise-free path (exact ground truth, RNG untouched), seeded reproducibility, sigma reporting
- Protocol conformance, including the duck-typed case that is the whole point of the module
- load_adapter: resolving the ``camera_adapter`` class path a scenario names, and failing loudly at load time when it is wrong
- one smoke test proving the estimates fit straight into the PerceptionMailbox

There is no outbound HTTP adapter to test: the out-of-process detector is a client of our own REST API, so that seam is tested in the API layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_sim.perception.adapter import PerceptionAdapter, StubPerceptionAdapter, load_adapter
from drone_sim.perception.camera import CameraView, VisibleDrone
from drone_sim.perception.mailbox import PerceptionMailbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _view(*, observer="ego", step=3, sim_time=0.3, position=(0.0, 0.0, 0.0), view_dir=(1.0, 0.0, 0.0),
          visible=(("d1", (5.0, 0.0, 0.0), 0.1),), image_png=None) -> CameraView:
   """Build a CameraView from (drone_id, position, radius) triples.

   No Drone and no physics are needed — the adapter only ever reads View attributes. Velocities are zeroed: they are not part of the adapter's
   contract in either direction.
   """
   return CameraView(observer_id=observer, step=step, sim_time=sim_time, position=np.array(position, dtype=float),
                     view_dir=np.array(view_dir, dtype=float), fov_deg=90.0, range_m=10.0,
                     visible=[VisibleDrone(drone_id=did, position=np.array(pos, dtype=float), velocity=np.zeros(3), radius=radius)
                              for did, pos, radius in visible],
                     image_png=image_png)


# ---------------------------------------------------------------------------
# StubPerceptionAdapter
# ---------------------------------------------------------------------------

class TestStubPerceptionAdapter:
   """Tests for the placeholder detector that runs on ground truth."""

   def test_one_estimate_per_visible(self):
      """Test every visible neighbor yields exactly one estimate under its own id."""
      view = _view(visible=(("d1", (5.0, 0.0, 0.0), 0.1), ("d2", (0.0, 4.0, 1.0), 0.2)))

      estimates = StubPerceptionAdapter().detect(view)

      assert [e.observed_id for e in estimates] == ["d1", "d2"]

   def test_zero_sigma_is_ground_truth(self):
      """Test noise_sigma=0 reproduces the visible positions bit-for-bit."""
      view = _view(visible=(("d1", (5.0, -1.0, 2.5), 0.1),))

      estimates = StubPerceptionAdapter(noise_sigma=0.0).detect(view)

      np.testing.assert_array_equal(estimates[0].position, np.array([5.0, -1.0, 2.5]))

   def test_zero_sigma_leaves_rng_untouched(self):
      """Test the noise-free path draws no random numbers, keeping a seeded stream aligned."""
      rng = np.random.default_rng(11)
      expected_next = np.random.default_rng(11).normal()

      StubPerceptionAdapter(noise_sigma=0.0, rng=rng).detect(
         _view(visible=(("d1", (1.0, 2.0, 3.0), 0.1), ("d2", (4.0, 5.0, 6.0), 0.1))))

      assert rng.normal() == expected_next

   def test_empty_view_returns_empty_list(self):
      """Test a View without visible neighbors detects nothing (and does not raise)."""
      assert StubPerceptionAdapter().detect(_view(visible=())) == []

   def test_passes_through_view_metadata(self):
      """Test observer/capture metadata comes from the View and received_time stays unstamped."""
      view = _view(observer="alpha", step=7, sim_time=0.7)

      estimate = StubPerceptionAdapter().detect(view)[0]

      assert estimate.observer_id == "alpha"
      assert estimate.captured_step == 7
      assert estimate.captured_time == pytest.approx(0.7)
      # Stamping received_time is the mailbox's job, not the adapter's.
      assert estimate.received_time == 0.0

   def test_noise_shifts_positions(self):
      """Test a positive sigma perturbs the position, but stays within a sane multiple of sigma."""
      view = _view(visible=(("d1", (5.0, 0.0, 0.0), 0.1),))

      estimate = StubPerceptionAdapter(noise_sigma=0.5, rng=np.random.default_rng(3)).detect(view)[0]

      offset = estimate.position - np.array([5.0, 0.0, 0.0])
      assert np.any(offset != 0.0)
      assert float(np.linalg.norm(offset)) < 10 * 0.5

   def test_seeded_rng_is_reproducible(self):
      """Test two adapters seeded identically produce identical noisy positions."""
      view = _view(visible=(("d1", (5.0, 0.0, 0.0), 0.1), ("d2", (0.0, 4.0, 0.0), 0.1)))

      first = StubPerceptionAdapter(noise_sigma=0.3, rng=np.random.default_rng(7)).detect(view)
      second = StubPerceptionAdapter(noise_sigma=0.3, rng=np.random.default_rng(7)).detect(view)

      for a, b in zip(first, second):
         np.testing.assert_array_equal(a.position, b.position)

   def test_sigma_reported(self):
      """Test the adapter reports its own noise level: None when perfect, the value otherwise."""
      view = _view()

      assert StubPerceptionAdapter(noise_sigma=0.0).detect(view)[0].sigma is None
      assert StubPerceptionAdapter(noise_sigma=0.5, rng=np.random.default_rng(1)).detect(view)[0].sigma == pytest.approx(0.5)

   def test_view_not_mutated(self):
      """Test detect() never writes into the View — it belongs to the simulator."""
      view = _view(visible=(("d1", (5.0, 0.0, 0.0), 0.1),))
      original = np.array(view.visible[0].position)

      StubPerceptionAdapter(noise_sigma=0.5, rng=np.random.default_rng(1)).detect(view)
      np.testing.assert_array_equal(view.visible[0].position, original)

      # Even the noise-free path hands out a copy, not the View's own array.
      estimate = StubPerceptionAdapter().detect(view)[0]
      assert estimate.position is not view.visible[0].position

   def test_negative_noise_sigma_raises(self):
      """Test a negative sigma is rejected, mirroring ScenarioConfig._validate_camera."""
      with pytest.raises(ValueError, match="noise_sigma"):
         StubPerceptionAdapter(noise_sigma=-0.1)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocolConformance:
   """Tests that detect() alone is the whole contract."""

   def test_stub_satisfies_protocol(self):
      """Test the stub adapter is recognized as a PerceptionAdapter."""
      assert isinstance(StubPerceptionAdapter(), PerceptionAdapter)

   def test_duck_typed_object_satisfies_protocol(self):
      """Test an unrelated object with detect() conforms — no inheritance needed for the external detector."""

      class _MyDetector:
         def detect(self, view):
            return []

      assert isinstance(_MyDetector(), PerceptionAdapter)


# ---------------------------------------------------------------------------
# Adapter -> mailbox
# ---------------------------------------------------------------------------

class TestAdapterFeedsMailbox:
   """Smoke test across the adapter/mailbox seam, before the worker sits in between."""

   def test_stub_estimates_land_in_mailbox(self):
      """Test detect() output can be posted as-is and comes back stamped."""
      view = _view(visible=(("d1", (5.0, 0.0, 0.0), 0.1),))
      mailbox = PerceptionMailbox()

      mailbox.post(view.observer_id, StubPerceptionAdapter().detect(view))

      latest = mailbox.latest("ego")
      np.testing.assert_array_equal(latest["d1"].position, np.array([5.0, 0.0, 0.0]))
      assert latest["d1"].received_time > 0.0


# ---------------------------------------------------------------------------
# load_adapter: the config seam
# ---------------------------------------------------------------------------

class LoadableDetector:
   """Stand-in for the separately developed detector, importable by class path like the real one will be."""

   def __init__(self, weights: str = "default.pt", threshold: float = 0.5) -> None:
      self.weights = weights
      self.threshold = threshold

   def detect(self, view):
      return []


class NotADetector:
   """Has no detect() — the mistake load_adapter has to catch before the worker swallows it per view."""


class TestLoadAdapter:
   """Tests for resolving ``ScenarioConfig.camera_adapter`` into an instance.

   Every failure must raise *here*, at scenario load: the perception worker isolates per-view exceptions by design, so a detector that only fails
   later would show up as drones quietly flying blind rather than as an error.
   """

   def test_loads_class_by_path(self):
      """Test the documented 'package.module:ClassName' form returns a usable instance."""
      adapter = load_adapter(f"{__name__}:LoadableDetector")

      assert isinstance(adapter, LoadableDetector)
      assert isinstance(adapter, PerceptionAdapter)

   def test_params_reach_the_constructor(self):
      """Test camera_adapter_params become keyword arguments — this is how a detector gets its weights path."""
      adapter = load_adapter(f"{__name__}:LoadableDetector", {"weights": "model.pt", "threshold": 0.9})

      assert adapter.weights == "model.pt"
      assert adapter.threshold == 0.9

   def test_no_params_uses_constructor_defaults(self):
      adapter = load_adapter(f"{__name__}:LoadableDetector", None)

      assert adapter.weights == "default.pt"

   @pytest.mark.parametrize("spec", ["my_project.detector.MyDetector", "MyDetector", "mod:", ":Cls", ""])
   def test_malformed_spec_raises(self, spec):
      """Test a path without the module:Class split is rejected with the expected form in the message."""
      with pytest.raises(ValueError, match="package.module:ClassName"):
         load_adapter(spec)

   def test_unimportable_module_raises_with_the_spec(self):
      """Test a typo in the module part names the offending string, not just an ImportError from deep inside importlib."""
      with pytest.raises(ValueError, match="cannot import module"):
         load_adapter("no_such_package.detector:MyDetector")

   def test_missing_class_raises(self):
      with pytest.raises(ValueError, match="has no 'NoSuchClass'"):
         load_adapter(f"{__name__}:NoSuchClass")

   def test_object_without_detect_raises_type_error(self):
      """Test the protocol check catches a class that forgot detect() — otherwise it would fail once per capture, inside the worker's try/except."""
      with pytest.raises(TypeError, match="not a PerceptionAdapter"):
         load_adapter(f"{__name__}:NotADetector")

   def test_constructor_errors_propagate(self):
      """Test a detector that rejects its params fails at load; wrapping this would hide the detector's own message."""
      with pytest.raises(TypeError):
         load_adapter(f"{__name__}:LoadableDetector", {"nonexistent_param": 1})
