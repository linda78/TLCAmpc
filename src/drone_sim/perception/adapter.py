"""Seam between the simulation and the (separately developed) video detector, for the **in-process** case.

Everything upstream of this module produces a :class:`~drone_sim.perception.camera.CameraView`; everything downstream consumes
:class:`~drone_sim.perception.mailbox.PositionEstimate` records. The detector can bridge that gap two ways, and only the first one goes through here:

- **in-process** — any object with ``detect(view) -> list[PositionEstimate]`` satisfies :class:`PerceptionAdapter` and can be handed to the perception
  worker directly. No inheritance, no registration; the duck type *is* the contract. This is the path for importing the detection project as a
  library and calling its internal functions.
- **out-of-process** — the detector is an HTTP *client* of TLCAmpc's own REST API: it pulls rendered camera images and pushes the resulting estimates
  straight into the :class:`~drone_sim.perception.mailbox.PerceptionMailbox` (see :mod:`drone_sim.api.app`). That path does not touch this module.

There is deliberately **no outbound HTTP adapter** here. Nothing in the perception pipeline calls out to a foreign service — there is no detector
endpoint to trigger, the detector drives the exchange itself. If you are looking for the REST seam, it is in the API layer, not here.

Two assumptions are baked into the contract and worth stating out loud:

* **Data association is assumed solved** (UVDAR-style blinking marker ids). Every estimate carries the ``drone_id`` of the neighbor it belongs to —
  :class:`StubPerceptionAdapter` passes the ground-truth id through, and the REST push contract demands one per estimate too. A detector that can only say
  "something is over there" without resolving *which* drone it is, is explicitly **not** covered here.
* **Adapters are transport, not trackers.** No filtering, no clipping, no plausibility check happens below. Estimates for neighbors that are not in
  ``view.visible``, or for ids the simulation has never heard of, are passed straight through — false positives are the detector's business, and
  swallowing them here would hide them from whoever has to debug the detector.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from drone_sim.perception.mailbox import PositionEstimate

if TYPE_CHECKING:
   from drone_sim.perception.camera import CameraView

_log = logging.getLogger(__name__)


@runtime_checkable
class PerceptionAdapter(Protocol):
   """Turns one :class:`~drone_sim.perception.camera.CameraView` into the neighbor positions a detector believes it sees.

   One call per ego drone per capture. The perception worker runs this on its own thread, so implementations must be either thread-safe or owned
   exclusively by one worker. Blocking is allowed — the worker exists to decouple a slow detector from the simulation loop — but the View must be
   treated as read-only: the simulator (and, with ``camera_render_images``, the renderer) still owns it.
   """

   def detect(self, view: CameraView) -> list[PositionEstimate]:
      """Estimate neighbor positions from one View.

      :param view: Capture of one ego drone at one simulation step.
      :return: One estimate per detected neighbor, in arbitrary order; empty list when nothing is detected.
      """
      ...


def _estimate_from_view(view: CameraView, observed_id: str, position: np.ndarray | Sequence[float],
                        sigma: float | None) -> PositionEstimate:
   """Wire one detection result together with the View metadata it belongs to.

   The two time axes of :mod:`drone_sim.perception.mailbox` are joined here and nowhere else: ``captured_step`` / ``captured_time`` come from the
   View (simulation time), while ``received_time`` stays at its ``0.0`` default — stamping it is :meth:`PerceptionMailbox.post`'s job alone.

   The ``reshape(3)`` is deliberate: a truncated server answer such as ``[x, y]`` fails here, with the offending detection in hand, instead of
   several layers later inside the solver.

   :param view: View the detection was made on.
   :param observed_id: Identifier of the detected neighbor.
   :param position: Estimated world position, anything array-like of length 3.
   :param sigma: 1-sigma position uncertainty in meters, or ``None`` when the detector reports none.
   :return: The assembled estimate.
   """
   return PositionEstimate(observer_id=view.observer_id, observed_id=observed_id,
                           position=np.asarray(position, dtype=float).reshape(3), captured_step=int(view.step),
                           captured_time=float(view.sim_time), sigma=float(sigma) if sigma is not None else None)


def _log_detection(kind: str, view: CameraView, estimates: list[PositionEstimate], *, extra: str = "") -> None:
   """Emit one grep-able DEBUG summary per detect() call.

   The interesting number is ``visible -> estimates``: a detector that sees fewer neighbors than the geometry offers is missing detections, one that
   sees more is producing false positives — and the adapter passes both through unchanged by design.
   Per-call lines are at DEBUG so the default INFO log stays clean.
   """
   if not _log.isEnabledFor(logging.DEBUG):
      return
   _log.debug("Perception %s detect: observer=%s step=%d t=%.3f | %d visible -> %d estimates %s%s", kind, view.observer_id, view.step, view.sim_time,
              len(view.visible), len(estimates), [e.observed_id for e in estimates], f" | {extra}" if extra else "")


class StubPerceptionAdapter:
   """Placeholder detector: ground truth plus Gaussian noise, no image processing at all.

   It exists so the whole pipeline — capture, worker, mailbox, DMPC bridge — is testable and runnable before the real detector is finished.
   ``noise_sigma=0`` models a perfect sensor and reproduces ``view.visible`` exactly; a positive value scatters each position independently.
   Neighbor ids are passed through unchanged (data association is assumed solved, see the module docstring).

   The injectable ``rng`` is the determinism lever for tests. At ``noise_sigma == 0`` the generator is not touched at all, so a seeded stream stays
   aligned across runs that mix noisy and noise-free adapters.

   :param noise_sigma: Standard deviation of the per-axis Gaussian position noise in meters, must be non-negative.
   :param rng: Random generator to draw the noise from. ``None`` creates an unseeded ``np.random.default_rng()``.
   """

   def __init__(self, noise_sigma: float = 0.0, rng: np.random.Generator | None = None) -> None:
      # Mirrors ScenarioConfig._validate_camera — a StubPerceptionAdapter is also constructed directly (tests, scripts), not only from a validated config.
      if noise_sigma < 0:
         raise ValueError("noise_sigma must be non-negative")
      self._noise_sigma = float(noise_sigma)
      self._rng = rng if rng is not None else np.random.default_rng()
      _log.info("StubPerceptionAdapter ready: noise_sigma=%.3f seeded_rng=%s", self._noise_sigma, rng is not None)

   def detect(self, view: CameraView) -> list[PositionEstimate]:
      """Report one estimate per visible neighbor.

      :param view: Capture of one ego drone; ``view.visible`` carries the ground truth.
      :return: One estimate per entry in ``view.visible``, ids unchanged; empty list for an empty View.
      """
      sigma = self._noise_sigma if self._noise_sigma > 0 else None

      estimates: list[PositionEstimate] = []
      for visible in view.visible:
         if sigma is None:
            # Copy, and deliberately no RNG draw: the noise-free path must not consume the stream.
            position = np.array(visible.position, dtype=float)
         else:
            position = np.asarray(visible.position, dtype=float) + self._rng.normal(0.0, self._noise_sigma, 3)
         estimates.append(_estimate_from_view(view, visible.drone_id, position, sigma))

      _log_detection("stub", view, estimates, extra=f"noise_sigma={self._noise_sigma}")
      return estimates
