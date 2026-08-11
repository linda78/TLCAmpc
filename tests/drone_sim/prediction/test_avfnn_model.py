"""Unit tests for AVFNNModel and a training smoke test for train_avfnn.

Modelled on ``test_model.py``. AVFNNModel is the variational feedforward
counterpart to AVLSTMModel; the AVFNN-specific tests assert the two
properties that distinguish it: eval-mode forward is FULLY deterministic
(mu and sigma), and ``Y_gt`` / ``teacher_forcing_ratio`` have no effect.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from drone_sim.prediction.avfnn_model import AVFNNModel
from drone_sim.prediction.model_loader import LSTMModelLoader

# ---------------------------------------------------------------------------
# Constants matching AVFNNModel defaults
# ---------------------------------------------------------------------------
BATCH = 4
M = 20   # history steps
T = 80   # prediction horizon
N = 6    # state dimension (px, py, pz, vx, vy, vz)
SIGMA_MIN = 0.1
D_Z = 32  # default latent dimension


@pytest.fixture
def model() -> AVFNNModel:
    """Default AVFNNModel with spec hyperparameters."""
    return AVFNNModel()


@pytest.fixture
def x_hat() -> torch.Tensor:
    """Synthetic noisy history: (batch, m, n) float32."""
    torch.manual_seed(42)
    return torch.randn(BATCH, M, N)


# ---------------------------------------------------------------------------
# TestAVFNNModel
# ---------------------------------------------------------------------------
class TestAVFNNModel:
    def test_output_shapes(self, model: AVFNNModel, x_hat: torch.Tensor) -> None:
        """Forward pass produces the AVLSTM-compatible 4-tuple with correct shapes."""
        model.eval()
        with torch.no_grad():
            mu, sigma, mu_z, logvar_z = model(x_hat)
        assert mu.shape == (BATCH, T, N), f"mu shape {mu.shape}"
        assert sigma.shape == (BATCH, T, N), f"sigma shape {sigma.shape}"
        assert mu_z.shape == (BATCH, D_Z), f"mu_z shape {mu_z.shape}"
        assert logvar_z.shape == (BATCH, D_Z), f"logvar_z shape {logvar_z.shape}"

    def test_sigma_floor(self, model: AVFNNModel, x_hat: torch.Tensor) -> None:
        """Sigma must be >= sigma_min everywhere — no sigma collapse possible."""
        model.eval()
        with torch.no_grad():
            _, sigma, _, _ = model(x_hat)
        assert (sigma >= SIGMA_MIN).all(), (
            f"sigma below sigma_min={SIGMA_MIN}: min={sigma.min().item():.6f}"
        )

    def test_float32_dtype(self, model: AVFNNModel, x_hat: torch.Tensor) -> None:
        """All four output tensors must be float32."""
        model.eval()
        with torch.no_grad():
            mu, sigma, mu_z, logvar_z = model(x_hat)
        assert mu.dtype == torch.float32, f"mu dtype: {mu.dtype}"
        assert sigma.dtype == torch.float32, f"sigma dtype: {sigma.dtype}"
        assert mu_z.dtype == torch.float32, f"mu_z dtype: {mu_z.dtype}"
        assert logvar_z.dtype == torch.float32, f"logvar_z dtype: {logvar_z.dtype}"

    def test_batch_size_one(self, model: AVFNNModel) -> None:
        """Single-sample batch (batch_size=1) must work correctly."""
        x = torch.randn(1, M, N)
        model.eval()
        with torch.no_grad():
            mu, sigma, mu_z, logvar_z = model(x)
        assert mu.shape == (1, T, N)
        assert sigma.shape == (1, T, N)
        assert mu_z.shape == (1, D_Z)
        assert logvar_z.shape == (1, D_Z)
        assert (sigma >= SIGMA_MIN).all()

    def test_kl_divergence_computable(self, model: AVFNNModel, x_hat: torch.Tensor) -> None:
        """mu_z/logvar_z yield a finite scalar KL divergence."""
        model.train()
        _, _, mu_z, logvar_z = model(x_hat)
        kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        assert kl.shape == (), "KL should be a scalar"
        assert torch.isfinite(kl), f"KL is not finite: {kl.item()}"

    def test_eval_mode_fully_deterministic(self, model: AVFNNModel, x_hat: torch.Tensor) -> None:
        """In eval mode, repeated forwards give identical mu AND sigma.

        This is the AVFNN-specific contrast with AVLSTMModel, whose decoder
        still samples x_prev between steps. AVFNN has no autoregression, so
        eval()-mode inference is fully deterministic.
        """
        model.eval()
        with torch.no_grad():
            mu1, sigma1, _, _ = model(x_hat)
            mu2, sigma2, _, _ = model(x_hat)
        assert torch.allclose(mu1, mu2), "mu not deterministic in eval mode"
        assert torch.allclose(sigma1, sigma2), "sigma not deterministic in eval mode"

    def test_teacher_forcing_ignored(self, model: AVFNNModel, x_hat: torch.Tensor) -> None:
        """Y_gt and teacher_forcing_ratio must have no effect on the output."""
        model.eval()
        with torch.no_grad():
            mu_a, sigma_a, _, _ = model(x_hat, Y_gt=None, teacher_forcing_ratio=0.0)
            mu_b, sigma_b, _, _ = model(
                x_hat, Y_gt=torch.randn(BATCH, T, N), teacher_forcing_ratio=1.0
            )
        assert torch.allclose(mu_a, mu_b), "Y_gt/teacher_forcing_ratio changed mu"
        assert torch.allclose(sigma_a, sigma_b), "Y_gt/teacher_forcing_ratio changed sigma"

    def test_train_mode_stochastic_latent(self, model: AVFNNModel, x_hat: torch.Tensor) -> None:
        """In train mode the encoder is deterministic but the sampled latent is not.

        mu_z is identical across calls (deterministic encoder), but the
        reparameterized sample z makes mu differ — confirming the variational
        bottleneck is active during training.
        """
        model.train()
        mu1, _, mu_z1, _ = model(x_hat)
        mu2, _, mu_z2, _ = model(x_hat)
        assert torch.allclose(mu_z1, mu_z2), "mu_z should be a deterministic encoder output"
        assert not torch.allclose(mu1, mu2), "mu identical — reparameterization not active"


# ---------------------------------------------------------------------------
# Slow integration test — training smoke test
# ---------------------------------------------------------------------------
def _write_npz(path: Path, num_samples: int = 8) -> None:
    """Write a synthetic NPZ with float64 arrays (the real dataset format)."""
    X = np.random.randn(num_samples, M, N).astype(np.float64)
    Y = np.random.randn(num_samples, T, N).astype(np.float64)
    np.savez_compressed(path, X=X, Y=Y)


@pytest.mark.slow
class TestAVFNNTraining:
    def test_train_avfnn_end_to_end(self, tmp_path: Path) -> None:
        """train_avfnn runs on tiny synthetic data and yields a usable checkpoint."""
        from paper3_lstm.train_avfnn import train_avfnn

        np.random.seed(0)
        _write_npz(tmp_path / "data.npz", num_samples=8)
        output_path = tmp_path / "avfnn.pt"

        train_avfnn(
            data_dir=tmp_path,
            output_path=output_path,
            n_epochs=2,
            batch_size=8,
            lr=1e-3,
        )

        assert output_path.exists(), "train_avfnn did not write the checkpoint"
        ckpt = torch.load(output_path, map_location="cpu")
        assert ckpt["arch"] == "avfnn", ckpt.get("arch")
        assert "state_dict" in ckpt and "model_kwargs" in ckpt

        loader = LSTMModelLoader(checkpoint_path=output_path)
        assert isinstance(loader.model, AVFNNModel), type(loader.model)
        with torch.inference_mode():
            mu, sigma, mu_z, logvar_z = loader.model(
                torch.randn(1, M, N), Y_gt=None, teacher_forcing_ratio=0.0
            )
        assert mu.shape == (1, T, N)
        assert sigma.shape == (1, T, N)
        assert mu_z.shape == (1, D_Z)
        assert logvar_z.shape == (1, D_Z)
