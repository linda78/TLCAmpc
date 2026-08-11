"""AV-FNN trajectory prediction model (variational feedforward variant).

A feedforward counterpart to ``AVLSTMModel`` for the AV-FNN vs AV-LSTM
ablation. It satisfies the same forward contract —
``forward(X_hat, Y_gt=None, teacher_forcing_ratio=...) -> (mu, sigma, mu_z, logvar_z)``
— so it is a drop-in replacement at inference (``LSTMModelLoader`` /
``LSTMSafetyZoneProvider``) and in training (NLL + KL loss).

Architecture (Option A — direct parallel prediction):
  1. MLP encoder: flatten ``(batch, m, n) -> (batch, m*n)`` -> context ``e in R^d``
  2. Variational bottleneck: ``Q(z|X) = N(mu_z, diag(exp(logvar_z)))``;
     ``z`` is sampled via the reparameterization trick during training and set
     to ``z = mu_z`` at inference.
  3. MLP decoder: ``z -> hidden -> hidden`` then two parallel heads that emit the
     entire ``(T, n)`` mu and sigma in a single forward pass (no autoregression).
  4. ``sigma = softplus(raw_sigma) + sigma_min`` — same floor convention as
     ``AVLSTMModel`` so the downstream UncertaintyPropagator is unaffected.

Unlike ``AVLSTMModel`` there is no autoregressive decoding and no teacher
forcing: ``Y_gt`` and ``teacher_forcing_ratio`` are accepted only for interface
parity and are ignored. The eval-mode forward pass is therefore fully
deterministic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drone_sim.prediction.model import reparameterize


class AVFNNModel(nn.Module):
   """Variational feedforward trajectory forecaster.

   Args:
       n:          State dimension (default: 6 for [px, py, pz, vx, vy, vz]).
       m:          History length in steps (default: 20). Fixed — the encoder
                   flattens (m, n), so inference windows MUST have this length.
       T:          Prediction horizon in steps (default: 80).
       d:          Encoder context dimension (default: 64).
       d_z:        Latent (variational bottleneck) dimension (default: 32).
       enc_hidden: Encoder MLP hidden width (default: 256).
       dec_hidden: Decoder MLP hidden width (default: 400). Chosen so the total
                   parameter count lands within +-15% of AVLSTMModel defaults.
       sigma_min:  Minimum sigma value to prevent collapse (default: 0.1).
   """

   def __init__(self, n: int = 6, m: int = 20, T: int = 80, d: int = 64,
                d_z: int = 32, enc_hidden: int = 256, dec_hidden: int = 400,
                sigma_min: float = 0.1) -> None:
      super().__init__()
      self.n = n
      self.m = m
      self.T = T
      self.d_z = d_z
      self.sigma_min = sigma_min

      # MLP encoder: flatten(m*n) -> context e in R^d
      self.encoder = nn.Sequential(
         nn.Linear(m * n, enc_hidden),
         nn.ReLU(),
         nn.Linear(enc_hidden, enc_hidden // 2),
         nn.ReLU(),
         nn.Linear(enc_hidden // 2, d),
      )

      # Variational posterior Q(z|X) = N(mu_z, diag(exp(logvar_z)))
      self.mu_z_head = nn.Linear(d, d_z)
      self.logvar_z_head = nn.Linear(d, d_z)

      # MLP decoder: z -> hidden, then parallel mu / raw-sigma heads spanning
      # the whole horizon (T*n outputs each).
      self.decoder = nn.Sequential(
         nn.Linear(d_z, dec_hidden),
         nn.ReLU(),
         nn.Linear(dec_hidden, dec_hidden),
         nn.ReLU(),
      )
      self.mu_head = nn.Linear(dec_hidden, T * n)
      self.sigma_head = nn.Linear(dec_hidden, T * n)

   def forward(self, X_hat: torch.Tensor, Y_gt: torch.Tensor | None = None,
               teacher_forcing_ratio: float = 1.0,
               **_kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      """Run the model forward pass.

      Args:
          X_hat: Noisy history tensor of shape ``(batch, m, n)``.
          Y_gt:  Ignored. Accepted only for interface parity with AVLSTMModel.
          teacher_forcing_ratio: Ignored. The model is not autoregressive.

      Returns:
          A tuple ``(mu, sigma, mu_z, logvar_z)``:
          - ``mu, sigma``: each ``(batch, T, n)``; ``sigma >= sigma_min``.
          - ``mu_z, logvar_z``: each ``(batch, d_z)``; variational posterior
            parameters for the KL term.
      """
      batch = X_hat.shape[0]
      x = X_hat.reshape(batch, -1)                       # (batch, m*n)

      e = self.encoder(x)                                # (batch, d)
      mu_z = self.mu_z_head(e)                           # (batch, d_z)
      logvar_z = self.logvar_z_head(e)                   # (batch, d_z)

      # Reparameterization trick (sample during training, mean at inference) — shared with AVLSTMModel,
      # so the ablation compares architectures rather than two sampling rules.
      z = reparameterize(mu_z, logvar_z, self.training)  # (batch, d_z)

      h = self.decoder(z)                                # (batch, dec_hidden)
      mu = self.mu_head(h).reshape(batch, self.T, self.n)
      raw_sigma = self.sigma_head(h).reshape(batch, self.T, self.n)
      sigma = F.softplus(raw_sigma) + self.sigma_min
      return mu, sigma, mu_z, logvar_z
