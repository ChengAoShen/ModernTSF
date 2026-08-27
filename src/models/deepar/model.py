"""DeepAR model implementation.

Vendored/adapted from
https://github.com/GestaltCogTeam/BasicTS/blob/79641b1c75246ab2d8c53bb52f2ac72588be0cdc/baselines/DeepAR/arch/deepar_arch.py
(and the sibling ``distributions.py``), Apache-2.0 License.

DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks
(International Journal of Forecasting 2020, https://arxiv.org/abs/1704.04110).

Adapted for ModernTSF:
- The upstream BasicTS wrapper forward signature
  ``forward(history_data (B, L, N, C), future_data, train, ...)`` is replaced
  by the TSLib-style contract ``forward(x_enc, x_mark_enc, x_dec, x_mark_dec)``
  with tensors shaped ``(B, T, C)``.
- Each of the ``enc_in`` channels is treated as an independent series ("node")
  sharing one LSTM, matching the upstream per-node autoregressive scheme.
- Covariate features are taken from the temporal marks (``x_mark_*``) when
  present; otherwise the model runs with no exogenous covariates.
- DeepAR is probabilistic: it emits a rank-4 tensor
  ``(B, pred_len, C, 2) = (loc, scale)`` from the Gaussian likelihood head
  (``loc = mu``, ``scale = softplus(...) + 1e-6 > 0``). It is trained with the
  ``nll_gaussian`` loss and evaluated with CRPS / coverage / width plus the
  point metrics computed on the mean (``loc``). Autoregressive feedback uses the
  deterministic mean; sampling is out of scope for Phase 1.

The shared Gaussian parameter head retains the upstream ``distributions.py``
projection and literal positive-scale expression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.gaussian_parameter_head import GaussianParameterHead


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        embedding_size=32,
        hidden_size=64,
        num_layers=2,
        cov_feat_size=0,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features
        self.enc_in = enc_in
        self.c_out = 1 if features == "MS" else enc_in
        # Probabilistic output contract (Phase 1): emit (loc, scale) per element.
        self.output_type = "distribution"
        self.distribution_family = "gaussian"
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # ``cov_feat_size`` is the configured number of covariate (time-mark)
        # features. The actual count used at runtime is min(configured, available).
        self.cov_feat_size = cov_feat_size

        # input embedding for the scalar value at each step
        self.input_embed = nn.Linear(1, embedding_size)
        # LSTM encoder shared across channels/nodes
        self.encoder = nn.LSTM(
            embedding_size + cov_feat_size,
            hidden_size,
            num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Gaussian likelihood head
        self.likelihood_layer = GaussianParameterHead(
            hidden_size, 1, eps=1e-6, scale_transform="log1pexp"
        )

    def _build_covar(self, x_mark, B, N, T):
        """Build a per-step covariate tensor of shape (B*N, T, cov_feat_size).

        ``x_mark`` is (B, T, F) shared across channels. We broadcast it across
        the N channels. If covariates are disabled or unavailable, returns None.
        """
        if self.cov_feat_size == 0 or x_mark is None:
            return None
        F_avail = x_mark.shape[-1]
        f = min(self.cov_feat_size, F_avail)
        cov = x_mark[..., :f]  # (B, T, f)
        if f < self.cov_feat_size:
            pad = x_mark.new_zeros(B, T, self.cov_feat_size - f)
            cov = torch.cat([cov, pad], dim=-1)
        # broadcast across N channels -> (B, N, T, cov) -> (B*N, T, cov)
        cov = cov.unsqueeze(1).expand(B, N, T, self.cov_feat_size)
        cov = cov.reshape(B * N, T, self.cov_feat_size)
        return cov

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, _, N = x_enc.shape
        L_in = self.seq_len
        L_out = self.pred_len
        device = x_enc.device

        # Full value sequence: history values, then placeholders for the future.
        # During decoding the placeholders are overwritten autoregressively.
        future_holder = x_enc.new_zeros(B, L_out, N)
        values_full = torch.cat([x_enc, future_holder], dim=1)  # (B, L_in+L_out, N)

        # Per-step covariates over the full horizon, when temporal marks exist.
        x_mark_full = None
        if x_mark_enc is not None and self.cov_feat_size > 0:
            if x_mark_dec is not None:
                dec_marks = x_mark_dec[:, -L_out:, :]
                x_mark_full = torch.cat([x_mark_enc, dec_marks], dim=1)
            else:
                pad = x_mark_enc.new_zeros(B, L_out, x_mark_enc.shape[-1])
                x_mark_full = torch.cat([x_mark_enc, pad], dim=1)
        cov_full = self._build_covar(x_mark_full, B, N, L_in + L_out)

        h = c = None
        history_next = None  # (B, 1, N)
        mus = []
        sigmas = []

        for t in range(1, L_in + L_out):
            # The prediction made at iteration t is for sequence position t.
            collect = t >= L_in
            # Teacher forcing: feed the real value at input position t-1 whenever
            # that position lies inside the conditioning window (t-1 < L_in).
            if t <= L_in:
                history_next = values_full[:, t - 1 : t, :]
            # embed scalar value: (B, 1, N) -> (B*N, 1, 1) -> (B*N, 1, embed)
            val = history_next.transpose(1, 2).reshape(B * N, 1, 1)
            embed_feat = self.input_embed(val)
            if cov_full is not None:
                covar_feat = cov_full[:, t : t + 1, :]  # (B*N, 1, cov)
                encoder_input = torch.cat([embed_feat, covar_feat], dim=-1)
            else:
                encoder_input = embed_feat

            if t == 1:
                _, (h, c) = self.encoder(encoder_input)
            else:
                _, (h, c) = self.encoder(encoder_input, (h, c))

            mu, sigma = self.likelihood_layer(F.relu(h[-1, :, :]))  # (B*N, 1) each
            # AR feedback still uses the deterministic mean.
            history_next = mu.view(B, N, 1).transpose(1, 2)  # (B, 1, N)

            if collect:
                mus.append(mu.view(B, N, 1).transpose(1, 2))  # (B, 1, N)
                sigmas.append(sigma.view(B, N, 1).transpose(1, 2))  # (B, 1, N)

        dec_mu = torch.cat(mus, dim=1)  # (B, L_out, N)
        dec_sigma = torch.cat(sigmas, dim=1)  # (B, L_out, N)
        return dec_mu, dec_sigma

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_mu, dec_sigma = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.features == "MS":
            dec_mu = dec_mu[:, :, -1:]
            dec_sigma = dec_sigma[:, :, -1:]
        dec_mu = dec_mu[:, -self.pred_len :, :]  # (B, pred_len, c_out)
        dec_sigma = dec_sigma[:, -self.pred_len :, :]  # (B, pred_len, c_out)
        out = torch.stack([dec_mu, dec_sigma], dim=-1)  # (B, pred_len, c_out, 2)
        return out  # out[..., 0] = loc, out[..., 1] = scale > 0
