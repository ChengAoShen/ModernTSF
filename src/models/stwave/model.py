"""ModernTSF adapter for the STWave spatiotemporal graph forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/STWave), Apache-2.0.

STWave (When Spatio-Temporal Meet Wavelets, ICDE 2023) disentangles a traffic
signal into low- / high-frequency components with a discrete wavelet transform
and processes each with a dual spatiotemporal stack: temporal attention / TCN
plus an efficient *sparse spatial attention* that uses a spectral graph
positional encoding (eigenvectors / eigenvalues of the normalized Laplacian)
and a graph-attention neighbour sampling.

STWave REQUIRES a predefined graph. From the injected ``(N, N)`` ``adj_mx`` this
adapter builds, in pure torch / numpy (no SciPy ``eigsh``):

  * ``graphwave`` = the top-``hidden_size`` eigenpairs of the symmetric
    normalized Laplacian of ``adj + I`` (via :func:`torch.linalg.eigh`), used as
    the spatial positional encoding,
  * ``adj_gat`` = the ``k`` nearest neighbours of every node by shortest-path
    distance (SciPy Dijkstra), used to restrict the sparse spatial attention.

The upstream arch keeps the BasicTS signature
``forward(history_data, future_data, batch_seen, epoch, train, **kwargs)`` with
``history_data`` shaped ``(B, L, N, C)`` and returns ``(B, pred_len, N, 1)`` at
inference. This adapter converts ModernTSF's ``(x_enc, x_mark_enc)`` into
``(B, L, N, input_dim)`` via :func:`models._external.marks.to_spatiotemporal`
(channel 0 the value, then calendar ``[time_in_day, day_in_week]``), always runs
the inference path (so the standard MAE loss sees ``(B, pred_len, N)``), and
squeezes the output channel. All hardcoded CUDA calls in the vendored arch are
removed; internally-created tensors follow the input tensor's device.
"""

from __future__ import annotations

import math

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from models._external.marks import to_spatiotemporal


# --------------------------------------------------------------------------- #
# Graph construction (adapted from BasicTS baselines/STWave/PEMS04.py).
# The original uses SciPy ``eigsh``; we use a dense ``torch.linalg.eigh`` and
# take the top-``k`` eigenpairs, which avoids the ``k < N`` restriction and
# keeps everything in torch.
# --------------------------------------------------------------------------- #
def _symmetric_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """Return ``L = I - D^{-1/2} A D^{-1/2}`` (symmetric normalized Laplacian)."""
    adj = np.asarray(adj, dtype=np.float64)
    n = adj.shape[0]
    degree = adj.sum(axis=0)
    d_inv_sqrt = np.zeros_like(degree)
    nz = degree > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(degree[nz])
    d_mat = np.diag(d_inv_sqrt)
    return np.eye(n) - d_mat @ adj @ d_mat


def _largest_k_eigv(laplacian: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` eigenpairs of a symmetric Laplacian via ``torch.linalg.eigh``.

    Returns ``(eigvalues (k,), eigvectors (N, k))`` with eigenvalues in
    descending magnitude (matching BasicTS' ``which='LM'``). ``k`` is clamped to
    ``N``; if fewer than ``k`` are available the result is zero-padded so the
    spectral positional encoding always has width ``k`` (= ``hidden_size``).
    """
    n = laplacian.shape[0]
    lap = torch.from_numpy(np.asarray(laplacian, dtype=np.float32))
    # eigh returns ascending eigenvalues; take the largest-magnitude k.
    evals, evecs = torch.linalg.eigh(lap)  # evals (N,), evecs (N, N)
    order = torch.argsort(evals.abs(), descending=True)
    take = min(k, n)
    idx = order[:take]
    lamb = evals[idx]
    vec = evecs[:, idx]  # (N, take)
    if take < k:
        lamb = F.pad(lamb, (0, k - take))
        vec = F.pad(vec, (0, k - take))
    return lamb.numpy().astype(np.float32), vec.numpy().astype(np.float32)


def _load_graph(
    adj_mx: np.ndarray, hidden_size: int, log_samples: int
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Build ``(adj_gat, graphwave)`` from a dense ``(N, N)`` adjacency.

    ``graphwave`` = top-``hidden_size`` eigenpairs of the normalized Laplacian of
    ``adj + I``. ``adj_gat`` = the ``sampled_nodes_number`` nearest neighbours of
    each node by shortest-path distance. Mirrors BasicTS' ``loadGraph``.
    """
    adj_mx = np.asarray(adj_mx, dtype=np.float32)
    n = adj_mx.shape[0]
    graphwave = _largest_k_eigv(
        _symmetric_normalized_laplacian(adj_mx + np.eye(n, dtype=np.float32)),
        hidden_size,
    )
    sampled_nodes_number = int(np.around(math.log(n)) + 2) * log_samples
    sampled_nodes_number = max(1, min(sampled_nodes_number, n))
    graph = csr_matrix(adj_mx)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix == 0] = dist_matrix.max() + 10
    adj_gat = np.argpartition(dist_matrix, sampled_nodes_number - 1, -1)[
        :, :sampled_nodes_number
    ]
    return adj_gat, graphwave


# --------------------------------------------------------------------------- #
# Vendored STWave layers (adapted from BasicTS baselines/STWave/arch).
# Tensors created internally follow the input device (no ``.to('cuda')``).
# --------------------------------------------------------------------------- #
class FeedForward(nn.Module):
    """Stacked linear layers with optional residual + LayerNorm."""

    def __init__(self, fea, res_ln: bool = False) -> None:
        super().__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList(
            [nn.Linear(fea[i], fea[i + 1]) for i in range(self.L)]
        )
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L - 1:
                x = F.relu(x)
        if self.res_ln:
            x = x + inputs
            x = self.ln(x)
        return x


class temporalEmbedding(nn.Module):
    """One-hot day-of-week / time-of-day embedding.

    ``in_dim`` is ``day_in_week_size + time_in_day_size`` (295 = 7 + 288 in the
    original PEMS config); kept configurable so smaller calendar vocabularies
    (e.g. 24 steps/day) work without shape mismatches.
    """

    def __init__(self, D: int, in_dim: int = 295) -> None:
        super().__init__()
        self.ff_te = FeedForward([in_dim, D, D])

    def forward(self, TE, T=288, W=7):
        # TE: [B, T, 2] of integer indices [dow, tod].
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], W, device=TE.device)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T, device=TE.device)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % W, W)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % T, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # [B, T, W + T]
        TE = TE.unsqueeze(dim=2)  # [B, T, 1, 295]
        TE = self.ff_te(TE)  # [B, T, 1, F]
        return TE


class sparseSpatialAttention(nn.Module):
    """Efficient sparse spatial attention with spectral positional encoding."""

    def __init__(self, hidden_size: int, log_samples: int) -> None:
        super().__init__()
        self.qfc = FeedForward([hidden_size, hidden_size])
        self.kfc = FeedForward([hidden_size, hidden_size])
        self.vfc = FeedForward([hidden_size, hidden_size])
        self.ofc = FeedForward([hidden_size, hidden_size])

        self._hidden_size = hidden_size
        self._log_samples = log_samples

        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size, hidden_size, hidden_size], True)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x, adj, eigvec, eigvalue):
        # x: [B, T, N, D]
        x_ = x + torch.matmul(
            eigvec.transpose(0, 1).squeeze(-1), torch.diag_embed(eigvalue)
        )

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)

        B, T, N, D = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, T, N, N, D)
        K_sample = K_expand[:, :, torch.arange(N).unsqueeze(1), adj, :]
        V_expand = V.unsqueeze(-3).expand(B, T, N, N, D)
        V_sample = V_expand[:, :, torch.arange(N).unsqueeze(1), adj, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1))
        GAT_results = torch.matmul(Q_K_sample, V_sample).squeeze(-2)
        M = self.proj(GAT_results).squeeze(-1)
        samples = max(1, min(int(self._log_samples * math.log(N, 2)), N))
        M_top = M.topk(samples, sorted=False)[1]

        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(T)[None, :, None],
            M_top,
            :,
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        Q_K = Q_K / (self._hidden_size ** 0.5)

        attn = torch.softmax(Q_K, dim=-1)

        cp = attn.argmax(dim=-2, keepdim=True).transpose(-2, -1)
        value = (
            torch.matmul(attn, V)
            .unsqueeze(-3)
            .expand(B, T, N, M_top.shape[-1], V.shape[-1])[
                torch.arange(B)[:, None, None, None],
                torch.arange(T)[None, :, None, None],
                torch.arange(N)[None, None, :, None],
                cp,
                :,
            ]
            .squeeze(-2)
        )

        value = self.ofc(value) + x_
        value = self.ln(value)
        return self.ff(value)


class temporalAttention(nn.Module):
    """Masked (causal) temporal self-attention."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.qfc = FeedForward([hidden_size, hidden_size])
        self.kfc = FeedForward([hidden_size, hidden_size])
        self.vfc = FeedForward([hidden_size, hidden_size])
        self.ofc = FeedForward([hidden_size, hidden_size])
        self._hidden_size = hidden_size
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size, hidden_size, hidden_size], True)

    def forward(self, x, te, Mask=True):
        x = x + te
        query = self.qfc(x).permute(0, 2, 1, 3)  # [B, N, T, F]
        key = self.kfc(x).permute(0, 2, 3, 1)
        value = self.vfc(x).permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)  # [B, N, T, T]
        attention = attention / (self._hidden_size ** 0.5)

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps, device=x.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(batch_size, num_vertexs, 1, 1)
            mask = mask.to(torch.bool)
            zero_vec = (-(2 ** 15) + 1) * torch.ones_like(attention)
            attention = torch.where(mask, attention, zero_vec)

        attention = F.softmax(attention, -1)

        value = torch.matmul(attention, value).permute(0, 2, 1, 3)
        value = self.ofc(value)
        value = value + x
        value = self.ln(value)
        return self.ff(value)


class Chomp1d(nn.Module):
    """Remove the trailing padded steps introduced by a causal conv."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, : -self.chomp_size].contiguous()


class temporalConvNet(nn.Module):
    """Dilated causal temporal convolution stack."""

    def __init__(self, hidden_size, kernel_size=2, dropout=0.2, levels=1) -> None:
        super().__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            conv = nn.Conv2d(
                hidden_size,
                hidden_size,
                (1, kernel_size),
                dilation=(1, dilation_size),
                padding=(0, padding),
            )
            chomp = Chomp1d(padding)
            relu = nn.ReLU()
            drop = nn.Dropout(dropout)
            layers += [nn.Sequential(conv, chomp, relu, drop)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        return self.tcn(xh.transpose(1, 3)).transpose(1, 3)


class adaptiveFusion(nn.Module):
    """Adaptive fusion of the low- / high-frequency predictions."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.qlfc = FeedForward([hidden_size, hidden_size])
        self.khfc = FeedForward([hidden_size, hidden_size])
        self.vhfc = FeedForward([hidden_size, hidden_size])
        self.ofc = FeedForward([hidden_size, hidden_size])
        self._hidden_size = hidden_size
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ff = FeedForward([hidden_size, hidden_size, hidden_size], True)

    def forward(self, xl, xh, te, Mask=True):
        xl = xl + te
        xh = xh + te

        query = self.qlfc(xl).permute(0, 2, 1, 3)  # [B, N, T, F]
        keyh = torch.relu(self.khfc(xh)).permute(0, 2, 3, 1)
        valueh = torch.relu(self.vhfc(xh)).permute(0, 2, 1, 3)

        attentionh = torch.matmul(query, keyh)  # [B, N, T, T]

        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps, device=xl.device)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(batch_size, num_vertexs, 1, 1)
            mask = mask.to(torch.bool)
            zero_vec = (-(2 ** 15) + 1) * torch.ones_like(attentionh)
            attentionh = torch.where(mask, attentionh, zero_vec)
        attentionh = attentionh / (self._hidden_size ** 0.5)
        attentionh = F.softmax(attentionh, -1)

        value = torch.matmul(attentionh, valueh).permute(0, 2, 1, 3)
        value = self.ofc(value)
        value = value + xl
        value = self.ln(value)
        return self.ff(value)


class dualEncoder(nn.Module):
    """Dual (low / high) spatiotemporal encoder block."""

    def __init__(self, hidden_size, log_samples, adj_gat, graphwave) -> None:
        super().__init__()
        self.tcn = temporalConvNet(hidden_size)
        self.tatt = temporalAttention(hidden_size)

        self.ssal = sparseSpatialAttention(hidden_size, log_samples)
        self.ssah = sparseSpatialAttention(hidden_size, log_samples)

        eigvalue = torch.from_numpy(graphwave[0].astype(np.float32))
        self.eigvalue = nn.Parameter(eigvalue, requires_grad=True)
        eigvec = (
            torch.from_numpy(graphwave[1].astype(np.float32))
            .transpose(0, 1)
            .unsqueeze(-1)
        )
        # Register graph tensors as buffers so they follow the model device.
        self.register_buffer("eigvec", eigvec)
        self.register_buffer("adj", torch.from_numpy(np.asarray(adj_gat)).long())

    def forward(self, xl, xh, te):
        xl = self.tatt(xl, te)
        xh = self.tcn(xh)

        spa_statesl = self.ssal(xl, self.adj, self.eigvec, self.eigvalue)
        spa_statesh = self.ssah(xh, self.adj, self.eigvec, self.eigvalue)
        xl = spa_statesl + xl
        xh = spa_statesh + xh
        return xl, xh


def disentangle(x, w, j):
    """Single-level wavelet disentangle into low / high frequency parts."""
    x = x.transpose(0, 3, 2, 1)  # [B, D, N, T]
    coef = pywt.wavedec(x, w, level=j)
    coefl = [coef[0]] + [None] * (len(coef) - 1)
    coefh = [None] + [coef[i + 1] for i in range(len(coef) - 1)]
    xl = pywt.waverec(coefl, w).transpose(0, 3, 2, 1)
    xh = pywt.waverec(coefh, w).transpose(0, 3, 2, 1)
    return torch.from_numpy(xl), torch.from_numpy(xh)


class STWave(nn.Module):
    """Upstream STWave backbone (BasicTS signature).

    Paper: When Spatio-Temporal Meet Wavelets (ICDE 2023). Disentangled traffic
    forecasting via efficient spectral graph attention networks.
    """

    def __init__(
        self,
        input_dim,
        hidden_size,
        layers,
        seq_len,
        horizon,
        log_samples,
        adj_gat,
        graphwave,
        time_in_day_size,
        day_in_week_size,
        wave_type,
        wave_levels,
    ) -> None:
        super().__init__()
        self.start_emb_l = FeedForward([input_dim, hidden_size, hidden_size])
        self.start_emb_h = FeedForward([input_dim, hidden_size, hidden_size])
        self.te_emb = temporalEmbedding(
            hidden_size, in_dim=day_in_week_size + time_in_day_size
        )

        self.dual_encoder = nn.ModuleList(
            [
                dualEncoder(hidden_size, log_samples, adj_gat, graphwave)
                for _ in range(layers)
            ]
        )
        self.adaptive_fusion = adaptiveFusion(hidden_size)

        self.pre_l = nn.Conv2d(seq_len, horizon, (1, 1))
        self.pre_h = nn.Conv2d(seq_len, horizon, (1, 1))

        self.end_emb = FeedForward([hidden_size, hidden_size, input_dim])
        self.end_emb_l = FeedForward([hidden_size, hidden_size, input_dim])

        self.td = time_in_day_size
        self.dw = day_in_week_size
        self.id = input_dim
        self.wt = wave_type
        self.wl = wave_levels

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        # history_data: [B, T, N, C]; channels 1,2 are normalized [tod, dow].
        x = history_data
        te = torch.cat([x[:, :, 0, 1:2] * self.td, x[:, :, 0, 2:] * self.dw], -1)
        ADD = torch.arange(te.shape[1], device=x.device).unsqueeze(0).unsqueeze(2) + 1
        TEYTOD = (te[:, -1:, 0:1] + ADD) % self.td
        TEYDOW = (
            torch.floor((te[:, -1:, 0:1] + ADD) / self.td) + te[..., 1:2]
        ) % self.dw
        te = torch.cat([te, torch.cat([TEYTOD, TEYDOW], -1)], 1)
        te = te[..., [1, 0]]

        inputs = x[..., : self.id]
        xl, xh = disentangle(inputs[..., 0:1].cpu().numpy(), self.wt, self.wl)

        xl, xh, TE = (
            self.start_emb_l(xl.to(x.device)),
            self.start_emb_h(xh.to(x.device)),
            self.te_emb(te, self.td, self.dw),
        )

        for enc in self.dual_encoder:
            xl, xh = enc(xl, xh, TE[:, : xl.shape[1], :, :])

        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)

        hat_y = self.adaptive_fusion(hat_y_l, hat_y_h, TE[:, xl.shape[1] :, :, :])
        hat_y = self.end_emb(hat_y)
        # Always return the inference prediction (B, horizon, N, input_dim); the
        # ModernTSF trainer uses a standard loss on (B, pred_len, N), so we never
        # emit the upstream training-mode concat that needs the custom STWave loss.
        return hat_y


# --------------------------------------------------------------------------- #
# ModernTSF adapter.
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """ModernTSF adapter wrapping the upstream STWave backbone.

    Parameters
    ----------
    seq_len : int
        Input sequence length (``T``).
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N`` (= ``enc_in``). Injected from the dataset.
    adj_mx : np.ndarray | None
        ``(N, N)`` adjacency matrix injected by the runner. Required for the
        spectral positional encoding + GAT neighbour sampling; a ring graph is
        used as a fallback when absent so the model is still constructible.
    input_dim : int
        Number of history channels fed to STWave (value + calendar
        ``[tod, dow]``); STWave reads channels 1,2 as normalized
        ``[time_in_day, day_in_week]`` for its temporal embedding, while the
        wavelet branch operates on the value channel 0 only.
    hidden_size : int
        Channel width. Must be ``<= num_nodes`` (it is the number of Laplacian
        eigenvectors used for the spatial positional encoding); clamped if larger.
    layers : int
        Number of dual encoder blocks.
    log_samples : int
        Sparse-attention sampling factor.
    time_in_day_size : int
        Number of samples per day (time-of-day vocabulary size).
    day_in_week_size : int
        Number of days per week (day-of-week vocabulary size).
    wave_type : str
        PyWavelets wavelet name (e.g. ``sym2``).
    wave_levels : int
        Wavelet decomposition level.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        hidden_size: int = 16,
        layers: int = 1,
        log_samples: int = 1,
        time_in_day_size: int = 24,
        day_in_week_size: int = 7,
        wave_type: str = "sym2",
        wave_levels: int = 1,
    ) -> None:
        super().__init__()
        # ``input_dim`` is the number of *history* channels we feed STWave
        # (value + calendar). The wavelet branch / start- and end-embeddings
        # operate on the single value channel only, so the upstream module's own
        # ``input_dim`` (the value dim) is fixed to 1.
        self.input_dim = max(input_dim, 3)
        self.value_dim = 1
        self.pred_len = pred_len
        self.num_nodes = num_nodes

        # hidden_size doubles as the number of Laplacian eigenvectors used for
        # the spatial positional encoding, so it cannot exceed N.
        hidden_size = min(hidden_size, num_nodes)

        if adj_mx is None:
            # Fallback ring graph so the model is constructible without a dataset
            # adjacency (each node linked to its two neighbours + self).
            adj_np = np.eye(num_nodes, dtype=np.float32)
            for i in range(num_nodes):
                adj_np[i, (i + 1) % num_nodes] = 1.0
                adj_np[i, (i - 1) % num_nodes] = 1.0
        else:
            adj_np = np.asarray(adj_mx, dtype=np.float32)

        adj_gat, graphwave = _load_graph(adj_np, hidden_size, log_samples)

        self.net = STWave(
            input_dim=self.value_dim,
            hidden_size=hidden_size,
            layers=layers,
            seq_len=seq_len,
            horizon=pred_len,
            log_samples=log_samples,
            adj_gat=adj_gat,
            graphwave=graphwave,
            time_in_day_size=time_in_day_size,
            day_in_week_size=day_in_week_size,
            wave_type=wave_type,
            wave_levels=wave_levels,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future node values.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input values of shape ``(B, seq_len, N)``.
        x_mark_enc : torch.Tensor, optional
            Node covariates ``(B, seq_len, N, F)`` or raw calendar stamps
            ``(B, seq_len, 6)``.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        # (B, L, N, 1 + F); channel 0 value, then calendar [tod, dow] / covariates.
        history = to_spatiotemporal(x_enc, x_mark_enc)
        # Keep only value + calendar features; STWave reads channels 1,2 as
        # normalized [time_in_day, day_in_week].
        history = history[..., : self.input_dim]
        if history.shape[-1] < self.input_dim:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - history.shape[-1])
            )
            history = torch.cat([history, pad], dim=-1)

        out = self.net(history, None, batch_seen=0, epoch=0, train=self.training)
        # out is (B, pred_len, N, input_dim); the value channel is channel 0.
        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
