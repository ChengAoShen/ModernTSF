"""STPGNN: Spatio-Temporal Pivotal Graph Neural Networks for traffic forecasting.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/STPGNN), Apache-2.0.

Paper: Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow
Forecasting (AAAI 2024). The model fuses a learned *adaptive* graph (built from
node embeddings) with a data-driven *pivotal* graph, and propagates over a
WaveNet-style stack of dilated temporal convolutions.

ModernTSF adaptations
---------------------
* The upstream ``end_conv_1`` had a ``in_channels`` value hardcoded for an input
  length of 12 (the BasicTS PEMS08 config). Here it is a ``nn.LazyConv2d`` whose
  input width is inferred on the first forward, so any ``seq_len`` works.
* All ``.to('cuda')`` / device assumptions were removed; internally-created
  tensors use the input tensor's device.
* The public :class:`Model` exposes plain keyword arguments and the ModernTSF
  ``forward(x_enc, x_mark_enc, ...)`` contract, returning ``(B, pred_len, N)``.

STPGNN learns its graph from node embeddings, so the injected ``adj_mx`` is
optional; it is accepted (and kept as a buffer for reference) but the forward
pass relies on the adaptive + pivotal graphs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models._external.marks import to_spatiotemporal


class nconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,nwv->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return h


class pgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2, temp=1):
        super().__init__()
        self.nconv = nconv()
        self.temp = temp
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = h[:, :, :, -h.size(3): -self.temp]
        return h


class STPGNN(nn.Module):
    def __init__(
        self,
        num_nodes,
        dropout,
        topk,
        out_dim,
        residual_channels,
        dilation_channels,
        end_channels,
        kernel_size,
        blocks,
        layers,
        days,
        time_of_day_size,
        dims,
        order,
        in_dim,
        normalization,
        seq_len,
    ):
        super().__init__()
        skip_channels = 8
        self.alpha = nn.Parameter(torch.tensor(-5.0))
        self.topk = topk
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.time_of_day_size = time_of_day_size
        self.days = days

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.normal = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.normal_a = nn.ModuleList()
        self.pgconv = nn.ModuleList()

        self.start_conv_a = nn.Conv2d(
            in_channels=in_dim, out_channels=1, kernel_size=(1, 1)
        )
        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )

        receptive_field = 1
        self.supports_len = 1
        self.nodevec_p1 = nn.Parameter(torch.randn(days, dims), requires_grad=True)
        self.nodevec_p2 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_p3 = nn.Parameter(torch.randn(num_nodes, dims), requires_grad=True)
        self.nodevec_pk = nn.Parameter(
            torch.randn(dims, dims, dims), requires_grad=True
        )

        # The temporal width entering each layer's LayerNorm depends on the
        # (padded) input length; track it so the optional "layer" normalisation
        # is sized correctly for any seq_len.
        in_len = max(seq_len, 1) + 1  # +1 for the forward-time left pad
        cur_len = in_len
        for _b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.residual_convs_a.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                self.pgconv.append(
                    pgcn(
                        dilation_channels,
                        residual_channels,
                        dropout,
                        support_len=self.supports_len,
                        order=order,
                        temp=new_dilation,
                    )
                )
                self.gconv.append(
                    gcn(
                        dilation_channels,
                        residual_channels,
                        dropout,
                        support_len=self.supports_len,
                        order=order,
                    )
                )
                # Temporal width after this layer's gated dilated conv.
                cur_len = cur_len - new_dilation * (kernel_size - 1)
                norm_len = max(cur_len, 1)
                if normalization == "batch":
                    self.normal.append(nn.BatchNorm2d(residual_channels))
                    self.normal_a.append(nn.BatchNorm2d(residual_channels))
                elif normalization == "layer":
                    self.normal.append(
                        nn.LayerNorm([residual_channels, num_nodes, norm_len])
                    )
                    self.normal_a.append(
                        nn.LayerNorm([residual_channels, num_nodes, norm_len])
                    )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.relu = nn.ReLU(inplace=True)

        # Upstream hardcoded ``skip_channels * (12+10+9+7+6+4+3+1)`` for input
        # length 12. The concatenated skip width varies with seq_len, so defer
        # to a lazy conv that infers it on the first forward.
        self.end_conv_1 = nn.LazyConv2d(
            out_channels=end_channels, kernel_size=(1, 1), bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.receptive_field = receptive_field

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum("ai, ijk->ajk", time_embedding, core_embedding)
        adp = torch.einsum("bj, ajk->abk", source_embedding, adp)
        adp = torch.einsum("ck, abk->abc", target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        return adp

    def pivotalconstruct(self, x, adj, k):
        x = x.squeeze(1)
        x = x.sum(dim=0)
        y = x.sum(dim=1).unsqueeze(0)
        adjp = torch.einsum("ij, jk->ik", x[:, :-1], x.transpose(0, 1)[1:, :]) / y
        adjp = adjp * adj
        score = adjp.sum(dim=0) + adjp.sum(dim=1)
        N = x.size(0)
        k = min(k, N)
        _, topk_indices = torch.topk(score, k)
        mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        mask[topk_indices] = True
        masked_matrix = adjp * mask.unsqueeze(1) * mask.unsqueeze(0)
        adjp = F.softmax(F.relu(masked_matrix), dim=1)
        return adjp.unsqueeze(0)

    def forward(self, history_data, **kwargs):
        # [B, T, N, F] -> [B, F, N, T]
        inputs = history_data[..., [0]].permute(0, 3, 2, 1).contiguous()
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))
        ind = (
            (history_data[:, -1, 0, 1] * self.time_of_day_size).long()
        ) % self.days

        in_len = inputs.size(3)
        num_nodes = inputs.size(2)
        if in_len < self.receptive_field:
            xo = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        else:
            xo = inputs
        x = self.start_conv(xo[:, [0]])
        x_a = self.start_conv_a(xo[:, [0]])
        skip = 0
        adj = self.dgconstruct(
            self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk
        )
        pivweight = torch.randn(num_nodes, num_nodes, device=x.device)
        adj_p = self.pivotalconstruct(x_a, pivweight, self.topk)
        supports = [adj]
        supports_a = [adj_p]

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x_a = self.pgconv[i](residual, supports_a)
            x = self.gconv[i](x, supports)
            alpha_sigmoid = torch.sigmoid(self.alpha)
            x = alpha_sigmoid * x_a + (1 - alpha_sigmoid) * x
            x = x + residual[:, :, :, -x.size(3):]
            s = x
            s = self.skip_convs[i](s)
            if isinstance(skip, int):  # B F N T
                skip = (
                    s.transpose(2, 3)
                    .reshape([s.shape[0], -1, s.shape[2], 1])
                    .contiguous()
                )
            else:
                skip = torch.cat(
                    [
                        s.transpose(2, 3).reshape([s.shape[0], -1, s.shape[2], 1]),
                        skip,
                    ],
                    dim=1,
                ).contiguous()
            x = self.normal[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x  # (B, out_dim, N, 1)


class Model(nn.Module):
    """ModernTSF adapter wrapping the upstream STPGNN architecture.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon (mapped to the upstream ``out_dim``).
    num_nodes : int
        Number of spatial nodes ``N`` (injected from the dataset; falls back to
        ``enc_in``).
    adj_mx : np.ndarray, optional
        ``(N, N)`` adjacency injected by the runner. STPGNN learns its graph
        from node embeddings, so this is kept only as a reference buffer.
    input_dim : int
        Number of spatiotemporal input channels assembled by
        :func:`to_spatiotemporal` (value + calendar covariates). Channel 0 is
        the value; the upstream model consumes channel 0 and reads channel 1 as
        the normalised time-of-day index.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        dropout: float = 0.1,
        topk: int = 4,
        residual_channels: int = 16,
        dilation_channels: int = 16,
        end_channels: int = 64,
        kernel_size: int = 2,
        blocks: int = 2,
        layers: int = 2,
        days: int = 7,
        time_of_day_size: int = 24,
        dims: int = 16,
        order: int = 2,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len

        if adj_mx is not None:
            adj_t = torch.as_tensor(np.asarray(adj_mx), dtype=torch.float32)
            self.register_buffer("adj_mx", adj_t, persistent=False)
        else:
            self.adj_mx = None

        self.net = STPGNN(
            num_nodes=num_nodes,
            dropout=dropout,
            topk=topk,
            out_dim=pred_len,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            end_channels=end_channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
            days=days,
            time_of_day_size=time_of_day_size,
            dims=dims,
            order=order,
            in_dim=1,
            normalization=normalization,
            seq_len=seq_len,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future values.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input values ``(B, seq_len, N)``.
        x_mark_enc : torch.Tensor, optional
            Node-structured covariates ``(B, seq_len, N, F)`` or raw calendar
            stamps ``(B, seq_len, 6)``.

        Returns
        -------
        torch.Tensor
            Forecast ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 1 + F)
        out = self.net(history, batch_seen=0, epoch=0, train=self.training)
        # out: (B, pred_len, N, 1)
        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
