"""ModernTSF adapter for the STGCN spatiotemporal graph forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/STGCN), Apache-2.0.

STGCN (Spatio-Temporal Graph Convolutional Networks, IJCAI 2018) stacks gated
temporal convolutions with Chebyshev graph convolutions. It REQUIRES a
predefined graph: the spatial conv multiplies by a graph shift operator (GSO),
here the symmetric normalized Laplacian built from the injected ``(N, N)``
``adj_mx`` (matching BasicTS' ``normlap`` setting). The upstream arch keeps the
BasicTS signature ``forward(history_data, future_data, batch_seen, epoch,
train, **kwargs)`` with ``history_data`` shaped ``(B, L, N, C)`` and returns
``(B, pred_len, N, 1)``.

This adapter:
  * converts ModernTSF's ``(x_enc, x_mark_enc)`` into ``(B, L, N, input_dim)``
    via :func:`models._external.marks.to_spatiotemporal` (channel 0 the value,
    then calendar ``[time_in_day, day_in_week]``),
  * builds the GSO from ``adj_mx`` and registers it as a device-following
    buffer (no hardcoded CUDA),
  * drives the upstream module with the BasicTS signature and squeezes the
    output channel back to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models._external.marks import to_spatiotemporal


# --------------------------------------------------------------------------- #
# Graph shift operator (GSO) construction.
# --------------------------------------------------------------------------- #
def _symmetric_normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute ``L = I - D^{-1/2} A D^{-1/2}`` (symmetric normalized Laplacian).

    Mirrors BasicTS ``calculate_symmetric_normalized_laplacian`` / the
    ``normlap`` adjacency used by the STGCN baseline configs.

    Parameters
    ----------
    adj : np.ndarray
        ``(N, N)`` adjacency matrix.

    Returns
    -------
    np.ndarray
        ``(N, N)`` symmetric normalized Laplacian (dense, float32).
    """
    adj = np.asarray(adj, dtype=np.float64)
    n = adj.shape[0]
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat = np.diag(d_inv_sqrt)
    laplacian = np.eye(n) - d_mat @ adj @ d_mat
    return laplacian.astype(np.float32)


# --------------------------------------------------------------------------- #
# Vendored STGCN layers (adapted from BasicTS baselines/STGCN/arch).
# Hardcoded ``.to('cuda')`` calls are removed; tensors created internally use
# the input tensor's device via ``.to(x)`` / the GSO buffer.
# --------------------------------------------------------------------------- #
class Align(nn.Module):
    """Channel alignment via 1x1 conv (or zero-padding when widening)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            pad = torch.zeros(
                [batch_size, self.c_out - self.c_in, timestep, n_vertex]
            ).to(x)
            x = torch.cat([x, pad], dim=1)
        return x


class CausalConv2d(nn.Conv2d):
    """Causal 2-D convolution (left/top padding only)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        enable_padding: bool = False,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding:
            self.__padding = [
                int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))
            ]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.__padding != 0:
            x = F.pad(x, (self.left_padding[1], 0, self.left_padding[0], 0))
        return super().forward(x)


class TemporalConvLayer(nn.Module):
    """Gated temporal convolution layer (GLU / GTU / relu / ...)."""

    def __init__(self, Kt: int, c_in: int, c_out: int, n_vertex: int, act_func: str) -> None:
        super().__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        if act_func in ("glu", "gtu"):
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=2 * c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1,
            )
        else:
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1,
            )
        self.act_func = act_func
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.align(x)[:, :, self.Kt - 1 :, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func in ("glu", "gtu"):
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out :, :, :]
            if self.act_func == "glu":
                x = torch.mul((x_p + x_in), self.sigmoid(x_q))
            else:
                x = torch.mul(self.tanh(x_p + x_in), self.sigmoid(x_q))
        elif self.act_func == "relu":
            x = self.relu(x_causal_conv + x_in)
        elif self.act_func == "leaky_relu":
            x = self.leaky_relu(x_causal_conv + x_in)
        elif self.act_func == "silu":
            x = self.silu(x_causal_conv + x_in)
        else:
            raise NotImplementedError(
                f"ERROR: The activation function {self.act_func} is not implemented."
            )
        return x


class ChebGraphConv(nn.Module):
    """Chebyshev graph convolution using a precomputed GSO."""

    def __init__(self, c_in: int, c_out: int, Ks: int, gso: torch.Tensor, bias: bool) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, (0, 2, 3, 1))
        gso = self.gso.to(x.device)

        if self.Ks - 1 < 0:
            raise ValueError(
                f"ERROR: the graph convolution kernel size Ks has to be a positive "
                f"integer, but received {self.Ks}."
            )
        if self.Ks - 1 == 0:
            x_list = [x]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum("hi,btij->bthj", gso, x)
            x_list = [x_0, x_1]
        else:
            x_0 = x
            x_1 = torch.einsum("hi,btij->bthj", gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(
                    torch.einsum("hi,btij->bthj", 2 * gso, x_list[k - 1]) - x_list[k - 2]
                )

        x = torch.stack(x_list, dim=2)
        cheb_graph_conv = torch.einsum("btkhi,kij->bthj", x, self.weight)
        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        return cheb_graph_conv


class GraphConv(nn.Module):
    """First-order GCN graph convolution using a precomputed GSO."""

    def __init__(self, c_in: int, c_out: int, gso: torch.Tensor, bias: bool) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.permute(x, (0, 2, 3, 1))
        gso = self.gso.to(x.device)
        first_mul = torch.einsum("hi,btij->bthj", gso, x)
        second_mul = torch.einsum("bthi,ij->bthj", first_mul, self.weight)
        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul
        return graph_conv


class GraphConvLayer(nn.Module):
    """Graph convolution layer with residual alignment."""

    def __init__(
        self,
        graph_conv_type: str,
        c_in: int,
        c_out: int,
        Ks: int,
        gso: torch.Tensor,
        bias: bool,
    ) -> None:
        super().__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        if self.graph_conv_type == "cheb_graph_conv":
            self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
        elif self.graph_conv_type == "graph_conv":
            self.graph_conv = GraphConv(c_out, c_out, gso, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gc_in = self.align(x)
        if self.graph_conv_type == "cheb_graph_conv":
            x_gc = self.cheb_graph_conv(x_gc_in)
        elif self.graph_conv_type == "graph_conv":
            x_gc = self.graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)
        return x_gc_out


class STConvBlock(nn.Module):
    """Spatio-temporal conv block: T-G-T-N-D."""

    def __init__(
        self,
        Kt: int,
        Ks: int,
        n_vertex: int,
        last_block_channel: int,
        channels,
        act_func: str,
        graph_conv_type: str,
        gso: torch.Tensor,
        bias: bool,
        droprate: float,
    ) -> None:
        super().__init__()
        self.tmp_conv1 = TemporalConvLayer(
            Kt, last_block_channel, channels[0], n_vertex, act_func
        )
        self.graph_conv = GraphConvLayer(
            graph_conv_type, channels[0], channels[1], Ks, gso, bias
        )
        self.tmp_conv2 = TemporalConvLayer(
            Kt, channels[1], channels[2], n_vertex, act_func
        )
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x


class OutputBlock(nn.Module):
    """Output block: T-N-F-F."""

    def __init__(
        self,
        Ko: int,
        last_block_channel: int,
        channels,
        end_channel: int,
        n_vertex: int,
        act_func: str,
        bias: bool,
        droprate: float,
    ) -> None:
        super().__init__()
        self.tmp_conv1 = TemporalConvLayer(
            Ko, last_block_channel, channels[0], n_vertex, act_func
        )
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x


class STGCNChebGraphConv(nn.Module):
    """Upstream STGCN backbone (Chebyshev / GCN spatial conv).

    Contains a ``TGTND TGTND TNFF`` structure. ``forward`` keeps the BasicTS
    signature.
    """

    def __init__(
        self,
        Kt: int,
        Ks: int,
        blocks,
        T: int,
        n_vertex: int,
        act_func: str,
        graph_conv_type: str,
        gso: torch.Tensor,
        bias: bool,
        droprate: float,
    ) -> None:
        super().__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                STConvBlock(
                    Kt,
                    Ks,
                    n_vertex,
                    blocks[l][-1],
                    blocks[l + 1],
                    act_func,
                    graph_conv_type,
                    gso,
                    bias,
                    droprate,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = T - (len(blocks) - 3) * 2 * (Kt - 1)
        self.Ko = Ko
        assert Ko != 0, "Ko = 0."
        self.output = OutputBlock(
            Ko,
            blocks[-3][-1],
            blocks[-2],
            blocks[-1][0],
            n_vertex,
            act_func,
            bias,
            droprate,
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        history_data : torch.Tensor
            Historical data of shape ``(B, L, N, C)``.

        Returns
        -------
        torch.Tensor
            Prediction of shape ``(B, pred_len, N, 1)``.
        """
        x = history_data.permute(0, 3, 1, 2).contiguous()  # (B, C, L, N)
        x = self.st_blocks(x)
        x = self.output(x)  # (B, pred_len, 1, N)
        x = x.transpose(2, 3)  # (B, pred_len, N, 1)
        return x


# --------------------------------------------------------------------------- #
# ModernTSF adapter.
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """ModernTSF adapter wrapping the upstream STGCN backbone.

    Parameters
    ----------
    seq_len : int
        Input sequence length (``T``).
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N`` (= ``enc_in``). Injected from the dataset.
    adj_mx : np.ndarray | None
        ``(N, N)`` adjacency matrix injected by the runner from the dataset.
        Required for STGCN; an identity graph is used as a fallback when absent.
    input_dim : int
        Number of input channels fed to STGCN (value + calendar covariates).
    Kt : int
        Temporal kernel size.
    Ks : int
        Chebyshev order (spatial kernel size).
    hidden_dim : int
        Channel width of the ST blocks.
    bottleneck_dim : int
        Bottleneck width inside each ST block.
    out_hidden_dim : int
        Hidden width of the output block.
    act_func : str
        Gated temporal activation (``glu`` / ``gtu`` / ...).
    graph_conv_type : str
        ``cheb_graph_conv`` or ``graph_conv``.
    bias : bool
        Whether conv / linear layers carry bias.
    droprate : float
        Dropout probability inside the ST blocks.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        Kt: int = 3,
        Ks: int = 3,
        hidden_dim: int = 64,
        bottleneck_dim: int = 16,
        out_hidden_dim: int = 128,
        act_func: str = "glu",
        graph_conv_type: str = "cheb_graph_conv",
        bias: bool = True,
        droprate: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        self.num_nodes = num_nodes

        # Build the graph shift operator (symmetric normalized Laplacian) from
        # the injected adjacency. Falls back to an identity graph when none is
        # supplied so the model is still constructible.
        if adj_mx is None:
            adj_np = np.eye(num_nodes, dtype=np.float32)
        else:
            adj_np = np.asarray(adj_mx, dtype=np.float32)
        gso_np = _symmetric_normalized_laplacian(adj_np)
        gso = torch.from_numpy(gso_np)
        # Register as a buffer so it follows the model's device and is saved
        # with the state dict; ChebGraphConv reads ``self.gso``.
        self.register_buffer("gso", gso)

        # STGCN block plan:  [input] [st1] [st2] [out] [horizon]
        # Two ST blocks consume 2 * 2 * (Kt - 1) temporal steps; the output
        # block consumes Ko more. Validate the budget up front.
        n_st_blocks = 2
        ko = seq_len - n_st_blocks * 2 * (Kt - 1)
        if ko <= 0:
            raise ValueError(
                f"STGCN: seq_len={seq_len} too short for Kt={Kt} with "
                f"{n_st_blocks} ST blocks (Ko={ko}). Increase seq_len or "
                f"decrease Kt."
            )

        blocks = [
            [input_dim],
            [hidden_dim, bottleneck_dim, hidden_dim],
            [hidden_dim, bottleneck_dim, hidden_dim],
            [out_hidden_dim, out_hidden_dim],
            [pred_len],
        ]

        self.net = STGCNChebGraphConv(
            Kt=Kt,
            Ks=Ks,
            blocks=blocks,
            T=seq_len,
            n_vertex=num_nodes,
            act_func=act_func,
            graph_conv_type=graph_conv_type,
            gso=self.gso,
            bias=bias,
            droprate=droprate,
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
        # Keep only the first ``input_dim`` channels (value + calendar features).
        history = history[..., : self.input_dim]
        if history.shape[-1] < self.input_dim:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - history.shape[-1])
            )
            history = torch.cat([history, pad], dim=-1)

        out = self.net(history, None, batch_seen=0, epoch=0, train=self.training)
        # out is (B, pred_len, N, 1).
        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
