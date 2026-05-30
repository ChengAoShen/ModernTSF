"""Vendored D2STGNN architecture.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/D2STGNN), Apache-2.0.

The original BasicTS implementation spreads the architecture across several
modules (``decouple``, ``difusion_block``, ``inherent_block``,
``dynamic_graph_conv``). Here every layer / utility is consolidated into a
single module with the relative imports flattened and all hardcoded CUDA
device usage removed (created tensors follow the input device). Functional
behaviour is otherwise unchanged.
"""

from __future__ import annotations

import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


def remove_nan_inf(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN / Inf entries of ``tensor`` with zeros."""
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


# ---------------------------------------------------------------------------
# decouple/residual_decomp.py
# ---------------------------------------------------------------------------
class ResidualDecomp(nn.Module):
    """Residual decomposition."""

    def __init__(self, input_shape):
        super().__init__()
        self.ln = nn.LayerNorm(input_shape[-1])
        self.ac = nn.ReLU()

    def forward(self, x, y):
        u = x - self.ac(y)
        u = self.ln(u)
        return u


# ---------------------------------------------------------------------------
# decouple/estimation_gate.py
# ---------------------------------------------------------------------------
class EstimationGate(nn.Module):
    """The spatial gate module."""

    def __init__(self, node_emb_dim, time_emb_dim, hidden_dim, input_seq_len):
        super().__init__()
        self.FC1 = nn.Linear(2 * node_emb_dim + time_emb_dim * 2, hidden_dim)
        self.act = nn.ReLU()
        self.FC2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_embedding1, node_embedding2, T_D, D_W, X):
        B, L, N, D = T_D.shape
        spatial_gate_feat = torch.cat(
            [
                T_D,
                D_W,
                node_embedding1.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
                node_embedding2.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1),
            ],
            dim=-1,
        )
        hidden = self.FC1(spatial_gate_feat)
        hidden = self.act(hidden)
        spatial_gate = torch.sigmoid(self.FC2(hidden))[:, -X.shape[1]:, :, :]
        X = X * spatial_gate
        return X


# ---------------------------------------------------------------------------
# dynamic_graph_conv/utils/mask.py
# ---------------------------------------------------------------------------
class Mask(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.mask = model_args["adjs"]

    def _mask(self, index, adj):
        mask = self.mask[index].to(adj.device) + torch.ones_like(
            self.mask[index].to(adj.device)
        ) * 1e-7
        return mask * adj

    def forward(self, adj):
        result = []
        for index, _ in enumerate(adj):
            result.append(self._mask(index, _))
        return result


# ---------------------------------------------------------------------------
# dynamic_graph_conv/utils/normalizer.py
# ---------------------------------------------------------------------------
class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree = torch.sum(graph, dim=2)
        degree = remove_nan_inf(1 / degree)
        degree = torch.diag_embed(degree)
        P = torch.bmm(degree, graph)
        return P

    def forward(self, adj):
        return [self._norm(_) for _ in adj]


class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order = order

    def _multi_order(self, graph):
        graph_ordered = []
        k_1_order = graph  # 1 order
        mask = torch.eye(graph.shape[1], device=graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask)
        for _k in range(2, self.order + 1):
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]


# ---------------------------------------------------------------------------
# dynamic_graph_conv/utils/distance.py
# ---------------------------------------------------------------------------
class DistanceFunction(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.hidden_dim = model_args["num_hidden"]
        self.node_dim = model_args["node_hidden"]
        self.time_slot_emb_dim = self.hidden_dim
        self.input_seq_len = model_args["seq_length"]
        self.dropout = nn.Dropout(model_args["dropout"])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        self.time_slot_embedding = nn.Linear(
            model_args["time_emb_dim"], self.time_slot_emb_dim
        )
        self.all_feat_dim = (
            self.ts_feat_dim + self.node_dim + model_args["time_emb_dim"] * 2
        )
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

    def forward(self, X, E_d, E_u, T_D, D_W):
        T_D = T_D[:, -1, :, :]
        D_W = D_W[:, -1, :, :]
        X = X[:, :, :, 0].transpose(1, 2).contiguous()
        [batch_size, num_nodes, seq_len] = X.shape
        X = X.view(batch_size * num_nodes, seq_len)
        dy_feat = self.fc_ts_emb2(
            self.dropout(self.bn(F.relu(self.fc_ts_emb1(X))))
        )
        dy_feat = dy_feat.view(batch_size, num_nodes, -1)
        emb1 = E_d.unsqueeze(0).expand(batch_size, -1, -1)
        emb2 = E_u.unsqueeze(0).expand(batch_size, -1, -1)
        X1 = torch.cat([dy_feat, T_D, D_W, emb1], dim=-1)
        X2 = torch.cat([dy_feat, T_D, D_W, emb2], dim=-1)
        X = [X1, X2]
        adjacent_list = []
        for _ in X:
            Q = self.WQ(_)
            K = self.WK(_)
            QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            W = torch.softmax(QKT, dim=-1)
            adjacent_list.append(W)
        return adjacent_list


# ---------------------------------------------------------------------------
# dynamic_graph_conv/dy_graph_conv.py
# ---------------------------------------------------------------------------
class DynamicGraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.k_s = model_args["k_s"]
        self.k_t = model_args["k_t"]
        self.hidden_dim = model_args["num_hidden"]
        self.node_dim = model_args["node_hidden"]

        self.distance_function = DistanceFunction(**model_args)
        self.mask = Mask(**model_args)
        self.normalizer = Normalizer()
        self.multi_order = MultiOrder(order=self.k_s)

    def st_localization(self, graph_ordered):
        st_local_graph = []
        for modality_i in graph_ordered:
            for k_order_graph in modality_i:
                k_order_graph = k_order_graph.unsqueeze(-2).expand(
                    -1, -1, self.k_t, -1
                )
                k_order_graph = k_order_graph.reshape(
                    k_order_graph.shape[0],
                    k_order_graph.shape[1],
                    k_order_graph.shape[2] * k_order_graph.shape[3],
                )
                st_local_graph.append(k_order_graph)
        return st_local_graph

    def forward(self, **inputs):
        X = inputs["X"]
        E_d = inputs["E_d"]
        E_u = inputs["E_u"]
        T_D = inputs["T_D"]
        D_W = inputs["D_W"]
        dist_mx = self.distance_function(X, E_d, E_u, T_D, D_W)
        dist_mx = self.mask(dist_mx)
        dist_mx = self.normalizer(dist_mx)
        mul_mx = self.multi_order(dist_mx)
        dynamic_graphs = self.st_localization(mul_mx)
        return dynamic_graphs


# ---------------------------------------------------------------------------
# difusion_block/dif_model.py
# ---------------------------------------------------------------------------
class STLocalizedConv(nn.Module):
    def __init__(
        self,
        hidden_dim,
        pre_defined_graph=None,
        use_pre=None,
        dy_graph=None,
        sta_graph=None,
        **model_args,
    ):
        super().__init__()
        self.k_s = model_args["k_s"]
        self.k_t = model_args["k_t"]
        self.hidden_dim = hidden_dim

        self.pre_defined_graph = pre_defined_graph
        self.use_predefined_graph = use_pre
        self.use_dynamic_hidden_graph = dy_graph
        self.use_static__hidden_graph = sta_graph

        self.support_len = (
            len(self.pre_defined_graph) + int(dy_graph) + int(sta_graph)
        )
        self.num_matric = (
            int(use_pre) * len(self.pre_defined_graph)
            + len(self.pre_defined_graph) * int(dy_graph)
            + int(sta_graph)
        ) * self.k_s + 1
        self.dropout = nn.Dropout(model_args["dropout"])
        self.pre_defined_graph = self.get_graph(self.pre_defined_graph)

        self.fc_list_updt = nn.Linear(
            self.k_t * hidden_dim, self.k_t * hidden_dim, bias=False
        )
        self.gcn_updt = nn.Linear(
            self.hidden_dim * self.num_matric, self.hidden_dim
        )

        self.bn = nn.BatchNorm2d(self.hidden_dim)
        self.activation = nn.ReLU()

    def gconv(self, support, X_k, X_0):
        out = [X_0]
        for graph in support:
            if len(graph.shape) == 2:  # static or predefined graph
                pass
            else:
                graph = graph.unsqueeze(1)
            H_k = torch.matmul(graph, X_k)
            out.append(H_k)
        out = torch.cat(out, dim=-1)
        out = self.gcn_updt(out)
        out = self.dropout(out)
        return out

    def get_graph(self, support):
        # Used for static (incl. static hidden) and predefined graphs only.
        graph_ordered = []
        if len(support) == 0:
            return []
        mask = 1 - torch.eye(support[0].shape[0], device=support[0].device)
        for graph in support:
            k_1_order = graph  # 1 order
            graph_ordered.append(k_1_order * mask)
            for _k in range(2, self.k_s + 1):
                k_1_order = torch.matmul(graph, k_1_order)
                graph_ordered.append(k_1_order * mask)
        st_local_graph = []
        for graph in graph_ordered:
            graph = graph.unsqueeze(-2).expand(-1, self.k_t, -1)
            graph = graph.reshape(graph.shape[0], graph.shape[1] * graph.shape[2])
            st_local_graph.append(graph)
        return st_local_graph

    def forward(self, X, dynamic_graph, static_graph):
        # X: [bs, seq, nodes, feat]
        X = X.unfold(1, self.k_t, 1).permute(0, 1, 2, 4, 3)
        batch_size, seq_len, num_nodes, kernel_size, num_feat = X.shape

        support = []
        if self.use_predefined_graph:
            support = support + self.pre_defined_graph
        if self.use_dynamic_hidden_graph:
            support = support + dynamic_graph
        if self.use_static__hidden_graph:
            support = support + self.get_graph(static_graph)

        X = X.reshape(batch_size, seq_len, num_nodes, kernel_size * num_feat)
        out = self.fc_list_updt(X)
        out = self.activation(out)
        out = out.view(batch_size, seq_len, num_nodes, kernel_size, num_feat)
        X_0 = torch.mean(out, dim=-2)
        X_k = out.transpose(-3, -2).reshape(
            batch_size, seq_len, kernel_size * num_nodes, num_feat
        )
        hidden = self.gconv(support, X_k, X_0)
        return hidden


# ---------------------------------------------------------------------------
# difusion_block/forecast.py
# ---------------------------------------------------------------------------
class DifForecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args["k_t"]
        self.output_seq_len = model_args["seq_length"]
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)
        self.model_args = model_args

    def forward(self, X, H, st_l_conv, dynamic_graph, static_graph):
        [B, seq_len_remain, N, D] = H.shape
        [B, seq_len_input, N, D] = X.shape

        predict = []
        history = X
        predict.append(H[:, -1, :, :].unsqueeze(1))
        for _ in range(int(self.output_seq_len / self.model_args["gap"]) - 1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            predict.append(st_l_conv(_1, dynamic_graph, static_graph))
        predict = torch.cat(predict, dim=1)
        predict = self.forecast_fc(predict)
        return predict


# ---------------------------------------------------------------------------
# difusion_block/dif_block.py
# ---------------------------------------------------------------------------
class DifBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        fk_dim=256,
        use_pre=None,
        dy_graph=None,
        sta_graph=None,
        **model_args,
    ):
        super().__init__()
        self.pre_defined_graph = model_args["adjs"]
        self.localized_st_conv = STLocalizedConv(
            hidden_dim,
            pre_defined_graph=self.pre_defined_graph,
            use_pre=use_pre,
            dy_graph=dy_graph,
            sta_graph=sta_graph,
            **model_args,
        )
        self.residual_decompose = ResidualDecomp([-1, -1, -1, hidden_dim])
        self.forecast_branch = DifForecast(hidden_dim, fk_dim=fk_dim, **model_args)
        self.backcast_branch = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, X_spa, dynamic_graph, static_graph):
        Z = self.localized_st_conv(X_spa, dynamic_graph, static_graph)
        forecast_hidden = self.forecast_branch(
            X_spa, Z, self.localized_st_conv, dynamic_graph, static_graph
        )
        backcast_seq = self.backcast_branch(Z)
        X = X[:, -backcast_seq.shape[1]:, :, :]
        backcast_seq_res = self.residual_decompose(X, backcast_seq)
        return backcast_seq_res, forecast_hidden


# ---------------------------------------------------------------------------
# inherent_block/inh_model.py
# ---------------------------------------------------------------------------
class RNNLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        [batch_size, seq_len, num_nodes, hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim
        )
        hx = th.zeros_like(X[:, 0, :])
        output = []
        for _ in range(X.shape[1]):
            hx = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = th.stack(output, dim=0)
        output = self.dropout(output)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, bias=bias
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, K, V):
        Z = self.multi_head_self_attention(X, K, V)[0]
        Z = self.dropout(Z)
        return Z


# ---------------------------------------------------------------------------
# inherent_block/forecast.py
# ---------------------------------------------------------------------------
class InhForecast(nn.Module):
    def __init__(self, hidden_dim, fk_dim, **model_args):
        super().__init__()
        self.output_seq_len = model_args["seq_length"]
        self.model_args = model_args
        self.forecast_fc = nn.Linear(hidden_dim, fk_dim)

    def forward(self, X, RNN_H, Z, transformer_layer, rnn_layer, pe):
        [B, L, N, D] = X.shape
        [L, B_N, D] = RNN_H.shape
        [L, B_N, D] = Z.shape

        predict = [Z[-1, :, :].unsqueeze(0)]
        for _ in range(int(self.output_seq_len / self.model_args["gap"]) - 1):
            _gru = rnn_layer.gru_cell(predict[-1][0], RNN_H[-1]).unsqueeze(0)
            RNN_H = torch.cat([RNN_H, _gru], dim=0)
            if pe is not None:
                RNN_H = pe(RNN_H)
            _Z = transformer_layer(_gru, K=RNN_H, V=RNN_H)
            predict.append(_Z)

        predict = torch.cat(predict, dim=0)
        predict = predict.reshape(-1, B, N, D)
        predict = predict.transpose(0, 1)
        predict = self.forecast_fc(predict)
        return predict


# ---------------------------------------------------------------------------
# inherent_block/inh_block.py
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=None, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, X):
        X = X + self.pe[: X.size(0)]
        X = self.dropout(X)
        return X


class InhBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=4,
        bias=True,
        fk_dim=256,
        first=None,
        **model_args,
    ):
        super().__init__()
        self.num_feat = hidden_dim
        self.hidden_dim = hidden_dim

        if first:
            self.pos_encoder = PositionalEncoding(
                hidden_dim, model_args["dropout"]
            )
        else:
            self.pos_encoder = None
        self.rnn_layer = RNNLayer(hidden_dim, model_args["dropout"])
        self.transformer_layer = TransformerLayer(
            hidden_dim, num_heads, model_args["dropout"], bias
        )
        self.forecast_block = InhForecast(hidden_dim, fk_dim, **model_args)
        self.backcast_fc = nn.Linear(hidden_dim, hidden_dim)
        self.sub_and_norm = ResidualDecomp([-1, -1, -1, hidden_dim])

    def forward(self, X):
        [batch_size, seq_len, num_nodes, num_feat] = X.shape
        RNN_H_raw = self.rnn_layer(X)
        if self.pos_encoder is not None:
            RNN_H = self.pos_encoder(RNN_H_raw)
        else:
            RNN_H = RNN_H_raw
        Z = self.transformer_layer(RNN_H, RNN_H, RNN_H)

        forecast_hidden = self.forecast_block(
            X,
            RNN_H_raw,
            Z,
            self.transformer_layer,
            self.rnn_layer,
            self.pos_encoder,
        )

        Z = Z.reshape(seq_len, batch_size, num_nodes, num_feat)
        Z = Z.transpose(0, 1)
        backcast_seq = self.backcast_fc(Z)
        backcast_seq_res = self.sub_and_norm(X, backcast_seq)
        return backcast_seq_res, forecast_hidden


# ---------------------------------------------------------------------------
# d2stgnn_arch.py
# ---------------------------------------------------------------------------
class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, first=False, **model_args):
        super().__init__()
        self.spatial_gate = EstimationGate(
            model_args["node_hidden"],
            model_args["time_emb_dim"],
            64,
            model_args["seq_length"],
        )
        self.dif_layer = DifBlock(hidden_dim, fk_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(
            hidden_dim, fk_dim=fk_dim, first=first, **model_args
        )

    def forward(self, X, dynamic_graph, static_graph, E_u, E_d, T_D, D_W):
        X_spa = self.spatial_gate(E_u, E_d, T_D, D_W, X)
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            X=X, X_spa=X_spa, dynamic_graph=dynamic_graph, static_graph=static_graph
        )
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(
            dif_backcast_seq_res
        )
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    """Decoupled Dynamic Spatial-Temporal Graph Neural Network (VLDB 2022)."""

    def __init__(self, **model_args):
        super().__init__()
        self._in_feat = model_args["num_feat"]
        self._hidden_dim = model_args["num_hidden"]
        self._node_dim = model_args["node_hidden"]
        self._forecast_dim = model_args.get("forecast_dim", 256)
        self._output_hidden = model_args.get("output_hidden", 512)
        self._output_dim = model_args["seq_length"]

        self._num_nodes = model_args["num_nodes"]
        self._k_s = model_args["k_s"]
        self._k_t = model_args["k_t"]
        self._num_layers = model_args.get("num_layers", 5)
        self._time_in_day_size = model_args["time_in_day_size"]
        self._day_in_week_size = model_args["day_in_week_size"]

        model_args["use_pre"] = False
        model_args["dy_graph"] = True
        model_args["sta_graph"] = True

        self._model_args = model_args

        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)

        self.T_i_D_emb = nn.Parameter(
            torch.empty(self._time_in_day_size, model_args["time_emb_dim"])
        )
        self.D_i_W_emb = nn.Parameter(
            torch.empty(self._day_in_week_size, model_args["time_emb_dim"])
        )

        self.layers = nn.ModuleList(
            [
                DecoupleLayer(
                    self._hidden_dim,
                    fk_dim=self._forecast_dim,
                    first=True,
                    **model_args,
                )
            ]
        )
        for _ in range(self._num_layers - 1):
            self.layers.append(
                DecoupleLayer(
                    self._hidden_dim, fk_dim=self._forecast_dim, **model_args
                )
            )

        if model_args["dy_graph"]:
            self.dynamic_graph_constructor = DynamicGraphConstructor(**model_args)

        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim)
        )
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim)
        )

        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args["gap"])

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs["E_d"]
        E_u = inputs["E_u"]
        if self._model_args["sta_graph"]:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args["dy_graph"]:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _prepare_inputs(self, X):
        num_feat = self._model_args["num_feat"]
        node_emb_u = self.node_emb_u
        node_emb_d = self.node_emb_d
        # time-of-day / day-of-week features are normalized to [0, 1); scale by
        # the vocabulary size to obtain integer embedding indices.
        T_i_D = self.T_i_D_emb[
            (X[:, :, :, num_feat] * self._time_in_day_size).long()
        ]
        D_i_W = self.D_i_W_emb[
            (X[:, :, :, num_feat + 1] * self._day_in_week_size).long()
        ]
        X = X[:, :, :, :num_feat]
        return X, node_emb_u, node_emb_d, T_i_D, D_i_W

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Forward.

        Args:
            history_data (Tensor): input with shape [B, L, N, C].

        Returns:
            torch.Tensor: outputs with shape [B, L_out, N, 1].
        """
        X = history_data
        X, E_u, E_d, T_D, D_W = self._prepare_inputs(X)

        static_graph, dynamic_graph = self._graph_constructor(
            E_u=E_u, E_d=E_d, X=X, T_D=T_D, D_W=D_W
        )

        X = self.embedding(X)

        spa_forecast_hidden_list = []
        tem_forecast_hidden_list = []

        tem_backcast_seq_res = X
        for _index, layer in enumerate(self.layers):
            tem_backcast_seq_res, spa_forecast_hidden, tem_forecast_hidden = layer(
                tem_backcast_seq_res,
                dynamic_graph,
                static_graph,
                E_u,
                E_d,
                T_D,
                D_W,
            )
            spa_forecast_hidden_list.append(spa_forecast_hidden)
            tem_forecast_hidden_list.append(tem_forecast_hidden)

        spa_forecast_hidden = sum(spa_forecast_hidden_list)
        tem_forecast_hidden = sum(tem_forecast_hidden_list)
        forecast_hidden = spa_forecast_hidden + tem_forecast_hidden

        forecast = self.out_fc_2(F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1
        )
        forecast = forecast.transpose(1, 2).unsqueeze(-1)
        return forecast
