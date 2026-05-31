"""Verbatim GCLSTM model source.

Vendored from CauAir (src/models/gclstm.py).
BaseModel replaced with nn.Module; explicit params stored on self.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(
            torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x, hx, cx):
        gates = (F.linear(x, self.weight_ih, self.bias_ih)
                 + F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        cy = f * cx + i * g
        hy = o * torch.tanh(cy)
        return hy, cy

class GCLSTM(nn.Module):
    """Graph Convolutional LSTM for spatiotemporal forecasting."""

    def __init__(self, gso, node_num, input_dim, output_dim,
                 seq_len, horizon):
        super(GCLSTM, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

        self.hist_len = self.seq_len
        self.fut_len = self.horizon
        self.out_dim = self.output_dim
        self.in_dim = self.input_dim

        # Model configuration
        self.hidden_size = 128
        self.dropout_rate = 0.1

        self.mlp_in = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size),
        )
        self.lstm_cell = LSTMCell(self.hidden_size, self.hidden_size)
        self.conv = ChebGraphConv(
            self.hidden_size, self.hidden_size, Ks=2, gso=gso)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.fut_len * self.out_dim),
        )

    def forward(self, inputs, labels=None, adj=None):
        aqi_hist = inputs[..., :1]
        mete_hist = inputs[..., 1:]
        x_hist = mete_hist

        # Use zeros for future covariates in inference
        if labels is not None:
            x_fut = labels
        else:
            bs, _, n, _ = inputs.shape
            x_fut = torch.zeros(
                bs, self.fut_len, n, mete_hist.shape[-1],
                device=inputs.device, dtype=inputs.dtype)

        x_in = torch.cat([x_hist, x_fut], dim=1)
        bs, ts, n, _ = x_in.shape

        h_t = torch.zeros(
            bs * n, self.hidden_size,
            device=x_in.device, dtype=x_in.dtype)
        c_t = torch.zeros(
            bs * n, self.hidden_size,
            device=x_in.device, dtype=x_in.dtype)

        aqi_t = aqi_hist[:, 0]
        for t in range(self.hist_len):
            if t < self.hist_len:
                aqi_t = aqi_hist[:, t]
            x_t = torch.cat((x_in[:, t], aqi_t), dim=-1)
            x_t = self.mlp_in(x_t)
            x_t = self.conv(x_t)
            h_t, c_t = self.lstm_cell(
                x_t.view(bs * n, -1), h_t, c_t)

        aqi_t = self.decoder(h_t)
        aqi_t = aqi_t.view(
            bs, n, self.fut_len, self.out_dim).permute(0, 2, 1, 3)
        return aqi_t

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(
            torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.Ks - 1 < 0:
            raise ValueError(
                f'ERROR: Ks must be positive, got {self.Ks}.')
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,bij->bhj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,bij->bhj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(
                    torch.einsum(
                        'hi,bij->bhj', 2 * self.gso,
                        x_list[k - 1]) - x_list[k - 2])
        x = torch.stack(x_list, dim=1)
        cheb_graph_conv = torch.einsum(
            'bkhi,kij->bhj', x, self.weight)
        cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        return cheb_graph_conv
