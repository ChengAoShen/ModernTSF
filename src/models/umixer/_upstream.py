"""Upstream UMixer model from CauAir.

Verbatim port with BaseModel replaced by nn.Module and explicit params.
"""

import torch
import torch.nn as nn
import torch.fft
import math


class UMixer(nn.Module):
    def __init__(self, node_num: int, seq_len: int, horizon: int,
                 stride: int = 24, patch_len: int = 24,
                 d_model: int = 128, dropout: float = 0.1,
                 e_layers: int = 2, d_layers: int = 1,
                 enc_in: int = 1, c_out: int = 1):
        super(UMixer, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.horizon = horizon
        self.pred_len = horizon
        self.stride = stride
        self.patch_len = patch_len
        self.d_model = d_model
        self.dropout = dropout
        self.Pnum = int((self.pred_len + self.seq_len - self.patch_len) / self.stride + 2)

        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.enc_in = enc_in
        self.c_out = c_out

        model_args = dict(
            pred_len=self.pred_len, seq_len=self.seq_len,
            stride=self.stride, patch_len=self.patch_len,
            d_model=self.d_model, dropout=self.dropout,
            e_layers=self.e_layers, d_layers=self.d_layers,
            enc_in=self.enc_in, c_out=self.c_out,
        )

        self.mlp_tempmix_md = nn.ModuleList([
            tempolMix_CI_pat(**model_args) for _ in range(self.e_layers)])
        self.mlp_chanmix_md = nn.ModuleList([
            channelMix_CI_pat(**model_args) for _ in range(self.e_layers)])
        self.mlp_tempmix_mu = nn.ModuleList([
            tempolMix_CI_pat(**model_args) for _ in range(self.e_layers)])
        self.mlp_chanmix_mu = nn.ModuleList([
            channelMix_CI_pat(**model_args) for _ in range(self.e_layers)])

        self.mlp_trend_ci = nn.ModuleList(
            nn.Linear(self.pred_len, self.d_model) for _ in range(self.c_out))
        self.mlp_trend2_ci = nn.ModuleList(
            nn.Linear(self.d_model, self.pred_len) for _ in range(self.c_out))

        self.revin = RevIN(self.enc_in)
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, self.stride, self.dropout)
        self.head = Flatten_Head(
            self.enc_in, self.d_model * self.Pnum, self.pred_len,
            head_dropout=self.dropout)
        self.comb = nn.Linear(self.e_layers, 1)

    def forecast(self, x_input):
        x_ori = x_input.contiguous()
        x_input = self.revin(x_input, 'norm')
        x_input = self.predict_linear(x_input.permute(0, 2, 1))
        x_input, n_vars = self.patch_embedding(x_input)

        x_old, _ = self.patch_embedding(x_ori.permute(0, 2, 1))

        x_all = torch.zeros(
            [x_input.shape[0], x_input.shape[1], x_input.shape[2], self.layer],
            device=x_input.device)
        for i in range(self.layer):
            x_ud = self.mlp_tempmix_md[i](x_input)
            x_ud = self.mlp_chanmix_md[i](x_ud)
            for j in range(i, -1, -1):
                x_ud = self.mlp_tempmix_mu[j](x_ud)
                x_ud = self.mlp_chanmix_mu[j](x_ud)
            x_all[:, :, :, i] = x_ud
        x_input = self.comb(x_all).squeeze(-1)
        x_input = S_Correction(
            self.layer_norm(x_old),
            self.layer_norm(x_input[:, :x_old.shape[1], :])) * x_input
        x_input = torch.reshape(
            x_input, (-1, n_vars, x_input.shape[-2], x_input.shape[-1]))
        x_input = x_input.permute(0, 1, 3, 2)

        x_input = self.head(x_input)
        x_input = x_input.permute(0, 2, 1)
        x_input = self.revin(x_input, 'denorm')

        x = x_input[:, -self.pred_len:, :]
        return x


def S_Correction(x, x_pre):
    x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
    x_pre_fft = torch.fft.rfft(x_pre, dim=1, norm='ortho')
    x_fft = x_fft * torch.conj(x_fft)
    x_pre_fft = x_pre_fft * torch.conj(x_pre_fft)
    x_ifft = torch.fft.irfft(x_fft, dim=1)
    x_pre_ifft = torch.fft.irfft(x_pre_fft, dim=1)
    x_ifft = torch.clamp(x_ifft, min=0)
    x_pre_ifft = torch.clamp(x_pre_ifft, min=0)
    alpha = (torch.sum(x_ifft * x_pre_ifft, dim=1, keepdim=True) /
             (torch.sum(x_pre_ifft * x_pre_ifft, dim=1, keepdim=True) + 0.001))
    return torch.sqrt(alpha)


class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            avg = func(x)
            moving_mean.append(avg)
            sea = x - avg
            res.append(sea)
        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class channelMix_CI_pat(nn.Module):
    def __init__(self, **model_args):
        super(channelMix_CI_pat, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.stride = model_args['stride']
        self.patch_len = model_args['patch_len']
        self.d_model = model_args['d_model']
        self.dropout = model_args["dropout"]
        self.Pnum = int((self.pred_len + self.seq_len - self.patch_len) / self.stride + 2)
        self.conv1 = nn.ModuleList(
            nn.Linear(self.Pnum, self.Pnum) for _ in range(self.d_model))
        self.conv2 = nn.ModuleList(
            nn.Linear(self.Pnum, self.Pnum) for _ in range(self.d_model))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.d_model)
        self.channels = self.d_model

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.channels):
            o[:, :, i] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, :, i]))))
        res = o + x
        res = self.norm(res)
        return res


class tempolMix_CI_pat(nn.Module):
    def __init__(self, **model_args):
        super(tempolMix_CI_pat, self).__init__()
        self.pred_len = model_args['pred_len']
        self.seq_len = model_args['seq_len']
        self.stride = model_args['stride']
        self.patch_len = model_args['patch_len']
        self.d_model = model_args['d_model']
        self.dropout = model_args["dropout"]
        self.Pnum = int((self.pred_len + self.seq_len - self.patch_len) / self.stride + 2)
        self.conv1 = nn.ModuleList(
            nn.Linear(self.d_model, self.d_model) for _ in range(self.Pnum))
        self.conv2 = nn.ModuleList(
            nn.Linear(self.d_model, self.d_model) for _ in range(self.Pnum))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.d_model)
        self.channels = self.Pnum

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.channels):
            o[:, i, :] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, i, :]))))
        res = o + x
        res = self.norm(res)
        return res


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.value_embedding = TokenEmbedding(patch_len, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

