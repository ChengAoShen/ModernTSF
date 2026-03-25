"""TimeMixer model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.autoformer_encdec import series_decomp
from models.module.embed import DataEmbedding_wo_pos
from models.module.standard_norm import Normalize


class DFTSeriesDecomp(nn.Module):
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, _ = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        if len(season_list) <= 1:
            return [season_list[0].permute(0, 2, 1)]
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window**i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window**i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        if len(trend_list) <= 1:
            return [trend_list[0].permute(0, 2, 1)]
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        d_ff,
        decomp_method,
        down_sampling_window,
        down_sampling_layers,
        channel_independence,
        moving_avg,
        top_k,
        dropout,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == "moving_avg":
            self.decompsition = series_decomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFTSeriesDecomp(top_k)
        else:
            raise ValueError("decompsition is error")

        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len=seq_len,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
        )

        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len=seq_len,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, t, _ = x.size()
            length_list.append(t)

        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixerModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        c_out,
        e_layers,
        d_model,
        d_ff,
        down_sampling_window,
        down_sampling_layers,
        down_sampling_method,
        channel_independence,
        moving_avg,
        embed,
        top_k,
        dropout,
        freq,
        use_norm,
        decomp_method,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.layer = e_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        self.channel_independence = channel_independence

        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    seq_len,
                    pred_len,
                    d_model,
                    d_ff,
                    decomp_method,
                    down_sampling_window,
                    down_sampling_layers,
                    channel_independence,
                    moving_avg,
                    top_k,
                    dropout,
                )
                for _ in range(e_layers)
            ]
        )

        self.preprocess = series_decomp(moving_avg)

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, d_model, embed, freq, dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                enc_in, d_model, embed, freq, dropout
            )

        self.normalize_layers = nn.ModuleList(
            [
                Normalize(
                    self.enc_in,
                    affine=True,
                    non_norm=True if use_norm == 0 else False,
                )
                for _ in range(down_sampling_layers + 1)
            ]
        )

        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(
                    seq_len // (down_sampling_window**i),
                    pred_len,
                )
                for i in range(down_sampling_layers + 1)
            ]
        )

        if self.channel_independence:
            self.projection_layer = nn.Linear(d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(d_model, self.c_out, bias=True)

            self.out_res_layers = nn.ModuleList(
                [
                    nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window**i),
                    )
                    for i in range(down_sampling_layers + 1)
                ]
            )

            self.regression_layers = nn.ModuleList(
                [
                    nn.Linear(
                        seq_len // (down_sampling_window**i),
                        pred_len,
                    )
                    for i in range(down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        out1_list = []
        out2_list = []
        for x in x_list:
            x_1, x_2 = self.preprocess(x)
            out1_list.append(x_1)
            out2_list.append(x_2)
        return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.down_sampling_method == "max":
            down_pool = nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == "avg":
            down_pool = nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc

        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]
        x_mark_sampling_list = [x_mark_enc]

        for _ in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[:, :: self.down_sampling_window, :]
                )
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[
                    :, :: self.down_sampling_window, :
                ]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None
        return x_enc, x_mark_enc

    def future_multi_mixing(self, b, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                dec_out = self.projection_layer(dec_out)
                dec_out = (
                    dec_out.reshape(b, self.c_out, self.pred_len)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(
                range(len(x_list[0])), enc_out_list, x_list[1]
            ):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1
                )
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        if torch.is_tensor(x_enc):
            x_enc = [x_enc]
        if x_mark_enc is not None and torch.is_tensor(x_mark_enc):
            x_mark_enc = [x_mark_enc]

        b = x_enc[0].shape[0]

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                b, t, n = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(b * n, t, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(n, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                b, t, n = x.size()
                x = self.normalize_layers[i](x, "norm")
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(b * n, t, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out_list = self.future_multi_mixing(b, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        c_out,
        e_layers,
        d_model,
        d_ff,
        down_sampling_window,
        down_sampling_layers,
        down_sampling_method,
        channel_independence,
        moving_avg,
        embed,
        top_k,
        dropout,
        freq,
        use_norm,
        decomp_method,
    ):
        super().__init__()
        self.model = TimeMixerModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            c_out=c_out,
            e_layers=e_layers,
            d_model=d_model,
            d_ff=d_ff,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
            down_sampling_method=down_sampling_method,
            channel_independence=channel_independence,
            moving_avg=moving_avg,
            embed=embed,
            top_k=top_k,
            dropout=dropout,
            freq=freq,
            use_norm=use_norm,
            decomp_method=decomp_method,
        )

    def forward(self, x, *args):
        return self.model(x, *args)
