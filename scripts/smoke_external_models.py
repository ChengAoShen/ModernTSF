"""Standalone smoke tests for the ported external model adapters.

Each test instantiates the benchmark-facing ``Model`` adapter with tiny
dimensions and runs a forward (and a backward for the default MAE loss) on
random data shaped exactly as ModernTSF's trainer would supply it:

* ``x_enc``      : (B, seq_len, N)
* ``x_mark_enc`` : (B, seq_len, 6)        raw [year,month,day,weekday,hour,minute]
* ``x_dec``      : (B, label_len+pred_len, N)
* ``x_mark_dec`` : (B, label_len+pred_len, 6)

The goal is only to confirm each adapter runs end to end and returns the
expected ``(B, pred_len, N)`` shape on CPU.
"""

from __future__ import annotations

import sys

import torch


def _make_batch(b, seq_len, label_len, pred_len, n):
    """Create a random benchmark-style batch with raw integer marks."""
    x_enc = torch.randn(b, seq_len, n)
    dec_len = label_len + pred_len

    def marks(length):
        m = torch.zeros(length, 6)
        m[:, 0] = 2020  # year
        m[:, 1] = torch.randint(1, 13, (length,))  # month
        m[:, 2] = torch.randint(1, 29, (length,))  # day
        m[:, 3] = torch.arange(length) % 7  # weekday
        m[:, 4] = torch.arange(length) % 24  # hour
        m[:, 5] = 0  # minute
        return m

    x_mark_enc = torch.stack([marks(seq_len) for _ in range(b)])
    x_mark_dec = torch.stack([marks(dec_len) for _ in range(b)])
    x_dec = torch.randn(b, dec_len, n)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def _check(name, model, batch, pred_len, n, loss_fn=None):
    """Run a forward/backward and assert the output shape."""
    x_enc, x_mark_enc, x_dec, x_mark_dec = batch
    model.train()
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    out = out[:, -pred_len:, :]
    assert out.shape == (x_enc.shape[0], pred_len, n), (
        f"{name}: expected {(x_enc.shape[0], pred_len, n)}, got {tuple(out.shape)}"
    )
    target = torch.randn_like(out)
    criterion = loss_fn if loss_fn is not None else torch.nn.L1Loss()
    loss = criterion(out, target)
    loss.backward()
    print(f"  {name}: OK  out={tuple(out.shape)}  loss={loss.item():.4f}")


def main() -> int:
    """Run all adapter smoke tests; return non-zero on failure."""
    sys.path.insert(0, "src")
    torch.manual_seed(0)

    B, SEQ, LABEL, PRED, N = 4, 24, 12, 12, 6
    batch = _make_batch(B, SEQ, LABEL, PRED, N)
    # Air models require pred_len == seq_len; use a square-horizon batch.
    air_batch = _make_batch(B, SEQ, LABEL, SEQ, N)

    failures = []

    # --- MoFo (time series) ---
    try:
        from models.mofo.model import Model as MoFoModel

        m = MoFoModel(seq_len=SEQ, pred_len=PRED, enc_in=N, d_model=32, periodic=24, head=4)
        _check("MoFo", m, batch, PRED, N)
    except Exception as exc:  # noqa: BLE001
        failures.append(("MoFo", exc))
        print(f"  MoFo: FAIL {exc!r}")

    # --- PHAT (time series, reconstructed PHAT_Attention) ---
    try:
        from models.phat.model import Model as PHATModel

        m = PHATModel(seq_len=SEQ, pred_len=PRED, enc_in=N, d_model=32, n_heads=4)
        _check("PHAT", m, batch, PRED, N)
    except Exception as exc:  # noqa: BLE001
        failures.append(("PHAT", exc))
        print(f"  PHAT: FAIL {exc!r}")

    # --- BiST (spatiotemporal) ---
    try:
        from models.bist.model import Model as BiSTModel

        m = BiSTModel(seq_len=SEQ, pred_len=PRED, enc_in=N)
        _check("BiST", m, batch, PRED, N)
    except Exception as exc:  # noqa: BLE001
        failures.append(("BiST", exc))
        print(f"  BiST: FAIL {exc!r}")

    # --- MAGE (spatiotemporal) ---
    try:
        from models.mage.model import Model as MAGEModel

        m = MAGEModel(seq_len=SEQ, pred_len=PRED, enc_in=N)
        _check("MAGE", m, batch, PRED, N)
    except Exception as exc:  # noqa: BLE001
        failures.append(("MAGE", exc))
        print(f"  MAGE: FAIL {exc!r}")

    # --- STOP (spatiotemporal) ---
    try:
        from models.stop.model import Model as STOPModel

        m = STOPModel(seq_len=SEQ, pred_len=PRED, enc_in=N)
        _check("STOP", m, batch, PRED, N)
    except Exception as exc:  # noqa: BLE001
        failures.append(("STOP", exc))
        print(f"  STOP: FAIL {exc!r}")

    # --- CauAir (air quality, needs future marks) ---
    try:
        from benchmark.losses_external import FrequencyMAELoss
        from models.cauair.model import Model as CauAirModel

        m = CauAirModel(seq_len=SEQ, pred_len=SEQ, enc_in=N, dim=32)
        _check("CauAir", m, air_batch, SEQ, N)
    except Exception as exc:  # noqa: BLE001
        failures.append(("CauAir", exc))
        print(f"  CauAir: FAIL {exc!r}")

    # --- AirCade (air quality, needs future marks + FFT loss) ---
    try:
        from benchmark.losses_external import FrequencyMAELoss
        from models.aircade.model import Model as AirCadeModel

        m = AirCadeModel(seq_len=SEQ, pred_len=SEQ, enc_in=N)
        _check("AirCade", m, air_batch, SEQ, N, loss_fn=FrequencyMAELoss())
    except Exception as exc:  # noqa: BLE001
        failures.append(("AirCade", exc))
        print(f"  AirCade: FAIL {exc!r}")

    # --- Node-structured modes: 4-D covariate marks (B, T, N, F) ---
    # Spatiotemporal mode feeds (value, cov) directly; covariate mode also feeds
    # a future covariate block via x_mark_dec.
    #
    # Calendar-covariate models (BiST / MAGE / STOP) read covariate channels as
    # time-of-day / day-of-week embedding indices, so their covariates must be
    # normalized to [0, 1) with F == 2. Air-quality models (CauAir / AirCade)
    # accept an arbitrary covariate count.
    st_x = torch.randn(B, SEQ, N)
    cal_cov = torch.rand(B, SEQ, N, 2)  # [time_in_day, day_in_week] in [0, 1)
    cal_fut = torch.rand(B, SEQ, N, 2)
    st_cal_batch = (st_x, cal_cov, torch.randn(B, SEQ, N), cal_fut)

    F = 4  # arbitrary covariate count for the air-quality path
    arb_cov = torch.randn(B, SEQ, N, F)
    arb_fut = torch.randn(B, SEQ, N, F)
    air_cov_batch = (st_x, arb_cov, torch.randn(B, SEQ, N), arb_fut)

    try:
        from models.bist.model import Model as BiSTModel

        m = BiSTModel(seq_len=SEQ, pred_len=PRED, enc_in=N)
        _check("BiST[ST node-cov]", m, st_cal_batch, PRED, N)
    except Exception as exc:  # noqa: BLE001
        failures.append(("BiST[ST node-cov]", exc))
        print(f"  BiST[ST node-cov]: FAIL {exc!r}")

    try:
        from models.cauair.model import Model as CauAirModel

        # cov_dim must match the covariate count for the future block.
        m = CauAirModel(seq_len=SEQ, pred_len=SEQ, enc_in=N, cov_dim=F, dim=32)
        _check("CauAir[air node-cov]", m, air_cov_batch, SEQ, N)
    except Exception as exc:  # noqa: BLE001
        failures.append(("CauAir[air node-cov]", exc))
        print(f"  CauAir[air node-cov]: FAIL {exc!r}")

    print()
    if failures:
        print(f"{len(failures)} adapter(s) failed:")
        for name, exc in failures:
            print(f"  - {name}: {exc!r}")
        return 1
    print("All adapter smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
