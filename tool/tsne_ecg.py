"""
t-SNE visualization comparing ECG vs ECG_Generate distributions.

Layout (2x2):
  Row 0 — Cross-dataset, per channel:  Ch0: ECG vs ECG_Generate | Ch1: ECG vs ECG_Generate
  Row 1 — Cross-channel, per dataset:  ECG: Ch0 vs Ch1          | ECG_Generate: Ch0 vs Ch1

Usage:
    uv run python tool/tsne_ecg.py \
        --ecg dataset/ECG/train.npz \
        --gen dataset/ECG_Generate/train.npz \
        [--n-samples 2000] \
        [--output work_dirs/tsne_ecg.png]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os


def load_samples(path: str, n_samples: int, rng: np.random.Generator):
    data = np.load(path)
    x = data["x"]  # (N, T, C)
    if n_samples < len(x):
        idx = rng.choice(len(x), n_samples, replace=False)
        x = x[idx]
    return x


def run_tsne(features: np.ndarray, perplexity: float) -> np.ndarray:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                max_iter=1000, learning_rate="auto", init="pca")
    return tsne.fit_transform(features_scaled)


def scatter(ax, emb, labels, color_map, alpha=0.45, s=8):
    for label, color in color_map.items():
        mask = labels == label
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=color, alpha=alpha, s=s, rasterized=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg", default="dataset/ECG/train.npz")
    parser.add_argument("--gen", default="dataset/ECG_Generate/train.npz")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--output", default="work_dirs/tsne_ecg.png")
    parser.add_argument("--perplexity", type=float, default=40.0)
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    print(f"Loading {args.ecg} ...")
    ecg = load_samples(args.ecg, args.n_samples, rng)   # (N1, T, 2)
    print(f"Loading {args.gen} ...")
    gen = load_samples(args.gen, args.n_samples, rng)   # (N2, T, 2)

    n_ecg, n_gen = len(ecg), len(gen)
    n_ch = ecg.shape[-1]
    print(f"Samples — ECG: {n_ecg}, ECG_Generate: {n_gen}, channels: {n_ch}")

    # Color palettes
    dataset_colors = {"ECG": "#2196F3", "ECG_Generate": "#FF5722"}
    channel_colors = {f"Ch{i}": c for i, c in enumerate(["#4CAF50", "#9C27B0"])}

    fig, axes = plt.subplots(2, n_ch, figsize=(7 * n_ch, 12))

    # ── Row 0: cross-dataset comparison, one subplot per channel ──────────────
    for ch_idx in range(n_ch):
        ax = axes[0, ch_idx]
        feat_ecg = ecg[:, :, ch_idx]
        feat_gen = gen[:, :, ch_idx]
        combined = np.vstack([feat_ecg, feat_gen])
        labels = np.array(["ECG"] * n_ecg + ["ECG_Generate"] * n_gen)

        print(f"  [Row 0] t-SNE Channel {ch_idx} ECG vs ECG_Generate (n={len(combined)}) ...")
        emb = run_tsne(combined, args.perplexity)
        scatter(ax, emb, labels, dataset_colors)

        ax.set_title(f"Channel {ch_idx}  —  ECG vs ECG_Generate",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.legend(handles=[
            mpatches.Patch(color=dataset_colors["ECG"], label=f"ECG (n={n_ecg})"),
            mpatches.Patch(color=dataset_colors["ECG_Generate"], label=f"ECG_Generate (n={n_gen})"),
        ], fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # ── Row 1: cross-channel comparison, one subplot per dataset ──────────────
    for ds_idx, (ds_name, ds_data) in enumerate([("ECG", ecg), ("ECG_Generate", gen)]):
        ax = axes[1, ds_idx]
        n_ds = len(ds_data)
        feat_ch0 = ds_data[:, :, 0]
        feat_ch1 = ds_data[:, :, 1]
        combined = np.vstack([feat_ch0, feat_ch1])
        labels = np.array(["Ch0"] * n_ds + ["Ch1"] * n_ds)

        print(f"  [Row 1] t-SNE {ds_name} Ch0 vs Ch1 (n={len(combined)}) ...")
        emb = run_tsne(combined, args.perplexity)
        scatter(ax, emb, labels, channel_colors)

        ax.set_title(f"{ds_name}  —  Ch0 vs Ch1", fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.legend(handles=[
            mpatches.Patch(color=channel_colors["Ch0"], label=f"Ch0 (n={n_ds})"),
            mpatches.Patch(color=channel_colors["Ch1"], label=f"Ch1 (n={n_ds})"),
        ], fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle("t-SNE: ECG vs ECG_Generate (train split, x windows)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
