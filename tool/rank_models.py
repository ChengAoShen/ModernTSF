#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank models by pred_len/seed for MSE and MAE."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("work_dirs"),
        help="Root directory containing dataset/model subfolders with performance.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETTh1",
        help="Dataset name to filter (default: ETTh1)",
    )
    parser.add_argument(
        "--out-mse",
        type=Path,
        default=None,
        help="Output wide MSE rankings CSV (model names)",
    )
    parser.add_argument(
        "--out-mae",
        type=Path,
        default=None,
        help="Output wide MAE rankings CSV (model names)",
    )
    parser.add_argument(
        "--out-long",
        type=Path,
        default=None,
        help="Output long rankings CSV (for plotting)",
    )
    return parser.parse_args()


def read_performance_files(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("**/performance.csv"))
    if not files:
        raise FileNotFoundError(f"No performance.csv found under {root}")

    frames = []
    for file in files:
        df = pd.read_csv(file)
        required = {"model", "pred_len", "seed", "mse", "mae"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {file}")
        if "dataset" not in df.columns:
            dataset = file.parent.parent.name
            df["dataset"] = dataset
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_rankings(
    df: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = df[["dataset", "model", "pred_len", "seed", "mse", "mae"]].copy()
    base = base[base["dataset"] == dataset].copy()
    if base.empty:
        raise ValueError(f"No rows found for dataset: {dataset}")

    long_frames = []
    for metric in ["mse", "mae"]:
        metric_df = base[["dataset", "model", "pred_len", "seed", metric]].copy()
        metric_df = metric_df.rename(columns={metric: "value"})
        metric_df["metric"] = metric
        metric_df["rank"] = (
            metric_df.groupby(["dataset", "pred_len", "seed", "metric"])["value"]
            .rank(method="min", ascending=True)
            .astype("Int64")
        )
        long_frames.append(metric_df)

    long_df = pd.concat(long_frames, ignore_index=True)

    def build_metric_table(metric: str) -> pd.DataFrame:
        metric_df = long_df[long_df["metric"] == metric].copy()
        metric_df = metric_df.sort_values(
            ["dataset", "pred_len", "seed", "rank", "model"]
        )
        metric_df["setting"] = (
            "pl"
            + metric_df["pred_len"].astype(str)
            + "_seed"
            + metric_df["seed"].astype(str)
        )

        table = metric_df.pivot_table(
            index="rank",
            columns="setting",
            values="model",
            aggfunc="first",
        ).sort_index()

        def sort_key(col: str) -> tuple[int, int]:
            parts = col.split("_")
            pred = int(parts[0].replace("pl", ""))
            seed = int(parts[1].replace("seed", ""))
            return pred, seed

        sorted_cols = sorted(table.columns, key=sort_key)
        table = table[sorted_cols].reset_index()
        return table

    mse_table = build_metric_table("mse")
    mae_table = build_metric_table("mae")

    return mse_table, mae_table, long_df


def main() -> None:
    args = parse_args()
    df = read_performance_files(args.input_root)
    mse_table, mae_table, long_df = build_rankings(df, args.dataset)

    if args.out_mse is None:
        args.out_mse = Path("work_dirs") / args.dataset / "model_rankings_mse.csv"
    if args.out_mae is None:
        args.out_mae = Path("work_dirs") / args.dataset / "model_rankings_mae.csv"
    if args.out_long is None:
        args.out_long = Path("work_dirs") / args.dataset / "model_rankings_long.csv"

    args.out_mse.parent.mkdir(parents=True, exist_ok=True)
    args.out_mae.parent.mkdir(parents=True, exist_ok=True)
    args.out_long.parent.mkdir(parents=True, exist_ok=True)

    mse_table.to_csv(args.out_mse, index=False)
    mae_table.to_csv(args.out_mae, index=False)
    long_df.to_csv(args.out_long, index=False)


if __name__ == "__main__":
    main()
