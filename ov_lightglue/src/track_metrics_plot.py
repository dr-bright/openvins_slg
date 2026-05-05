#!/usr/bin/env python3
"""Plot generation-based tracker metrics exported by evaluate_tracker."""


"""
/root/catkin_ws/src/openvins_lightglue/ov_lightglue/src/track_metrics_plot.py \
  /data/gopro10/slow_fast/ov_lightglue_tests/klt_metrics.csv \
  /data/gopro10/slow_fast/ov_lightglue_tests/slg_metrics.csv \
  --out-dir /data/gopro10/slow_fast/ov_lightglue_tests/plots \
  --display 0

"""

import argparse
from pathlib import Path
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GEN_RE = re.compile(r"^gen(\d+)$")


def parse_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def generation_columns(df: pd.DataFrame) -> List[str]:
    indexed = []
    for col in df.columns:
        match = GEN_RE.match(col)
        if match:
            indexed.append((int(match.group(1)), col))

    if not indexed:
        raise ValueError("missing genN columns")

    indexed.sort()
    expected = list(range(indexed[-1][0] + 1))
    actual = [idx for idx, _ in indexed]
    if actual != expected:
        raise ValueError(f"genN columns must be contiguous from gen0, got {actual}")

    return [col for _, col in indexed]


def load_metrics(csv_path: Path) -> Dict:
    df = pd.read_csv(csv_path)
    if "frame" not in df.columns or "timestamp" not in df.columns:
        raise ValueError(f"{csv_path}: missing frame/timestamp columns")

    gen_cols = generation_columns(df)
    df = df.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in gen_cols:
      df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    gen = df[gen_cols].to_numpy(dtype=float)
    all_count = gen.sum(axis=1)
    density = np.divide(gen, all_count[:, None], out=np.zeros_like(gen), where=all_count[:, None] > 0)

    # death[t, g] is the number of features that were generation g at frame t-1
    # and did not survive into generation g+1 at frame t. The final genP column
    # is a saturated P+ tail, so its deaths are the observable residual after
    # lower-generation deaths are removed from total deaths.
    death = np.zeros_like(gen)
    if len(df) > 1:
        previous = gen[:-1, :]
        current = gen[1:, :]
        death[1:, :-1] = np.maximum(previous[:, :-1] - current[:, 1:], 0.0)
        total_deaths = np.maximum(previous.sum(axis=1) - current[:, 1:].sum(axis=1), 0.0)
        lower_deaths = death[1:, :-1].sum(axis=1)
        death[1:, -1] = np.maximum(total_deaths - lower_deaths, 0.0)

    return {
        "label": csv_path.stem,
        "path": csv_path,
        "df": df,
        "gen_cols": gen_cols,
        "P": len(gen_cols) - 1,
        "gen": gen,
        "density": density,
        "death": death,
        "all": all_count,
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: List[float]) -> np.ndarray:
    mask = weights > 0
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.full(len(quantiles), np.nan)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return np.interp(np.asarray(quantiles) * cumulative[-1], cumulative, values)


def death_stats(dataset: Dict) -> Dict:
    P = dataset["P"]
    totals = dataset["death"].sum(axis=0)
    values = np.arange(P, dtype=float)
    weights = totals[:P]
    total = weights.sum()

    if total <= 0:
        return {"total": 0.0, "tail": totals[P], "mean": np.nan, "std": np.nan, "quantiles": np.full(7, np.nan)}

    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return {
        "total": total,
        "tail": totals[P],
        "mean": mean,
        "std": np.sqrt(variance),
        "min": values[weights > 0].min(),
        "max": values[weights > 0].max(),
        "quantiles": weighted_quantile(values, weights, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
    }


def category_series(dataset: Dict, split_n: int) -> Dict:
    gen = dataset["gen"]
    density = dataset["density"]
    death = dataset["death"]
    P = dataset["P"]

    older = slice(split_n, P + 1)
    return {
        "count": gen[:, older].sum(axis=1),
        "density": density[:, older].sum(axis=1),
        "death_count": death[:, older].sum(axis=1),
    }


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.jpg", dpi=160)


def plot_split_comparison(datasets: List[Dict], out_dir: Path, split_n: int, field: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    for dataset in datasets:
        values = category_series(dataset, split_n)[field]
        frames = dataset["df"]["frame"].to_numpy()
        ax.plot(frames, values, linewidth=1.3, label=dataset["label"])

    ax.set_title(f"older{split_n} {field}")
    ax.set_xlabel("Frame")
    ax.set_ylabel(field.replace("_", " "))
    ax.grid(True, alpha=0.3)
    ax.legend()
    if field == "density":
        ax.set_ylim(0, 1.05)

    save_figure(fig, out_dir, f"split_{split_n}_{field}")
    return fig


def plot_all_splits(datasets: List[Dict], out_dir: Path, display_split: int) -> Tuple[List[plt.Figure], List[plt.Figure]]:
    display = []
    max_split = min(dataset["P"] for dataset in datasets)
    for split_n in range(1, max_split + 1):
        for field in ["count", "density", "death_count"]:
            fig = plot_split_comparison(datasets, out_dir, split_n, field)
            if split_n == display_split:
                display.append(fig)
            else:
                plt.close(fig)
    return display, display


def plot_lifetime_hist_from(datasets: List[Dict], out_dir: Path, start_n: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    max_p = max(dataset["P"] for dataset in datasets)
    width = 0.8 / max(1, len(datasets))

    for i, dataset in enumerate(datasets):
        P = dataset["P"]
        totals = dataset["death"].sum(axis=0)[start_n:P+1]
        all_total = totals.sum()
        xs = np.arange(start_n, P + 1)
        offset = (i - (len(datasets) - 1) * 0.5) * width
        values = np.divide(totals, all_total, out=np.zeros_like(totals), where=all_total > 0)
        ax.bar(xs + offset, values, width=width, label=dataset["label"])

    ax.set_title(f"Feature death generation PDF from {start_n}")
    ax.set_xlabel("Generation at death")
    ax.set_ylabel("Probability")
    ax.set_xticks(np.arange(start_n, max_p + 1))
    ax.set_xticklabels([str(i) for i in range(start_n, max_p)] + [f"{max_p}+"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    save_figure(fig, out_dir, f"lifetime_death_hist_from_{start_n}")
    return fig


def plot_lifetime_hist_family(datasets: List[Dict], out_dir: Path, display_n: int) -> List[plt.Figure]:
    display = []
    max_start = min(dataset["P"] for dataset in datasets)
    for start_n in range(1, max_start + 1):
        fig = plot_lifetime_hist_from(datasets, out_dir, start_n)
        if start_n == display_n:
            display.append(fig)
        else:
            plt.close(fig)
    return display


def print_lifetime_stats(datasets: List[Dict]) -> None:
    for dataset in datasets:
        stats = death_stats(dataset)
        q = stats["quantiles"]
        print(f"[{dataset['label']}]")
        print(f"  P: {dataset['P']}")
        print(f"  deaths_total_gen0_to_gen{dataset['P'] - 1}: {stats['total']:.0f}")
        print(f"  deaths_gen{dataset['P']}_plus_omitted: {stats['tail']:.0f}")
        print(f"  mean_generation_at_death: {stats['mean']:.3f}")
        print(f"  std_generation_at_death: {stats['std']:.3f}")
        print(f"  min_generation_at_death: {stats.get('min', np.nan):.3f}")
        print(f"  max_generation_at_death: {stats.get('max', np.nan):.3f}")
        print(f"  q10,q25,q50,q75,q90,q95,q99: {','.join(f'{value:.3f}' for value in q)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot tracker generation metrics.")
    parser.add_argument("csvs", nargs="+", help="CSV files exported by evaluate_tracker")
    parser.add_argument("--out-dir", required=True, help="Directory where JPG figures will be written")
    parser.add_argument("--display", nargs="?", const=True, default=False, type=parse_bool, help="Show selected figures on screen")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [load_metrics(Path(csv)) for csv in args.csvs]
    split_n = max(1, min(dataset["P"] for dataset in datasets) // 2)

    print_lifetime_stats(datasets)

    figures, display_figures = plot_all_splits(datasets, out_dir, split_n)
    figures.extend(plot_lifetime_hist_family(datasets, out_dir, 2))

    if args.display:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
