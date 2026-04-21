#!/usr/bin/env python3
"""Analyze tracker CSV metrics and plot feature/propagation trends."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {"frame", "active", "carried"}


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing required columns: {sorted(missing)}")

    df = df.copy()
    df["active"] = pd.to_numeric(df["active"], errors="coerce")
    df["carried"] = pd.to_numeric(df["carried"], errors="coerce")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")

    # Propagation percentage is carried / total active features.
    df["propagation_pct"] = (df["carried"] / df["active"].replace(0, float("inf"))) * 100.0
    return df


def print_stats(name: str, values: pd.Series, units: str = "") -> None:
    values = values.dropna()
    if values.empty:
        print(f"{name}: no valid data")
        return

    suffix = f" {units}" if units else ""
    print(f"  avg:    {values.mean():.3f}{suffix}")
    print(f"  median: {values.median():.3f}{suffix}")
    print(f"  stddev: {values.std(ddof=1):.3f}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read one or more tracker CSVs, print summary stats, and plot trends."
    )
    parser.add_argument("csvs", nargs="+", help="Paths to CSV files exported by bag-to-mp4 tools")
    args = parser.parse_args()

    datasets = []
    for csv in args.csvs:
        path = Path(csv)
        df = load_metrics(path)
        label = path.stem
        datasets.append((label, df))

    for label, df in datasets:
        print(f"[{label}]")
        print_stats("feature count", df["active"])
        print_stats("propagation", df["propagation_pct"], "%")
        print("")

    # Figure 1: feature counts.
    plt.figure("Active feature Count", figsize=(11, 5))
    for label, df in datasets:
        plt.plot(df["frame"], df["carried"], linewidth=1.5, label=label)
    plt.title("Carried (active) feature Count per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Carried (active) features")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Figure 2: propagation percentage.
    plt.figure("Propagation Percentage", figsize=(11, 5))
    for label, df in datasets:
        plt.plot(df["frame"], df["propagation_pct"], linewidth=1.5, label=label)
    plt.title("Propagation Percentage per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Carried / Active [%]")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
