"""Analyze the LIF parameter sweep with effect-size and correlation summaries."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


PARAMETERS = [
    "threshold",
    "leak",
    "weight_scale",
    "refractory_steps",
]

OUTPUTS = [
    "total_spikes",
    "motor_spikes",
    "mean_locomotion",
    "mean_turn_bias",
    "transition_rate",
    "dominant_period",
]


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation with robust handling for constant series."""
    if x.nunique() <= 1 or y.nunique() <= 1:
        return 0.0
    return float(x.corr(y))


def summarize_parameter_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-parameter relationship statistics for each output metric."""
    rows = []

    for param in PARAMETERS:
        for output in OUTPUTS:
            x = df[param].astype(float)
            y = df[output].astype(float)

            corr = _safe_corr(x, y)
            mean_y = float(y.mean())
            x_unique = sorted(x.unique().tolist())
            if len(x_unique) < 2:
                delta_mean = 0.0
                effect_size = 0.0
            else:
                low = y[x == x_unique[0]].mean()
                high = y[x == x_unique[-1]].mean()
                delta_mean = float(high - low)
                pooled = np.sqrt(
                    (
                        ((x == x_unique[0]).sum() - 1) * (y[x == x_unique[0]].std(ddof=1) ** 2)
                        + ((x == x_unique[-1]).sum() - 1) * (y[x == x_unique[-1]].std(ddof=1) ** 2)
                    )
                    / (len(x[x == x_unique[0]]) + len(x[x == x_unique[-1]]) - 2)
                )
                effect_size = float(delta_mean / pooled) if pooled > 0 else 0.0

            rows.append(
                {
                    "parameter": param,
                    "output": output,
                    "pearson_r": corr,
                    "abs_r": abs(corr),
                    "mean_output": mean_y,
                    "delta_mean_high_minus_low": delta_mean,
                    "effect_size": effect_size,
                }
            )

    out = pd.DataFrame(rows)
    out["rank_by_abs_r"] = out.groupby("output")["abs_r"].rank(method="dense", ascending=False)
    return out.sort_values(["output", "abs_r", "parameter"], ascending=[True, False, True]).reset_index(drop=True)


def load_sweep_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "condition",
        "threshold",
        "leak",
        "weight_scale",
        "refractory_steps",
        "total_spikes",
        "motor_spikes",
        "mean_locomotion",
        "mean_turn_bias",
        "transition_rate",
        "dominant_period",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def print_summary(summary: pd.DataFrame) -> None:
    """Print a compact but interpretable ranking summary."""
    print("\n=== SENSITIVITY ANALYSIS ===")
    print("\nTop parameter-output correlations by output:")
    for output in OUTPUTS:
        subset = summary[summary["output"] == output].sort_values("abs_r", ascending=False)
        top = subset.iloc[0]
        print(
            f"- {output}: {top['parameter']} | r={top['pearson_r']:.4f} | "
            f"delta_mean={top['delta_mean_high_minus_low']:.4f} | effect_size={top['effect_size']:.4f}"
        )

    print("\nDetailed table:")
    print(summary.to_string(index=False, formatters={
        "pearson_r": lambda v: f"{v:.4f}",
        "abs_r": lambda v: f"{v:.4f}",
        "delta_mean_high_minus_low": lambda v: f"{v:.4f}",
        "effect_size": lambda v: f"{v:.4f}",
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="results/sensitivity_summary.csv", help="CSV produced by sensitivity_sweep.py")
    parser.add_argument("--output", type=str, default="results/sensitivity_analysis.csv", help="CSV summary output")
    args = parser.parse_args()

    df = load_sweep_csv(args.input)
    summary = summarize_parameter_effects(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Saved analysis summary to {out_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()
