"""Analyze timestep-level directional signals from a closed-loop trajectory CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


CONDITIONS = ("left", "right", "front")
ACTIONS = ("turn_left", "turn_right", "walk_forward", "rest", "idle")
EPSILON = 1e-9


def load_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped = {condition: [] for condition in CONDITIONS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] in grouped:
                grouped[row["condition"]].append(row)
    return grouped


def mean(rows: list[dict[str, str]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def action_probabilities(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    counts = Counter(row[field] for row in rows)
    return {action: counts[action] / len(rows) for action in ACTIONS}


def bias_sign_distribution(rows: list[dict[str, str]]) -> dict[str, float]:
    signs = Counter()
    for row in rows:
        bias = float(row["turn_bias"])
        if bias < -EPSILON:
            signs["negative"] += 1
        elif bias > EPSILON:
            signs["positive"] += 1
        else:
            signs["zero"] += 1
    return {sign: signs[sign] / len(rows) for sign in ("negative", "zero", "positive")}


def summarize(condition: str, rows: list[dict[str, str]]) -> dict[str, object]:
    action_probs = action_probabilities(rows, "action")
    requested_probs = action_probabilities(rows, "requested_action")
    signs = bias_sign_distribution(rows)
    return {
        "condition": condition,
        "steps": len(rows),
        "mean_left_drive": mean(rows, "left_drive"),
        "mean_right_drive": mean(rows, "right_drive"),
        "mean_turn_bias": mean(rows, "turn_bias"),
        "std_turn_bias": (
            sum((float(row["turn_bias"]) - mean(rows, "turn_bias")) ** 2 for row in rows)
            / len(rows)
        ) ** 0.5,
        "p_turn_left": action_probs["turn_left"],
        "p_turn_right": action_probs["turn_right"],
        "p_walk_forward": action_probs["walk_forward"],
        "p_rest": action_probs["rest"],
        "p_bias_negative": signs["negative"],
        "p_bias_zero": signs["zero"],
        "p_bias_positive": signs["positive"],
        "requested_action_distribution": dict(Counter(row["requested_action"] for row in rows)),
        "action_distribution": dict(Counter(row["action"] for row in rows)),
        "requested_p_turn_left": requested_probs["turn_left"],
        "requested_p_turn_right": requested_probs["turn_right"],
    }


def paired_left_right(left: list[dict[str, str]], right: list[dict[str, str]]) -> dict[str, float]:
    by_step_left = {int(row["step"]): row for row in left}
    by_step_right = {int(row["step"]): row for row in right}
    steps = sorted(set(by_step_left) & set(by_step_right))
    deltas = [float(by_step_left[step]["turn_bias"]) - float(by_step_right[step]["turn_bias"]) for step in steps]
    opposite_sign = sum(
        float(by_step_left[step]["turn_bias"]) * float(by_step_right[step]["turn_bias"]) < 0
        for step in steps
    )
    return {
        "paired_steps": len(steps),
        "mean_left_minus_right_bias": sum(deltas) / len(deltas),
        "p_left_bias_greater": sum(delta > EPSILON for delta in deltas) / len(deltas),
        "p_right_bias_greater": sum(delta < -EPSILON for delta in deltas) / len(deltas),
        "p_opposite_bias_sign": opposite_sign / len(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/directional_behavior/trajectory.csv")
    parser.add_argument("--output-dir", default="results/directional_behavior")
    args = parser.parse_args()

    grouped = load_rows(Path(args.input))
    missing = [condition for condition, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"Missing conditions in {args.input}: {missing}")

    summaries = [summarize(condition, grouped[condition]) for condition in CONDITIONS]
    paired = paired_left_right(grouped["left"], grouped["right"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = [
        "condition", "steps", "mean_left_drive", "mean_right_drive", "mean_turn_bias",
        "std_turn_bias", "p_turn_left", "p_turn_right", "p_walk_forward", "p_rest",
        "p_bias_negative", "p_bias_zero", "p_bias_positive", "requested_action_distribution",
        "action_distribution", "requested_p_turn_left", "requested_p_turn_right",
    ]
    with (output_dir / "temporal_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)

    paired_fields = list(paired)
    with (output_dir / "left_right_paired.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerow(paired)

    print("condition mean_bias std_bias p_turn_left p_turn_right p_bias_negative p_bias_zero p_bias_positive")
    for summary in summaries:
        print(
            f"{summary['condition']:<9} {summary['mean_turn_bias']:10.4f} "
            f"{summary['std_turn_bias']:8.4f} {summary['p_turn_left']:12.3f} "
            f"{summary['p_turn_right']:13.3f} {summary['p_bias_negative']:15.3f} "
            f"{summary['p_bias_zero']:11.3f} {summary['p_bias_positive']:14.3f}"
        )
    print("left_vs_right", paired)
    print(f"Saved temporal summary to {output_dir / 'temporal_summary.csv'}")
    print(f"Saved paired comparison to {output_dir / 'left_right_paired.csv'}")


if __name__ == "__main__":
    main()
