"""Analyze motor-command to Body2D causality from navigation trajectories."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ACTIONS = ("turn_left", "turn_right", "walk_forward", "rest", "idle")
BIAS_GROUPS = ("negative", "zero", "positive")

SUMMARY_FIELDS = [
    "requested_action", "count", "mean_delta_heading", "mean_delta_x", "mean_delta_y",
    "mean_delta_distance", "mean_turn_bias", "collision_rate",
]
BIAS_FIELDS = ["bias_group", "count", "mean_delta_heading", "mean_delta_x", "mean_delta_y", "mean_delta_distance"]
TIMESTEP_FIELDS = [
    "scenario", "step", "requested_action", "action", "turn_bias", "delta_heading",
    "delta_x", "delta_y", "delta_distance", "collision",
]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def bias_group(bias: float) -> str:
    if bias < -1e-9:
        return "negative"
    if bias > 1e-9:
        return "positive"
    return "zero"


def read_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["target_side"]].append(row)
    return grouped


def derive_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    derived = []
    for previous, current in zip(rows, rows[1:]):
        previous_heading = float(previous["heading"])
        current_heading = float(current["heading"])
        derived.append({
            "scenario": current["target_side"],
            "step": int(current["step"]),
            "requested_action": current["requested_action"],
            "action": current["action"],
            "turn_bias": float(current["turn_bias"]),
            "delta_heading": normalize_angle(current_heading - previous_heading),
            "delta_x": float(current["x"]) - float(previous["x"]),
            "delta_y": float(current["y"]) - float(previous["y"]),
            "delta_distance": float(current["distance_to_target"]) - float(previous["distance_to_target"]),
            "collision": current["collision"] == "True",
        })
    return derived


def summarize(rows: list[dict[str, object]], key: str, values: tuple[str, ...]) -> list[dict[str, object]]:
    output = []
    for value in values:
        selected = [row for row in rows if row[key] == value]
        if not selected:
            continue
        output.append({
            key: value,
            "count": len(selected),
            "mean_delta_heading": sum(float(row["delta_heading"]) for row in selected) / len(selected),
            "mean_delta_x": sum(float(row["delta_x"]) for row in selected) / len(selected),
            "mean_delta_y": sum(float(row["delta_y"]) for row in selected) / len(selected),
            "mean_delta_distance": sum(float(row["delta_distance"]) for row in selected) / len(selected),
            "mean_turn_bias": sum(float(row["turn_bias"]) for row in selected) / len(selected),
            "collision_rate": sum(bool(row["collision"]) for row in selected) / len(selected),
        })
    return output


def summarize_bias(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for group in BIAS_GROUPS:
        selected = [row for row in rows if bias_group(float(row["turn_bias"])) == group]
        if not selected:
            continue
        output.append({
            "bias_group": group,
            "count": len(selected),
            "mean_delta_heading": sum(float(row["delta_heading"]) for row in selected) / len(selected),
            "mean_delta_x": sum(float(row["delta_x"]) for row in selected) / len(selected),
            "mean_delta_y": sum(float(row["delta_y"]) for row in selected) / len(selected),
            "mean_delta_distance": sum(float(row["delta_distance"]) for row in selected) / len(selected),
        })
    return output


def plot_causality(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"turn_left": "tab:blue", "turn_right": "tab:orange", "walk_forward": "tab:green", "rest": "gray", "idle": "black"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for action in ACTIONS:
        selected = [row for row in rows if row["requested_action"] == action]
        if not selected:
            continue
        axes[0].scatter(
            [float(row["turn_bias"]) for row in selected],
            [float(row["delta_heading"]) for row in selected],
            s=16,
            alpha=0.65,
            color=colors[action],
            label=action,
        )
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("turn_bias")
    axes[0].set_ylabel("delta_heading (radians)")
    axes[0].set_title("Motor bias vs heading change")
    axes[0].legend()
    actions = []
    means = []
    for action in ACTIONS:
        selected = [row for row in rows if row["requested_action"] == action]
        if selected:
            actions.append(action)
            means.append(sum(float(row["delta_heading"]) for row in selected) / len(selected))
    axes[1].bar(actions, means, color=[colors[action] for action in actions])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("mean delta_heading (radians)")
    axes[1].set_title("Action-to-body heading contract")
    axes[1].tick_params(axis="x", rotation=25)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/directional_navigation/trajectory.csv")
    parser.add_argument("--output-dir", default="results/directional_navigation")
    args = parser.parse_args()

    grouped = read_rows(Path(args.input))
    derived = []
    for rows in grouped.values():
        derived.extend(derive_rows(rows))
    if not derived:
        raise ValueError(f"No timestep deltas found in {args.input}")

    action_summary = summarize(derived, "requested_action", ACTIONS)
    bias_summary = summarize_bias(derived)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "motor_action_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(action_summary)
    with (output_dir / "motor_bias_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BIAS_FIELDS)
        writer.writeheader()
        writer.writerows(bias_summary)
    with (output_dir / "motor_body_timestep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESTEP_FIELDS)
        writer.writeheader()
        writer.writerows(derived)

    plot_causality(derived, output_dir / "motor_body_causality.png")
    print("requested_action count mean_delta_heading mean_delta_distance collision_rate")
    for row in action_summary:
        print(
            f"{row['requested_action']:<15} {row['count']:5d} "
            f"{row['mean_delta_heading']:17.4f} {row['mean_delta_distance']:18.4f} "
            f"{row['collision_rate']:14.3f}"
        )
    print("bias_group count mean_delta_heading mean_delta_distance")
    for row in bias_summary:
        print(
            f"{row['bias_group']:<10} {row['count']:5d} "
            f"{row['mean_delta_heading']:17.4f} {row['mean_delta_distance']:18.4f}"
        )
    print(f"Saved action summary to {output_dir / 'motor_action_summary.csv'}")
    print(f"Saved timestep causality to {output_dir / 'motor_body_timestep.csv'}")
    print(f"Saved causality plot to {output_dir / 'motor_body_causality.png'}")


if __name__ == "__main__":
    main()
