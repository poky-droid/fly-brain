"""Compare FlyWire actions with a geometric navigation reference."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

SCENARIOS = ("left", "right", "front")
ACTIONS = ("turn_left", "turn_right", "walk_forward")

SUMMARY_FIELDS = [
    "scenario", "steps", "action_match_rate", "turn_match_rate",
    "left_turn_confusion", "right_turn_confusion", "forward_confusion",
    "mean_abs_heading_error", "mean_turn_bias",
]
TIMESTEP_FIELDS = [
    "scenario", "step", "heading_error", "mean_abs_heading_error", "turn_bias",
    "flywire_action", "ideal_action", "action_match", "confusion_class",
]
MATRIX_FIELDS = ["scenario", "flywire_action", "ideal_action", "count", "rate"]


def read_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped = {scenario: [] for scenario in SCENARIOS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["target_side"] in grouped:
                grouped[row["target_side"]].append(row)
    return grouped


def geometric_action(heading_error: float, threshold: float) -> str:
    if heading_error > threshold:
        return "turn_left"
    if heading_error < -threshold:
        return "turn_right"
    return "walk_forward"


def confusion_class(flywire_action: str, ideal_action: str) -> str:
    if flywire_action == ideal_action:
        return "match"
    if ideal_action == "turn_left":
        return "left_turn_confusion"
    if ideal_action == "turn_right":
        return "right_turn_confusion"
    return "forward_confusion"


def analyze_scenario(scenario: str, rows: list[dict[str, str]], threshold: float):
    analyzed = []
    for row in rows:
        heading_error = float(row["heading_error"])
        flywire_action = row["requested_action"]
        ideal_action = geometric_action(heading_error, threshold)
        analyzed.append({
            "scenario": scenario,
            "step": row["step"],
            "heading_error": heading_error,
            "mean_abs_heading_error": abs(heading_error),
            "turn_bias": float(row["turn_bias"]),
            "flywire_action": flywire_action,
            "ideal_action": ideal_action,
            "action_match": flywire_action == ideal_action,
            "confusion_class": confusion_class(flywire_action, ideal_action),
        })

    total = len(analyzed)
    ideal_turns = [row for row in analyzed if row["ideal_action"] in {"turn_left", "turn_right"}]
    turn_matches = sum(row["flywire_action"] == row["ideal_action"] for row in ideal_turns)
    confusion_counts = Counter(row["confusion_class"] for row in analyzed)
    summary = {
        "scenario": scenario,
        "steps": total,
        "action_match_rate": sum(row["action_match"] for row in analyzed) / total,
        "turn_match_rate": turn_matches / len(ideal_turns) if ideal_turns else 0.0,
        "left_turn_confusion": confusion_counts["left_turn_confusion"] / total,
        "right_turn_confusion": confusion_counts["right_turn_confusion"] / total,
        "forward_confusion": confusion_counts["forward_confusion"] / total,
        "mean_abs_heading_error": sum(row["mean_abs_heading_error"] for row in analyzed) / total,
        "mean_turn_bias": sum(row["turn_bias"] for row in analyzed) / total,
    }
    matrix = []
    counts = Counter((row["flywire_action"], row["ideal_action"]) for row in analyzed)
    for flywire_action in ACTIONS:
        for ideal_action in ACTIONS:
            matrix.append({
                "scenario": scenario,
                "flywire_action": flywire_action,
                "ideal_action": ideal_action,
                "count": counts[(flywire_action, ideal_action)],
                "rate": counts[(flywire_action, ideal_action)] / total,
            })
    return summary, analyzed, matrix


def plot_analysis(matrix_rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, scenario in zip(axes, SCENARIOS):
        values = [
            [
                next(row["rate"] for row in matrix_rows
                     if row["scenario"] == scenario
                     and row["flywire_action"] == flywire_action
                     and row["ideal_action"] == ideal_action)
                for ideal_action in ACTIONS
            ]
            for flywire_action in ACTIONS
        ]
        image = axis.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_title(scenario)
        axis.set_xticks(range(3), ["ideal L", "ideal R", "ideal F"])
        axis.set_yticks(range(3), ["FlyWire L", "FlyWire R", "FlyWire F"])
        for row_index in range(3):
            for column_index in range(3):
                axis.text(column_index, row_index, f"{values[row_index][column_index]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axes, label="fraction")
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/directional_navigation/trajectory.csv")
    parser.add_argument("--output-dir", default="results/intervention")
    parser.add_argument("--heading-threshold", type=float, default=0.15)
    args = parser.parse_args()

    grouped = read_rows(Path(args.input))
    summaries = []
    timestep_rows = []
    matrix_rows = []
    for scenario in SCENARIOS:
        if not grouped[scenario]:
            raise ValueError(f"Missing scenario in {args.input}: {scenario}")
        summary, analyzed, matrix = analyze_scenario(scenario, grouped[scenario], args.heading_threshold)
        summaries.append(summary)
        timestep_rows.extend(analyzed)
        matrix_rows.extend(matrix)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "translation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "translation_timestep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESTEP_FIELDS)
        writer.writeheader()
        writer.writerows(timestep_rows)
    with (output_dir / "translation_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS)
        writer.writeheader()
        writer.writerows(matrix_rows)

    plot_analysis(matrix_rows, output_dir / "translation_analysis.png")
    print("scenario match_rate turn_match left_confusion right_confusion forward_confusion mean_bias")
    for row in summaries:
        print(
            f"{row['scenario']:<9} {row['action_match_rate']:10.3f} {row['turn_match_rate']:10.3f} "
            f"{row['left_turn_confusion']:14.3f} {row['right_turn_confusion']:15.3f} "
            f"{row['forward_confusion']:16.3f} {row['mean_turn_bias']:10.4f}"
        )
    print(f"Saved translation summary to {output_dir / 'translation_summary.csv'}")
    print(f"Saved timestep comparison to {output_dir / 'translation_timestep.csv'}")
    print(f"Saved confusion matrix to {output_dir / 'translation_confusion_matrix.csv'}")
    print(f"Saved plot to {output_dir / 'translation_analysis.png'}")


if __name__ == "__main__":
    main()
