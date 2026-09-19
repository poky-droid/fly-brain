"""Analyze and plot directional navigation failures from a trajectory CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

TARGETS = {
    "left": (15.0, 20.0),
    "right": (15.0, 0.0),
    "front": (25.0, 10.0),
}

SUMMARY_FIELDS = [
    "scenario", "initial_distance", "final_distance", "distance_reduction",
    "initial_heading_error", "final_heading_error", "min_heading_error",
    "mean_abs_heading_error", "turn_count", "movement_fraction", "collision_rate",
]

HEADING_FIELDS = [
    "scenario", "step", "x", "y", "heading", "target_x", "target_y",
    "distance_to_target", "heading_error", "abs_heading_error", "left_drive",
    "right_drive", "turn_bias", "requested_action", "action", "collision",
]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def read_trajectory(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {name: [] for name in TARGETS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["target_side"] in grouped:
                grouped[row["target_side"]].append(row)
    return grouped


def analyze_rows(scenario: str, rows: list[dict[str, str]]) -> tuple[dict, list[dict]]:
    target_x, target_y = TARGETS[scenario]
    initial_x, initial_y, initial_heading = 15.0, 10.0, 0.0
    heading_rows = []
    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        heading = float(row["heading"])
        distance = math.hypot(target_x - x, target_y - y)
        heading_error = normalize_angle(math.atan2(target_y - y, target_x - x) - heading)
        heading_rows.append({
            "scenario": scenario,
            "step": row["step"],
            "x": x,
            "y": y,
            "heading": heading,
            "target_x": target_x,
            "target_y": target_y,
            "distance_to_target": distance,
            "heading_error": heading_error,
            "abs_heading_error": abs(heading_error),
            "left_drive": row["left_drive"],
            "right_drive": row["right_drive"],
            "turn_bias": row["turn_bias"],
            "requested_action": row["requested_action"],
            "action": row["action"],
            "collision": row["collision"],
        })

    final = heading_rows[-1]
    initial_distance = math.hypot(target_x - initial_x, target_y - initial_y)
    initial_heading_error = normalize_angle(
        math.atan2(target_y - initial_y, target_x - initial_x) - initial_heading
    )
    final_distance = float(final["distance_to_target"])
    return (
        {
            "scenario": scenario,
            "initial_distance": initial_distance,
            "final_distance": final_distance,
            "distance_reduction": initial_distance - final_distance,
            "initial_heading_error": initial_heading_error,
            "final_heading_error": final["heading_error"],
            "min_heading_error": min(row["abs_heading_error"] for row in heading_rows),
            "mean_abs_heading_error": sum(row["abs_heading_error"] for row in heading_rows) / len(heading_rows),
            "turn_count": sum(row["action"] in {"turn_left", "turn_right"} for row in heading_rows),
            "movement_fraction": sum(
                (float(row["x"]), float(row["y"])) != (
                    float(rows[index - 1]["x"]),
                    float(rows[index - 1]["y"]),
                )
                for index, row in enumerate(rows)
                if index > 0
            ) / len(rows),
            "collision_rate": sum(row["collision"] == "True" for row in heading_rows) / len(heading_rows),
        },
        heading_rows,
    )


def plot_results(rows_by_scenario: dict[str, list[dict]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, (trajectory_axis, error_axis) = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    colors = {"left": "tab:blue", "right": "tab:orange", "front": "tab:green"}
    for scenario, rows in rows_by_scenario.items():
        color = colors[scenario]
        trajectory_axis.plot(
            [float(row["x"]) for row in rows],
            [float(row["y"]) for row in rows],
            color=color,
            label=scenario,
        )
        trajectory_axis.scatter(float(rows[0]["x"]), float(rows[0]["y"]), color=color, marker="o")
        target_x, target_y = TARGETS[scenario]
        trajectory_axis.scatter(target_x, target_y, color=color, marker="*", s=120)
        error_axis.plot(
            [int(row["step"]) for row in rows],
            [float(row["heading_error"]) for row in rows],
            color=color,
            label=scenario,
        )

    trajectory_axis.set_title("Directional navigation trajectories")
    trajectory_axis.set_xlabel("x")
    trajectory_axis.set_ylabel("y")
    trajectory_axis.set_aspect("equal")
    trajectory_axis.grid(alpha=0.25)
    trajectory_axis.legend()
    error_axis.set_title("Heading error over time")
    error_axis.set_xlabel("step")
    error_axis.set_ylabel("error (radians)")
    error_axis.axhline(0.0, color="black", linewidth=0.8)
    error_axis.grid(alpha=0.25)
    error_axis.legend()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/directional_navigation/trajectory.csv")
    parser.add_argument("--output-dir", default="results/directional_navigation")
    args = parser.parse_args()

    grouped = read_trajectory(Path(args.input))
    missing = [scenario for scenario, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"Missing scenarios in {args.input}: {missing}")

    summaries = []
    heading_rows = []
    for scenario, rows in grouped.items():
        summary, analyzed = analyze_rows(scenario, rows)
        summaries.append(summary)
        heading_rows.extend(analyzed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "failure_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "heading_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADING_FIELDS)
        writer.writeheader()
        writer.writerows(heading_rows)

    plot_results(grouped, output_dir / "trajectory_plot.png")
    print("scenario distance_reduction final_heading_error min_abs_heading_error mean_abs_heading_error")
    for summary in summaries:
        print(
            f"{summary['scenario']:<9} {summary['distance_reduction']:18.3f} "
            f"{summary['final_heading_error']:20.3f} {summary['min_heading_error']:22.3f} "
            f"{summary['mean_abs_heading_error']:23.3f}"
        )
    print(f"Saved failure summary to {output_dir / 'failure_summary.csv'}")
    print(f"Saved heading analysis to {output_dir / 'heading_analysis.csv'}")
    print(f"Saved trajectory plot to {output_dir / 'trajectory_plot.png'}")


if __name__ == "__main__":
    main()
