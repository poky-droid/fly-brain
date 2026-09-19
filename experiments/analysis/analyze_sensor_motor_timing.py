"""Analyze sensor-to-motor-to-body timing in closed-loop trajectories."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

SCENARIOS = ("left", "right", "front")
EPSILON = 1e-9

SUMMARY_FIELDS = [
    "scenario", "steps", "sensor_change_count", "motor_change_count", "action_change_count",
    "best_sensor_motor_lag", "best_sensor_action_lag", "best_action_heading_lag",
    "sensor_motor_peak_correlation", "sensor_action_peak_correlation",
    "action_heading_peak_correlation", "mean_heading_response",
]

LAG_FIELDS = ["scenario", "signal_pair", "lag", "correlation"]
TIMING_FIELDS = [
    "scenario", "step", "sensor_asymmetry", "delta_sensor_asymmetry", "turn_bias",
    "delta_turn_bias", "action_code", "action_changed", "delta_heading", "delta_x",
    "delta_y", "delta_distance", "collision",
]


def read_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["target_side"]].append(row)
    return grouped


def action_code(action: str) -> float:
    if action == "turn_left":
        return -1.0
    if action == "turn_right":
        return 1.0
    if action == "walk_forward":
        return 0.5
    return 0.0


def derive_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    timing = []
    for index, (previous, current) in enumerate(zip(rows, rows[1:]), start=1):
        previous_sensor = float(previous["left_sensor"]) - float(previous["right_sensor"])
        current_sensor = float(current["left_sensor"]) - float(current["right_sensor"])
        previous_bias = float(previous["turn_bias"])
        current_bias = float(current["turn_bias"])
        timing.append({
            "scenario": current["target_side"],
            "step": int(current["step"]),
            "sensor_asymmetry": current_sensor,
            "delta_sensor_asymmetry": current_sensor - previous_sensor,
            "turn_bias": current_bias,
            "delta_turn_bias": current_bias - previous_bias,
            "action_code": action_code(current["requested_action"]),
            "action_changed": current["requested_action"] != previous["requested_action"],
            "delta_heading": math.atan2(
                math.sin(float(current["heading"]) - float(previous["heading"])),
                math.cos(float(current["heading"]) - float(previous["heading"])),
            ),
            "delta_x": float(current["x"]) - float(previous["x"]),
            "delta_y": float(current["y"]) - float(previous["y"]),
            "delta_distance": float(current["distance_to_target"]) - float(previous["distance_to_target"]),
            "collision": current["collision"] == "True",
        })
    return timing


def correlation_at_lag(first: list[float], second: list[float], lag: int) -> float:
    if lag >= len(first) or lag >= len(second):
        return 0.0
    if lag == 0:
        left, right = first, second
    else:
        left, right = first[:-lag], second[lag:]
    if len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def lag_profile(first: list[float], second: list[float], max_lag: int) -> list[tuple[int, float]]:
    return [(lag, correlation_at_lag(first, second, lag)) for lag in range(max_lag + 1)]


def best_lag(profile: list[tuple[int, float]]) -> tuple[int, float]:
    return max(profile, key=lambda item: abs(item[1]))


def analyze_scenario(scenario: str, timing: list[dict[str, object]], max_lag: int):
    sensor_changes = [abs(float(row["delta_sensor_asymmetry"])) for row in timing]
    motor_changes = [abs(float(row["delta_turn_bias"])) for row in timing]
    action_changes = [1.0 if row["action_changed"] else 0.0 for row in timing]
    heading_changes = [abs(float(row["delta_heading"])) for row in timing]
    sensor_values = [float(row["sensor_asymmetry"]) for row in timing]
    motor_values = [float(row["turn_bias"]) for row in timing]
    action_values = [float(row["action_code"]) for row in timing]

    profiles = {
        "sensor_change_to_motor_change": lag_profile(sensor_changes, motor_changes, max_lag),
        "sensor_change_to_action_change": lag_profile(sensor_changes, action_changes, max_lag),
        "action_change_to_heading_change": lag_profile(action_changes, heading_changes, max_lag),
        "sensor_asymmetry_to_turn_bias": lag_profile(sensor_values, motor_values, max_lag),
        "turn_bias_to_heading_change": lag_profile(motor_values, heading_changes, max_lag),
    }
    lag_rows = [
        {"scenario": scenario, "signal_pair": pair, "lag": lag, "correlation": correlation}
        for pair, profile in profiles.items()
        for lag, correlation in profile
    ]
    sensor_motor_lag, sensor_motor_corr = best_lag(profiles["sensor_change_to_motor_change"])
    sensor_action_lag, sensor_action_corr = best_lag(profiles["sensor_change_to_action_change"])
    action_heading_lag, action_heading_corr = best_lag(profiles["action_change_to_heading_change"])
    summary = {
        "scenario": scenario,
        "steps": len(timing) + 1,
        "sensor_change_count": sum(value > EPSILON for value in sensor_changes),
        "motor_change_count": sum(value > EPSILON for value in motor_changes),
        "action_change_count": int(sum(action_changes)),
        "best_sensor_motor_lag": sensor_motor_lag,
        "best_sensor_action_lag": sensor_action_lag,
        "best_action_heading_lag": action_heading_lag,
        "sensor_motor_peak_correlation": sensor_motor_corr,
        "sensor_action_peak_correlation": sensor_action_corr,
        "action_heading_peak_correlation": action_heading_corr,
        "mean_heading_response": sum(float(row["delta_heading"]) for row in timing) / len(timing),
    }
    return summary, lag_rows


def plot_timing(grouped_timing: dict[str, list[dict[str, object]]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    colors = {"left": "tab:blue", "right": "tab:orange", "front": "tab:green"}
    for scenario, rows in grouped_timing.items():
        color = colors[scenario]
        steps = [int(row["step"]) for row in rows]
        axes[0].plot(steps, [row["sensor_asymmetry"] for row in rows], color=color, label=scenario)
        axes[1].plot(steps, [row["turn_bias"] for row in rows], color=color, label=scenario)
        axes[2].plot(steps, [row["delta_heading"] for row in rows], color=color, label=scenario)
    axes[0].set_ylabel("sensor L-R")
    axes[1].set_ylabel("turn bias")
    axes[2].set_ylabel("delta heading")
    axes[2].set_xlabel("step")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/directional_navigation/trajectory.csv")
    parser.add_argument("--output-dir", default="results/directional_navigation")
    parser.add_argument("--max-lag", type=int, default=10)
    args = parser.parse_args()

    grouped = read_rows(Path(args.input))
    summaries = []
    lag_rows = []
    grouped_timing = {}
    for scenario in SCENARIOS:
        if scenario not in grouped:
            raise ValueError(f"Missing scenario in {args.input}: {scenario}")
        timing = derive_rows(grouped[scenario])
        grouped_timing[scenario] = timing
        summary, scenario_lags = analyze_scenario(scenario, timing, args.max_lag)
        summaries.append(summary)
        lag_rows.extend(scenario_lags)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "timing_lag_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LAG_FIELDS)
        writer.writeheader()
        writer.writerows(lag_rows)
    with (output_dir / "sensor_motor_timing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        for rows in grouped_timing.values():
            writer.writerows(rows)

    plot_timing(grouped_timing, output_dir / "sensor_motor_timing.png")
    print("scenario sensor_changes motor_changes action_changes sensor_motor_lag sensor_action_lag action_heading_lag mean_heading_response")
    for summary in summaries:
        print(
            f"{summary['scenario']:<9} {summary['sensor_change_count']:14d} "
            f"{summary['motor_change_count']:13d} {summary['action_change_count']:14d} "
            f"{summary['best_sensor_motor_lag']:16d} {summary['best_sensor_action_lag']:17d} "
            f"{summary['best_action_heading_lag']:18d} {summary['mean_heading_response']:23.4f}"
        )
    print(f"Saved timing summary to {output_dir / 'timing_summary.csv'}")
    print(f"Saved lag analysis to {output_dir / 'timing_lag_analysis.csv'}")
    print(f"Saved timestep timing to {output_dir / 'sensor_motor_timing.csv'}")
    print(f"Saved timing plot to {output_dir / 'sensor_motor_timing.png'}")


if __name__ == "__main__":
    main()
