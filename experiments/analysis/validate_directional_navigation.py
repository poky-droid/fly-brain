"""Measure closed-loop navigation toward targets on the left, right, and front."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from virtual_brain import load_connectome
from virtual_fly import Body2D, Environment2D, FlyWireBrain, LightSource, VirtualFlySimulator


TARGETS = {
    "left": (15.0, 20.0),
    "right": (15.0, 0.0),
    "front": (25.0, 10.0),
}

SUMMARY_FIELDS = [
    "target_side", "steps", "initial_distance", "final_distance", "min_distance",
    "distance_change", "distance_reduction", "initial_heading_error", "final_heading_error",
    "turn_count", "movement_fraction", "collision_rate", "mean_turn_bias", "mean_locomotion",
]

TRAJECTORY_FIELDS = [
    "target_side", "step", "x", "y", "heading", "distance_to_target", "heading_error",
    "left_sensor", "right_sensor", "requested_action", "action", "locomotion_drive",
    "left_drive", "right_drive", "turn_bias", "collision",
]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def target_metrics(body: Body2D, target: tuple[float, float]) -> tuple[float, float]:
    dx = target[0] - body.x
    dy = target[1] - body.y
    distance = math.hypot(dx, dy)
    heading_error = normalize_angle(math.atan2(dy, dx) - body.heading)
    return distance, heading_error


def evaluate_target(brain: FlyWireBrain, target_side: str, steps: int) -> tuple[dict, list[dict]]:
    target = TARGETS[target_side]
    body = Body2D(x=15.0, y=10.0, heading=0.0, velocity=0.5)
    environment = Environment2D(
        width=30.0,
        height=30.0,
        light=LightSource(target[0], target[1], radius=30.0),
    )
    brain.reset()
    simulator = VirtualFlySimulator(body, environment, brain=brain)
    initial_distance, initial_heading_error = target_metrics(body, target)
    initial_x, initial_y = body.x, body.y
    trajectory = simulator.run(steps)

    rows = []
    previous_x, previous_y = initial_x, initial_y
    moved = 0
    for state in trajectory:
        distance, heading_error = target_metrics(
            Body2D(x=state.x, y=state.y, heading=state.heading), target
        )
        if (state.x, state.y) != (previous_x, previous_y):
            moved += 1
        previous_x, previous_y = state.x, state.y
        rows.append({
            "target_side": target_side,
            "step": state.step,
            "x": state.x,
            "y": state.y,
            "heading": state.heading,
            "distance_to_target": distance,
            "heading_error": heading_error,
            "left_sensor": state.left_sensor,
            "right_sensor": state.right_sensor,
            "requested_action": state.requested_action,
            "action": state.action,
            "locomotion_drive": state.locomotion_drive,
            "left_drive": state.left_drive,
            "right_drive": state.right_drive,
            "turn_bias": state.right_drive - state.left_drive,
            "collision": state.collision,
        })

    final = rows[-1]
    collision_count = sum(row["collision"] for row in rows)
    summary = {
        "target_side": target_side,
        "steps": steps,
        "initial_distance": initial_distance,
        "final_distance": final["distance_to_target"],
        "min_distance": min(row["distance_to_target"] for row in rows),
        "distance_change": final["distance_to_target"] - initial_distance,
        "distance_reduction": initial_distance - final["distance_to_target"],
        "initial_heading_error": initial_heading_error,
        "final_heading_error": final["heading_error"],
        "turn_count": sum(row["action"] in {"turn_left", "turn_right"} for row in rows),
        "movement_fraction": moved / steps,
        "collision_rate": collision_count / steps,
        "mean_turn_bias": sum(row["turn_bias"] for row in rows) / steps,
        "mean_locomotion": sum(row["locomotion_drive"] for row in rows) / steps,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-sensory", type=int, default=50)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--output-dir", default="results/directional_navigation")
    args = parser.parse_args()

    print("Loading FlyWire connectome...")
    brain = FlyWireBrain(load_connectome(), max_sensory=args.max_sensory, hops=args.hops)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    trajectories = []
    for target_side in TARGETS:
        summary, rows = evaluate_target(brain, target_side, args.steps)
        summaries.append(summary)
        trajectories.extend(rows)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        writer.writerows(trajectories)

    print("target initial_distance final_distance reduction initial_error final_error turns movement collisions mean_bias mean_locomotion")
    for row in summaries:
        print(
            f"{row['target_side']:<6} {row['initial_distance']:15.3f} {row['final_distance']:14.3f} "
            f"{row['distance_reduction']:9.3f} {row['initial_heading_error']:13.3f} "
            f"{row['final_heading_error']:11.3f} {row['turn_count']:6d} "
            f"{row['movement_fraction']:8.3f} {row['collision_rate']:10.3f} "
            f"{row['mean_turn_bias']:10.3f} {row['mean_locomotion']:15.3f}"
        )
    print(f"Saved summary to {output_dir / 'summary.csv'}")
    print(f"Saved trajectory to {output_dir / 'trajectory.csv'}")


if __name__ == "__main__":
    main()
