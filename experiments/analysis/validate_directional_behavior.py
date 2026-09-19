"""Evaluate closed-loop left/right/front behavior without changing the brain."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from virtual_brain import load_connectome
from virtual_fly import Body2D, Environment2D, FlyWireBrain, LightSource, VirtualFlySimulator


CONDITIONS = {
    "left": LightSource(x=15.0, y=25.0, radius=30.0),
    "right": LightSource(x=15.0, y=0.0, radius=30.0),
    "front": LightSource(x=25.0, y=10.0, radius=30.0),
}


SUMMARY_FIELDS = [
    "condition",
    "steps",
    "mean_left_sensor",
    "mean_right_sensor",
    "mean_left_drive",
    "mean_right_drive",
    "mean_locomotion",
    "mean_turn_bias",
    "turn_count",
    "collision_count",
    "collision_rate",
    "movement_fraction",
    "action_distribution",
]

TRAJECTORY_FIELDS = [
    "condition",
    "step",
    "x",
    "y",
    "heading",
    "left_sensor",
    "right_sensor",
    "requested_action",
    "action",
    "locomotion_drive",
    "left_drive",
    "right_drive",
    "turn_bias",
    "collision",
]


def evaluate_condition(brain: FlyWireBrain, condition: str, steps: int) -> tuple[dict, list[dict]]:
    body = Body2D(x=15.0, y=10.0, heading=0.0, velocity=0.5)
    environment = Environment2D(width=30.0, height=30.0, light=CONDITIONS[condition])
    brain.reset()
    simulator = VirtualFlySimulator(body, environment, brain=brain)
    initial_x, initial_y = body.x, body.y
    trajectory = simulator.run(steps)

    rows = []
    previous_x, previous_y = initial_x, initial_y
    moved = 0
    for state in trajectory:
        if (state.x, state.y) != (previous_x, previous_y):
            moved += 1
        previous_x, previous_y = state.x, state.y
        rows.append({
            "condition": condition,
            "step": state.step,
            "x": state.x,
            "y": state.y,
            "heading": state.heading,
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

    actions = Counter(row["action"] for row in rows)
    collision_count = sum(row["collision"] for row in rows)
    summary = {
        "condition": condition,
        "steps": steps,
        "mean_left_sensor": sum(row["left_sensor"] for row in rows) / steps,
        "mean_right_sensor": sum(row["right_sensor"] for row in rows) / steps,
        "mean_left_drive": sum(row["left_drive"] for row in rows) / steps,
        "mean_right_drive": sum(row["right_drive"] for row in rows) / steps,
        "mean_locomotion": sum(row["locomotion_drive"] for row in rows) / steps,
        "mean_turn_bias": sum(row["turn_bias"] for row in rows) / steps,
        "turn_count": sum(row["action"] in {"turn_left", "turn_right"} for row in rows),
        "collision_count": collision_count,
        "collision_rate": collision_count / steps,
        "movement_fraction": moved / steps,
        "action_distribution": dict(actions),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-sensory", type=int, default=50)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--output-dir", default="results/directional_behavior")
    args = parser.parse_args()

    print("Loading FlyWire connectome...")
    brain = FlyWireBrain(load_connectome(), max_sensory=args.max_sensory, hops=args.hops)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    trajectories = []
    for condition in CONDITIONS:
        summary, rows = evaluate_condition(brain, condition, args.steps)
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

    print("condition mean_sensor_L mean_sensor_R mean_motor_L mean_motor_R mean_bias turns collisions movement actions")
    for summary in summaries:
        print(
            f"{summary['condition']:<9} {summary['mean_left_sensor']:13.3f} "
            f"{summary['mean_right_sensor']:13.3f} {summary['mean_left_drive']:12.3f} "
            f"{summary['mean_right_drive']:12.3f} {summary['mean_turn_bias']:10.3f} "
            f"{summary['turn_count']:6d} {summary['collision_count']:10d} "
            f"{summary['movement_fraction']:8.3f} {summary['action_distribution']}"
        )
    print(f"Saved summary to {output_dir / 'summary.csv'}")
    print(f"Saved trajectory to {output_dir / 'trajectory.csv'}")


if __name__ == "__main__":
    main()
