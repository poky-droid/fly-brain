"""Evaluate closed-loop VirtualFly behavior across controlled scenarios."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from virtual_brain import load_connectome
from virtual_fly import Body2D, Environment2D, FlyWireBrain, LightSource, Obstacle, VirtualFlySimulator


@dataclass(frozen=True)
class Scenario:
    name: str
    body: Body2D
    environment: Environment2D


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            "light_source",
            Body2D(x=5.0, y=5.0, heading=0.0, velocity=0.5),
            Environment2D(width=30.0, height=30.0, light=LightSource(25.0, 20.0, radius=35.0)),
        ),
        Scenario(
            "obstacle",
            Body2D(x=5.0, y=5.0, heading=0.0, velocity=0.5),
            Environment2D(
                width=30.0,
                height=30.0,
                light=LightSource(25.0, 5.0, radius=35.0),
                obstacles=[Obstacle(10.0, 0.0, 3.0, 10.0)],
            ),
        ),
        Scenario(
            "left_stimulus",
            Body2D(x=15.0, y=10.0, heading=0.0, velocity=0.5),
            Environment2D(width=30.0, height=30.0, light=LightSource(15.0, 25.0, radius=30.0)),
        ),
        Scenario(
            "right_stimulus",
            Body2D(x=15.0, y=10.0, heading=0.0, velocity=0.5),
            Environment2D(width=30.0, height=30.0, light=LightSource(15.0, 0.0, radius=30.0)),
        ),
        Scenario(
            "front_stimulus",
            Body2D(x=15.0, y=10.0, heading=0.0, velocity=0.5),
            Environment2D(width=30.0, height=30.0, light=LightSource(25.0, 10.0, radius=30.0)),
        ),
    ]


def evaluate_scenario(brain: FlyWireBrain, scenario: Scenario, steps: int):
    brain.reset()
    simulator = VirtualFlySimulator(scenario.body, scenario.environment, brain=brain)
    light = scenario.environment.light
    initial_distance = math.hypot(light.x - scenario.body.x, light.y - scenario.body.y) if light else math.inf
    initial_x, initial_y = scenario.body.x, scenario.body.y
    trajectory = simulator.run(steps)
    final = trajectory[-1]
    distances = [
        math.hypot(light.x - state.x, light.y - state.y) if light else math.inf
        for state in trajectory
    ]
    mean_locomotion = sum(state.locomotion_drive for state in trajectory) / len(trajectory)
    mean_turn_bias = sum(state.right_drive - state.left_drive for state in trajectory) / len(trajectory)
    turn_count = sum(state.action in {"turn_left", "turn_right"} for state in trajectory)
    previous_x, previous_y = initial_x, initial_y
    movement_steps = 0
    for state in trajectory:
        if (state.x, state.y) != (previous_x, previous_y):
            movement_steps += 1
        previous_x, previous_y = state.x, state.y

    return {
        "scenario": scenario.name,
        "initial_distance": initial_distance,
        "final_distance": distances[-1],
        "min_distance": min(distances),
        "final_x": final.x,
        "final_y": final.y,
        "final_heading": final.heading,
        "mean_locomotion": mean_locomotion,
        "mean_turn_bias": mean_turn_bias,
        "turn_count": turn_count,
        "collision_count": sum(state.collision for state in trajectory),
        "collision_rate": sum(state.collision for state in trajectory) / len(trajectory),
        "movement_fraction": movement_steps / len(trajectory),
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-sensory", type=int, default=50)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--output-dir", default="results/behavioral_scenarios")
    args = parser.parse_args()

    print("Loading FlyWire connectome...")
    brain = FlyWireBrain(load_connectome(), max_sensory=args.max_sensory, hops=args.hops)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_scenario(brain, scenario, args.steps) for scenario in build_scenarios()]

    summary_fields = [
        "scenario", "initial_distance", "final_distance", "min_distance",
        "final_x", "final_y", "final_heading", "mean_locomotion",
        "mean_turn_bias", "turn_count", "collision_count",
        "collision_rate", "movement_fraction",
    ]
    with (output_dir / "scenario_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in summary_fields})

    trajectory_fields = [
        "scenario", "step", "x", "y", "heading", "left_sensor", "right_sensor",
        "locomotion_drive", "left_drive", "right_drive", "turn_bias", "action", "collision",
        "requested_action",
    ]
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_fields)
        writer.writeheader()
        for result in results:
            for state in result["trajectory"]:
                writer.writerow({
                    "scenario": result["scenario"],
                    "step": state.step,
                    "x": state.x,
                    "y": state.y,
                    "heading": state.heading,
                    "left_sensor": state.left_sensor,
                    "right_sensor": state.right_sensor,
                    "locomotion_drive": state.locomotion_drive,
                    "left_drive": state.left_drive,
                    "right_drive": state.right_drive,
                    "turn_bias": state.right_drive - state.left_drive,
                    "action": state.action,
                    "collision": state.collision,
                    "requested_action": state.requested_action,
                })

    print("scenario initial_distance final_distance mean_locomotion mean_turn_bias turns collisions collision_rate movement_fraction")
    for result in results:
        print(
            f"{result['scenario']:<16} {result['initial_distance']:15.3f} "
            f"{result['final_distance']:14.3f} {result['mean_locomotion']:16.3f} "
            f"{result['mean_turn_bias']:15.3f} {result['turn_count']:6d} "
            f"{result['collision_count']:10d} {result['collision_rate']:14.3f} "
            f"{result['movement_fraction']:17.3f}"
        )
    print(f"Saved metrics to {output_dir / 'scenario_summary.csv'}")
    print(f"Saved trajectories to {output_dir / 'trajectory.csv'}")


if __name__ == "__main__":
    main()