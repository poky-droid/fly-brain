"""Compare FlyWire, oracle, and minimal-intervention navigation controls."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from virtual_brain import load_connectome
from virtual_fly import Body2D, Environment2D, FlyWireBrain, LightSource, MotorCommand, VirtualFlySimulator


TARGETS = {
    "left": (15.0, 20.0),
    "right": (15.0, 0.0),
    "front": (25.0, 10.0),
}
MODES = ("baseline", "oracle", "intervention")
SUMMARY_FIELDS = [
    "mode", "scenario", "steps", "initial_distance", "final_distance", "min_distance",
    "distance_reduction", "final_heading_error", "turn_count", "movement_fraction",
    "collision_rate", "mean_turn_bias", "mean_locomotion", "action_distribution",
]
TRAJECTORY_FIELDS = [
    "mode", "scenario", "step", "x", "y", "heading", "distance_to_target", "heading_error",
    "requested_action", "action", "left_drive", "right_drive", "turn_bias",
    "locomotion_drive", "collision",
]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def target_error(body: Body2D, target: tuple[float, float]) -> tuple[float, float]:
    dx = target[0] - body.x
    dy = target[1] - body.y
    return math.hypot(dx, dy), normalize_angle(math.atan2(dy, dx) - body.heading)


class OracleController:
    def __init__(self, body: Body2D, target: tuple[float, float]) -> None:
        self.body = body
        self.target = target

    def decide(self, sensor: dict[str, float | bool]) -> MotorCommand:
        _, error = target_error(self.body, self.target)
        if abs(error) > 0.15:
            action = "turn_left" if error > 0.0 else "turn_right"
            drive = min(1.0, max(0.2, abs(error) / math.pi))
            return MotorCommand(action, drive, 0.0, 0.0)
        return MotorCommand("walk_forward", 1.0, 0.0, 0.0)


class HeadingGuard:
    """Diagnostic intervention that changes only misaligned brain actions."""

    def __init__(self, brain: FlyWireBrain, body: Body2D, target: tuple[float, float]) -> None:
        self.brain = brain
        self.body = body
        self.target = target

    def reset(self) -> None:
        self.brain.reset()

    def step(self, sensor: dict[str, float | bool]) -> MotorCommand:
        command = self.brain.step(sensor)
        _, error = target_error(self.body, self.target)
        if abs(error) <= 0.15:
            return command
        desired = "turn_left" if error > 0.0 else "turn_right"
        opposite = "turn_right" if desired == "turn_left" else "turn_left"
        if command.action in {"rest", "idle", "walk_forward", opposite}:
            return MotorCommand(
                desired,
                command.locomotion_drive,
                command.left_drive,
                command.right_drive,
                command.endocrine_drive,
            )
        return command


def scenario_environment(target: tuple[float, float]) -> Environment2D:
    return Environment2D(
        width=30.0,
        height=30.0,
        light=LightSource(target[0], target[1], radius=30.0),
    )


def evaluate(
    mode: str,
    scenario: str,
    brain: FlyWireBrain,
    steps: int,
) -> tuple[dict, list[dict]]:
    target = TARGETS[scenario]
    body = Body2D(x=15.0, y=10.0, heading=0.0, velocity=0.5)
    environment = scenario_environment(target)
    controller = None
    active_brain = None
    if mode == "oracle":
        controller = OracleController(body, target)
    elif mode == "baseline":
        active_brain = brain
        brain.reset()
    else:
        active_brain = HeadingGuard(brain, body, target)
        active_brain.reset()

    simulator = VirtualFlySimulator(body, environment, controller=controller, brain=active_brain)
    initial_distance, _ = target_error(body, target)
    initial_x, initial_y = body.x, body.y
    trajectory = simulator.run(steps)
    rows = []
    previous_x, previous_y = initial_x, initial_y
    moved = 0
    for state in trajectory:
        distance, error = target_error(Body2D(state.x, state.y, state.heading), target)
        if (state.x, state.y) != (previous_x, previous_y):
            moved += 1
        previous_x, previous_y = state.x, state.y
        rows.append({
            "mode": mode,
            "scenario": scenario,
            "step": state.step,
            "x": state.x,
            "y": state.y,
            "heading": state.heading,
            "distance_to_target": distance,
            "heading_error": error,
            "requested_action": state.requested_action,
            "action": state.action,
            "left_drive": state.left_drive,
            "right_drive": state.right_drive,
            "turn_bias": state.right_drive - state.left_drive,
            "locomotion_drive": state.locomotion_drive,
            "collision": state.collision,
        })

    collision_count = sum(row["collision"] for row in rows)
    summary = {
        "mode": mode,
        "scenario": scenario,
        "steps": steps,
        "initial_distance": initial_distance,
        "final_distance": rows[-1]["distance_to_target"],
        "min_distance": min(row["distance_to_target"] for row in rows),
        "distance_reduction": initial_distance - rows[-1]["distance_to_target"],
        "final_heading_error": rows[-1]["heading_error"],
        "turn_count": sum(row["action"] in {"turn_left", "turn_right"} for row in rows),
        "movement_fraction": moved / steps,
        "collision_rate": collision_count / steps,
        "mean_turn_bias": sum(row["turn_bias"] for row in rows) / steps,
        "mean_locomotion": sum(row["locomotion_drive"] for row in rows) / steps,
        "action_distribution": dict(Counter(row["action"] for row in rows)),
    }
    return summary, rows


def plot_results(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"baseline": "tab:blue", "oracle": "tab:green", "intervention": "tab:orange"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for scenario_index, scenario in enumerate(TARGETS):
        axis = axes[scenario_index]
        for mode in MODES:
            selected = [row for row in rows if row["scenario"] == scenario and row["mode"] == mode]
            axis.plot(
                [row["x"] for row in selected],
                [row["y"] for row in selected],
                color=colors[mode],
                label=mode,
            )
        target = TARGETS[scenario]
        axis.scatter(target[0], target[1], color="black", marker="*", s=120)
        axis.set_title(scenario)
        axis.set_xlim(0, 30)
        axis.set_ylim(0, 30)
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-sensory", type=int, default=50)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--output-dir", default="results/intervention")
    args = parser.parse_args()

    print("Loading FlyWire connectome...")
    brain = FlyWireBrain(load_connectome(), max_sensory=args.max_sensory, hops=args.hops)
    summaries = []
    trajectories = []
    for mode in MODES:
        for scenario in TARGETS:
            summary, rows = evaluate(mode, scenario, brain, args.steps)
            summaries.append(summary)
            trajectories.extend(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        writer.writerows(trajectories)
    for mode in MODES:
        with (output_dir / f"{mode}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(row for row in summaries if row["mode"] == mode)

    plot_results(trajectories, output_dir / "intervention_trajectory.png")
    print("mode scenario distance_reduction final_distance turns movement collisions")
    for row in summaries:
        print(
            f"{row['mode']:<13} {row['scenario']:<7} {row['distance_reduction']:18.3f} "
            f"{row['final_distance']:14.3f} {row['turn_count']:5d} "
            f"{row['movement_fraction']:8.3f} {row['collision_rate']:10.3f}"
        )
    print(f"Saved comparison to {output_dir / 'comparison.csv'}")
    print(f"Saved trajectories to {output_dir / 'trajectory.csv'}")
    print(f"Saved plot to {output_dir / 'intervention_trajectory.png'}")


if __name__ == "__main__":
    main()
