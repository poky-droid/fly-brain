"""Visualize FlyWire behavior when the fly encounters an obstacle."""

from __future__ import annotations

import argparse
from pathlib import Path

from virtual_brain import load_connectome
from virtual_fly import Body2D, Environment2D, FlyWireBrain, LightSource, Obstacle, VirtualFlySimulator


def build_obstacle_scene() -> tuple[Body2D, Environment2D]:
    body = Body2D(x=5.0, y=5.0, heading=0.0, velocity=0.5)
    environment = Environment2D(
        width=30.0,
        height=30.0,
        light=LightSource(x=25.0, y=5.0, radius=35.0),
        obstacles=[Obstacle(x=10.0, y=0.0, width=3.0, height=10.0)],
    )
    return body, environment


def plot_obstacle_run(trajectory, environment: Environment2D, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, (arena_axis, telemetry_axis) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1.0]}, constrained_layout=True
    )
    arena_axis.set_title("FlyWire VirtualFly: obstacle interaction")
    arena_axis.set_xlim(0, environment.width)
    arena_axis.set_ylim(0, environment.height)
    arena_axis.set_aspect("equal")
    arena_axis.set_xlabel("x")
    arena_axis.set_ylabel("y")
    arena_axis.grid(alpha=0.25)

    for obstacle in environment.obstacles:
        arena_axis.add_patch(
            plt.Rectangle(
                (obstacle.x, obstacle.y),
                obstacle.width,
                obstacle.height,
                facecolor="crimson",
                edgecolor="darkred",
                alpha=0.35,
                label="obstacle",
            )
        )
    if environment.light is not None:
        arena_axis.scatter(
            environment.light.x,
            environment.light.y,
            color="gold",
            edgecolor="black",
            s=120,
            marker="*",
            label="light",
            zorder=4,
        )

    arena_axis.plot(
        [state.x for state in trajectory],
        [state.y for state in trajectory],
        color="tab:blue",
        linewidth=2,
        label="trajectory",
        zorder=3,
    )
    arena_axis.scatter(trajectory[0].x, trajectory[0].y, color="green", s=70, label="start", zorder=5)
    collisions = [state for state in trajectory if state.collision]
    if collisions:
        arena_axis.scatter(
            [state.x for state in collisions],
            [state.y for state in collisions],
            color="black",
            marker="x",
            s=55,
            label="collision",
            zorder=6,
        )
    arena_axis.legend(loc="upper left")

    steps = [state.step for state in trajectory]
    bias = [state.right_drive - state.left_drive for state in trajectory]
    telemetry_axis.plot(steps, bias, color="tab:purple", label="turn bias")
    telemetry_axis.plot(
        steps,
        [state.locomotion_drive for state in trajectory],
        color="tab:green",
        label="locomotion",
    )
    collision_steps = [state.step for state in collisions]
    if collision_steps:
        telemetry_axis.scatter(
            collision_steps,
            [0.0] * len(collision_steps),
            color="crimson",
            marker="x",
            s=45,
            label="collision",
        )
    telemetry_axis.axhline(0.0, color="black", linewidth=0.8)
    telemetry_axis.set_title("Motor telemetry and collision steps")
    telemetry_axis.set_xlabel("step")
    telemetry_axis.set_ylabel("drive / bias")
    telemetry_axis.grid(alpha=0.25)
    telemetry_axis.legend()

    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-sensory", type=int, default=50)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--output", default="results/obstacle/obstacle_trajectory.png")
    args = parser.parse_args()

    print("Loading FlyWire connectome...")
    brain = FlyWireBrain(load_connectome(), max_sensory=args.max_sensory, hops=args.hops)
    body, environment = build_obstacle_scene()
    trajectory = VirtualFlySimulator(body, environment, brain=brain).run(args.steps)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_obstacle_run(trajectory, environment, output)
    collisions = sum(state.collision for state in trajectory)
    moved = sum(
        (current.x, current.y) != (previous.x, previous.y)
        for previous, current in zip(
            [Body2D(x=5.0, y=5.0), *trajectory[:-1]], trajectory
        )
    )
    print(f"steps={args.steps} collisions={collisions} collision_rate={collisions / args.steps:.3f}")
    print(f"movement_fraction={moved / args.steps:.3f}")
    print(f"saved={output}")


if __name__ == "__main__":
    main()