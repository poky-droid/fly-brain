"""Compare FlyWire motor output for left, right, and front light stimuli."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from virtual_brain import load_connectome
from virtual_fly import Body2D, Environment2D, FlyWireBrain, LightSource


@dataclass(frozen=True)
class StimulusSummary:
    name: str
    sensor_left: float
    sensor_right: float
    mean_left_drive: float
    mean_right_drive: float
    mean_locomotion: float
    actions: tuple[str, ...]


def run_condition(brain: FlyWireBrain, name: str, light: LightSource, steps: int) -> StimulusSummary:
    body = Body2D(x=15.0, y=10.0, heading=0.0)
    environment = Environment2D(width=30.0, height=30.0, light=light)
    sensor = environment.observe(body)
    brain.reset()
    outputs = [brain.step(sensor) for _ in range(steps)]
    return StimulusSummary(
        name=name,
        sensor_left=float(sensor["left"]),
        sensor_right=float(sensor["right"]),
        mean_left_drive=sum(output.left_drive for output in outputs) / steps,
        mean_right_drive=sum(output.right_drive for output in outputs) / steps,
        mean_locomotion=sum(output.locomotion_drive for output in outputs) / steps,
        actions=tuple(output.action for output in outputs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--max-sensory", type=int, default=50)
    parser.add_argument("--hops", type=int, default=1)
    args = parser.parse_args()

    print("Loading FlyWire connectome...")
    connectome = load_connectome()
    brain = FlyWireBrain(
        connectome,
        max_sensory=args.max_sensory,
        hops=args.hops,
    )
    print(
        f"sensory mapping={brain.mapping_source} "
        f"left={brain._left_input.size} right={brain._right_input.size}"
    )
    conditions = [
        ("left", LightSource(x=15.0, y=25.0, radius=30.0)),
        ("right", LightSource(x=15.0, y=-5.0, radius=30.0)),
        ("front", LightSource(x=25.0, y=10.0, radius=30.0)),
    ]

    print("condition  sensor_L  sensor_R  motor_L  motor_R  locomotion  actions")
    for name, light in conditions:
        result = run_condition(brain, name, light, args.steps)
        print(
            f"{result.name:<9} {result.sensor_left:8.3f} {result.sensor_right:8.3f} "
            f"{result.mean_left_drive:7.3f} {result.mean_right_drive:7.3f} "
            f"{result.mean_locomotion:10.3f} {'/'.join(result.actions)}"
        )


if __name__ == "__main__":
    main()