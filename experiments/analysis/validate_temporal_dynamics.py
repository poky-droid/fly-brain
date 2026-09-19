"""Measure FlyWire motor dynamics under controlled sensory protocols."""

from __future__ import annotations

import argparse
from collections import Counter

from virtual_brain import load_connectome
from virtual_fly import FlyWireBrain


PROTOCOLS = {
    "constant_left": lambda step: {"left": 1.0, "right": 0.0},
    "constant_right": lambda step: {"left": 0.0, "right": 1.0},
    "constant_front": lambda step: {"left": 1.0, "right": 1.0},
    "stimulus_none": lambda step: {"left": 1.0, "right": 0.0} if step < 10 else {"left": 0.0, "right": 0.0},
    "left_to_right": lambda step: {"left": 1.0, "right": 0.0} if step < 10 else {"left": 0.0, "right": 1.0},
    "right_to_left": lambda step: {"left": 0.0, "right": 1.0} if step < 10 else {"left": 1.0, "right": 0.0},
}


def run_protocol(brain: FlyWireBrain, name: str, steps: int) -> None:
    brain.reset()
    rows: list[tuple[int, int, int, int, float, float, float, str]] = []
    sensor_fn = PROTOCOLS[name]

    for step in range(steps):
        sensor = sensor_fn(step)
        output = brain.step(sensor)
        active = brain._previous_spikes
        rows.append(
            (
                step,
                int(active.sum()),
                int(active[brain.model._left_local].sum()),
                int(active[brain.model._right_local].sum()),
                output.locomotion_drive,
                output.left_drive,
                output.right_drive,
                output.action,
            )
        )

    actions = Counter(row[7] for row in rows)
    unique_motor_patterns = len({(row[2], row[3]) for row in rows})
    transitions = sum(a[7] != b[7] for a, b in zip(rows, rows[1:]))
    print(f"\n=== {name} ({steps} steps) ===")
    print("step active left_spikes right_spikes locomotion left_drive right_drive action")
    for row in rows:
        print(
            f"{row[0]:4d} {row[1]:6d} {row[2]:11d} {row[3]:12d} "
            f"{row[4]:10.3f} {row[5]:10.3f} {row[6]:11.3f} {row[7]}"
        )
    print(
        "summary: "
        f"motor_patterns={unique_motor_patterns} action_transitions={transitions} "
        f"actions={dict(actions)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--max-sensory", type=int, default=50)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument(
        "--protocol",
        choices=["all", *PROTOCOLS],
        default="all",
    )
    args = parser.parse_args()

    print("Loading FlyWire connectome...")
    brain = FlyWireBrain(
        load_connectome(),
        max_sensory=args.max_sensory,
        hops=args.hops,
    )
    print(
        f"mapping={brain.mapping_source} graph={brain.model._graph.shape[0]} "
        f"edges={brain.model._graph.nnz} motor_left={brain.model._left_local.size} "
        f"motor_right={brain.model._right_local.size} "
        f"descending={brain.model._descending_local.size}"
    )

    names = PROTOCOLS if args.protocol == "all" else {args.protocol: PROTOCOLS[args.protocol]}
    for name in names:
        run_protocol(brain, name, args.steps)


if __name__ == "__main__":
    main()
