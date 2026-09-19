"""Export simulation results into a tabular CSV file."""

import argparse
import csv
from pathlib import Path

import numpy as np

from virtual_brain import StimulusCondition, compare_stimuli, load_connectome


DEFAULT_CONDITIONS = [
    StimulusCondition("sensory", sensory_filter={"super_class": "sensory"}, max_sensory=50),
    StimulusCondition("ascending", sensory_filter={"super_class": "ascending"}, max_sensory=50),
]


def _export_rows(result) -> list[dict]:
    rows: list[dict] = []
    for condition_name, sr, behaviors in zip(result.conditions, result.sensory_results, result.fly_behaviors):
        motor_local = np.flatnonzero(np.isin(sr.subgraph_ids, sr.motor_ids))
        for timestep, behavior in enumerate(behaviors):
            if timestep >= len(sr.history):
                break
            active = np.asarray(sr.history[timestep], dtype=int)
            motor_active = np.intersect1d(active, motor_local)
            rows.append(
                {
                    "condition": condition_name,
                    "timestep": timestep,
                    "active_neurons": int(len(active)),
                    "motor_spikes": int(len(motor_active)),
                    "locomotion_drive": float(behavior.locomotion_drive),
                    "left_drive": float(behavior.left_drive),
                    "right_drive": float(behavior.right_drive),
                    "turn_bias": float(behavior.turn_bias),
                    "endocrine_drive": float(behavior.endocrine_drive),
                    "action": behavior.action,
                }
            )
    return rows


def export_results_csv(
    result,
    output_path: str | Path,
) -> Path:
    """Write the per-timestep behavior table to CSV."""
    rows = _export_rows(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "condition",
        "timestep",
        "active_neurons",
        "motor_spikes",
        "locomotion_drive",
        "left_drive",
        "right_drive",
        "turn_bias",
        "endocrine_drive",
        "action",
    ]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="results/simulation_results.csv")
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-motor", type=int, default=200)
    parser.add_argument("--model", choices=["lif", "threshold"], default="lif")
    parser.add_argument("--lif-threshold", type=float, default=0.5)
    parser.add_argument("--lif-leak", type=float, default=0.9)
    parser.add_argument("--lif-weight-scale", type=float, default=0.04)
    parser.add_argument("--lif-refractory", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--decay", type=float, default=0.0)
    args = parser.parse_args()

    if args.model == "lif":
        model_kwargs = dict(
            threshold=args.lif_threshold,
            leak=args.lif_leak,
            weight_scale=args.lif_weight_scale,
            refractory_steps=args.lif_refractory,
        )
    else:
        model_kwargs = dict(threshold=args.threshold, decay=args.decay)

    brain = load_connectome()
    result = compare_stimuli(
        brain,
        conditions=DEFAULT_CONDITIONS,
        max_motor=args.max_motor,
        hops=args.hops,
        steps=args.steps,
        model=args.model,
        **model_kwargs,
    )

    output = export_results_csv(result, args.output)
    print(f"Exported CSV to {output}")
    print(f"Rows: {len(_export_rows(result))}")


if __name__ == "__main__":
    main()
