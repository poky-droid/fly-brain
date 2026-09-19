"""Run a reproducible sensitivity sweep over LIF parameters."""

import argparse
import csv
from itertools import product
from pathlib import Path

from virtual_brain import StimulusCondition, analyze_oscillation, compare_stimuli, load_connectome


DEFAULT_CONDITIONS = [
    StimulusCondition("sensory", sensory_filter={"super_class": "sensory"}, max_sensory=50),
    StimulusCondition("ascending", sensory_filter={"super_class": "ascending"}, max_sensory=50),
]


def run_sensitivity_sweep(
    brain,
    thresholds,
    leaks,
    weight_scales,
    refractory_steps_list,
    conditions=None,
    max_motor=200,
    hops=1,
    steps=8,
):
    """Return one row per condition and parameter combination."""
    conditions = conditions or DEFAULT_CONDITIONS
    rows = []

    for threshold, leak, weight_scale, refractory_steps in product(
        thresholds,
        leaks,
        weight_scales,
        refractory_steps_list,
    ):
        baseline = compare_stimuli(
            brain,
            conditions=conditions,
            max_motor=max_motor,
            hops=hops,
            steps=steps,
            model="lif",
            threshold=threshold,
            leak=leak,
            weight_scale=weight_scale,
            refractory_steps=refractory_steps,
        )
        oscillations = analyze_oscillation(
            brain,
            conditions=conditions,
            max_motor=max_motor,
            hops=hops,
            steps=steps,
            model="lif",
            threshold=threshold,
            leak=leak,
            weight_scale=weight_scale,
            refractory_steps=refractory_steps,
        )

        osc_by_condition = {item.condition: item for item in oscillations}
        summary_by_condition = {row["condition"]: row for row in baseline.summary()}

        for cond in conditions:
            row = summary_by_condition.get(cond.name)
            osc = osc_by_condition.get(cond.name)
            if row is None or osc is None:
                continue

            rows.append(
                {
                    "condition": cond.name,
                    "threshold": float(threshold),
                    "leak": float(leak),
                    "weight_scale": float(weight_scale),
                    "refractory_steps": int(refractory_steps),
                    "total_spikes": int(row["total_spikes"]),
                    "motor_spikes": int(row["motor_spikes"]),
                    "mean_locomotion": float(row["mean_locomotion"]),
                    "mean_turn_bias": float(row["mean_turn_bias"]),
                    "transition_rate": float(osc.transition_rate),
                    "dominant_period": int(osc.dominant_period),
                }
            )

    return rows


def export_sensitivity_csv(rows, output_path):
    """Write sweep rows to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "condition",
        "threshold",
        "leak",
        "weight_scale",
        "refractory_steps",
        "total_spikes",
        "motor_spikes",
        "mean_locomotion",
        "mean_turn_bias",
        "transition_rate",
        "dominant_period",
    ]

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="results/sensitivity_summary.csv")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-motor", type=int, default=200)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--leaks", nargs="+", type=float, default=[0.8, 0.9, 0.95])
    parser.add_argument("--weight-scales", nargs="+", type=float, default=[0.02, 0.04, 0.06])
    parser.add_argument("--refractory-steps", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    brain = load_connectome()
    rows = run_sensitivity_sweep(
        brain,
        thresholds=args.thresholds,
        leaks=args.leaks,
        weight_scales=args.weight_scales,
        refractory_steps_list=args.refractory_steps,
        conditions=DEFAULT_CONDITIONS,
        max_motor=args.max_motor,
        hops=args.hops,
        steps=args.steps,
    )

    output = export_sensitivity_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {output}")
    print(
        "Columns: condition, threshold, leak, weight_scale, refractory_steps, "
        "total_spikes, motor_spikes, mean_locomotion, mean_turn_bias, transition_rate, dominant_period"
    )


if __name__ == "__main__":
    main()
