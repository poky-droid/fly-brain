"""Compare activity propagation across multiple stimulus conditions.

Default conditions:
  sensory   — afferent neurons with super_class = sensory
  ascending — afferent neurons with super_class = ascending
"""

import argparse

import numpy as np

from virtual_brain import StimulusCondition, compare_stimuli, load_connectome


DEFAULT_CONDITIONS = [
    StimulusCondition("sensory",   sensory_filter={"super_class": "sensory"},   max_sensory=50),
    StimulusCondition("ascending", sensory_filter={"super_class": "ascending"}, max_sensory=50),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-motor", type=int, default=200)
    parser.add_argument("--model", choices=["lif", "threshold"], default="lif")
    # LIF params
    parser.add_argument("--lif-threshold", type=float, default=0.5)
    parser.add_argument("--lif-leak", type=float, default=0.9)
    parser.add_argument("--lif-weight-scale", type=float, default=0.04)
    parser.add_argument("--lif-refractory", type=int, default=2)
    # threshold params
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

    print("Loading connectome...")
    brain = load_connectome()

    print(f"Running {len(DEFAULT_CONDITIONS)} conditions × {args.steps} steps ({args.model})...")
    result = compare_stimuli(
        brain,
        conditions=DEFAULT_CONDITIONS,
        max_motor=args.max_motor,
        hops=args.hops,
        steps=args.steps,
        model=args.model,
        **model_kwargs,
    )

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print("=== MULTI-STIMULUS SUMMARY ===")
    print()
    header = (
        f"{'condition':<12} {'seeds':>6} {'total_spk':>10} {'motor_spk':>10} "
        f"{'mean_loco':>10} {'mean_bias':>10} {'dominant_action':<16}"
    )
    print(header)
    print("-" * len(header))
    for row in result.summary():
        print(
            f"{row['condition']:<12} {row['seed_neurons']:>6} "
            f"{row['total_spikes']:>10} {row['motor_spikes']:>10} "
            f"{row['mean_locomotion']:>10.4f} {row['mean_turn_bias']:>+10.4f} "
            f"{row['dominant_action']:<16}"
        )

    # ── Per-step comparison ──────────────────────────────────────────────────
    print()
    print("=== PER-STEP COMPARISON ===")
    print()

    names = result.conditions
    col_w = 14

    # Header
    header_parts = [f"{'step':>4}"]
    for name in names:
        header_parts.append(f"{'active':>{col_w}}({name})")
    for name in names:
        header_parts.append(f"{'motor':>{col_w}}({name})")
    for name in names:
        header_parts.append(f"{'action':>{col_w}}({name})")
    print("  ".join(header_parts))
    print("-" * (6 + len(names) * (col_w + 2) * 3))

    for t in range(args.steps + 1):
        row_parts = [f"{t:4}"]
        # active counts
        for sr in result.sensory_results:
            active = len(sr.history[t]) if t < len(sr.history) else 0
            row_parts.append(f"{active:>{col_w + len(names[0]) + 2}}")
        # motor counts
        motor_per_cond = [sr.motor_activity() for sr in result.sensory_results]
        for ma in motor_per_cond:
            m = len(ma[t]) if t < len(ma) else 0
            row_parts.append(f"{m:>{col_w + len(names[0]) + 2}}")
        # actions
        for behaviors in result.fly_behaviors:
            act = behaviors[t].action if t < len(behaviors) else "-"
            row_parts.append(f"{act:>{col_w + len(names[0]) + 2}}")
        print("  ".join(row_parts))

    # ── Locomotion + turn_bias per step ─────────────────────────────────────
    print()
    print("=== LOCOMOTION + TURN BIAS PER STEP ===")
    print()
    loco_header = f"{'step':>4}"
    for name in names:
        loco_header += f"  {'loco_' + name:>12}  {'bias_' + name:>12}"
    print(loco_header)
    print("-" * len(loco_header))
    for t in range(args.steps + 1):
        line = f"{t:4}"
        for behaviors in result.fly_behaviors:
            if t < len(behaviors):
                b = behaviors[t]
                line += f"  {b.locomotion_drive:>12.4f}  {b.turn_bias:>+12.4f}"
            else:
                line += f"  {'N/A':>12}  {'N/A':>12}"
        print(line)


if __name__ == "__main__":
    main()
