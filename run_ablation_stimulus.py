"""Ablation × stimulus experiment.

For each named stimulus condition, ablate the top-N most influential neurons in
its local subgraph and compare the resulting behavior against the baseline.
The output ranks neurons by behavioral disruption rather than raw spike count.
"""

import argparse

from virtual_brain import StimulusCondition, load_connectome, run_ablation_stimulus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--model", choices=["lif", "threshold"], default="lif")
    parser.add_argument("--max-motor", type=int, default=200)
    parser.add_argument("--hops", type=int, default=1)

    # LIF
    parser.add_argument("--lif-threshold", type=float, default=0.5)
    parser.add_argument("--lif-leak", type=float, default=0.9)
    parser.add_argument("--lif-weight-scale", type=float, default=0.04)
    parser.add_argument("--lif-refractory", type=int, default=2)

    # threshold
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

    conditions = [
        StimulusCondition("sensory", sensory_filter={"super_class": "sensory"}, max_sensory=50),
        StimulusCondition("ascending", sensory_filter={"super_class": "ascending"}, max_sensory=50),
    ]

    print("Loading connectome...")
    brain = load_connectome()

    print(f"Running ablation × stimulus experiment for {len(conditions)} conditions...")
    results = run_ablation_stimulus(
        brain,
        conditions=conditions,
        top_n=args.top_n,
        max_motor=args.max_motor,
        hops=args.hops,
        steps=args.steps,
        model=args.model,
        **model_kwargs,
    )

    print("\n=== ABLATION × STIMULUS RESULTS ===")
    print(f"{'condition':<12} {'neuron':>9} {'spike_loss':>11} {'Δlocomotion':>12} {'Δturn_bias':>12} {'action_changed':>14}")
    print("-" * 82)
    for r in results[:20]:
        print(
            f"{r.condition:<12} {r.neuron_id:>9} {r.spike_loss:>11} "
            f"{r.delta_locomotion:>+12.4f} {r.delta_turn_bias:>+12.4f} "
            f"{str(r.action_changed):>14}"
        )


if __name__ == "__main__":
    main()
