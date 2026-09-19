"""Sensory-driven simulation: afferent neurons → network → efferent motor output."""

import argparse

from virtual_brain import load_connectome, run_sensory_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--max-sensory", type=int, default=50,
                        help="Max afferent seed neurons")
    parser.add_argument("--max-motor", type=int, default=200,
                        help="Max efferent neurons to track")
    parser.add_argument("--sensory-superclass", type=str, default=None,
                        help="Filter afferent by super_class, e.g. 'visual'")
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

    sensory_filter = {}
    if args.sensory_superclass:
        sensory_filter["super_class"] = args.sensory_superclass

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

    print(f"Running sensory experiment...")
    result = run_sensory_experiment(
        brain,
        sensory_filter=sensory_filter or None,
        max_sensory=args.max_sensory,
        max_motor=args.max_motor,
        hops=args.hops,
        steps=args.steps,
        model=args.model,
        **model_kwargs,
    )

    motor_activity = result.motor_activity()

    print()
    print("=== SENSORY EXPERIMENT RESULT ===")
    print(f"Afferent (sensory) seeds : {len(result.sensory_ids)}")
    print(f"Efferent (motor) tracked : {len(result.motor_ids)}")
    print(f"Sub-connectome           : {len(result.subgraph_ids)} neurons")
    print(f"Model                    : {args.model}")
    print(f"Steps                    : {result.steps}")
    print(f"Total spikes             : {result.total_spikes()}")
    print(f"Total motor spikes       : {result.total_motor_spikes()}")
    print()
    print("step | total active | motor active")
    print("-----+-------------+--------------")
    for step, (active, motor) in enumerate(zip(result.history, motor_activity)):
        print(f"{step:4} | {len(active):11} | {len(motor):12}")


if __name__ == "__main__":
    main()
