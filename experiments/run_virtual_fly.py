"""Run a virtual fly simulation and print behavioral output per timestep."""

import argparse

from virtual_brain import VirtualFly, load_connectome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--max-sensory", type=int, default=50)
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

    print("Initialising VirtualFly...")
    fly = VirtualFly(brain, max_sensory=args.max_sensory, hops=args.hops)
    print(f"  Sub-connectome : {fly._graph.shape[0]} neurons, {fly._graph.nnz} edges")
    print(f"  Descending     : {fly._descending_local.size}")
    print(f"  Motor (direct) : {fly._motor_local.size}")
    print(f"  Endocrine      : {fly._endocrine_local.size}")
    print(f"  Left nerve     : {fly._left_local.size}")
    print(f"  Right nerve    : {fly._right_local.size}")
    print()

    behaviors = fly.run(steps=args.steps, model=args.model, **model_kwargs)

    print(f"=== VIRTUAL FLY BEHAVIOR ({args.model.upper()}, {args.steps} steps) ===")
    print()
    print(f"{'step':>4}  {'action':<14}  {'loco':>6}  {'turn_bias':>10}  "
          f"{'left':>6}  {'right':>6}  {'endo':>6}")
    print("-" * 60)
    for b in behaviors:
        print(
            f"{b.step:4}  {b.action:<14}  {b.locomotion_drive:6.3f}  "
            f"{b.turn_bias:+10.3f}  {b.left_drive:6.3f}  "
            f"{b.right_drive:6.3f}  {b.endocrine_drive:6.3f}"
        )


if __name__ == "__main__":
    main()
