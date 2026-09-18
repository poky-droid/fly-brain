"""Run Virtual Fly activity-propagation experiments.

Supports two neuron models:
  threshold  — binary threshold + decay  (fast, simple)
  lif        — Leaky Integrate-and-Fire  (biologically closer)
"""

import argparse

from virtual_brain import ablate, lif_propagate, load_connectome, propagate_activity


def _run_threshold(graph, seed_pos, args):
    normal = propagate_activity(
        graph, [seed_pos],
        steps=args.steps,
        threshold=args.threshold,
        decay=args.decay,
    )
    ablated = propagate_activity(
        ablate(graph, seed_pos), [seed_pos],
        steps=args.steps,
        threshold=args.threshold,
        decay=args.decay,
    )
    return [(len(s), len(a)) for s, a in zip(normal, ablated)]


def _run_lif(graph, seed_pos, args):
    normal = lif_propagate(
        graph, [seed_pos],
        steps=args.steps,
        threshold=args.lif_threshold,
        leak=args.lif_leak,
        weight_scale=args.lif_weight_scale,
        refractory_steps=args.lif_refractory,
    )
    ablated = lif_propagate(
        ablate(graph, seed_pos), [seed_pos],
        steps=args.steps,
        threshold=args.lif_threshold,
        leak=args.lif_leak,
        weight_scale=args.lif_weight_scale,
        refractory_steps=args.lif_refractory,
    )
    return [(len(s.active_indices), len(a.active_indices)) for s, a in zip(normal, ablated)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90883)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--model", choices=["threshold", "lif"], default="threshold")

    # threshold model params
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--decay", type=float, default=0.0)

    # LIF model params
    parser.add_argument("--lif-threshold", type=float, default=0.5)
    parser.add_argument("--lif-leak", type=float, default=0.9)
    parser.add_argument("--lif-weight-scale", type=float, default=0.04)
    parser.add_argument("--lif-refractory", type=int, default=2)

    args = parser.parse_args()

    brain = load_connectome()
    neuron_ids, graph = brain.induced_subgraph([args.seed], hops=args.hops)
    seed_pos = int((neuron_ids == args.seed).nonzero()[0][0])

    if args.model == "lif":
        results = _run_lif(graph, seed_pos, args)
        model_info = (
            f"threshold={args.lif_threshold}  leak={args.lif_leak}  "
            f"weight_scale={args.lif_weight_scale}  refractory={args.lif_refractory}"
        )
    else:
        results = _run_threshold(graph, seed_pos, args)
        model_info = f"threshold={args.threshold}  decay={args.decay}"

    print("=== PROPAGATION EXPERIMENT ===")
    print(f"Seed neuron       : {args.seed}")
    print(f"Model             : {args.model}")
    print(f"Sub-connectome    : {graph.shape[0]} neurons, {graph.nnz} edges")
    print(f"Steps             : {args.steps}")
    print(f"Params            : {model_info}")
    print()
    print("step | normal active | ablated active")
    print("-----+---------------+---------------")
    for step, (n_active, a_active) in enumerate(results):
        print(f"{step:4} | {n_active:13} | {a_active:14}")


if __name__ == "__main__":
    main()
