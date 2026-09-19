"""Batch ablation experiment: ablate N neurons and compare activity metrics."""

import argparse

from virtual_brain import ablate_batch, load_connectome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90883,
                        help="Seed neuron for subgraph extraction")
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--top-n", type=int, default=20,
                        help="Ablate the top-N highest out-degree neurons in the subgraph")
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

    brain = load_connectome()
    neuron_ids, graph = brain.induced_subgraph([args.seed], hops=args.hops)
    seed_pos = int((neuron_ids == args.seed).nonzero()[0][0])

    # Pick top-N local neurons by out-degree (most influential)
    import numpy as np
    out_degree = np.diff(graph.tocsr().indptr)
    top_local = out_degree.argsort()[::-1][: args.top_n].tolist()

    if args.model == "lif":
        model_kwargs = dict(
            threshold=args.lif_threshold,
            leak=args.lif_leak,
            weight_scale=args.lif_weight_scale,
            refractory_steps=args.lif_refractory,
        )
    else:
        model_kwargs = dict(threshold=args.threshold, decay=args.decay)

    print(f"=== BATCH ABLATION EXPERIMENT ===")
    print(f"Seed neuron    : {args.seed}")
    print(f"Sub-connectome : {graph.shape[0]} neurons, {graph.nnz} edges")
    print(f"Model          : {args.model}")
    print(f"Steps          : {args.steps}")
    print(f"Ablating top-{args.top_n} out-degree neurons")
    print()

    results = ablate_batch(
        graph,
        neurons=top_local,
        steps=args.steps,
        model=args.model,
        **model_kwargs,
    )

    # Sort by spike_delta descending (most impactful ablations first)
    results.sort(key=lambda r: r.spike_delta, reverse=True)

    print(f"{'rank':>4} {'local_id':>8} {'global_id':>9} "
          f"{'spikes_normal':>13} {'spikes_ablated':>14} "
          f"{'Δspikes':>8} {'Δcoverage':>10}")
    print("-" * 72)
    for rank, r in enumerate(results, 1):
        global_id = int(neuron_ids[r.neuron_id])
        print(
            f"{rank:4} {r.neuron_id:8} {global_id:9} "
            f"{r.total_spikes_normal:13} {r.total_spikes_ablated:14} "
            f"{r.spike_delta:+8} {r.coverage_delta:+10.3f}"
        )


if __name__ == "__main__":
    main()
