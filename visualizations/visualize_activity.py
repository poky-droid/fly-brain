"""Visualize activity propagation in 3D anatomical space."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from virtual_brain import ablate, lif_propagate, load_connectome, propagate_activity


def plot_activity(
    all_coords: np.ndarray,
    subgraph_ids: np.ndarray,
    history: list[np.ndarray],
    title: str,
    save_path: str | None = None,
) -> None:
    """Plot each propagation timestep as a 3D scatter in anatomical space."""

    steps = len(history)
    cols = min(steps, 3)
    rows = (steps + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 5, rows * 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for step, active_local in enumerate(history):
        ax = fig.add_subplot(rows, cols, step + 1, projection="3d")

        sub_coords = all_coords[subgraph_ids]
        ax.scatter(
            sub_coords[:, 0], sub_coords[:, 1], sub_coords[:, 2],
            c="lightgrey", s=1, alpha=0.2, linewidths=0,
        )

        if active_local.size > 0:
            active_global = subgraph_ids[active_local]
            act_coords = all_coords[active_global]
            ax.scatter(
                act_coords[:, 0], act_coords[:, 1], act_coords[:, 2],
                c="crimson", s=8, alpha=0.7, linewidths=0,
            )

        ax.set_title(f"t = {step}  ({len(active_local)} active)", fontsize=9)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=90883)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--model", choices=["threshold", "lif"], default="threshold")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--save", type=str, default=None)

    # threshold params
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--decay", type=float, default=0.0)

    # LIF params
    parser.add_argument("--lif-threshold", type=float, default=0.5)
    parser.add_argument("--lif-leak", type=float, default=0.9)
    parser.add_argument("--lif-weight-scale", type=float, default=0.04)
    parser.add_argument("--lif-refractory", type=int, default=2)

    args = parser.parse_args()

    print("Loading connectome...")
    brain = load_connectome()

    print(f"Extracting {args.hops}-hop sub-connectome from neuron {args.seed}...")
    neuron_ids, graph = brain.induced_subgraph([args.seed], hops=args.hops)
    seed_pos = int((neuron_ids == args.seed).nonzero()[0][0])

    model_label = args.model.upper()
    ablate_label = " (ablated)" if args.ablate else ""
    title = f"{model_label} propagation — seed {args.seed}{ablate_label}"

    if args.ablate:
        graph = ablate(graph, seed_pos)

    print(f"Simulating {args.steps} timesteps with {args.model} model...")
    if args.model == "lif":
        lif_history = lif_propagate(
            graph, [seed_pos],
            steps=args.steps,
            threshold=args.lif_threshold,
            leak=args.lif_leak,
            weight_scale=args.lif_weight_scale,
            refractory_steps=args.lif_refractory,
        )
        history = [state.active_indices for state in lif_history]
    else:
        history = propagate_activity(
            graph, [seed_pos],
            steps=args.steps,
            threshold=args.threshold,
            decay=args.decay,
        )

    print("Rendering 3D plot...")
    plot_activity(
        all_coords=brain.coordinates,
        subgraph_ids=neuron_ids,
        history=history,
        title=title,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
