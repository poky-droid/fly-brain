"""Run and optionally plot the VirtualFly V1 closed-loop simulation."""

import argparse

from virtual_brain import load_connectome
from virtual_fly import Body2D, Environment2D, FlyWireBrain, LightSource, VirtualFlySimulator
from virtual_fly.visualize_fly import plot_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", default="results/virtual_fly_v1.png")
    parser.add_argument("--max-sensory", type=int, default=50)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--leak", type=float, default=0.9)
    parser.add_argument("--weight-scale", type=float, default=0.04)
    parser.add_argument("--refractory-steps", type=int, default=2)
    parser.add_argument("--rule-based", action="store_true", help="Use the simple controller instead of FlyWire")
    args = parser.parse_args()

    body = Body2D(x=5.0, y=5.0, heading=0.0, velocity=0.5)
    environment = Environment2D(
        width=30.0,
        height=30.0,
        light=LightSource(x=25.0, y=20.0, intensity=1.0, radius=35.0),
    )
    brain = None
    if not args.rule_based:
        print("Loading FlyWire connectome...")
        connectome = load_connectome()
        brain = FlyWireBrain(
            connectome,
            max_sensory=args.max_sensory,
            hops=args.hops,
            threshold=args.threshold,
            leak=args.leak,
            weight_scale=args.weight_scale,
            refractory_steps=args.refractory_steps,
        )
        print(f"Using FlyWire brain: {brain.model._graph.shape[0]} neurons, {brain.model._graph.nnz} edges")

    trajectory = VirtualFlySimulator(body, environment, brain=brain).run(args.steps)

    for state in trajectory:
        turn_bias = state.right_drive - state.left_drive
        print(
            f"step={state.step:03d} pos=({state.x:.3f},{state.y:.3f}) "
            f"heading={state.heading:+.3f} "
            f"sensor=(L={state.left_sensor:.3f},R={state.right_sensor:.3f}) "
            f"motor=(L={state.left_drive:.3f},R={state.right_drive:.3f},"
            f"F={state.locomotion_drive:.3f}) "
            f"bias={turn_bias:+.3f} action={state.action}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_trajectory(trajectory, environment)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved trajectory plot to {args.output}")


if __name__ == "__main__":
    main()
