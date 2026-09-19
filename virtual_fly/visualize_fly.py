"""Plot a V1 fly trajectory."""

from __future__ import annotations

from .environment import Environment2D


def plot_trajectory(steps, environment: Environment2D, ax=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if ax is None:
        _, ax = plt.subplots()
    ax.set_xlim(0, environment.width)
    ax.set_ylim(0, environment.height)
    ax.set_aspect("equal")

    for obstacle in environment.obstacles:
        ax.add_patch(
            plt.Rectangle(
                (obstacle.x, obstacle.y),
                obstacle.width,
                obstacle.height,
                color="dimgray",
                alpha=0.7,
            )
        )
    if environment.light is not None:
        ax.scatter(environment.light.x, environment.light.y, color="gold", s=80)

    if steps:
        ax.plot([item.x for item in steps], [item.y for item in steps], "o-", color="tab:blue")
        collisions = [item for item in steps if getattr(item, "collision", False)]
        if collisions:
            ax.scatter(
                [item.x for item in collisions],
                [item.y for item in collisions],
                color="crimson",
                marker="x",
                s=55,
                label="collision",
            )
            ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax
