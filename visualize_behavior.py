"""Visualize behavioral readout of multi-stimulus experiments.

Produces a figure with:
  Row 1 — Active neuron count per timestep (per condition)
  Row 2 — Motor spike count per timestep (per condition)
  Row 3 — Locomotion drive + turn bias per timestep (per condition)
  Row 4 — Action label timeline (colored spans, per condition)
  Row 5 — Action distribution bar chart (all conditions, side-by-side)

Usage
-----
  python visualize_behavior.py --steps 12 --model lif --save output_behavior.png
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from virtual_brain import (
    MultiStimulusResult,
    StimulusCondition,
    compare_stimuli,
    load_connectome,
)

# ── Color palette ─────────────────────────────────────────────────────────────
CONDITION_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

ACTION_COLORS = {
    "rest":         "#BDBDBD",
    "walk_forward": "#66BB6A",
    "turn_left":    "#29B6F6",
    "turn_right":   "#EF5350",
    "modulated":    "#FFA726",
}


def _action_color(action: str) -> str:
    return ACTION_COLORS.get(action, "#9E9E9E")


def plot_behavior(
    result: MultiStimulusResult,
    title: str = "Behavioral Readout",
    save_path: str | None = None,
) -> None:
    """Render behavioral comparison figure from a MultiStimulusResult."""

    n_cond = len(result.conditions)
    steps = result.sensory_results[0].steps
    t = np.arange(steps + 1)

    # ── Precompute per-condition series ───────────────────────────────────────
    active_counts = []
    motor_counts  = []
    loco_series   = []
    bias_series   = []
    left_series   = []
    right_series  = []
    endo_series   = []
    action_series = []

    for sr, behaviors in zip(result.sensory_results, result.fly_behaviors):
        active_counts.append([len(sr.history[i]) if i < len(sr.history) else 0 for i in t])
        ma = sr.motor_activity()
        motor_counts.append([len(ma[i]) if i < len(ma) else 0 for i in t])
        loco_series.append([b.locomotion_drive for b in behaviors])
        bias_series.append([b.turn_bias        for b in behaviors])
        left_series.append([b.left_drive       for b in behaviors])
        right_series.append([b.right_drive     for b in behaviors])
        endo_series.append([b.endocrine_drive  for b in behaviors])
        action_series.append([b.action         for b in behaviors])

    # ── Figure layout ─────────────────────────────────────────────────────────
    # 5 rows × n_cond columns, last row spans all columns
    n_rows = 5
    fig = plt.figure(figsize=(6 * n_cond, 18))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(
        n_rows, n_cond,
        height_ratios=[2.5, 2, 3, 1.5, 2.5],
        hspace=0.55, wspace=0.3,
    )

    all_actions = sorted(ACTION_COLORS.keys())

    for col, (name, color) in enumerate(zip(result.conditions, CONDITION_COLORS)):

        # ── Row 0: Active neuron count ────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.fill_between(t, active_counts[col], alpha=0.25, color=color)
        ax0.plot(t, active_counts[col], color=color, linewidth=2, marker="o", markersize=4)
        ax0.set_title(f"[{name}]  Active neurons / step", fontsize=10, fontweight="bold")
        ax0.set_xlabel("timestep")
        ax0.set_ylabel("# neurons")
        ax0.grid(True, alpha=0.3)
        ax0.set_xlim(0, steps)

        # ── Row 1: Motor spike count ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.fill_between(t, motor_counts[col], alpha=0.25, color=color)
        ax1.plot(t, motor_counts[col], color=color, linewidth=2, marker="s", markersize=4)
        ax1.set_title(f"[{name}]  Motor (efferent) spikes / step", fontsize=10, fontweight="bold")
        ax1.set_xlabel("timestep")
        ax1.set_ylabel("# motor neurons")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, steps)

        # ── Row 2: Locomotion + turn bias + drives ────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        ax2.plot(t, loco_series[col],  color="#1B5E20", linewidth=2,   label="locomotion",  marker="o", markersize=3)
        ax2.plot(t, left_series[col],  color="#0288D1", linewidth=1.5, label="left_drive",  linestyle="--")
        ax2.plot(t, right_series[col], color="#E53935", linewidth=1.5, label="right_drive", linestyle="--")
        ax2.plot(t, endo_series[col],  color="#7B1FA2", linewidth=1.2, label="endocrine",   linestyle=":")

        ax2b = ax2.twinx()
        ax2b.plot(t, bias_series[col], color="#FF6F00", linewidth=2, label="turn_bias", marker="^", markersize=4)
        ax2b.axhline(0, color="#FF6F00", linewidth=0.8, linestyle="--", alpha=0.5)
        ax2b.set_ylabel("turn_bias (R−L)", color="#FF6F00", fontsize=8)
        ax2b.tick_params(axis="y", labelcolor="#FF6F00", labelsize=7)

        ax2.set_title(f"[{name}]  Drives + turn bias", fontsize=10, fontweight="bold")
        ax2.set_xlabel("timestep")
        ax2.set_ylabel("drive (0–1)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, steps)

        lines2, labels2 = ax2.get_legend_handles_labels()
        lines2b, labels2b = ax2b.get_legend_handles_labels()
        ax2.legend(lines2 + lines2b, labels2 + labels2b, fontsize=7, loc="upper right")

        # ── Row 3: Action label timeline (colored spans) ──────────────────────
        ax3 = fig.add_subplot(gs[3, col])
        ax3.set_xlim(0, steps)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([])
        ax3.set_title(f"[{name}]  Action timeline", fontsize=10, fontweight="bold")
        ax3.set_xlabel("timestep")

        for i, action in enumerate(action_series[col]):
            ax3.axvspan(i, i + 1, color=_action_color(action), alpha=0.85)
            ax3.text(
                i + 0.5, 0.5, action.replace("_", "\n"),
                ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold",
            )

        # legend for action colors
        patches = [
            mpatches.Patch(color=c, label=a)
            for a, c in ACTION_COLORS.items()
        ]
        ax3.legend(handles=patches, fontsize=6, loc="upper right",
                   bbox_to_anchor=(1.0, 1.5), ncol=len(patches))

    # ── Row 4: Action distribution bar chart (spans all columns) ─────────────
    ax4 = fig.add_subplot(gs[4, :])
    x = np.arange(len(all_actions))
    bar_w = 0.8 / n_cond
    for i, (name, color) in enumerate(zip(result.conditions, CONDITION_COLORS)):
        counts = [action_series[i].count(a) for a in all_actions]
        offset = (i - (n_cond - 1) / 2) * bar_w
        bars = ax4.bar(x + offset, counts, bar_w, label=name, color=color, alpha=0.85)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(count),
                    ha="center", va="bottom", fontsize=8,
                )

    ax4.set_xticks(x)
    ax4.set_xticklabels(all_actions, fontsize=10)
    ax4.set_ylabel("timesteps with this action")
    ax4.set_title("Action Distribution — All Conditions", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, axis="y", alpha=0.3)

    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.97,
                        hspace=0.55, wspace=0.3)

    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hops",      type=int,   default=1)
    parser.add_argument("--steps",     type=int,   default=12)
    parser.add_argument("--max-motor", type=int,   default=200)
    parser.add_argument("--model",     choices=["lif", "threshold"], default="lif")
    parser.add_argument("--save",      type=str,   default=None,
                        help="Path to save figure, e.g. output_behavior.png")

    # LIF params
    parser.add_argument("--lif-threshold",    type=float, default=0.5)
    parser.add_argument("--lif-leak",         type=float, default=0.9)
    parser.add_argument("--lif-weight-scale", type=float, default=0.04)
    parser.add_argument("--lif-refractory",   type=int,   default=2)

    # threshold params
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--decay",     type=float, default=0.0)

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
        StimulusCondition("sensory",   sensory_filter={"super_class": "sensory"},   max_sensory=50),
        StimulusCondition("ascending", sensory_filter={"super_class": "ascending"}, max_sensory=50),
    ]

    print("Loading connectome...")
    brain = load_connectome()

    print(f"Running {len(conditions)} conditions × {args.steps} steps ({args.model})...")
    result = compare_stimuli(
        brain,
        conditions=conditions,
        max_motor=args.max_motor,
        hops=args.hops,
        steps=args.steps,
        model=args.model,
        **model_kwargs,
    )

    print("Rendering behavior visualization...")
    plot_behavior(
        result,
        title=f"Virtual Fly Brain — Behavioral Readout ({args.model.upper()}, {args.steps} steps)",
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
