"""
Paper figures from the five-seed results.

Run (from the project root) after scripts/aggregate_seeds.py:

    python scripts/make_paper_figures.py

Reads results/seed_summary.csv and results/seed_overall.csv.
Writes to figures/ (PNG, 300 dpi):

    paper_fig_task_reward.png   task reward vs perturbation, four axes
    paper_fig_fall_rate.png     fall rate vs perturbation, four axes
    paper_fig_per_seed.png      per-seed comparison, aware vs no-morphology

Caption wording
---------------
Four-axis figures: "Lines are means over five training seeds for the two
PPO policies; shaded bands show the minimum to maximum across seeds.
Direct, CPG and stand-still are deterministic (mean of 10 evaluation
episodes), so they have no seed spread. Direct and stand-still never fall."

Per-seed figure: "Each pair of points is one training seed (mean over the
21 distinct conditions); black bars are means over seeds. The dotted line
is the stand-still reference."
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

AWARE = "morphology-aware"
BLIND = "ppo-no-morphology"

STYLE = {
    "direct": ("Direct", "#1f77b4", "s", "-"),
    "cpg": ("CPG-style", "#d62728", "^", "--"),
    AWARE: ("Morphology-aware PPO", "#2ca02c", "o", "-"),
    BLIND: ("Morphology-blind PPO", "#ff7f0e", "v", "-."),
    "standstill": ("Stand-still", "#7f7f7f", "D", ":"),
}

AXES = [
    ("mass", "(a) Mass"),
    ("limb_length", "(b) Limb length"),
    ("joint_strength", "(c) Joint strength"),
    ("friction", "(d) Friction"),
]


def panel_figure(df, metric, scale, ylabel, ylim, filename, outdir):
    """Four panels; PPO bands are min..max across training seeds."""

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4), sharey=True)

    for ax, (axis, title) in zip(axes.flat, AXES):

        for controller, (label, color, marker, line) in STYLE.items():

            d = df[
                (df["controller"] == controller) & (df["axis"] == axis)
            ].sort_values("level")

            if d.empty:
                continue

            x = d["level"] * 100.0
            m = d[f"{metric}_mean"] * scale

            ax.plot(
                x,
                m,
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=2.0,
                markersize=6,
                label=label,
            )

            if controller in (AWARE, BLIND):

                lo = np.clip(d[f"{metric}_min"] * scale, ylim[0], ylim[1])
                hi = np.clip(d[f"{metric}_max"] * scale, ylim[0], ylim[1])

                ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)

        ax.set_ylim(*ylim)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Perturbation (%)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(labelleft=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    path = os.path.join(outdir, filename)

    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(path)


def per_seed_figure(overall, outdir):
    """One line per training seed: aware vs no-morphology."""

    ps = overall[overall["controller"].isin([AWARE, BLIND])]

    stand = float(
        overall[overall["controller"] == "standstill"]["task_reward"].mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))

    specs = [
        ("task_reward", 1.0, "Mean task reward", None),
        ("fall_rate", 100.0, "Fall rate (%)", (0, 25)),
    ]

    for ax, (column, scale, label, fixed) in zip(axes, specs):

        wide = ps.pivot(
            index="train_seed",
            columns="controller",
            values=column,
        ) * scale

        values = wide.to_numpy().ravel()

        if fixed is not None:
            ylim = (fixed[0], max(fixed[1], float(values.max()) + 2.0))
        else:
            ylim = (
                min(float(values.min()), stand) - 15.0,
                float(values.max()) + 15.0,
            )

        ax.set_ylim(*ylim)

        gap = 0.055 * (ylim[1] - ylim[0])

        labels = []

        for seed, row in wide.iterrows():

            a, b = float(row[AWARE]), float(row[BLIND])

            ax.plot([0, 1], [a, b], color="#aaaaaa", linewidth=1.2, zorder=1)
            ax.scatter([0], [a], color=STYLE[AWARE][1], s=45, zorder=2)
            ax.scatter([1], [b], color=STYLE[BLIND][1], s=45, zorder=2)

            labels.append((a, int(seed)))

        # Seed labels to the left of the aware points, pushed apart so
        # they never overlap.
        previous = -1e18

        for a, seed in sorted(labels):

            y = max(a, previous + gap)

            ax.annotate(
                f"s{seed}",
                xy=(0, a),
                xytext=(-0.2, y),
                textcoords="data",
                ha="right",
                va="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.8),
            )

            previous = y

        for x, controller in ((0, AWARE), (1, BLIND)):

            ax.hlines(
                float(wide[controller].mean()),
                x - 0.18,
                x + 0.18,
                color="black",
                linewidth=2.5,
                zorder=3,
            )

        if column == "task_reward":

            ax.axhline(stand, color="#7f7f7f", linestyle=":", linewidth=1.5)

            ax.text(
                1.45,
                stand,
                "stand-still",
                ha="right",
                va="bottom",
                fontsize=9,
                color="#555555",
            )

        ax.set_xlim(-0.7, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Morphology-aware", "Morphology-blind"])
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()

    path = os.path.join(outdir, "paper_fig_per_seed.png")

    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(path)


def main():

    parser = argparse.ArgumentParser(description="Paper figures.")

    parser.add_argument(
        "--summary", default=os.path.join("results", "seed_summary.csv")
    )
    parser.add_argument(
        "--overall", default=os.path.join("results", "seed_overall.csv")
    )
    parser.add_argument("--outdir", default="figures")

    args = parser.parse_args()

    for path in (args.summary, args.overall):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run scripts/aggregate_seeds.py first."
            )

    os.makedirs(args.outdir, exist_ok=True)

    summary = pd.read_csv(args.summary)
    overall = pd.read_csv(args.overall)

    panel_figure(
        summary,
        "task_reward",
        1.0,
        "Mean task reward",
        (0, 500),
        "paper_fig_task_reward.png",
        args.outdir,
    )

    panel_figure(
        summary,
        "fall_rate",
        100.0,
        "Fall rate (%)",
        (0, 100),
        "paper_fig_fall_rate.png",
        args.outdir,
    )

    per_seed_figure(overall, args.outdir)


if __name__ == "__main__":
    main()