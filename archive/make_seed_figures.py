"""Five-seed figures from results/seed_summary.csv (written by aggregate_seeds.py).
Lines = mean over training seeds; bands (PPO arms only) = +/-1 SD across seeds.
Direct, CPG and stand-still are deterministic: no seed spread is drawn."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

STYLE = {
    "direct": ("Direct", "#1f77b4", "s", "-"),
    "cpg": ("CPG + PD", "#d62728", "^", "--"),
    "morphology-aware": ("Morphology-aware PPO", "#2ca02c", "o", "-"),
    "ppo-no-morphology": ("PPO, no morphology", "#ff7f0e", "v", "-."),
    "standstill": ("Stand-still", "#7f7f7f", "D", ":"),
}

AXES = [
    ("mass", "(a) Mass"),
    ("limb_length", "(b) Limb length"),
    ("joint_strength", "(c) Joint strength"),
    ("friction", "(d) Friction"),
]


def panel_figure(df, metric, scale, ylabel, filename, outdir):

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6), sharey=True)

    for ax, (axis, title) in zip(axes.flat, AXES):

        for controller, (label, color, marker, line) in STYLE.items():

            d = df[
                (df["controller"] == controller) & (df["axis"] == axis)
            ].sort_values("level")

            if d.empty:
                continue

            x = d["level"] * 100.0
            m = d[f"{metric}_mean"] * scale
            sd = d[f"{metric}_sd"].fillna(0.0) * scale

            ax.plot(x, m, color=color, marker=marker, linestyle=line,
                    linewidth=2.0, markersize=6, label=label)

            if controller in ("morphology-aware", "ppo-no-morphology"):
                ax.fill_between(x, m - sd, m + sd, color=color,
                                alpha=0.18, linewidth=0)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Perturbation (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.tick_params(labelleft=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=os.path.join("results", "seed_summary.csv"))
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.summary)

    print(panel_figure(df, "task_reward", 1.0, "Mean task reward",
                       "fig_task_reward_5seeds.png", args.outdir))
    print(panel_figure(df, "fall_rate", 100.0, "Fall rate (%)",
                       "fig_fall_rate_5seeds.png", args.outdir))


if __name__ == "__main__":
    main()