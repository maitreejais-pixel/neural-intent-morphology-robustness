"""
Make paper figures 3, 4 and 5 directly from the analysis summary.

No numbers are typed in: everything is read from summary.csv, which
scripts/analyze_results.py writes. Re-run this script whenever the
summary changes.

Usage (from the project root):

    python scripts/make_figures.py

Optional:

    python scripts/make_figures.py \
        --summary results_with_standstill/summary.csv \
        --outdir figures

Outputs (PNG, 300 dpi):

    fig3_reward_vs_perturbation.png   mean episode reward (+/-1 SD), 4 axes
    fig4_fall_rate_limb_length.png    fall rate vs limb-length perturbation
    fig5_energy_vs_mass.png           mean mechanical energy vs mass perturbation
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# Controller name in summary.csv -> (label, color, marker, line style, z-order)
STYLE = {
    "direct": ("Direct (A)", "#1f77b4", "s", "-", 4),
    "cpg": ("CPG + PD (B)", "#d62728", "^", "--", 3),
    "morphology-aware": ("Morphology-Aware PPO (C)", "#2ca02c", "o", "-", 5),
    "standstill": ("Stand-still reference", "#7f7f7f", "D", ":", 2),
}

# Order in which controllers are drawn and listed in legends.
ORDER = ["direct", "cpg", "morphology-aware", "standstill"]

AXES = [
    ("mass", "(a) Mass Axis", "Mass Perturbation (%)"),
    ("limb_length", "(b) Limb Length Axis", "Limb-Length Perturbation (%)"),
    ("joint_strength", "(c) Joint Strength Axis", "Joint-Strength Perturbation (%)"),
    ("friction", "(d) Friction Axis", "Friction Perturbation (%)"),
]


def load(summary_path):

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Could not find {summary_path}. "
            "Run scripts/analyze_results.py first."
        )

    df = pd.read_csv(summary_path)

    needed = {
        "controller",
        "axis",
        "level",
        "reward_mean",
        "reward_sd",
        "fall_rate_percent",
        "energy_mean",
    }

    missing = needed - set(df.columns)

    if missing:
        raise ValueError(
            f"{summary_path} is missing columns: {sorted(missing)}"
        )

    df["percent"] = (df["level"] * 100.0).round().astype(int)

    return df


def series(df, controller, axis):

    d = df[
        (df["controller"] == controller)
        & (df["axis"] == axis)
    ].sort_values("percent")

    return d


def present_controllers(df):

    return [c for c in ORDER if c in set(df["controller"])]


def style_axes(ax, xlabel, ylabel):

    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


# ======================================================================
# Figure 3: reward vs perturbation (2 x 2)
# ======================================================================

def figure3(df, outdir):

    controllers = present_controllers(df)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8.6),
        sharey=True,
    )

    for ax, (axis, title, xlabel) in zip(axes.flat, AXES):

        for c in controllers:

            label, color, marker, ls, z = STYLE[c]

            d = series(df, c, axis)

            x = d["percent"].to_numpy()
            m = d["reward_mean"].to_numpy()
            sd = d["reward_sd"].fillna(0.0).to_numpy()

            # Lower error bars stop at 0 so every panel shares the
            # (0, 0) origin. This truncation is stated in the caption.
            lower = [min(s, max(v, 0.0)) for v, s in zip(m, sd)]

            ax.errorbar(
                x,
                m,
                yerr=[lower, sd],
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=3,
                alpha=0.85,
                zorder=z,
                clip_on=False,
            )

            ax.plot(
                x,
                m,
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=2.0,
                markersize=6,
                zorder=z + 0.5,
                clip_on=False,
                label=label,
            )

        ax.set_ylim(0, 800)
        ax.set_yticks(range(0, 801, 200))
        ax.set_title(title, fontweight="bold", fontsize=13)
        style_axes(ax, xlabel, "Mean Episode Reward")
        ax.tick_params(labelleft=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=10.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    path = os.path.join(outdir, "fig3_reward_vs_perturbation.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


# ======================================================================
# Figure 4: fall rate vs limb-length perturbation
# ======================================================================

def figure4(df, outdir):

    controllers = present_controllers(df)

    fig, ax = plt.subplots(figsize=(8, 4.6))

    for c in controllers:

        label, color, marker, ls, z = STYLE[c]

        d = series(df, c, "limb_length")

        # The stand-still reference is drawn larger and underneath, so
        # it stays visible where it overlaps the PPO line at 0%.
        size = 11 if c == "standstill" else 8

        ax.plot(
            d["percent"],
            d["fall_rate_percent"],
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.2,
            markersize=size,
            markerfacecolor="none" if c == "standstill" else color,
            zorder=z,
            clip_on=False,
            label=label,
        )

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(
        "Fall Rate vs. Limb-Length Perturbation",
        fontweight="bold",
        fontsize=13,
    )
    style_axes(ax, "Limb-Length Perturbation (%)", "Fall Rate (%)")

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=9.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=(0, 0.09, 1, 1))

    path = os.path.join(outdir, "fig4_fall_rate_limb_length.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


# ======================================================================
# Figure 5: mechanical energy vs mass perturbation
# ======================================================================

def figure5(df, outdir):

    controllers = present_controllers(df)

    fig, ax = plt.subplots(figsize=(8, 5.2))

    top = 0.0

    for c in controllers:

        label, color, marker, ls, z = STYLE[c]

        d = series(df, c, "mass")

        top = max(top, float(d["energy_mean"].max()))

        ax.plot(
            d["percent"],
            d["energy_mean"],
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.2,
            markersize=7,
            zorder=z,
            clip_on=False,
            label=label,
        )

    # Round the upper limit up to the next 1000.
    ymax = int(-(-top // 1000) * 1000)

    ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, ymax + 1, 1000))
    ax.set_title(
        "Mean Mechanical Energy per Episode vs. Mass Perturbation",
        fontweight="bold",
        fontsize=12.5,
    )
    style_axes(
        ax,
        "Mass Perturbation (%)",
        "Mean Mechanical Energy per Episode",
    )

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=9.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))

    path = os.path.join(outdir, "fig5_energy_vs_mass.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return path


def main():

    parser = argparse.ArgumentParser(
        description="Make paper figures from summary.csv."
    )

    parser.add_argument(
        "--summary",
        default=os.path.join(
            "results_with_standstill",
            "summary.csv",
        ),
        help="Summary CSV written by analyze_results.py.",
    )

    parser.add_argument(
        "--outdir",
        default="figures",
        help="Folder for the PNG files.",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load(args.summary)

    print(f"Read {len(df)} summary rows from {args.summary}")
    print(f"Controllers: {present_controllers(df)}")

    for make in (figure3, figure4, figure5):
        print(f"Saved {make(df, args.outdir)}")


if __name__ == "__main__":
    main()