"""
Make paper figures 3, 4 and 5 directly from the analysis summary.

No numbers are typed in: everything is read from summary.csv, which
scripts/analyze_results.py writes. Re-run this script whenever the
summary changes.

Usage (from the project root):

    python scripts/make_figures.py

Optional:

    python scripts/make_figures.py \
        --summary results/summary.csv \
        --outdir figures \
        --metric task_reward_mean

Outputs (PNG, 300 dpi):

    fig3_reward_vs_perturbation.png   mean reward (+/-1 SD), 4 axes
    fig4_fall_rate_limb_length.png    fall rate vs limb-length perturbation
    fig5_energy_vs_mass.png           mean mechanical energy vs mass perturbation

CHANGES vs. the previous version
--------------------------------
* Figure 3 plots TASK reward by default (the primary metric now: a
  stand-still robot scores only ~140 on it). Use --metric reward_mean
  for the total episode reward (which includes the constant uprightness
  bonus). If the summary has no task-reward column (an older results
  file), reward_mean is used automatically.
* The y-axis of figure 3 is no longer fixed to 0-800. The scale follows
  the data, and it extends below zero when a controller's mean reward is
  negative (controllers that fall early score below zero). Lower error
  bars are truncated at the axis floor, which is 0 when no mean is
  negative (as before). Say so in the caption.
* The no-morphology PPO ablation ("ppo-no-morphology") has its own
  style and appears in all figures when it is in the summary.
* If --summary is not given, results/summary.csv is used; if that does
  not exist, results_with_standstill/summary.csv (the old default) is
  tried.
"""

import argparse
import math
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
    "ppo-no-morphology": ("PPO, no morphology input", "#ff7f0e", "v", "-.", 5),
    "standstill": ("Stand-still reference", "#7f7f7f", "D", ":", 2),
}

# Order in which controllers are drawn and listed in legends.
ORDER = [
    "direct",
    "cpg",
    "morphology-aware",
    "ppo-no-morphology",
    "standstill",
]

AXES = [
    ("mass", "(a) Mass Axis", "Mass Perturbation (%)"),
    ("limb_length", "(b) Limb Length Axis", "Limb-Length Perturbation (%)"),
    ("joint_strength", "(c) Joint Strength Axis", "Joint-Strength Perturbation (%)"),
    ("friction", "(d) Friction Axis", "Friction Perturbation (%)"),
]

# Summary files tried when --summary is not given.
DEFAULT_SUMMARIES = (
    os.path.join("results", "summary.csv"),
    os.path.join("results_with_standstill", "summary.csv"),
)


def default_summary_path():

    for path in DEFAULT_SUMMARIES:
        if os.path.exists(path):
            return path

    return DEFAULT_SUMMARIES[0]


def load(summary_path, metric=None):
    """
    Read summary.csv. Returns (dataframe, metric, sd_column).

    metric=None picks task_reward_mean when the summary has it,
    otherwise reward_mean.
    """

    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Could not find {summary_path}. "
            "Run scripts/analyze_results.py first."
        )

    df = pd.read_csv(summary_path)

    if metric is None:
        metric = (
            "task_reward_mean"
            if "task_reward_mean" in df.columns
            else "reward_mean"
        )

    sd_column = metric.replace("_mean", "_sd")

    needed = {
        "controller",
        "axis",
        "level",
        metric,
        sd_column,
        "fall_rate_percent",
        "energy_mean",
    }

    missing = needed - set(df.columns)

    if missing:
        raise ValueError(
            f"{summary_path} is missing columns: {sorted(missing)}"
        )

    df["percent"] = (df["level"] * 100.0).round().astype(int)

    return df, metric, sd_column


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


def legend_columns(n_labels):
    """One row for up to 4 entries, otherwise 3 columns."""

    return n_labels if n_labels <= 4 else 3


def reward_axis_limits(df, controllers, metric, sd_column):
    """
    y-limits for figure 3, shared by all four panels.

    The floor is 0 unless some mean is negative, in which case it drops
    to the next multiple of 100 below the lowest mean. The ceiling is
    the highest (mean + SD), rounded up to a multiple of 100.
    """

    values = df[df["controller"].isin(controllers)]

    lowest = float(values[metric].min())

    highest = float(
        (values[metric] + values[sd_column].fillna(0.0)).max()
    )

    floor = (
        0.0
        if lowest >= 0.0
        else math.floor(lowest / 100.0) * 100.0
    )

    ceiling = max(
        100.0,
        math.ceil(1.05 * highest / 100.0) * 100.0,
    )

    span = ceiling - floor

    step = max(50, int(round(span / 5.0 / 50.0)) * 50)

    return floor, ceiling, step


# ======================================================================
# Figure 3: reward vs perturbation (2 x 2)
# ======================================================================

def figure3(df, metric, sd_column, outdir):

    controllers = present_controllers(df)

    ylabel = (
        "Mean Task Reward"
        if metric.startswith("task")
        else "Mean Episode Reward"
    )

    floor, ceiling, step = reward_axis_limits(
        df,
        controllers,
        metric,
        sd_column,
    )

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
            m = d[metric].to_numpy()
            sd = d[sd_column].fillna(0.0).to_numpy()

            # Lower error bars stop at the axis floor (0 when no mean
            # is negative). This truncation is stated in the caption.
            lower = [
                min(s, max(v - floor, 0.0))
                for v, s in zip(m, sd)
            ]

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

        ax.set_ylim(floor, ceiling)
        ax.set_yticks(range(int(floor), int(ceiling) + 1, step))
        ax.set_title(title, fontweight="bold", fontsize=13)
        style_axes(ax, xlabel, ylabel)
        ax.tick_params(labelleft=True)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_columns(len(labels)),
        fontsize=10.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(
        rect=(0, 0.05 if len(labels) <= 4 else 0.08, 1, 1)
    )

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
        ncol=legend_columns(len(labels)),
        fontsize=9.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(
        rect=(0, 0.09 if len(labels) <= 4 else 0.13, 1, 1)
    )

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

    # Round the upper limit up to the next 1000 (at least 1000).
    ymax = max(1000, int(-(-top // 1000) * 1000))

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
        ncol=legend_columns(len(labels)),
        fontsize=9.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(
        rect=(0, 0.08 if len(labels) <= 4 else 0.12, 1, 1)
    )

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
        default=None,
        help="Summary CSV written by analyze_results.py. Default: "
        "results/summary.csv, else results_with_standstill/summary.csv.",
    )

    parser.add_argument(
        "--outdir",
        default="figures",
        help="Folder for the PNG files.",
    )

    parser.add_argument(
        "--metric",
        default=None,
        choices=["task_reward_mean", "reward_mean"],
        help="Quantity plotted in figure 3. Default: task_reward_mean "
        "when the summary has it, else reward_mean.",
    )

    args = parser.parse_args()

    summary_path = args.summary or default_summary_path()

    os.makedirs(args.outdir, exist_ok=True)

    df, metric, sd_column = load(
        summary_path,
        args.metric,
    )

    print(f"Read {len(df)} summary rows from {summary_path}")
    print(f"Controllers: {present_controllers(df)}")
    print(f"Figure 3 metric: {metric}")

    print(f"Saved {figure3(df, metric, sd_column, args.outdir)}")
    print(f"Saved {figure4(df, args.outdir)}")
    print(f"Saved {figure5(df, args.outdir)}")


if __name__ == "__main__":
    main()