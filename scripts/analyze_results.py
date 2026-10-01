import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "physics_results.csv",
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "summary.csv",
)

# FIXED: this script previously used DISTANCE (raw net displacement)
# as the primary "retained performance" metric and as the sole basis
# for "best_controller_by_condition". That metric has a real flaw for
# an INTENTION-FOLLOWING benchmark: net displacement rewards a
# controller for moving far from its start point regardless of
# whether it actually executed the commanded Forward/Left/Right/Stop
# schedule. A policy that ignores the Stop and turning commands and
# simply keeps walking forward the whole episode will rack up MORE
# net distance than one that correctly stops and turns as
# instructed -- so "distance" can reward *ignoring* the task.
#
# Concretely: at limb_length=0.5, PPO's distance-based "retained"
# figure was 468% (looks like a huge improvement under perturbation),
# but its reward_mean there was 378.58 -- actually WORSE than
# Direct's 617.25 at the same condition. The reward the environment
# already computes (env.py's _intention_reward, which is
# intention-specific: tracks commanded forward speed during Forward,
# commanded yaw during Left/Right, stillness during Stop, plus an
# uprightness bonus and a per-step fall penalty) does NOT have this
# blind spot, and it is already logged in the "reward" column of
# every episode. This script now treats REWARD as the primary
# performance metric, with distance and fall rate kept as supporting,
# clearly-labeled context rather than the headline number.


def main():

    # ---------------------------------------------------------
    # Load results
    # ---------------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 90)
    print("FULL-PHYSICS MUJOCO RESULTS ANALYSIS")
    print("=" * 90)

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Controllers: {df['controller'].unique().tolist()}"
    )

    print(
        f"Axes: {df['axis'].unique().tolist()}"
    )

    print(
        f"Perturbation levels: "
        f"{sorted(df['level'].unique().tolist())}"
    )

    # ---------------------------------------------------------
    # Handle fall column
    # ---------------------------------------------------------

    if "fallen" in df.columns:
        fall_column = "fallen"
    elif "fall" in df.columns:
        fall_column = "fall"
    else:
        raise ValueError(
            "Results CSV must contain either "
            "'fallen' or 'fall'."
        )

    if df[fall_column].dtype == bool:
        df["fall_numeric"] = df[fall_column].astype(int)
    else:
        df["fall_numeric"] = (
            df[fall_column]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                }
            )
        )

    if df["fall_numeric"].isna().any():
        raise ValueError(
            f"Could not interpret some values in "
            f"{fall_column} as fall/no-fall."
        )

    # ---------------------------------------------------------
    # Nominal performance (level == 0), for BOTH distance and reward
    # ---------------------------------------------------------

    nominal = (
        df[df["level"] == 0]
        .groupby("controller")[["distance", "reward"]]
        .mean()
        .rename(
            columns={
                "distance": "nominal_distance",
                "reward": "nominal_reward",
            }
        )
    )

    df = df.join(
        nominal,
        on="controller",
    )

    if df["nominal_distance"].isna().any() or df["nominal_reward"].isna().any():
        raise ValueError(
            "Some controllers do not have nominal "
            "level=0 results."
        )

    # ---------------------------------------------------------
    # Retained distance (kept for reference -- see caveat above)
    # ---------------------------------------------------------

    df["retained_percent"] = (
        100.0
        * df["distance"]
        / df["nominal_distance"]
    )

    # ---------------------------------------------------------
    # FIXED: retained REWARD -- the primary robustness metric.
    #
    # Unlike retained_percent (distance-based), this cannot be
    # inflated by a controller that ignores commanded turns/stops and
    # just walks in one direction, because the reward itself already
    # penalizes not following the current intention.
    #
    # Nominal reward can be negative (e.g. CPG falls even at 0%
    # perturbation in this run), which makes a simple percentage
    # misleading (a very negative nominal reward makes any less-bad
    # reward look like a huge positive "improvement"). We therefore
    # only compute reward_retained_percent when nominal_reward is
    # positive, and report the raw reward difference alongside it in
    # all cases so the numbers are never silently misleading.
    # ---------------------------------------------------------

    df["reward_retained_percent"] = df.apply(
        lambda row: (
            100.0 * row["reward"] / row["nominal_reward"]
            if row["nominal_reward"] > 0
            else float("nan")
        ),
        axis=1,
    )

    df["reward_delta_from_nominal"] = (
        df["reward"] - df["nominal_reward"]
    )

    # ---------------------------------------------------------
    # Recovery time
    # ---------------------------------------------------------

    if "recovery_time" in df.columns:
        df["recovery_time_numeric"] = pd.to_numeric(
            df["recovery_time"],
            errors="coerce",
        )
    else:
        df["recovery_time_numeric"] = float("nan")

    # ---------------------------------------------------------
    # Main summary
    # ---------------------------------------------------------

    aggregations = {
        "distance_mean": ("distance", "mean"),
        "distance_sd": ("distance", "std"),
        "retained_mean": ("retained_percent", "mean"),
        "retained_sd": ("retained_percent", "std"),
        "reward_mean": ("reward", "mean"),
        "reward_sd": ("reward", "std"),
        "reward_retained_mean": ("reward_retained_percent", "mean"),
        "reward_delta_mean": ("reward_delta_from_nominal", "mean"),
        "fall_rate": ("fall_numeric", "mean"),
        "energy_mean": ("energy", "mean"),
    }

    if "forward_distance" in df.columns:
        aggregations["forward_distance_mean"] = (
            "forward_distance",
            "mean",
        )

    if "yaw_change" in df.columns:
        aggregations["yaw_change_mean"] = ("yaw_change", "mean")
        aggregations["yaw_change_sd"] = ("yaw_change", "std")

    if "steps" in df.columns:
        aggregations["steps_mean"] = ("steps", "mean")

    if "recovery_time" in df.columns:
        aggregations["recovery_time_mean"] = (
            "recovery_time_numeric",
            "mean",
        )
        aggregations["recovery_time_sd"] = (
            "recovery_time_numeric",
            "std",
        )

    summary = (
        df
        .groupby(["controller", "axis", "level"])
        .agg(**aggregations)
        .reset_index()
    )

    summary["fall_rate_percent"] = 100.0 * summary["fall_rate"]

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary.to_csv(SUMMARY_FILE, index=False)

    print()
    print(f"Saved summary to {SUMMARY_FILE}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        "NOTE: reward_mean / reward_retained_mean is the primary "
        "robustness comparison. distance_mean / retained_mean is kept "
        "for reference only -- it can look artificially good for a "
        "controller that ignores the Left/Right/Stop commands and "
        "just keeps walking forward, since raw net displacement does "
        "not penalize that. See the note at the top of this script."
    )
    print()

    print(summary.to_string(index=False))

    # ---------------------------------------------------------
    # Controller-level nominal performance
    # ---------------------------------------------------------

    nominal_summary = (
        df[df["level"] == 0]
        .groupby("controller")
        .agg(
            nominal_distance=("distance", "mean"),
            nominal_reward=("reward", "mean"),
            nominal_energy=("energy", "mean"),
            nominal_fall_rate=("fall_numeric", "mean"),
        )
        .reset_index()
    )

    nominal_summary["nominal_fall_rate_percent"] = (
        100.0 * nominal_summary["nominal_fall_rate"]
    )

    nominal_summary.to_csv(
        os.path.join(RESULTS_DIR, "nominal_summary.csv"),
        index=False,
    )

    # ---------------------------------------------------------
    # Robustness plots -- now one for distance (reference) and one
    # for reward (primary), per axis, plus the existing fall-rate
    # plots.
    # ---------------------------------------------------------

    controllers = summary["controller"].dropna().unique()
    axes = summary["axis"].dropna().unique()

    for axis in axes:

        # --- Reward-based robustness (primary) ---
        plt.figure(figsize=(7, 5))
        for controller in controllers:
            d = summary[
                (summary["axis"] == axis)
                & (summary["controller"] == controller)
            ].sort_values("level")
            plt.plot(
                d["level"] * 100.0,
                d["reward_mean"],
                marker="o",
                label=controller,
            )
        plt.xlabel("Perturbation (%)")
        plt.ylabel("Mean episode reward")
        plt.title(
            f"Reward vs {axis.replace('_', ' ')} perturbation "
            "(primary robustness metric)"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        output = os.path.join(RESULTS_DIR, f"{axis}_reward.png")
        plt.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {output}")

        # --- Distance-based robustness (reference only) ---
        plt.figure(figsize=(7, 5))
        for controller in controllers:
            d = summary[
                (summary["axis"] == axis)
                & (summary["controller"] == controller)
            ].sort_values("level")
            plt.plot(
                d["level"] * 100.0,
                d["retained_mean"],
                marker="o",
                label=controller,
            )
        plt.xlabel("Perturbation (%)")
        plt.ylabel("Retained walking distance (%) -- reference only")
        plt.title(
            f"Distance vs {axis.replace('_', ' ')} perturbation "
            "(reference only -- can reward ignoring commands)"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        output = os.path.join(
            RESULTS_DIR, f"{axis}_robustness.png"
        )
        plt.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {output}")

    # ---------------------------------------------------------
    # Fall-rate plots
    # ---------------------------------------------------------

    for axis in axes:

        plt.figure(figsize=(7, 5))
        for controller in controllers:
            d = summary[
                (summary["axis"] == axis)
                & (summary["controller"] == controller)
            ].sort_values("level")
            plt.plot(
                d["level"] * 100.0,
                d["fall_rate_percent"],
                marker="o",
                label=controller,
            )
        plt.xlabel("Perturbation (%)")
        plt.ylabel("Fall rate (%)")
        plt.title(f"Falls under {axis.replace('_', ' ')} variation")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        output = os.path.join(RESULTS_DIR, f"{axis}_fall_rate.png")
        plt.savefig(output, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {output}")

    # ---------------------------------------------------------
    # Final endpoint comparison -- sorted by REWARD now, not distance
    # ---------------------------------------------------------

    endpoint = (
        summary[summary["level"] == 0.5]
        .copy()
        .sort_values(
            ["axis", "reward_mean"],
            ascending=[True, False],
        )
    )

    endpoint.to_csv(
        os.path.join(RESULTS_DIR, "endpoint_50_percent.csv"),
        index=False,
    )

    print()
    print("=" * 90)
    print("50% PERTURBATION ENDPOINT (sorted by reward_mean, primary metric)")
    print("=" * 90)
    print(endpoint.to_string(index=False))

    # ---------------------------------------------------------
    # Best controller at each perturbation -- FIXED: now ranked by
    # reward_mean, not distance-based retained_mean.
    # ---------------------------------------------------------

    best_rows = []

    for axis in axes:
        for level in sorted(summary["level"].unique()):
            d = summary[
                (summary["axis"] == axis)
                & (summary["level"] == level)
            ]
            if len(d) == 0:
                continue
            best = d.loc[d["reward_mean"].idxmax()]
            best_rows.append(
                {
                    "axis": axis,
                    "level": level,
                    "best_controller": best["controller"],
                    "reward_mean": best["reward_mean"],
                    "retained_percent_distance_reference": best[
                        "retained_mean"
                    ],
                    "fall_rate_percent": best["fall_rate_percent"],
                }
            )

    best_df = pd.DataFrame(best_rows)

    best_df.to_csv(
        os.path.join(RESULTS_DIR, "best_controller_by_condition.csv"),
        index=False,
    )

    print()
    print("=" * 90)
    print("BEST CONTROLLER BY CONDITION (ranked by reward_mean)")
    print("=" * 90)
    print(best_df.to_string(index=False))

    print()
    print("=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()