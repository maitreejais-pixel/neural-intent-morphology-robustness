import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


# Defaults reproduce the original behavior (same input file, same
# output folder). To analyze another results file WITHOUT overwriting
# the existing summary and plots, pass --input and --outdir, e.g.:
#
#   python scripts/analyze_results.py \
#       --input results/physics_results_run2.csv \
#       --outdir results_run2
#
DEFAULT_INPUT = os.path.join("results", "physics_results.csv")
DEFAULT_OUTDIR = "results"


# ---------------------------------------------------------------------
# HISTORY (kept from the earlier versions of this script)
# ---------------------------------------------------------------------
#
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
# Distance and fall rate are therefore kept only as supporting,
# clearly-labeled context rather than the headline number.
#
# ADDED: if the results file contains a "standstill" controller (zero
# torque at every step), the summary gains standstill_* columns and a
# table comparing every controller to the stand-still reference is
# printed. This calibrates the reward: a policy that does nothing must
# not score as well as one that follows the commands.
# "Best controller by condition" is ranked among the real controllers
# only (the stand-still reference is excluded from it).
#
# ---------------------------------------------------------------------
# CHANGES IN THIS VERSION
# ---------------------------------------------------------------------
#
# The environment reward changed (src/env.py): tracking terms are now
# baseline-subtracted, so a robot that does not move earns 0 on the
# Forward / Left / Right tracking terms. The results CSV therefore has
# new columns, and this script uses them when they are present:
#
#   task_reward     tracking reward only (no uprightness bonus, no
#                   control cost). A stand-still robot scores only
#                   about 150 here (the Stop segments), so this is the
#                   PRIMARY metric now.
#   fwd_vel_err     mean |forward speed - target| in Forward segments
#   yaw_rate_err    mean |yaw rate - target| in Left/Right segments
#   stop_speed      mean residual speed in Stop segments
#
# "reward" (total episode reward) is kept for continuity: it still
# includes the uprightness bonus, which is a constant for any policy
# that stays upright and does not separate controllers.
#
# Everything that was in the old summary is still written. Old results
# files without the new columns still work (the new columns are simply
# skipped).

REFERENCE_CONTROLLER = "standstill"

# Below this nominal distance (metres) a "retained distance" percentage
# is meaningless (it would explode), so it is set to NaN. NOTE: the
# stand-still robot is NOT necessarily below this threshold -- it
# settles and drifts a few tens of centimetres -- so its distance
# percentage can still be large; that is exactly why distance is
# reference-only (see the history note at the top of this script).
MIN_NOMINAL_DISTANCE = 0.05


def main():

    parser = argparse.ArgumentParser(
        description="Analyze full-physics MuJoCo results."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Per-episode results CSV.",
    )

    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Folder for summary CSVs and plots.",
    )

    args = parser.parse_args()

    INPUT_FILE = args.input
    RESULTS_DIR = args.outdir

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    SUMMARY_FILE = os.path.join(
        RESULTS_DIR,
        "summary.csv",
    )

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
        f"Input file: {INPUT_FILE}"
    )

    print(
        f"Output folder: {RESULTS_DIR}"
    )

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

    has_task_reward = "task_reward" in df.columns

    if has_task_reward:
        print(
            "Task-reward columns found: task_reward is the "
            "primary metric."
        )
    else:
        print(
            "No task_reward column (older results file): "
            "reward is the primary metric."
        )

    # The primary metric: task_reward when available, else reward.
    PRIMARY = (
        "task_reward_mean"
        if has_task_reward
        else "reward_mean"
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
    # Nominal performance (level == 0)
    # ---------------------------------------------------------

    nominal_columns = ["distance", "reward"]

    if has_task_reward:
        nominal_columns.append("task_reward")

    nominal = (
        df[df["level"] == 0]
        .groupby("controller")[nominal_columns]
        .mean()
        .rename(
            columns={
                "distance": "nominal_distance",
                "reward": "nominal_reward",
                "task_reward": "nominal_task_reward",
            }
        )
    )

    df = df.join(
        nominal,
        on="controller",
    )

    required_nominal = [
        "nominal_distance",
        "nominal_reward",
    ]

    if has_task_reward:
        required_nominal.append("nominal_task_reward")

    if df[required_nominal].isna().any().any():
        raise ValueError(
            "Some controllers do not have nominal "
            "level=0 results."
        )

    # ---------------------------------------------------------
    # Retained distance (kept for reference -- see caveat above)
    #
    # Note: this percentage is not meaningful for a controller whose
    # nominal distance is (near) zero, and it is set to NaN below
    # MIN_NOMINAL_DISTANCE. It is also not meaningful for the
    # stand-still reference, which can show a large "retained
    # distance" from settling/drift alone.
    # ---------------------------------------------------------

    df["retained_percent"] = (
        100.0
        * df["distance"]
        / df["nominal_distance"]
    ).where(
        df["nominal_distance"] >= MIN_NOMINAL_DISTANCE
    )

    # ---------------------------------------------------------
    # Retained REWARD (total reward and task reward).
    #
    # Nominal reward can be negative or near zero (e.g. CPG falls even
    # at 0% perturbation), which makes a simple percentage misleading.
    # We therefore only compute the retained percentage when the
    # nominal value is positive, and report the raw difference
    # alongside it in all cases so the numbers are never silently
    # misleading.
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

    if has_task_reward:

        df["task_reward_retained_percent"] = df.apply(
            lambda row: (
                100.0
                * row["task_reward"]
                / row["nominal_task_reward"]
                if row["nominal_task_reward"] > 0
                else float("nan")
            ),
            axis=1,
        )

        df["task_reward_delta_from_nominal"] = (
            df["task_reward"] - df["nominal_task_reward"]
        )

    # ---------------------------------------------------------
    # Recovery time
    # (always empty in the current environment, because episodes end
    # at the first fall; kept so older results files still work)
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
        "n_episodes": ("reward", "size"),
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

    if has_task_reward:
        aggregations["task_reward_mean"] = ("task_reward", "mean")
        aggregations["task_reward_sd"] = ("task_reward", "std")
        aggregations["task_reward_retained_mean"] = (
            "task_reward_retained_percent",
            "mean",
        )
        aggregations["task_reward_delta_mean"] = (
            "task_reward_delta_from_nominal",
            "mean",
        )

    for column, name in (
        ("fwd_vel_err", "fwd_vel_err_mean"),
        ("yaw_rate_err", "yaw_rate_err_mean"),
        ("stop_speed", "stop_speed_mean"),
    ):
        if column in df.columns:
            aggregations[name] = (column, "mean")

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
    # Stand-still reference comparison (only if present)
    # ---------------------------------------------------------

    has_reference = (
        REFERENCE_CONTROLLER
        in summary["controller"].values
    )

    if has_reference:

        reference_columns = ["axis", "level", "reward_mean"]

        reference_rename = {
            "reward_mean": "standstill_reward_mean",
        }

        if has_task_reward:
            reference_columns.append("task_reward_mean")
            reference_rename["task_reward_mean"] = (
                "standstill_task_reward_mean"
            )

        reference = (
            summary[
                summary["controller"]
                == REFERENCE_CONTROLLER
            ][reference_columns]
            .rename(
                columns=reference_rename
            )
        )

        summary = summary.merge(
            reference,
            on=["axis", "level"],
            how="left",
        )

        summary["reward_minus_standstill"] = (
            summary["reward_mean"]
            - summary["standstill_reward_mean"]
        )

        if has_task_reward:
            summary["task_minus_standstill"] = (
                summary["task_reward_mean"]
                - summary["standstill_task_reward_mean"]
            )

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
        f"NOTE: {PRIMARY} is the primary robustness comparison. "
        "distance_mean / retained_mean is kept for reference only -- "
        "it can look artificially good for a controller that ignores "
        "the Left/Right/Stop commands and just keeps walking forward, "
        "since raw net displacement does not penalize that. See the "
        "note at the top of this script."
    )
    print()

    print(summary.to_string(index=False))

    if has_reference:

        comparison_column = (
            "task_minus_standstill"
            if has_task_reward
            else "reward_minus_standstill"
        )

        print()
        print("=" * 90)
        print(
            f"{comparison_column.upper()} "
            "(negative = scores below a do-nothing policy)"
        )
        print("=" * 90)

        comparison = (
            summary[
                summary["controller"]
                != REFERENCE_CONTROLLER
            ]
            .pivot_table(
                index=["axis", "level"],
                columns="controller",
                values=comparison_column,
            )
            .reset_index()
        )

        print(comparison.to_string(index=False))

        # Both comparison tables are saved when available.
        reward_comparison = (
            summary[
                summary["controller"]
                != REFERENCE_CONTROLLER
            ]
            .pivot_table(
                index=["axis", "level"],
                columns="controller",
                values="reward_minus_standstill",
            )
            .reset_index()
        )

        reward_comparison.to_csv(
            os.path.join(
                RESULTS_DIR,
                "reward_minus_standstill.csv",
            ),
            index=False,
        )

        if has_task_reward:

            comparison.to_csv(
                os.path.join(
                    RESULTS_DIR,
                    "task_minus_standstill.csv",
                ),
                index=False,
            )

    # ---------------------------------------------------------
    # Controller-level nominal performance
    # ---------------------------------------------------------

    nominal_aggregations = {
        "nominal_distance": ("distance", "mean"),
        "nominal_reward": ("reward", "mean"),
        "nominal_energy": ("energy", "mean"),
        "nominal_fall_rate": ("fall_numeric", "mean"),
    }

    if has_task_reward:
        nominal_aggregations["nominal_task_reward"] = (
            "task_reward",
            "mean",
        )

    for column, name in (
        ("fwd_vel_err", "nominal_fwd_vel_err"),
        ("yaw_rate_err", "nominal_yaw_rate_err"),
        ("stop_speed", "nominal_stop_speed"),
    ):
        if column in df.columns:
            nominal_aggregations[name] = (column, "mean")

    nominal_summary = (
        df[df["level"] == 0]
        .groupby("controller")
        .agg(**nominal_aggregations)
        .reset_index()
    )

    nominal_summary["nominal_fall_rate_percent"] = (
        100.0 * nominal_summary["nominal_fall_rate"]
    )

    nominal_summary.to_csv(
        os.path.join(RESULTS_DIR, "nominal_summary.csv"),
        index=False,
    )

    print()
    print("=" * 90)
    print("NOMINAL SUMMARY")
    print("=" * 90)
    print(nominal_summary.to_string(index=False))

    # ---------------------------------------------------------
    # Robustness plots -- per axis: task reward (primary, when
    # available), total reward, and distance (reference only), plus
    # the fall-rate plots below.
    # ---------------------------------------------------------

    controllers = summary["controller"].dropna().unique()
    axes = summary["axis"].dropna().unique()

    for axis in axes:

        # --- Task-reward robustness (primary) ---
        if has_task_reward:

            plt.figure(figsize=(7, 5))
            for controller in controllers:
                d = summary[
                    (summary["axis"] == axis)
                    & (summary["controller"] == controller)
                ].sort_values("level")
                plt.plot(
                    d["level"] * 100.0,
                    d["task_reward_mean"],
                    marker="o",
                    label=controller,
                )
            plt.xlabel("Perturbation (%)")
            plt.ylabel("Mean task reward")
            plt.title(
                f"Task reward vs {axis.replace('_', ' ')} "
                "perturbation (primary metric)"
            )
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            output = os.path.join(
                RESULTS_DIR, f"{axis}_task_reward.png"
            )
            plt.savefig(output, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved {output}")

        # --- Total-reward robustness ---
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
            f"Reward vs {axis.replace('_', ' ')} perturbation"
            + (
                ""
                if has_task_reward
                else " (primary robustness metric)"
            )
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
    # Final endpoint comparison -- sorted by the PRIMARY metric
    # ---------------------------------------------------------

    endpoint = (
        summary[summary["level"] == 0.5]
        .copy()
        .sort_values(
            ["axis", PRIMARY],
            ascending=[True, False],
        )
    )

    endpoint.to_csv(
        os.path.join(RESULTS_DIR, "endpoint_50_percent.csv"),
        index=False,
    )

    print()
    print("=" * 90)
    print(
        f"50% PERTURBATION ENDPOINT (sorted by {PRIMARY}, "
        "primary metric)"
    )
    print("=" * 90)
    print(endpoint.to_string(index=False))

    # ---------------------------------------------------------
    # Best controller at each perturbation -- ranked by the PRIMARY
    # metric. The stand-still reference is a calibration baseline, not
    # a candidate controller, so it is excluded from this ranking (its
    # value is reported alongside instead).
    # ---------------------------------------------------------

    best_rows = []

    candidates = summary[
        summary["controller"] != REFERENCE_CONTROLLER
    ]

    for axis in axes:
        for level in sorted(summary["level"].unique()):
            d = candidates[
                (candidates["axis"] == axis)
                & (candidates["level"] == level)
            ]
            if len(d) == 0:
                continue
            best = d.loc[d[PRIMARY].idxmax()]
            row = {
                "axis": axis,
                "level": level,
                "best_controller": best["controller"],
                PRIMARY: best[PRIMARY],
                "reward_mean": best["reward_mean"],
                "retained_percent_distance_reference": best[
                    "retained_mean"
                ],
                "fall_rate_percent": best["fall_rate_percent"],
            }
            if has_reference:
                row["standstill_reward_mean"] = best[
                    "standstill_reward_mean"
                ]
                if has_task_reward:
                    row["standstill_task_reward_mean"] = best[
                        "standstill_task_reward_mean"
                    ]
            best_rows.append(row)

    best_df = pd.DataFrame(best_rows)

    best_df.to_csv(
        os.path.join(RESULTS_DIR, "best_controller_by_condition.csv"),
        index=False,
    )

    print()
    print("=" * 90)
    print(
        f"BEST CONTROLLER BY CONDITION (ranked by {PRIMARY}, "
        "stand-still reference excluded)"
    )
    print("=" * 90)
    print(best_df.to_string(index=False))

    print()
    print("=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()