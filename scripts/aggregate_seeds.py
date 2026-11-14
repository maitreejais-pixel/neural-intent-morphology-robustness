"""
Combine the experiment results of several TRAINING seeds.

Each training seed gives one results file, produced by

    python scripts/run_experiments.py --controllers direct cpg ppo ppo_nomorph standstill \
        --model models/final_aware_s42.zip --model-nomorph models/final_nomorph_s42.zip \
        --output results/physics_results_s42.csv

(for the other seeds only the two PPO arms are needed:
 --controllers ppo ppo_nomorph). Then:

    python scripts/aggregate_seeds.py > results/aggregate_report_v2.txt

What it does
------------
* Statistics are computed per training seed first (mean over the
  evaluation episodes of that seed). The evaluation episodes of one
  trained policy differ only through small reset noise, so the real
  uncertainty is the variation BETWEEN TRAINING SEEDS. That is what is
  summarized here.
* Controllers that do not depend on the training seed (direct, cpg,
  standstill) give identical rows in every file; they are taken once.
* DISTINCT CONDITIONS: the nominal point (level 0) is identical on all
  four axes, so every overall mean counts it once. The study has 21
  distinct conditions (1 nominal + 4 axes x 5 levels), not 24 rows.
  (An earlier version of this script counted nominal four times; its
  overall means and p-values are superseded.)
* Morphology-aware minus no-morphology is reported for: all 21
  conditions, nominal only, perturbed only (10-50%), each axis, and
  leaving out one training seed at a time.
* The two arms do not share an initialization, so "paired by seed" only
  means matched by seed INDEX. The unpaired Welch test is the main test;
  the matched test is shown for completeness. With 5 seeds both have very
  little power: treat p-values as descriptive.
* There is deliberately NO "drop from nominal" table. That quantity is
  (perturbed difference) - (nominal difference); when the nominal scores
  of the two arms differ, it measures the nominal gap, not robustness.
  Compare the NOMINAL and PERTURBED blocks instead.

Outputs (in --outdir, default results/):
    seed_summary.csv            every controller and condition: mean / sd / min / max across seeds
    seed_overall.csv            per seed, mean over the 21 distinct conditions
    morph_vs_nomorph_split.csv  all comparison blocks below, one row per block and metric
    leave_one_out.csv           aware-minus-blind difference with each seed left out
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

AWARE = "morphology-aware"
BLIND = "ppo-no-morphology"
SEED_FREE = ("direct", "cpg", "standstill")

METRICS = [
    "task_reward",
    "reward",
    "fall_rate",
    "fwd_vel_err",
    "yaw_rate_err",
    "stop_speed",
    "energy",
    "distance",
]

AXES = ["mass", "limb_length", "joint_strength", "friction"]

COMPARED = ("task_reward", "fall_rate")
COMPARED_ALL = ("task_reward", "fall_rate", "fwd_vel_err", "yaw_rate_err")


# ======================================================================
# Loading and per-seed tables
# ======================================================================

def load_files(patterns):

    paths = []

    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))

    paths = sorted(set(paths))

    if not paths:
        raise FileNotFoundError(f"No files match {patterns}")

    frames = []

    for index, path in enumerate(paths):

        match = re.search(r"_s(\d+)\.csv$", path)

        seed = int(match.group(1)) if match else index

        df = pd.read_csv(path)

        df["train_seed"] = seed

        if "fallen" in df.columns:
            df["fall_rate"] = df["fallen"].astype(float)

        frames.append(df)

        print(f"  {path}: training seed {seed}, {len(df)} episode rows")

    return pd.concat(frames, ignore_index=True)


def per_seed_table(df):
    """Mean over evaluation episodes for every seed and condition."""

    present = [m for m in METRICS if m in df.columns]

    table = (
        df.groupby(["train_seed", "controller", "axis", "level"])[present]
        .mean()
        .reset_index()
    )

    return table, present


def distinct_only(table):
    """Keep each distinct condition once (nominal is the same on every axis)."""

    return table[(table["level"] > 0) | (table["axis"] == AXES[0])]


def across_seeds(table, present):
    """Mean / sd / min / max over training seeds, for every condition."""

    rows = []

    for (controller, axis, level), g in table.groupby(
        ["controller", "axis", "level"]
    ):

        if controller in SEED_FREE:
            # identical in every file: use the first seed only
            g = g[g["train_seed"] == g["train_seed"].min()]

        row = {
            "controller": controller,
            "axis": axis,
            "level": level,
            "n_seeds": len(g),
        }

        for metric in present:

            values = g[metric].astype(float)

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())

        rows.append(row)

    out = pd.DataFrame(rows)

    if "fall_rate_mean" in out.columns:
        out["fall_rate_percent"] = 100.0 * out["fall_rate_mean"]

    return out


# ======================================================================
# Aware minus no-morphology
# ======================================================================

def seed_means(sub, controller, metrics):

    g = sub[sub["controller"] == controller]

    return g.groupby("train_seed")[list(metrics)].mean()


def compare_block(block, sub, metrics=COMPARED):
    """One row per metric: aware minus no-morphology over training seeds."""

    metrics = [m for m in metrics if m in sub.columns]

    a = seed_means(sub, AWARE, metrics)
    b = seed_means(sub, BLIND, metrics)

    seeds = a.index.intersection(b.index)

    a, b = a.loc[seeds], b.loc[seeds]

    rows = []

    for m in metrics:

        d = (a[m] - b[m]).astype(float)

        n = len(d)

        sd = float(d.std(ddof=1)) if n > 1 else float("nan")

        row = {
            "block": block,
            "metric": m,
            "n_seeds": n,
            "aware_mean": float(a[m].mean()),
            "blind_mean": float(b[m].mean()),
            "diff_mean": float(d.mean()),
            "diff_sd": sd,
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "seeds_aware_higher": int((d > 0).sum()),
            "per_seed_diffs": ", ".join(f"{x:+.3f}" for x in d),
            "p_welch": float("nan"),
            "p_matched_by_seed": float("nan"),
        }

        if n >= 3 and sd > 0:

            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)

            row["ci95_low"] = float(d.mean() - half)
            row["ci95_high"] = float(d.mean() + half)

            row["p_matched_by_seed"] = float(
                stats.ttest_1samp(d.to_numpy(), 0.0).pvalue
            )

            row["p_welch"] = float(
                stats.ttest_ind(
                    a[m].to_numpy(),
                    b[m].to_numpy(),
                    equal_var=False,
                ).pvalue
            )

        rows.append(row)

    return pd.DataFrame(rows)


def print_block(frame):

    for _, r in frame.iterrows():

        print(
            f"{r['metric']:13s} aware {r['aware_mean']:8.3f} | "
            f"blind {r['blind_mean']:8.3f} | "
            f"diff {r['diff_mean']:+8.3f} (sd {r['diff_sd']:.3f}, "
            f"95% CI [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]) | "
            f"aware higher in {int(r['seeds_aware_higher'])}/{int(r['n_seeds'])} seeds"
        )
        print(
            f"{'':13s} per seed [{r['per_seed_diffs']}] | "
            f"Welch p={r['p_welch']:.3f} | matched-by-seed p={r['p_matched_by_seed']:.3f}"
        )


def leave_one_out(distinct):

    rows = []

    for s in sorted(distinct["train_seed"].unique()):

        sub = distinct[distinct["train_seed"] != s]

        a = seed_means(sub, AWARE, COMPARED)
        b = seed_means(sub, BLIND, COMPARED)

        d = a - b

        rows.append(
            {
                "left_out_seed": int(s),
                "reward_diff": float(d["task_reward"].mean()),
                "fall_rate_diff": float(d["fall_rate"].mean()),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# Main
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Combine experiment results over training seeds."
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[os.path.join("results", "physics_results_s*.csv")],
        help="Results files (glob patterns allowed). The training seed is "
        "read from the _s<seed>.csv ending of the file name.",
    )

    parser.add_argument("--outdir", default="results")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Reading:")

    df = load_files(args.inputs)

    table, present = per_seed_table(df)

    seeds = sorted(int(v) for v in table["train_seed"].unique())

    print(f"\nTraining seeds: {seeds}")

    if len(seeds) < 3:
        print(
            "NOTE: fewer than 3 training seeds. Spread across seeds and "
            "the tests are not meaningful yet."
        )

    # ---- all conditions (for figures) --------------------------------

    summary = across_seeds(table, present)

    summary.to_csv(os.path.join(args.outdir, "seed_summary.csv"), index=False)

    # ---- distinct conditions (for every overall number) --------------

    distinct = distinct_only(table)

    n_conditions = (
        distinct[distinct["controller"] == AWARE][["axis", "level"]]
        .drop_duplicates()
        .shape[0]
    )

    print(f"Distinct conditions per policy: {n_conditions} (expected 21)")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    overall = (
        distinct.groupby(["train_seed", "controller"])[
            [m for m in ("task_reward", "fall_rate") if m in distinct.columns]
        ]
        .mean()
        .reset_index()
    )

    overall.to_csv(os.path.join(args.outdir, "seed_overall.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"MEAN OVER THE {n_conditions} DISTINCT CONDITIONS, PER TRAINING SEED")
    print("=" * 100)
    print(overall.to_string(index=False))

    reference = distinct[distinct["controller"].isin(SEED_FREE)]

    if not reference.empty:

        reference = reference[
            reference["train_seed"] == reference["train_seed"].min()
        ]

        print("\nSeed-free references (deterministic; same rows in every file):")
        print(
            reference.groupby("controller")[["task_reward", "fall_rate"]]
            .mean()
            .to_string()
        )

    # ---- aware minus no-morphology -----------------------------------

    print("\n" + "=" * 100)
    print("MORPHOLOGY-AWARE MINUS NO-MORPHOLOGY")
    print("(positive reward difference = aware better; positive fall-rate difference = aware falls MORE)")
    print("The two arms share no initialization: 'matched' means matched by seed index only.")
    print("Welch (unpaired) is the main test; with 5 seeds treat every p-value as descriptive.")
    print("=" * 100)

    blocks = [
        (f"ALL {n_conditions} distinct conditions", distinct, COMPARED_ALL),
        ("NOMINAL only (level 0)", distinct[distinct["level"] == 0], COMPARED),
        ("PERTURBED only (levels 10-50%)", distinct[distinct["level"] > 0], COMPARED),
    ]

    for axis in AXES:
        blocks.append(
            (
                f"PERTURBED, axis = {axis}",
                distinct[(distinct["level"] > 0) & (distinct["axis"] == axis)],
                COMPARED,
            )
        )

    frames = []

    for name, sub, metrics in blocks:

        frame = compare_block(name, sub, metrics)

        frames.append(frame)

        print(f"\n{name}")
        print("-" * 100)
        print_block(frame)

    pd.concat(frames, ignore_index=True).to_csv(
        os.path.join(args.outdir, "morph_vs_nomorph_split.csv"),
        index=False,
    )

    # ---- leave one seed out -------------------------------------------

    loo = leave_one_out(distinct)

    loo.to_csv(os.path.join(args.outdir, "leave_one_out.csv"), index=False)

    print("\n" + "=" * 100)
    print(f"LEAVE-ONE-SEED-OUT: aware minus blind over all {n_conditions} conditions")
    print("=" * 100)

    for _, r in loo.iterrows():
        print(
            f"without seed {int(r['left_out_seed'])}: "
            f"reward {r['reward_diff']:+8.2f} | "
            f"fall rate {r['fall_rate_diff']:+.3f}"
        )

    print(
        "\nNOTE: a 'drop from nominal' comparison is not reported. It equals the "
        "perturbed difference minus the nominal difference, so when the two arms "
        "differ at nominal it reflects that gap, not robustness under perturbation."
    )

    print(f"\nSaved tables to {args.outdir}/")


if __name__ == "__main__":
    main()