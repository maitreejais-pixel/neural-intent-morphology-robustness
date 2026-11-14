"""Five-seed comparison with each DISTINCT condition counted once.
Nominal (level 0) is identical on all four axes, so it is kept for one axis only.
Imports the loaders from aggregate_seeds.py; run from the project root:
    python scripts/aggregate_v2.py > results/aggregate_report_v2.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy import stats

from aggregate_seeds import AWARE, BLIND, load_files, per_seed_table

METRICS = ["task_reward", "fall_rate"]


def per_seed(sub, controller):
    return (
        sub[sub["controller"] == controller]
        .groupby("train_seed")[METRICS]
        .mean()
    )


def compare(title, sub):

    a, b = per_seed(sub, AWARE), per_seed(sub, BLIND)

    print(f"\n{title}")
    print("-" * 100)

    for m in METRICS:

        d = a[m] - b[m]

        paired = stats.ttest_1samp(d, 0.0).pvalue
        welch = stats.ttest_ind(a[m], b[m], equal_var=False).pvalue

        print(
            f"{m:12s} aware {a[m].mean():8.3f} | blind {b[m].mean():8.3f} | "
            f"diff {d.mean():+8.3f} (sd {d.std(ddof=1):.3f}) | "
            f"per seed [{', '.join(f'{x:+.3f}' for x in d)}] | "
            f"matched-by-seed p={paired:.3f} | Welch p={welch:.3f}"
        )


def main():

    print("Reading:")
    df = load_files([os.path.join("results", "physics_results_s*.csv")])

    table, _ = per_seed_table(df)

    distinct = table[(table["level"] > 0) | (table["axis"] == "mass")]

    seeds = sorted(table["train_seed"].unique())

    print(f"\nTraining seeds: {seeds}")
    print(f"Distinct conditions per policy: {distinct[distinct.controller == AWARE][['axis','level']].drop_duplicates().shape[0]} (expected 21)")

    print("\n" + "=" * 100)
    print("MEAN OVER THE 21 DISTINCT CONDITIONS, per training seed")
    print("=" * 100)

    overall = distinct.groupby(["train_seed", "controller"])[METRICS].mean().reset_index()
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nSeed-free references (deterministic):")
    ref = (
        distinct[distinct["controller"].isin(["direct", "cpg", "standstill"])]
        .groupby("controller")[METRICS].mean()
    )
    print(ref.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 100)
    print("AWARE MINUS NO-MORPHOLOGY (positive reward diff = aware better; positive fall diff = aware falls MORE)")
    print("Matched by seed INDEX only: the two arms share no initialization.")
    print("=" * 100)

    compare("ALL 21 distinct conditions", distinct)
    compare("NOMINAL only (level 0)", distinct[distinct["level"] == 0])
    compare("PERTURBED only (levels 10-50%, 20 conditions)", distinct[distinct["level"] > 0])

    for axis in ("mass", "limb_length", "joint_strength", "friction"):
        compare(
            f"PERTURBED, axis = {axis}",
            distinct[(distinct["level"] > 0) & (distinct["axis"] == axis)],
        )

    print("\n" + "=" * 100)
    print("LEAVE-ONE-SEED-OUT: mean aware-minus-blind difference over all 21 conditions")
    print("=" * 100)

    for s in seeds:

        sub = distinct[distinct["train_seed"] != s]
        a, b = per_seed(sub, AWARE), per_seed(sub, BLIND)

        print(
            f"without seed {s}: reward {np.mean(a['task_reward'] - b['task_reward']):+8.2f} | "
            f"fall rate {np.mean(a['fall_rate'] - b['fall_rate']):+.3f}"
        )

    print(
        "\nNOTE: the 'drop from nominal' used in the old ROBUSTNESS table is the "
        "perturbed difference minus the nominal difference. Compare the NOMINAL and "
        "PERTURBED blocks above: if the nominal gap is large, the drop reflects the "
        "gap at nominal, not robustness."
    )


if __name__ == "__main__":
    main()