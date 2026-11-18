"""
Re-evaluate the final models at NOMINAL morphology on FRESH start states.

Why: in the main experiment the morphology-aware policy scored lower and
fell more often than the no-morphology policy at nominal (level 0), where
only 10 start states per model (seeds 1000-1009) were used. This script
checks whether that gap reproduces on new start states (default seeds
2000-2019, 20 episodes per model, full 1000-step intention schedule).

Run from the project root:

    python scripts/nominal_check.py
    python scripts/nominal_check.py --seeds 42 43 44 45 46 --episodes 20 --seed0 2000

Writes results/nominal_check.csv (one row per training seed and arm) and
prints the arm comparison. It does not touch physics_results_s*.csv.

How to read it (decide BEFORE looking at the numbers):
  * If aware again scores lower and falls more at nominal, report the
    nominal gap as reproduced on fresh start states.
  * If the gap disappears, it was specific to the original 10 start
    states: drop the nominal-gap sentence and report the all-conditions
    comparison only.
"""

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

import numpy as np
import pandas as pd
from scipy import stats

from src.env import MorphologyAntEnv
from src.protocol import run_episode


def evaluate(model_path, morphology_observation, episodes, seed0, steps):
    """Mean task reward and fall rate of one model at nominal morphology."""

    from stable_baselines3 import PPO

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(model_path)

    def policy(obs, intention, step):

        action, _ = model.predict(obs, deterministic=True)

        return action

    env = MorphologyAntEnv(morphology_observation=morphology_observation)

    results = []

    try:
        for ep in range(episodes):

            results.append(
                run_episode(
                    env,
                    policy,
                    seed=seed0 + ep,
                    morphology=np.ones(4, dtype=np.float32),
                    steps=steps,
                )
            )
    finally:
        env.close()

    return {
        "task_reward": float(np.mean([r["task_reward"] for r in results])),
        "fall_rate": float(np.mean([r["fallen"] for r in results])),
        "episodes": len(results),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Nominal-morphology re-check on fresh start states."
    )

    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default=os.path.join("results", "nominal_check.csv"))

    args = parser.parse_args()

    rows = []

    for seed in args.seeds:

        for arm, morph in (("aware", True), ("nomorph", False)):

            path = os.path.join(args.models_dir, f"final_{arm}_s{seed}.zip")

            print(f"seed {seed}, {arm}: {path}", flush=True)

            score = evaluate(path, morph, args.episodes, args.seed0, args.steps)

            rows.append({"train_seed": seed, "arm": arm, **score})

    table = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    table.to_csv(args.output, index=False)

    wide = table.pivot(index="train_seed", columns="arm")

    pd.set_option("display.width", 200)

    print("\n" + "=" * 80)
    print(f"NOMINAL MORPHOLOGY, fresh start states ({args.episodes} episodes per model, "
          f"seeds {args.seed0}..{args.seed0 + args.episodes - 1})")
    print("=" * 80)
    print(wide[["task_reward", "fall_rate"]].round(3).to_string())

    print("\nAware minus no-morphology (positive reward = aware better; positive fall = aware falls MORE):")

    for metric in ("task_reward", "fall_rate"):

        a = wide[(metric, "aware")].to_numpy()
        b = wide[(metric, "nomorph")].to_numpy()

        d = a - b

        n = len(d)

        line = (
            f"  {metric:12s} aware {a.mean():8.3f} | blind {b.mean():8.3f} | "
            f"diff {d.mean():+8.3f} | per seed [{', '.join(f'{x:+.3f}' for x in d)}]"
        )

        if n >= 3 and d.std(ddof=1) > 0:

            welch = stats.ttest_ind(a, b, equal_var=False)

            ci = welch.confidence_interval(confidence_level=0.95)

            line += (
                f" | Welch 95% CI [{ci.low:+.3f}, {ci.high:+.3f}]"
                f" | Welch p={welch.pvalue:.3f}"
            )

        print(line)

    print(
        "\nEpisodes within a model differ only through reset noise, so there are "
        f"effectively {len(args.seeds)} independent samples per arm. Treat p-values as descriptive."
    )

    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()