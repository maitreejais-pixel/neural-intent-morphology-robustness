"""
Evaluate multiple saved PPO checkpoints (not just the final model)
under the fixed intention schedule, and report which checkpoint is
actually best.

Why this is needed: training on Forward/Left/Right/Stop simultaneously
means the policy shares one network across four, sometimes competing,
objectives. Training longer is NOT guaranteed to improve every
intention at once -- a checkpoint at step 5,000,000 can be WORSE on
some intentions than one at step 1,000,000, even though it was trained
for longer overall. train_policy.py's CheckpointCallback saves periodic
snapshots to models/checkpoints/<name>/ -- this script evaluates a
sample of them and reports the best one, instead of trusting the final
save by default.

SELECTION CRITERION (predeclared; this is what the paper should state)
----------------------------------------------------------------------
Mean TASK reward (tracking reward only: no uprightness bonus, no
control cost) over the fixed Forward -> Left -> Right -> Stop schedule
(150 steps each, repeating), averaged over:

    --episodes episodes at NOMINAL morphology, and
    --episodes episodes at RANDOM morphologies drawn from the training
    distribution.

Selection seeds (100+ and 500+) are disjoint from the evaluation seeds
(1000+ in run_experiments.py), and every checkpoint sees exactly the
same episodes.

OPTIONAL FALL CONSTRAINT (--max-fall-rate, e.g. 0.1): the task reward
does not include the fall penalty, so a checkpoint that falls often can
still rank high. With --max-fall-rate, the best checkpoint is the one
with the highest task reward AMONG those whose fall rate on the
selection episodes is at most that value; if none qualifies, the
checkpoint with the lowest fall rate is taken (ties by task reward) and
a warning is printed. State the value in the paper if you use it.

A policy that does not move scores only about 150 on this criterion
(the Stop segments), so an inactive checkpoint cannot win. The previous
version of this script ranked checkpoints with a heuristic that could
not tell standing from locomotion and compared a total yaw ANGLE
(radians, wrapped to [-pi, pi]) with a yaw RATE target (rad/s).

Run from the project root:

    python eval_checkpoints.py --name ppo_seed42
    python eval_checkpoints.py --name ppo_nomorph_seed42 --no-morph
"""

import argparse
import glob
import os
import re
import shutil
import sys

# Make "from src..." work from any working directory.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.abspath(__file__)
    ),
)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.env import MorphologyAntEnv
from src.protocol import run_episode, sample_morphology


# How many checkpoints to sample across the training run (evenly
# spaced by step count). Evaluating every single checkpoint would be
# slow; this gives a representative spread. Increase if you want finer
# resolution once you've narrowed down a promising region.
DEFAULT_N_SAMPLES = 10

# Selection seeds (disjoint from the evaluation seeds 1000+).
NOMINAL_SEED0 = 100
RANDOM_SEED0 = 500

# Fixed seed for the random-morphology draw, so every checkpoint (and
# every re-run of this script) is scored on the same morphologies.
MORPHOLOGY_RNG_SEED = 12345


def checkpoint_step_count(path):
    """Training timesteps encoded in a checkpoint file name."""

    match = re.search(r"_(\d+)_steps\.zip$", path)

    return int(match.group(1)) if match else -1


def find_checkpoints(
    checkpoint_dir,
    prefix,
    n_samples,
):

    paths = glob.glob(
        os.path.join(
            checkpoint_dir,
            f"{prefix}_*_steps.zip",
        )
    )

    paths = [
        p
        for p in paths
        if checkpoint_step_count(p) >= 0
    ]

    paths.sort(key=checkpoint_step_count)

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No checkpoints found matching "
            f"{os.path.join(checkpoint_dir, prefix + '_*_steps.zip')}. "
            "Check that train_policy.py's CheckpointCallback ran and "
            "saved files there (and that --name matches the run name)."
        )

    if len(paths) <= n_samples:
        return paths

    # Evenly sample n_samples checkpoints across the full range,
    # always including the very first and very last.
    idx = np.linspace(
        0,
        len(paths) - 1,
        n_samples,
    ).astype(int)

    idx = sorted(set(idx.tolist()))

    return [paths[i] for i in idx]


def build_conditions(episodes):
    """
    (seed, morphology) pairs used to score every checkpoint:
    `episodes` nominal-morphology episodes, then `episodes` episodes
    at random training-distribution morphologies.
    """

    rng = np.random.default_rng(
        MORPHOLOGY_RNG_SEED
    )

    conditions = [
        (
            NOMINAL_SEED0 + i,
            np.ones(4, dtype=np.float32),
        )
        for i in range(episodes)
    ]

    conditions += [
        (
            RANDOM_SEED0 + i,
            sample_morphology(rng),
        )
        for i in range(episodes)
    ]

    return conditions


def score_checkpoint(
    model,
    env,
    conditions,
    steps,
):
    """
    Higher task_reward is better (this is the selection criterion).

    Also reports, for context:
      - fall_rate     fraction of episodes that ended in a fall
      - fwd_vel_err   mean |forward speed - 1.0 m/s| in Forward segments
      - yaw_rate_err  mean |yaw rate - target| in Left/Right segments
    """

    def policy(obs, intention, step):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        return action

    results = [
        run_episode(
            env,
            policy,
            seed,
            morphology,
            steps,
        )
        for seed, morphology in conditions
    ]

    return {
        "task_reward": float(
            np.mean(
                [r["task_reward"] for r in results]
            )
        ),
        "reward": float(
            np.mean(
                [r["reward"] for r in results]
            )
        ),
        "fall_rate": float(
            np.mean(
                [r["fallen"] for r in results]
            )
        ),
        "fwd_vel_err": float(
            np.nanmean(
                [r["fwd_vel_err"] for r in results]
            )
        ),
        "yaw_rate_err": float(
            np.nanmean(
                [r["yaw_rate_err"] for r in results]
            )
        ),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Select a PPO checkpoint with a predeclared "
        "criterion (mean task reward under the intention schedule)."
    )

    parser.add_argument(
        "--name",
        default="morphology_ant_ppo",
        help="Run name used in train_policy.py (--name).",
    )

    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="Checkpoint folder. Default: models/checkpoints/<name>.",
    )

    parser.add_argument(
        "--no-morph",
        action="store_true",
        help="Use for the no-morphology ablation (109-dim "
        "observation).",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="How many checkpoints to sample across the run.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes at nominal morphology, and the same number "
        "at random morphologies.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum steps per episode.",
    )

    parser.add_argument(
        "--max-fall-rate",
        type=float,
        default=None,
        help="Only consider checkpoints whose fall rate on the "
        "selection episodes is at most this value (e.g. 0.1).",
    )

    parser.add_argument(
        "--copy-best-to",
        default=None,
        help="Copy the selected checkpoint to this path (e.g. "
        "models/final_aware_s42.zip).",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="CSV with the ranking. Default: "
        "results/checkpoint_selection_<name>.csv.",
    )

    args = parser.parse_args()

    checkpoint_dir = args.ckpt_dir or os.path.join(
        "models",
        "checkpoints",
        args.name,
    )

    output_csv = args.out or os.path.join(
        "results",
        f"checkpoint_selection_{args.name}.csv",
    )

    checkpoints = find_checkpoints(
        checkpoint_dir,
        args.name,
        args.n_samples,
    )

    print(
        f"Evaluating {len(checkpoints)} checkpoints from "
        f"{checkpoint_dir}"
    )
    print(
        "Training steps: "
        f"{[checkpoint_step_count(p) for p in checkpoints]}\n"
    )

    conditions = build_conditions(
        args.episodes
    )

    env = MorphologyAntEnv(
        morphology_observation=not args.no_morph
    )

    rows = []

    try:
        for path in checkpoints:

            model = PPO.load(path)

            scores = score_checkpoint(
                model,
                env,
                conditions,
                args.steps,
            )

            row = {
                "checkpoint": os.path.basename(path),
                "timesteps": checkpoint_step_count(path),
                **scores,
            }

            rows.append(row)

            print(
                f"{row['checkpoint']:45s} | "
                f"task_reward={row['task_reward']:8.2f} | "
                f"fall_rate={row['fall_rate']:.2f} | "
                f"fwd_vel_err={row['fwd_vel_err']:.3f} | "
                f"yaw_rate_err={row['yaw_rate_err']:.3f}"
            )

    finally:
        env.close()

    ranking = pd.DataFrame(rows).sort_values(
        "task_reward",
        ascending=False,
    )

    os.makedirs(
        os.path.dirname(output_csv) or ".",
        exist_ok=True,
    )

    ranking.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("=" * 90)
    print("BEST CHECKPOINTS (highest task_reward first)")
    print("=" * 90)

    for _, r in ranking.head(5).iterrows():
        print(
            f"{r['checkpoint']:45s} | "
            f"task_reward={r['task_reward']:8.2f} | "
            f"fall_rate={r['fall_rate']:.2f}"
        )

    if args.max_fall_rate is None:

        best = ranking.iloc[0]

    else:

        eligible = ranking[
            ranking["fall_rate"] <= args.max_fall_rate
        ]

        if len(eligible) > 0:

            best = eligible.iloc[0]

            print(
                f"Fall constraint: {len(eligible)} of "
                f"{len(ranking)} checkpoints have a fall rate <= "
                f"{args.max_fall_rate:.2f}."
            )

        else:

            best = ranking.sort_values(
                ["fall_rate", "task_reward"],
                ascending=[True, False],
            ).iloc[0]

            print(
                "WARNING: no checkpoint satisfies the fall "
                f"constraint (<= {args.max_fall_rate:.2f}); taking "
                "the one with the lowest fall rate."
            )

    best_path = os.path.join(
        checkpoint_dir,
        best["checkpoint"],
    )

    print()
    print(
        f"Best checkpoint: {best_path}  "
        f"(task reward {best['task_reward']:.1f}, "
        f"fall rate {best['fall_rate']:.2f})"
    )
    print(f"Saved the full ranking to {output_csv}")

    if args.copy_best_to:

        os.makedirs(
            os.path.dirname(args.copy_best_to) or ".",
            exist_ok=True,
        )

        shutil.copyfile(
            best_path,
            args.copy_best_to,
        )

        print(f"Copied the selected checkpoint to {args.copy_best_to}")
    print(
        "To use it going forward, copy it over the main model, e.g.:"
    )
    print(
        f'    Copy-Item "{best_path}" '
        f'"models\\{args.name}.zip" -Force'
    )


if __name__ == "__main__":
    main()