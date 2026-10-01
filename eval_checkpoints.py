"""
Evaluate multiple saved PPO checkpoints (not just the final model)
across all four intentions, and report which checkpoint is actually
best-balanced.

Why this is needed: training on Forward/Left/Right/Stop simultaneously
means the policy shares one network across four, sometimes competing,
objectives. Training longer is NOT guaranteed to improve every
intention at once -- a checkpoint at step 5,000,000 can be WORSE on
some intentions than one at step 1,000,000, even though it was trained
for longer overall. train_policy.py's CheckpointCallback already saves
periodic snapshots to models/checkpoints/ -- this script evaluates a
sample of them and reports the best one, instead of trusting the final
save by default.

Run from the project root:
    python eval_checkpoints.py
"""

import glob
import os
import re

import numpy as np
from stable_baselines3 import PPO

from src.env import MorphologyAntEnv


CHECKPOINT_DIR = "models/checkpoints"
CHECKPOINT_GLOB = "morphology_ant_ppo_*_steps.zip"

# How many checkpoints to sample across the training run (evenly
# spaced by step count). Evaluating every single checkpoint would be
# slow; this gives a representative spread. Increase if you want finer
# resolution once you've narrowed down a promising region.
N_SAMPLES = 10

INTENTIONS = {
    0: "Forward",
    1: "Left",
    2: "Right",
    3: "Stop",
}

TARGETS = {
    0: {"forward_velocity": 1.0, "yaw_velocity": 0.0},
    1: {"forward_velocity": 0.0, "yaw_velocity": 0.8},
    2: {"forward_velocity": 0.0, "yaw_velocity": -0.8},
    3: {"forward_velocity": 0.0, "yaw_velocity": 0.0},
}

MAX_STEPS = 1000


def find_checkpoints():

    paths = glob.glob(
        os.path.join(CHECKPOINT_DIR, CHECKPOINT_GLOB)
    )

    def step_count(path):
        match = re.search(r"_(\d+)_steps\.zip$", path)
        return int(match.group(1)) if match else -1

    paths = [p for p in paths if step_count(p) >= 0]
    paths.sort(key=step_count)

    if len(paths) == 0:
        raise FileNotFoundError(
            f"No checkpoints found matching "
            f"{os.path.join(CHECKPOINT_DIR, CHECKPOINT_GLOB)}. "
            "Check that train_policy.py's CheckpointCallback ran and "
            "saved files there."
        )

    if len(paths) <= N_SAMPLES:
        return paths

    # Evenly sample N_SAMPLES checkpoints across the full range,
    # always including the very first and very last.
    idx = np.linspace(0, len(paths) - 1, N_SAMPLES).astype(int)
    idx = sorted(set(idx.tolist()))
    return [paths[i] for i in idx]


def run_episode(model, env, intention, seed):

    obs, _ = env.reset(
        seed=seed,
        morphology=np.ones(4, dtype=np.float32),
        intention=intention,
    )

    fall_snapshot = None
    info = {}

    for step in range(MAX_STEPS):

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if fall_snapshot is None and info.get("fallen", False):
            fall_snapshot = dict(info)

        if terminated or truncated:
            break

    snapshot = fall_snapshot if fall_snapshot is not None else info

    return {
        "forward_velocity_proxy": float(
            snapshot.get("forward_distance", 0.0)
        )
        / (MAX_STEPS * 0.05),
        "yaw_change": float(snapshot.get("yaw_change", 0.0)),
        "energy": float(snapshot.get("energy", 0.0)),
        "fallen": bool(info.get("fallen", False)),
    }


def score_checkpoint(model, env, seeds=(100, 101, 102)):
    """
    Lower is better. Penalizes:
      - Forward: not moving forward, any yaw drift, falls
      - Left/Right: yaw magnitude far from +-0.8, falling
      - Stop: any movement or yaw drift at all
      - all: excessive energy (spinning/thrashing)
    """

    total_error = 0.0
    details = {}

    for intention, name in INTENTIONS.items():

        results = [
            run_episode(model, env, intention, seed)
            for seed in seeds
        ]

        avg_yaw = np.mean([r["yaw_change"] for r in results])
        avg_energy = np.mean([r["energy"] for r in results])
        any_fallen = any(r["fallen"] for r in results)

        target_yaw = TARGETS[intention]["yaw_velocity"]

        if intention == 0:
            avg_fwd = np.mean(
                [r["forward_velocity_proxy"] for r in results]
            )
            error = (
                abs(1.0 - avg_fwd) * 2.0
                + abs(avg_yaw) * 1.0
                + (5.0 if any_fallen else 0.0)
            )
        elif intention == 3:
            error = (
                abs(avg_yaw) * 2.0
                + (5.0 if any_fallen else 0.0)
            )
        else:
            error = (
                abs(target_yaw - avg_yaw) * 1.0
                + (5.0 if any_fallen else 0.0)
            )

        error += avg_energy / 5000.0  # mild energy penalty, all cases

        total_error += error
        details[name] = {
            "avg_yaw": avg_yaw,
            "avg_energy": avg_energy,
            "fallen": any_fallen,
        }

    return total_error, details


def main():

    checkpoints = find_checkpoints()
    print(f"Evaluating {len(checkpoints)} checkpoints...\n")

    env = MorphologyAntEnv(morphology_observation=True)

    results = []

    try:
        for path in checkpoints:

            model = PPO.load(path)
            score, details = score_checkpoint(model, env)
            results.append((path, score, details))

            print(f"{os.path.basename(path)} | total_error={score:.3f}")
            for name, d in details.items():
                print(
                    f"    {name:8s} | "
                    f"avg_yaw={d['avg_yaw']:+.3f} | "
                    f"avg_energy={d['avg_energy']:8.1f} | "
                    f"fallen={d['fallen']}"
                )
            print()

    finally:
        env.close()

    results.sort(key=lambda r: r[1])

    print("=" * 90)
    print("BEST CHECKPOINTS (lowest total_error first)")
    print("=" * 90)
    for path, score, _ in results[:5]:
        print(f"{os.path.basename(path):45s} | total_error={score:.3f}")

    print()
    print(f"Best checkpoint: {results[0][0]}")
    print(
        "To use it going forward, copy it over the main model, e.g.:"
    )
    print(
        f'    Copy-Item "{results[0][0]}" "models\\morphology_ant_ppo.zip" -Force'
    )


if __name__ == "__main__":
    main()