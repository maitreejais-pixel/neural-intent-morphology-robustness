import argparse
import csv
import os

import numpy as np
from stable_baselines3 import PPO

from src.env import MorphologyAntEnv
from src.controllers import (
    DirectTrajectoryController,
    FixedCPGController,
)

# NOTE on morphology_observation / morphology array handling:
# This file already called MorphologyAntEnv(morphology_observation=...)
# and checked env.morphology_observation, and it already passes
# morphology as a 4-element NumPy array to env.reset(morphology=...).
# Both of these match env.py's corrected parameter name and reset()'s
# array-accepting logic, so no change was needed for that.
#
# FIXED (this revision): two real bugs in the metric collection below.
#
# 1. Both run_classical() and run_ppo() only reported the reward from
#    the LAST step of the episode ("reward": float(reward)), not the
#    accumulated episode total. evaluate_intentions.py already did
#    this correctly (total_reward += float(reward)); this file did
#    not. Fixed by accumulating total_reward across the whole episode.
#
# 2. Neither function terminates the episode when the robot falls
#    (env.py has no fall-based termination), so the "distance",
#    "forward_distance", "yaw_change", and "energy" fields were being
#    read from last_info AFTER the full episode -- meaning, for any
#    episode where a fall occurred, those numbers included however
#    many remaining steps of post-fall tumbling happened afterward,
#    not just the real walking distance/energy. This is the exact
#    same contamination bug found and fixed in test_intentions.py
#    (see that file's history for the original diagnosis). Fixed by
#    snapshotting those four fields at the moment info["fallen"]
#    first becomes True, and using the final-step values only when
#    the episode never fell.
#
# "fallen", "fall_step", "recovery_time", and "steps" are left as
# final-episode values on purpose -- those are meant to describe the
# whole episode (whether/when it fell, whether/when it recovered),
# not a walking-phase snapshot.


# ======================================================================
# MORPHOLOGY
# ======================================================================

def params(axis, level):
    """
    Return morphology parameters:

        [mass, limb_length, joint_strength, friction]

    Nominal:
        [1.0, 1.0, 1.0, 1.0]

    Perturbation:
        mass            -> +level
        limb_length     -> +level
        joint_strength  -> -level
        friction        -> -level
    """

    p = np.ones(
        4,
        dtype=np.float32,
    )

    if axis == "mass":
        p[0] = 1.0 + level

    elif axis == "limb_length":
        p[1] = 1.0 + level

    elif axis == "joint_strength":
        p[2] = 1.0 - level

    elif axis == "friction":
        p[3] = 1.0 - level

    else:
        raise ValueError(
            f"Unknown morphology axis: {axis}"
        )

    return p


# ======================================================================
# EPISODE RUNNER
# ======================================================================

def run_classical(
    name,
    episodes,
    axis,
    level,
    seed0,
    episode_steps,
):
    """
    Evaluate a non-learning controller.

    Controllers:
        direct
        cpg
    """

    rows = []

    for ep in range(episodes):

        seed = seed0 + ep

        env = MorphologyAntEnv(
            morphology_observation=False
        )

        morphology = params(
            axis,
            level,
        )

        obs, _ = env.reset(
            seed=seed,
            morphology=morphology,
            intention=0,
        )

        if name == "direct":
            controller = DirectTrajectoryController()

        elif name == "cpg":
            controller = FixedCPGController()

        else:
            raise ValueError(
                f"Unknown classical controller: {name}"
            )

        controller.reset()

        last_info = {}

        # FIXED: accumulate total episode reward (was only keeping
        # the last step's reward).
        total_reward = 0.0

        # FIXED: snapshot distance/forward_distance/yaw_change/energy
        # at the moment of first fall, so post-fall tumbling doesn't
        # contaminate these metrics.
        fall_snapshot = None

        terminated = False
        truncated = False

        for step in range(
            episode_steps
        ):

            # Frozen intention sequence is deliberately
            # generated deterministically from the episode seed.
            #
            # This means every controller receives the same
            # intention sequence for a given episode.
            intention = int(
                (step // 150) % 4
            )

            env.set_intention(
                intention
            )

            action = controller.act(
                intention,
                step,
                obs,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                last_info,
            ) = env.step(
                action
            )

            total_reward += float(reward)

            if fall_snapshot is None and last_info.get("fallen", False):
                fall_snapshot = dict(last_info)

            if terminated or truncated:
                break

        snapshot = (
            fall_snapshot
            if fall_snapshot is not None
            else last_info
        )

        rows.append(
            {
                "controller": name,
                "axis": axis,
                "level": level,
                "episode": ep,
                "seed": seed,
                "distance": float(
                    snapshot.get(
                        "distance",
                        0.0,
                    )
                ),
                "forward_distance": float(
                    snapshot.get(
                        "forward_distance",
                        0.0,
                    )
                ),
                "yaw_change": float(
                    snapshot.get(
                        "yaw_change",
                        0.0,
                    )
                ),
                "energy": float(
                    snapshot.get(
                        "energy",
                        0.0,
                    )
                ),
                "reward": float(
                    total_reward
                ),
                "steps": int(
                    last_info.get(
                        "steps",
                        step + 1,
                    )
                ),
                "fallen": int(
                    last_info.get(
                        "fallen",
                        0,
                    )
                ),
                "fall_step": last_info.get(
                    "fall_step",
                    None,
                ),
                "recovery_time": last_info.get(
                    "recovery_time",
                    None,
                ),
            }
        )

        env.close()

    return rows


# ======================================================================
# PPO EVALUATION
# ======================================================================

def run_ppo(
    model,
    episodes,
    axis,
    level,
    seed0,
    episode_steps,
):
    """
    Evaluate the morphology-aware PPO controller.

    Important:
        MorphologyAntEnv already constructs the complete PPO
        observation:

            Ant state
            + morphology
            + intention one-hot

    Therefore we DO NOT manually concatenate observation
    components here.
    """

    rows = []

    for ep in range(episodes):

        seed = seed0 + ep

        env = MorphologyAntEnv(
            morphology_observation=True
        )

        morphology = params(
            axis,
            level,
        )

        # Start each episode with Forward intention.
        obs, _ = env.reset(
            seed=seed,
            morphology=morphology,
            intention=0,
        )

        last_info = {}

        # FIXED: accumulate total episode reward (was only keeping
        # the last step's reward).
        total_reward = 0.0

        # FIXED: snapshot distance/forward_distance/yaw_change/energy
        # at the moment of first fall, so post-fall tumbling doesn't
        # contaminate these metrics.
        fall_snapshot = None

        terminated = False
        truncated = False

        for step in range(
            episode_steps
        ):

            # Same deterministic intention schedule used
            # for the classical controllers.
            intention = int(
                (step // 150) % 4
            )

            env.set_intention(
                intention
            )

            # IMPORTANT:
            # env._obs() is not called here.
            #
            # The observation returned by env.step()
            # already contains the current intention.
            #
            # Therefore we need to refresh the observation
            # when intention changes before prediction.

            base_obs = (
                obs[
                    : env.env.observation_space.shape[0]
                ]
            )

            if env.morphology_observation:

                p = env.current_params

                intention_one_hot = np.zeros(
                    4,
                    dtype=np.float32,
                )

                intention_one_hot[
                    intention
                ] = 1.0

                obs_for_policy = np.concatenate(
                    [
                        base_obs,
                        p,
                        intention_one_hot,
                    ]
                ).astype(
                    np.float32
                )

            else:

                intention_one_hot = np.zeros(
                    4,
                    dtype=np.float32,
                )

                intention_one_hot[
                    intention
                ] = 1.0

                obs_for_policy = np.concatenate(
                    [
                        base_obs,
                        intention_one_hot,
                    ]
                ).astype(
                    np.float32
                )

            action, _ = model.predict(
                obs_for_policy,
                deterministic=True,
            )

            (
                obs,
                reward,
                terminated,
                truncated,
                last_info,
            ) = env.step(
                action
            )

            total_reward += float(reward)

            if fall_snapshot is None and last_info.get("fallen", False):
                fall_snapshot = dict(last_info)

            if terminated or truncated:
                break

        snapshot = (
            fall_snapshot
            if fall_snapshot is not None
            else last_info
        )

        rows.append(
            {
                "controller": "morphology-aware",
                "axis": axis,
                "level": level,
                "episode": ep,
                "seed": seed,
                "distance": float(
                    snapshot.get(
                        "distance",
                        0.0,
                    )
                ),
                "forward_distance": float(
                    snapshot.get(
                        "forward_distance",
                        0.0,
                    )
                ),
                "yaw_change": float(
                    snapshot.get(
                        "yaw_change",
                        0.0,
                    )
                ),
                "energy": float(
                    snapshot.get(
                        "energy",
                        0.0,
                    )
                ),
                "reward": float(
                    total_reward
                ),
                "steps": int(
                    last_info.get(
                        "steps",
                        step + 1,
                    )
                ),
                "fallen": int(
                    last_info.get(
                        "fallen",
                        0,
                    )
                ),
                "fall_step": last_info.get(
                    "fall_step",
                    None,
                ),
                "recovery_time": last_info.get(
                    "recovery_time",
                    None,
                ),
            }
        )

        env.close()

    return rows


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(
    rows,
    output_path,
):
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    fieldnames = [
        "controller",
        "axis",
        "level",
        "episode",
        "seed",
        "distance",
        "forward_distance",
        "yaw_change",
        "energy",
        "reward",
        "steps",
        "fallen",
        "fall_step",
        "recovery_time",
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run full-physics MuJoCo morphology "
            "robustness experiments."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of repeats per condition.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum MuJoCo steps per episode.",
    )

    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ],
        help="Morphology perturbation levels.",
    )

    parser.add_argument(
        "--model",
        default="models/morphology_ant_ppo.zip",
        help="Path to trained PPO model.",
    )

    parser.add_argument(
        "--output",
        default="results/physics_results.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load PPO
    # --------------------------------------------------------------

    if not os.path.exists(
        args.model
    ):
        raise FileNotFoundError(
            f"PPO model not found: {args.model}"
        )

    print()
    print("=" * 70)
    print("FULL-PHYSICS MUJOCO MORPHOLOGY EXPERIMENT")
    print("=" * 70)
    print()
    print(
        f"PPO model : {args.model}"
    )
    print(
        f"Episodes  : {args.episodes}"
    )
    print(
        f"Steps     : {args.steps}"
    )
    print(
        f"Levels    : {args.levels}"
    )
    print()

    model = PPO.load(
        args.model
    )

    rows = []

    axes = [
        "mass",
        "limb_length",
        "joint_strength",
        "friction",
    ]

    # --------------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------------

    for axis in axes:

        for level in args.levels:

            print(
                f"\n[{axis}] perturbation = "
                f"{level:.0%}"
            )

            # ------------------------------------------------------
            # Direct baseline
            # ------------------------------------------------------

            print(
                "  Running Direct..."
            )

            rows.extend(
                run_classical(
                    name="direct",
                    episodes=args.episodes,
                    axis=axis,
                    level=level,
                    seed0=1000,
                    episode_steps=args.steps,
                )
            )

            # ------------------------------------------------------
            # Fixed CPG baseline
            # ------------------------------------------------------

            print(
                "  Running CPG..."
            )

            rows.extend(
                run_classical(
                    name="cpg",
                    episodes=args.episodes,
                    axis=axis,
                    level=level,
                    seed0=2000,
                    episode_steps=args.steps,
                )
            )

            # ------------------------------------------------------
            # Morphology-aware PPO
            # ------------------------------------------------------

            print(
                "  Running morphology-aware PPO..."
            )

            rows.extend(
                run_ppo(
                    model=model,
                    episodes=args.episodes,
                    axis=axis,
                    level=level,
                    seed0=3000,
                    episode_steps=args.steps,
                )
            )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_results(
        rows,
        args.output,
    )

    print()
    print("=" * 70)
    print(
        f"Wrote {len(rows)} episode results"
    )
    print(
        f"Output: {args.output}"
    )
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()