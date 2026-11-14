"""
Forward-only evaluation: does a policy actually WALK forward?

Use it for the sanity test (train_policy.py --forward-only) and to check
any model's forward behaviour. The intention is fixed to Forward for the
whole episode; nothing else is scored.

    python scripts/eval_forward.py --model models/fwd_test.zip
    python scripts/eval_forward.py --model models/ppo_seed42.zip --episodes 20
    python scripts/eval_forward.py --model models/ppo_nomorph_seed42.zip --no-morph

Reported (means over episodes):
    forward speed       mean forward velocity in m/s  (target: 1.0;
                        a robot that stands still is about 0.0)
    forward distance    metres travelled along the starting heading
    task reward / step  tracking reward per step; standing still is
                        about 0.0 on Forward, perfect tracking is 1.0
    reward breakdown    the task reward split into its three terms:
                        forward tracking (positive), lateral-drift
                        penalty and yaw-rate penalty (both negative),
                        plus the mean |yaw rate| and |lateral speed|
                        behind them
    fall cause          'tilt' (torso tipped over) or 'height'
    falls               fraction of episodes that ended in a fall
    yaw drift           mean |final yaw change| in radians (staying
                        straight is about 0.0)

Rule of thumb: forward speed above about 0.3 m/s with no falls means the
policy has learned to walk; about 0.0 means it is standing still.
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

from src.env import MorphologyAntEnv
from src.protocol import morphology_params


def evaluate(
    policy,
    env,
    episodes,
    steps,
    seed0,
    morphology,
):
    """Run Forward-only episodes; return a list of per-episode dicts."""

    results = []

    for ep in range(episodes):

        env.reset(
            seed=seed0 + ep,
            morphology=morphology,
            intention=0,
        )

        speeds = []
        task = 0.0
        info = {}
        fall_snapshot = None

        tracking_terms = []
        lateral_terms = []
        yaw_terms = []
        abs_yaw = []
        abs_lateral = []

        for step in range(steps):

            obs = env.get_current_observation()

            action = policy(obs)

            _, _, terminated, truncated, info = env.step(action)

            speeds.append(info["forward_velocity"])
            task += info["task_reward"]

            # The three terms of the Forward task reward, exactly as
            # the environment computed them (including any smoothing
            # or weight overrides).
            tracking_terms.append(info["forward_tracking_term"])
            lateral_terms.append(info["forward_lateral_term"])
            yaw_terms.append(info["forward_yaw_term"])

            abs_yaw.append(abs(info["yaw_velocity"]))
            abs_lateral.append(abs(info["lateral_velocity"]))

            if fall_snapshot is None and info["fallen"]:
                fall_snapshot = dict(info)

            if terminated or truncated:
                break

        snapshot = (
            fall_snapshot
            if fall_snapshot is not None
            else info
        )

        n = len(speeds)

        fallen = int(bool(info["fallen"]))

        results.append(
            {
                "speed": float(np.mean(speeds)),
                "distance": float(snapshot["forward_distance"]),
                "task_per_step": task / n,
                "fallen": fallen,
                "yaw_drift": abs(float(snapshot["yaw_change"])),
                "steps": n,
                "tracking": float(np.mean(tracking_terms)),
                "lateral_penalty": float(np.mean(lateral_terms)),
                "yaw_penalty": float(np.mean(yaw_terms)),
                "abs_yaw_rate": float(np.mean(abs_yaw)),
                "abs_lateral_speed": float(np.mean(abs_lateral)),
                "fall_tilt": int(
                    fallen
                    and info["uprightness"] < env.FALL_UPRIGHTNESS
                ),
            }
        )

    return results


def summarize(results):

    return {
        key: float(np.mean([r[key] for r in results]))
        for key in (
            "speed",
            "distance",
            "task_per_step",
            "fallen",
            "yaw_drift",
            "steps",
            "tracking",
            "lateral_penalty",
            "yaw_penalty",
            "abs_yaw_rate",
            "abs_lateral_speed",
            "fall_tilt",
        )
    }


def main():

    parser = argparse.ArgumentParser(
        description="Forward-only evaluation of a trained PPO model."
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--no-morph", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument(
        "--axis",
        default=None,
        choices=["mass", "limb_length", "joint_strength", "friction"],
        help="Optional single perturbation axis.",
    )
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument(
        "--forward-yaw-weight", type=float, default=None,
        help="Same value as used in training (default 0.30).",
    )
    parser.add_argument(
        "--lateral-weight", type=float, default=None,
        help="Same value as used in training (default 0.20).",
    )
    parser.add_argument(
        "--penalty-tau", type=float, default=None,
        help="Same value as used in training (default none).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    from stable_baselines3 import PPO

    model = PPO.load(args.model)

    def policy(obs):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        return action

    morphology = (
        morphology_params(args.axis, args.level)
        if args.axis
        else np.ones(4, dtype=np.float32)
    )

    env_kwargs = {}

    if args.forward_yaw_weight is not None:
        env_kwargs["forward_yaw_weight"] = args.forward_yaw_weight

    if args.lateral_weight is not None:
        env_kwargs["forward_lateral_weight"] = args.lateral_weight

    if args.penalty_tau is not None:
        env_kwargs["penalty_tau"] = args.penalty_tau

    env = MorphologyAntEnv(
        morphology_observation=not args.no_morph,
        **env_kwargs,
    )

    try:
        results = evaluate(
            policy,
            env,
            args.episodes,
            args.steps,
            args.seed0,
            morphology,
        )
    finally:
        env.close()

    summary = summarize(results)

    print("=" * 70)
    print(f"FORWARD-ONLY EVALUATION: {args.model}")
    print("=" * 70)
    print(f"Episodes x steps    : {args.episodes} x {args.steps}")
    print(
        "Morphology          : "
        + (
            f"{args.axis} +/-{args.level:.0%}"
            if args.axis
            else "nominal"
        )
    )
    print(f"Forward speed       : {summary['speed']:+.3f} m/s  (target 1.0, standing ~ 0.0)")
    print(f"Forward distance    : {summary['distance']:+.2f} m")
    print(f"Task reward / step  : {summary['task_per_step']:+.3f}  (standing ~ 0.0, perfect 1.0)")
    print(f"Falls               : {summary['fallen']:.0%}")
    print(f"Yaw drift           : {summary['yaw_drift']:.2f} rad")
    print(f"Mean episode length : {summary['steps']:.0f} steps")
    print("-" * 70)
    print("Task reward per step, split into its terms:")
    print(f"  forward tracking  : {summary['tracking']:+.3f}")
    print(f"  lateral penalty   : {summary['lateral_penalty']:+.3f}   (mean |lateral speed| {summary['abs_lateral_speed']:.2f} m/s)")
    print(f"  yaw-rate penalty  : {summary['yaw_penalty']:+.3f}   (mean |yaw rate| {summary['abs_yaw_rate']:.2f} rad/s)")
    print(f"  total             : {summary['tracking'] + summary['lateral_penalty'] + summary['yaw_penalty']:+.3f}")
    if summary["fallen"] > 0:
        print(f"Falls by tilt       : {summary['fall_tilt'] / summary['fallen']:.0%} of the falls (the rest: torso too low)")
    print("=" * 70)

    if summary["speed"] > 0.3 and summary["fallen"] < 0.2:
        print("Reading: the policy walks forward.")
    elif summary["fallen"] >= 0.2:
        print("Reading: the policy falls in at least 20% of episodes; "
              "look at the speed and yaw drift before the fall.")
    elif abs(summary["speed"]) < 0.1:
        print("Reading: the policy is standing still (no forward speed).")
    else:
        print("Reading: partial forward motion without falling; "
              "compare the speed with the 1.0 m/s target.")


if __name__ == "__main__":
    main()