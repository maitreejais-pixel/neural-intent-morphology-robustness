"""
Per-intention evaluation: which commands has a policy actually learned?

Each intention is evaluated on its own, held fixed for the whole episode
(no switching), so the result says what the policy does for Forward,
Left, Right and Stop separately. (eval_checkpoints.py and
run_experiments.py score the mixed schedule, which hides this.)

    python scripts/evaluate_intentions.py --model models/multi_test.zip
    python scripts/evaluate_intentions.py --model models/ppo_nomorph_seed42.zip --no-morph
    python scripts/evaluate_intentions.py --model models/ppo_seed42.zip --axis limb_length --level 0.5

What counts as "learned": the response is compared with its target.
    >= 50% of the target and falls in < 20% of episodes   -> learned
    20% to 50% of the target (and stable)                 -> partial
    below 20%                                             -> no response
(Stop: residual speed below 0.1 counts as learned.) Reference values;
standing still is ~0 for the first three rows, ~0 residual speed for Stop:

    Forward   mean forward speed        target +1.0 m/s
    Left      mean yaw rate             target +0.8 rad/s
    Right     mean yaw rate             target -0.8 rad/s
    Stop      residual speed            target 0 (linear m/s + |yaw| rad/s)

Also reported per intention: falls, mean task reward per step (standing
still scores ~0 on Forward/Left/Right and ~1.0 on Stop), and the mean
|forward speed| during turns (a turn in place has ~0).
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

NAMES = {0: "Forward", 1: "Left", 2: "Right", 3: "Stop"}


def evaluate_intention(
    policy,
    env,
    intention,
    episodes,
    steps,
    seed0,
    morphology,
):
    """Mean statistics over episodes with a fixed intention."""

    rows = []

    for ep in range(episodes):

        env.reset(
            seed=seed0 + ep,
            morphology=morphology,
            intention=intention,
        )

        fwd, lat, yaw, task = [], [], [], []
        info = {}

        for step in range(steps):

            obs = env.get_current_observation()

            _, _, terminated, truncated, info = env.step(
                policy(obs)
            )

            fwd.append(info["forward_velocity"])
            lat.append(info["lateral_velocity"])
            yaw.append(info["yaw_velocity"])
            task.append(info["task_reward"])

            if terminated or truncated:
                break

        rows.append(
            {
                "fwd": float(np.mean(fwd)),
                "abs_fwd": float(np.mean(np.abs(fwd))),
                "yaw": float(np.mean(yaw)),
                "residual": float(
                    np.mean(
                        np.hypot(fwd, lat) + np.abs(yaw)
                    )
                ),
                "task": float(np.mean(task)),
                "fallen": int(bool(info["fallen"])),
                "steps": len(fwd),
            }
        )

    return {
        key: float(np.mean([r[key] for r in rows]))
        for key in rows[0]
    }


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate each intention separately."
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--no-morph", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument(
        "--axis",
        default=None,
        choices=["mass", "limb_length", "joint_strength", "friction"],
    )
    parser.add_argument("--level", type=float, default=0.0)

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

    env = MorphologyAntEnv(
        morphology_observation=not args.no_morph
    )

    results = {}

    try:
        for intention in NAMES:
            results[intention] = evaluate_intention(
                policy,
                env,
                intention,
                args.episodes,
                args.steps,
                args.seed0,
                morphology,
            )
    finally:
        env.close()

    print("=" * 78)
    print(f"PER-INTENTION EVALUATION: {args.model}")
    print("=" * 78)
    print(
        f"{args.episodes} episodes x {args.steps} steps per intention, "
        "morphology: "
        + (
            f"{args.axis} {args.level:+.0%}"
            if args.axis
            else "nominal"
        )
    )
    print("-" * 78)
    print(
        f"{'Intention':9s} | {'measured':>28s} | {'target':>8s} | "
        f"{'task/step':>9s} | {'falls':>5s}"
    )
    print("-" * 78)

    f, l, r, s = (results[i] for i in range(4))

    print(f"{'Forward':9s} | forward speed {f['fwd']:+8.3f} m/s   | {'+1.0':>8s} | {f['task']:+9.3f} | {f['fallen']:5.0%}")
    print(f"{'Left':9s} | yaw rate      {l['yaw']:+8.3f} rad/s | {'+0.8':>8s} | {l['task']:+9.3f} | {l['fallen']:5.0%}")
    print(f"{'Right':9s} | yaw rate      {r['yaw']:+8.3f} rad/s | {'-0.8':>8s} | {r['task']:+9.3f} | {r['fallen']:5.0%}")
    print(f"{'Stop':9s} | residual speed {s['residual']:7.3f}       | {'0':>8s} | {s['task']:+9.3f} | {s['fallen']:5.0%}")
    print("-" * 78)
    print(
        f"Mean |forward speed| while turning: Left {l['abs_fwd']:.2f}, "
        f"Right {r['abs_fwd']:.2f} m/s"
    )

    def status(fraction, fallen):
        """fraction = response / target (negative = wrong direction)."""

        if fallen >= 0.2:
            return "unstable (falls)"

        if fraction >= 0.5:
            return "learned"

        if fraction >= 0.2:
            return "partial"

        return "no response"

    rows = [
        ("Forward", f["fwd"] / 1.0, f["fallen"]),
        ("Left", l["yaw"] / 0.8, l["fallen"]),
        ("Right", r["yaw"] / -0.8, r["fallen"]),
    ]

    print("Response as a fraction of the target:")

    for name, fraction, fallen in rows:
        print(f"  {name:8s} {fraction:+6.0%}  -> {status(fraction, fallen)}")

    stop_ok = s["residual"] < 0.1 and s["fallen"] < 0.2

    print(
        f"  {'Stop':8s} residual {s['residual']:.3f}  -> "
        + ("learned" if stop_ok else "not still")
    )
    print("=" * 78)


if __name__ == "__main__":
    main()