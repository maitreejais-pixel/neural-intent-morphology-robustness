import numpy as np
from stable_baselines3 import PPO

from src.env import MorphologyAntEnv


MODEL_PATH = "models/morphology_ant_ppo.zip"

INTENTIONS = {
    0: "Forward",
    1: "Left",
    2: "Right",
    3: "Stop",
}

# These are the intended behavioral targets used by the environment.
#
# FIXED (earlier revision): yaw_velocity for Left/Right were
# previously swapped (Left was -0.8, Right was +0.8). Under this
# project's frame convention (x=forward, y=left, z=up) and
# yaw = atan2(rotation[3], rotation[0]), a POSITIVE yaw velocity is a
# LEFT turn. These values match env.py's corrected LEFT_YAW_TARGET /
# RIGHT_YAW_TARGET.
TARGETS = {
    0: {"forward_velocity": 1.0, "yaw_velocity": 0.0},
    1: {"forward_velocity": 0.0, "yaw_velocity": 0.8},
    2: {"forward_velocity": 0.0, "yaw_velocity": -0.8},
    3: {"forward_velocity": 0.0, "yaw_velocity": 0.0},
}


def run_episode(
    model,
    env,
    intention,
    seed=100,
    max_steps=1000,
):
    """
    Evaluate one intention under nominal morphology.

    Morphology:
        [mass, limb_length, joint_strength, friction]
        = [1, 1, 1, 1]
    """

    obs, _ = env.reset(
        seed=seed,
        morphology=np.ones(
            4,
            dtype=np.float32,
        ),
        intention=intention,
    )

    total_reward = 0.0
    steps = 0
    info = {}

    # FIXED: snapshot distance/forward_distance/yaw_change/energy at
    # the moment of first fall, instead of reading them only from the
    # final info after the full episode. env.py does not terminate
    # episodes on a fall, so a fallen robot keeps being simulated
    # (and the controller keeps issuing commands into it) for the
    # rest of the episode -- reading final-step info means these
    # fields would include however much post-fall tumbling happened
    # afterward, not just the real walking behavior. This is the same
    # contamination bug originally found and fixed in
    # test_intentions.py.
    fall_snapshot = None

    for step in range(max_steps):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += float(reward)
        steps = step + 1

        if fall_snapshot is None and info.get("fallen", False):
            fall_snapshot = dict(info)

        if terminated or truncated:
            break

    snapshot = (
        fall_snapshot
        if fall_snapshot is not None
        else info
    )

    recovery = info.get(
        "recovery_time",
        None,
    )

    return {
        "distance": float(
            snapshot.get("distance", 0.0)
        ),
        "forward_distance": float(
            snapshot.get("forward_distance", 0.0)
        ),
        "yaw_change": float(
            snapshot.get("yaw_change", 0.0)
        ),
        "energy": float(
            snapshot.get("energy", 0.0)
        ),
        "reward": float(total_reward),
        "steps": int(steps),
        "fallen": bool(
            info.get("fallen", False)
        ),
        "fall_step": info.get(
            "fall_step",
            None,
        ),
        "recovery_time": recovery,
    }


def main():

    print(
        "Loading trained PPO model..."
    )

    model = PPO.load(
        MODEL_PATH
    )

    env = MorphologyAntEnv(
        morphology_observation=True
    )

    print()
    print(
        "Intention behavior test"
    )
    print("=" * 110)

    print(
        "Nominal morphology: "
        "mass=1.00, limb_length=1.00, "
        "joint_strength=1.00, friction=1.00"
    )

    print("=" * 110)

    try:

        for intention, name in INTENTIONS.items():

            result = run_episode(
                model,
                env,
                intention,
                seed=100 + intention,
            )

            recovery = result[
                "recovery_time"
            ]

            if recovery is None:
                recovery_text = "N/A"
            else:
                recovery_text = (
                    f"{recovery:.3f}s"
                )

            target = TARGETS[intention]

            print(
                f"{name:8s} | "
                f"target_v="
                f"{target['forward_velocity']:+.2f} | "
                f"target_yaw="
                f"{target['yaw_velocity']:+.2f} | "
                f"distance="
                f"{result['distance']:+.3f} | "
                f"forward="
                f"{result['forward_distance']:+.3f} | "
                f"yaw="
                f"{result['yaw_change']:+.3f} | "
                f"energy="
                f"{result['energy']:.3f} | "
                f"reward="
                f"{result['reward']:.3f} | "
                f"fallen="
                f"{result['fallen']} | "
                f"fall_step="
                f"{result['fall_step']} | "
                f"recovery="
                f"{recovery_text} | "
                f"steps="
                f"{result['steps']}"
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()