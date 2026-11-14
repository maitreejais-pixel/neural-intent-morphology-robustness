from src.env import MorphologyAntEnv
from src.controllers import FixedCPGController
import numpy as np


INTENTIONS = {
    0: "Forward",
    1: "Left",
    2: "Right",
    3: "Stop",
}


for intention, name in INTENTIONS.items():

    env = MorphologyAntEnv()

    obs, _ = env.reset(
        seed=42,
        intention=intention,
    )

    controller = FixedCPGController()
    controller.reset()

    total_reward = 0.0
    info = {}

    # FIXED: capture a snapshot of distance/yaw_change at the exact
    # moment a fall is first detected, instead of only reading the
    # final info at the end of the episode.
    #
    # self.fallen in env.py never resets mid-episode, and the
    # environment does NOT terminate when a fall occurs (there is no
    # terminate_when_unhealthy / fall-based termination). That means
    # after falling, the robot spends the REST of the episode lying
    # on the ground while the controller keeps issuing leg commands
    # into it -- producing chaotic, physically meaningless tumbling.
    # Reading yaw_change/distance at step 500 was therefore measuring
    # that post-fall noise, not the intended walking/turning
    # behavior. Snapshotting at the moment of first fall gives a
    # clean read of what the controller actually did before things
    # went wrong.
    fall_snapshot = None

    for step in range(500):

        action = controller.act(
            intention=intention,
            step=step,
            obs=obs,
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward

        if fall_snapshot is None and info.get("fallen", False):
            fall_snapshot = dict(info)

        if terminated or truncated:
            break

    # If it never fell, the final info IS the clean snapshot.
    snapshot = (
        fall_snapshot
        if fall_snapshot is not None
        else info
    )

    distance = float(snapshot.get("distance", 0.0))
    yaw_change = float(snapshot.get("yaw_change", 0.0))
    fallen = bool(info.get("fallen", False))
    fall_step = info.get("fall_step", None)
    steps_taken = int(info.get("steps", step + 1))

    print(
        f"{name:>7} | "
        f"distance_at_fall_or_end={distance:.4f} | "
        f"yaw_change_at_fall_or_end={yaw_change:.4f} | "
        f"episode_reward={total_reward:.2f} | "
        f"fallen={fallen} | "
        f"fall_step={fall_step} | "
        f"total_steps_run={steps_taken}"
    )

    env.close()