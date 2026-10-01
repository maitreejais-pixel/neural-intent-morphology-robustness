import numpy as np
import mujoco
import gymnasium as gym


def get_yaw(env):
    """
    Extract torso yaw from the Ant's quaternion.
    """

    data = env.unwrapped.data

    quat = np.asarray(
        data.qpos[3:7],
        dtype=np.float64,
    )

    rotation = np.zeros(
        9,
        dtype=np.float64,
    )

    mujoco.mju_quat2Mat(
        rotation,
        quat,
    )

    return float(
        np.arctan2(
            rotation[3],
            rotation[0],
        )
    )


def wrapped_angle_difference(a, b):
    """
    Return the shortest signed angular difference
    between angles a and b.
    """

    return float(
        np.arctan2(
            np.sin(a - b),
            np.cos(a - b),
        )
    )


def main():

    env = gym.make("Ant-v5")

    env.reset(seed=42)

    dt = float(
        env.unwrapped.model.opt.timestep
    )

    print(
        "Testing qvel[5] against actual torso yaw change"
    )
    print("=" * 90)

    print(
        f"MuJoCo timestep = {dt:.6f} s"
    )

    print()
    print(
        "qvel[5] is the measured yaw angular-velocity "
        "component."
    )

    print(
        "The expected yaw increment is approximately "
        "qvel[5] * timestep."
    )

    print("=" * 90)

    # Zero action means there is no intentional actuator input.
    # Small yaw motion can still arise from the initial dynamics.
    action = np.zeros(
        8,
        dtype=np.float32,
    )

    previous_yaw = get_yaw(env)

    for step in range(100):

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        current_yaw = get_yaw(env)

        # Actual measured yaw change.
        yaw_delta = wrapped_angle_difference(
            current_yaw,
            previous_yaw,
        )

        # MuJoCo angular velocity.
        yaw_rate_qvel = float(
            env.unwrapped.data.qvel[5]
        )

        # First-order expected change over one timestep.
        expected_delta = (
            yaw_rate_qvel * dt
        )

        error = (
            yaw_delta
            - expected_delta
        )

        print(
            f"Step {step + 1:3d} | "
            f"qvel[5] = {yaw_rate_qvel:+.6f} | "
            f"yaw_delta = {yaw_delta:+.6f} | "
            f"qvel[5]*dt = {expected_delta:+.6f} | "
            f"error = {error:+.6f}"
        )

        previous_yaw = current_yaw

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()

