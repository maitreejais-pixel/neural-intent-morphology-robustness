import gymnasium as gym
import numpy as np


def main():

    env = gym.make("Ant-v5")

    env.reset(seed=42)

    model = env.unwrapped.model
    data = env.unwrapped.data

    print("=" * 80)
    print("ANT-V5 LEG / ACTUATOR MOVEMENT TEST")
    print("=" * 80)

    print("\nEach actuator will be activated individually.")
    print("We record the joint position change.\n")

    for actuator_id in range(model.nu):

        env.reset(seed=42)

        initial_qpos = data.qpos.copy()

        action = np.zeros(model.nu, dtype=np.float32)

        # Apply positive action to only this actuator
        action[actuator_id] = 1.0

        for _ in range(20):
            env.step(action)

        final_qpos = data.qpos.copy()

        print("-" * 80)

        joint_id = int(model.actuator_trnid[actuator_id, 0])

        joint_name = model.joint(
            joint_id
        ).name

        body_id = int(model.jnt_bodyid[joint_id])

        body_name = model.body(
            body_id
        ).name

        qpos_index = int(model.jnt_qposadr[joint_id])

        change = (
            final_qpos[qpos_index]
            - initial_qpos[qpos_index]
        )

        print(
            f"Actuator {actuator_id}"
        )

        print(
            f"  Joint : {joint_name}"
        )

        print(
            f"  Body  : {body_name}"
        )

        print(
            f"  qpos change after positive action: "
            f"{change:.6f}"
        )

    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()