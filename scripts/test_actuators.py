import gymnasium as gym
import numpy as np


env = gym.make("Ant-v5")

env.reset(seed=42)

model = env.unwrapped.model

print("=" * 80)
print("ANT-V5 ACTUATOR / JOINT MAPPING")
print("=" * 80)

print("\nACTUATORS")
print("-" * 80)

for actuator_id in range(model.nu):

    joint_id = model.actuator_trnid[actuator_id, 0]

    actuator_name = model.names[
        model.name_actuatoradr[actuator_id]:
    ].split(b"\x00")[0] if False else ""

    joint_name = model.joint(
        joint_id
    ).name

    print(
        f"Actuator {actuator_id}: "
        f"joint_id={joint_id}, "
        f"joint={joint_name}"
    )


print("\nJOINTS")
print("-" * 80)

for joint_id in range(model.njnt):

    joint_name = model.joint(
        joint_id
    ).name

    print(
        f"Joint {joint_id}: "
        f"{joint_name}"
    )


print("\nBODIES")
print("-" * 80)

for body_id in range(model.nbody):

    body_name = model.body(
        body_id
    ).name

    print(
        f"Body {body_id}: "
        f"{body_name}"
    )


print("\nACTUATOR GEAR")
print("-" * 80)

for actuator_id in range(model.nu):

    print(
        f"Actuator {actuator_id}: "
        f"gear={model.actuator_gear[actuator_id]}"
    )


print("=" * 80)

env.close()