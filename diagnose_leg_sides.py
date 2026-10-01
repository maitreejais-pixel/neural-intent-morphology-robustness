"""
Diagnostic: determine the REAL spatial side (left/right, front/back)
of each Ant-v5 actuator, by walking up the body tree from each
actuator's joint to its aux_N attachment point and reading that
body's actual (x, y) position relative to the torso.

Convention used elsewhere in this project: x = forward, y = left,
z = up. So:
    y > 0  -> physically LEFT
    y < 0  -> physically RIGHT
    x > 0  -> physically FRONT
    x < 0  -> physically BACK

Run this and share the printed table. It tells us, with certainty,
whether controllers.py's LEG_GROUPS / LEFT_LEGS / RIGHT_LEGS
assignment actually matches the real geometry of the Ant-v5 model
installed in this environment.
"""

import gymnasium as gym
import mujoco
import numpy as np


def main():

    env = gym.make("Ant-v5")
    env.reset(seed=42)

    model = env.unwrapped.model

    print("=" * 90)
    print("AUX BODY POSITIONS (leg attachment points, relative to torso)")
    print("=" * 90)

    aux_positions = {}

    for i in range(1, 5):
        name = f"aux_{i}"
        try:
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )
        except Exception:
            bid = -1

        if bid < 0:
            print(f"{name}: NOT FOUND")
            continue

        pos = model.body_pos[bid]
        aux_positions[bid] = (name, pos)

        side = "LEFT" if pos[1] > 0 else "RIGHT"
        front_back = "FRONT" if pos[0] > 0 else "BACK"

        print(
            f"{name:8s} | body_id={bid:2d} | "
            f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) | "
            f"=> physically {front_back}-{side}"
        )

    print()
    print("=" * 90)
    print("ACTUATOR -> JOINT -> PARENT AUX -> REAL SIDE")
    print("=" * 90)

    aux_ids = set(aux_positions.keys())

    for actuator_id in range(model.nu):

        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model.joint(joint_id).name

        body_id = int(model.jnt_bodyid[joint_id])
        body_name = model.body(body_id).name

        # Walk up the parent chain until we hit one of the aux_N
        # bodies (the leg's attachment point to the torso).
        walk_id = body_id
        guard = 0
        while walk_id not in aux_ids and walk_id != 0 and guard < 20:
            walk_id = int(model.body_parentid[walk_id])
            guard += 1

        if walk_id in aux_ids:
            aux_name, pos = aux_positions[walk_id]
            side = "LEFT" if pos[1] > 0 else "RIGHT"
            front_back = "FRONT" if pos[0] > 0 else "BACK"
            label = f"{front_back}-{side} (via {aux_name})"
        else:
            label = "UNRESOLVED"

        print(
            f"Actuator {actuator_id} | "
            f"joint={joint_name:12s} | "
            f"body={body_name:12s} | "
            f"REAL SIDE = {label}"
        )

    print("=" * 90)
    print()
    print("Compare this against controllers.py's LEG_GROUPS:")
    print("  leg 0 (labeled front-left)  -> actuators [2, 3]")
    print("  leg 1 (labeled front-right) -> actuators [4, 5]")
    print("  leg 2 (labeled back-left)   -> actuators [6, 7]")
    print("  leg 3 (labeled back-right)  -> actuators [0, 1]")
    print(
        "If the REAL SIDE printed above for actuators 4,5 says LEFT "
        "instead of RIGHT (or similar mismatches), that confirms the "
        "LEG_GROUPS labeling is wrong and needs to be reassigned to "
        "match the real geometry."
    )

    env.close()


if __name__ == "__main__":
    main()