"""
Diagnostic: determine the REAL spatial side (left/right, front/back)
of each Ant-v5 actuator, by walking up the body tree from each
actuator's joint to its aux_N attachment point and reading that
body's actual (x, y) position relative to the torso.

Convention used throughout this project: x = forward, y = left, z = up.
    y > 0  -> physically LEFT
    y < 0  -> physically RIGHT
    x > 0  -> physically FRONT
    x < 0  -> physically BACK

The result is compared automatically against the mapping used in
src/controllers.py (LEG_GROUPS), so no manual comparison is needed.
"""

import gymnasium as gym
import mujoco

# Mapping used by src/controllers.py (LEG_GROUPS / LEFT_LEGS / RIGHT_LEGS).
EXPECTED_SIDE = {
    0: "FRONT-RIGHT",
    1: "FRONT-RIGHT",
    2: "FRONT-LEFT",
    3: "FRONT-LEFT",
    4: "BACK-LEFT",
    5: "BACK-LEFT",
    6: "BACK-RIGHT",
    7: "BACK-RIGHT",
}


def side_label(pos):
    front_back = "FRONT" if pos[0] > 0 else "BACK"
    side = "LEFT" if pos[1] > 0 else "RIGHT"
    return f"{front_back}-{side}"


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
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        if bid < 0:
            print(f"{name}: NOT FOUND")
            continue

        pos = model.body_pos[bid]
        aux_positions[bid] = (name, pos)

        print(
            f"{name:8s} | body_id={bid:2d} | "
            f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) | "
            f"=> physically {side_label(pos)}"
        )

    print()
    print("=" * 90)
    print("ACTUATOR -> JOINT -> PARENT AUX -> REAL SIDE")
    print("=" * 90)

    all_match = True

    for actuator_id in range(model.nu):

        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model.joint(joint_id).name

        body_id = int(model.jnt_bodyid[joint_id])
        body_name = model.body(body_id).name or "(unnamed)"

        # Walk up the parent chain until we reach one of the aux_N
        # bodies (the leg's attachment point to the torso).
        walk_id = body_id
        guard = 0

        while (
            walk_id not in aux_positions
            and walk_id != 0
            and guard < 20
        ):
            walk_id = int(model.body_parentid[walk_id])
            guard += 1

        if walk_id in aux_positions:
            aux_name, pos = aux_positions[walk_id]
            real = side_label(pos)
            via = f"via {aux_name}"
        else:
            real = "UNRESOLVED"
            via = "no aux ancestor"

        ok = real == EXPECTED_SIDE[actuator_id]
        all_match = all_match and ok

        print(
            f"Actuator {actuator_id} | "
            f"joint={joint_name:8s} | "
            f"body={body_name:10s} | "
            f"REAL SIDE = {real:11s} ({via}) | "
            f"{'OK' if ok else 'MISMATCH vs LEG_GROUPS'}"
        )

    print("=" * 90)
    print()
    print("controllers.py LEG_GROUPS being checked:")
    print("  leg 0 (front-left)  -> actuators [2, 3]")
    print("  leg 1 (front-right) -> actuators [0, 1]")
    print("  leg 2 (back-left)   -> actuators [4, 5]")
    print("  leg 3 (back-right)  -> actuators [6, 7]")
    print()

    if all_match:
        print("RESULT: all 8 actuators match LEG_GROUPS.")
    else:
        print(
            "RESULT: MISMATCH found. Reassign LEG_GROUPS, LEFT_LEGS and "
            "RIGHT_LEGS in src/controllers.py to match the REAL SIDE "
            "column above."
        )

    env.close()


if __name__ == "__main__":
    main()