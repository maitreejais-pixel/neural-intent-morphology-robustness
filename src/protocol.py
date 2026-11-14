"""
Shared evaluation protocol, used by run_experiments.py and
eval_checkpoints.py so that every controller (and checkpoint selection)
sees exactly the same intention schedule and the same metrics.
"""

import numpy as np

SEGMENT_STEPS = 150  # Forward -> Left -> Right -> Stop, repeating


def intention_at(step):
    """Deterministic schedule; independent of the episode seed."""

    return int((step // SEGMENT_STEPS) % 4)


def morphology_params(axis, level):
    """
    [mass, limb_length, joint_strength, friction]

    mass / limb_length go UP by `level`; joint_strength / friction go
    DOWN by `level`.
    """

    p = np.ones(4, dtype=np.float32)

    if axis == "mass":
        p[0] = 1.0 + level
    elif axis == "limb_length":
        p[1] = 1.0 + level
    elif axis == "joint_strength":
        p[2] = 1.0 - level
    elif axis == "friction":
        p[3] = 1.0 - level
    else:
        raise ValueError(f"Unknown morphology axis: {axis}")

    return p


def sample_morphology(rng):
    """Training distribution (matches the evaluated perturbation ranges)."""

    return np.array(
        [
            rng.uniform(0.5, 1.5),  # mass            (eval: 1.0 .. 1.5)
            rng.uniform(0.5, 1.5),  # limb length     (eval: 1.0 .. 1.5)
            rng.uniform(0.5, 1.0),  # joint strength  (eval: 0.5 .. 1.0)
            rng.uniform(0.5, 1.0),  # friction        (eval: 0.5 .. 1.0)
        ],
        dtype=np.float32,
    )


def _mean_or_nan(values):

    return float(np.mean(values)) if len(values) else float("nan")


def run_episode(env, policy, seed, morphology, steps=1000):
    """
    Run one episode under the fixed intention schedule.

    policy(obs, intention, step) -> action (shape (8,)).

    Metrics that separate locomotion from standing:
        fwd_vel_err    mean |forward_velocity - target| in Forward segments
        yaw_rate_err   mean |yaw_velocity - target| in Left/Right segments
        stop_speed     mean linear + angular speed in Stop segments
    distance / forward_distance / yaw_change / energy are snapshotted at
    the first fall (or taken at the end if the robot never fell).
    """

    env.reset(seed=seed, morphology=morphology, intention=0)

    total_reward = 0.0
    total_task = 0.0
    info = {}
    fall_snapshot = None

    fwd_err, yaw_err, stop_speed = [], [], []

    for step in range(steps):

        intention = intention_at(step)

        env.set_intention(intention)
        obs = env.get_current_observation()

        action = policy(obs, intention, step)

        _, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        total_task += float(info["task_reward"])

        if fall_snapshot is None and info["fallen"]:
            fall_snapshot = dict(info)

        fwd_v = info["forward_velocity"]
        lat_v = info["lateral_velocity"]
        yaw_v = info["yaw_velocity"]

        if intention == 0:
            fwd_err.append(abs(fwd_v - env.FORWARD_TARGET))
        elif intention == 1:
            yaw_err.append(abs(yaw_v - env.LEFT_YAW_TARGET))
        elif intention == 2:
            yaw_err.append(abs(yaw_v - env.RIGHT_YAW_TARGET))
        else:
            stop_speed.append(float(np.hypot(fwd_v, lat_v)) + abs(yaw_v))

        if terminated or truncated:
            break

    snapshot = fall_snapshot if fall_snapshot is not None else info

    return {
        "distance": float(snapshot.get("distance", 0.0)),
        "forward_distance": float(snapshot.get("forward_distance", 0.0)),
        "yaw_change": float(snapshot.get("yaw_change", 0.0)),
        "energy": float(snapshot.get("energy", 0.0)),
        "reward": total_reward,
        "task_reward": total_task,
        "fwd_vel_err": _mean_or_nan(fwd_err),
        "yaw_rate_err": _mean_or_nan(yaw_err),
        "stop_speed": _mean_or_nan(stop_speed),
        "steps": int(info.get("steps", step + 1)),
        "fallen": int(bool(info.get("fallen", False))),
        "fall_step": info.get("fall_step", None),
        "recovery_time": None,
    }