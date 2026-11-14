"""
Re-test the best candidates from tune_cpg_round2.py across multiple
seeds, to check whether their performance is a robust property of the
gains or a lucky/unlucky artifact of a single seed=42 run.

The round-2 sweep showed high sensitivity to small frequency changes
at fixed (hip_amplitude, kd) -- e.g. hip_amp=0.30, kd=0.50 went from
fall_step=34 (freq=1.30) to NEVER (freq=1.50) to fall_step=28
(freq=1.70). That kind of narrow window is a sign the system may be
sensitive to initial conditions, so a single-seed "winner" deserves a
second look before becoming the new default.

Run from the project root:
    python robust_check.py
"""

import numpy as np

from src.env import MorphologyAntEnv
from src.controllers import FixedCPGController


EPISODE_STEPS = 500
SEEDS = [0, 1, 2, 3, 4, 42]

# Top candidates from round 2, plus the original default for
# reference.
CANDIDATES = [
    {"label": "round2 best-distance", "hip_amplitude": 0.35, "kd": 0.50, "frequency": 1.30},
    {"label": "round2 never-falls",   "hip_amplitude": 0.30, "kd": 0.50, "frequency": 1.50},
    {"label": "round2 longest-survival", "hip_amplitude": 0.35, "kd": 0.50, "frequency": 1.70},
    {"label": "original default",     "hip_amplitude": 0.65, "kd": 0.20, "frequency": 1.50},
]

KP = 1.8
ANKLE_AMPLITUDE = 0.40
ACTION_SCALE = 2.0


def run_once(hip_amplitude, kd, frequency, seed):

    env = MorphologyAntEnv()

    obs, _ = env.reset(
        seed=seed,
        intention=0,  # Forward
    )

    controller = FixedCPGController(
        frequency=frequency,
        hip_amplitude=hip_amplitude,
        ankle_amplitude=ANKLE_AMPLITUDE,
        kp=KP,
        kd=kd,
        action_scale=ACTION_SCALE,
    )
    controller.reset()

    info = {}
    fall_snapshot = None

    for step in range(EPISODE_STEPS):

        action = controller.act(
            intention=0,
            step=step,
            obs=obs,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        if fall_snapshot is None and info.get("fallen", False):
            fall_snapshot = dict(info)

        if terminated or truncated:
            break

    env.close()

    snapshot = fall_snapshot if fall_snapshot is not None else info

    return {
        "distance": float(snapshot.get("distance", 0.0)),
        "fallen": bool(info.get("fallen", False)),
        "fall_step": info.get("fall_step", None),
    }


def main():

    for candidate in CANDIDATES:

        print("=" * 90)
        print(
            f"{candidate['label']} | "
            f"hip_amp={candidate['hip_amplitude']} | "
            f"kd={candidate['kd']} | "
            f"freq={candidate['frequency']}"
        )
        print("=" * 90)

        distances = []
        fall_steps = []
        never_fell_count = 0

        for seed in SEEDS:

            result = run_once(
                candidate["hip_amplitude"],
                candidate["kd"],
                candidate["frequency"],
                seed,
            )

            distances.append(result["distance"])

            if result["fallen"]:
                fall_steps.append(result["fall_step"])
            else:
                never_fell_count += 1

            fall_step_display = (
                result["fall_step"]
                if result["fall_step"] is not None
                else "NEVER"
            )

            print(
                f"  seed={seed:3d} | "
                f"distance={result['distance']:.3f} | "
                f"fall_step={fall_step_display}"
            )

        avg_distance = float(np.mean(distances))
        min_distance = float(np.min(distances))

        print(
            f"  -> avg_distance={avg_distance:.3f} | "
            f"min_distance={min_distance:.3f} | "
            f"never_fell={never_fell_count}/{len(SEEDS)}"
        )

        if fall_steps:
            print(
                f"  -> avg_fall_step={np.mean(fall_steps):.1f} | "
                f"min_fall_step={min(fall_steps)}"
            )

        print()


if __name__ == "__main__":
    main()