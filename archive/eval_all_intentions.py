"""
Evaluate one set of FixedCPGController gains across ALL FOUR
intentions (Forward, Left, Right, Stop), not just Forward.

The earlier sweeps (tune_cpg.py, tune_cpg_round2.py, robust_check.py)
only ever tested intention=0 (Forward). That let us pick gains that
walk far in a straight line while accidentally breaking Left (barely
turns, falls almost instantly) and Stop (which previously never fell
at all, and now falls while drifting several meters) -- a controller
that only works for Forward isn't a fair baseline for a system meant
to follow four different commands.

Run from the project root:
    python eval_all_intentions.py
"""

import numpy as np

from src.env import MorphologyAntEnv
from src.controllers import FixedCPGController


EPISODE_STEPS = 500
SEEDS = [0, 1, 2, 3, 4, 42]

INTENTIONS = {
    0: "Forward",
    1: "Left",
    2: "Right",
    3: "Stop",
}

# ----------------------------------------------------------------------
# Candidates to compare. Edit / add rows here to test new ideas.
# ----------------------------------------------------------------------

CANDIDATES = [
    {"label": "current (post-tuning)", "hip_amplitude": 0.35, "kd": 0.50, "frequency": 1.30},
    {"label": "original default",      "hip_amplitude": 0.65, "kd": 0.20, "frequency": 1.50},
    {"label": "new amp/freq, old kd",  "hip_amplitude": 0.35, "kd": 0.20, "frequency": 1.30},
]

KP = 1.8
ANKLE_AMPLITUDE = 0.40
ACTION_SCALE = 2.0


def run_once(hip_amplitude, kd, frequency, intention, seed):

    env = MorphologyAntEnv()

    obs, _ = env.reset(
        seed=seed,
        intention=intention,
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
            intention=intention,
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
        "yaw_change": float(snapshot.get("yaw_change", 0.0)),
        "fallen": bool(info.get("fallen", False)),
        "fall_step": info.get("fall_step", None),
    }


def main():

    for candidate in CANDIDATES:

        print("=" * 100)
        print(
            f"{candidate['label']} | "
            f"hip_amp={candidate['hip_amplitude']} | "
            f"kd={candidate['kd']} | "
            f"freq={candidate['frequency']}"
        )
        print("=" * 100)

        for intention, name in INTENTIONS.items():

            fall_steps = []
            distances = []
            never_fell = 0

            for seed in SEEDS:

                result = run_once(
                    candidate["hip_amplitude"],
                    candidate["kd"],
                    candidate["frequency"],
                    intention,
                    seed,
                )

                distances.append(result["distance"])

                if result["fallen"]:
                    fall_steps.append(result["fall_step"])
                else:
                    never_fell += 1

            avg_distance = float(np.mean(distances))
            min_distance = float(np.min(distances))

            fall_step_summary = (
                f"avg_fall_step={np.mean(fall_steps):.1f}"
                if fall_steps
                else "n/a"
            )

            print(
                f"  {name:8s} | "
                f"avg_distance={avg_distance:.3f} | "
                f"min_distance={min_distance:.3f} | "
                f"never_fell={never_fell}/{len(SEEDS)} | "
                f"{fall_step_summary}"
            )

        print()


if __name__ == "__main__":
    main()