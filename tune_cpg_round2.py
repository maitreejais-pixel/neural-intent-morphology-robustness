"""
Sweep FixedCPGController gains under the Forward intention and report
how far/long each combination walks before falling.

FIXED: run_once() now snapshots distance/reward at the moment a fall
is first detected, instead of reading `info` only after the full
500-step loop. The environment does not terminate episodes on a fall
(no terminate_when_unhealthy / fall-based termination), so a fallen
robot spends the remainder of the episode being tossed around by
continued actuator commands. Reading final-step info therefore
measured post-fall tumbling noise, not real walking distance -- this
is why higher-damping (kd) configs previously showed LARGER "distance"
while falling SOONER: more damping was producing more violent
post-fall flailing, not better walking.

Run from the project root:
    python tune_cpg.py
"""

import itertools

import numpy as np

from src.env import MorphologyAntEnv
from src.controllers import FixedCPGController


EPISODE_STEPS = 500
SEED = 42

# ----------------------------------------------------------------------
# Grid to sweep. Kept small on purpose -- widen later once you see
# which direction helps.
# ----------------------------------------------------------------------

HIP_AMPLITUDES = [0.25, 0.30, 0.35]   # narrowed around round-1 standout (0.30)
KD_VALUES = [0.40, 0.50, 0.60, 0.80]  # narrowed around round-1 standout (0.60)
FREQUENCIES = [1.3, 1.5, 1.7]          # narrowed around round-1 standout (1.5)

# Keep these fixed for this first sweep.
KP = 1.8
ANKLE_AMPLITUDE = 0.40
ACTION_SCALE = 2.0


def run_once(hip_amplitude, kd, frequency):

    env = MorphologyAntEnv()

    obs, _ = env.reset(
        seed=SEED,
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
    total_reward = 0.0

    # FIXED: snapshot distance/energy at the moment of first fall,
    # exactly as in the corrected test_intentions.py.
    fall_snapshot = None

    for step in range(EPISODE_STEPS):

        action = controller.act(
            intention=0,
            step=step,
            obs=obs,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if fall_snapshot is None and info.get("fallen", False):
            fall_snapshot = dict(info)

        if terminated or truncated:
            break

    env.close()

    snapshot = fall_snapshot if fall_snapshot is not None else info

    return {
        "hip_amplitude": hip_amplitude,
        "kd": kd,
        "frequency": frequency,
        # Distance at the moment of falling (or at episode end if it
        # never fell) -- NOT the final-step distance, which would
        # include post-fall tumbling.
        "distance": float(snapshot.get("distance", 0.0)),
        "fallen": bool(info.get("fallen", False)),
        "fall_step": info.get("fall_step", None),
        "reward": float(total_reward),
    }


def main():

    results = []

    combos = list(
        itertools.product(HIP_AMPLITUDES, KD_VALUES, FREQUENCIES)
    )

    print(f"Running {len(combos)} configurations...\n")

    for hip_amplitude, kd, frequency in combos:

        result = run_once(hip_amplitude, kd, frequency)
        results.append(result)

        fall_step_display = (
            result["fall_step"]
            if result["fall_step"] is not None
            else "NEVER"
        )

        print(
            f"hip_amp={hip_amplitude:.2f} | "
            f"kd={kd:.2f} | "
            f"freq={frequency:.2f} | "
            f"distance_at_fall_or_end={result['distance']:.3f} | "
            f"fall_step={fall_step_display} | "
            f"fallen={result['fallen']}"
        )

    # ------------------------------------------------------------------
    # Rank: never-fell configs first (by distance descending -- a
    # never-fell config that also covers real ground is what we want),
    # then fell configs by fall_step descending (survived longest).
    # ------------------------------------------------------------------

    def sort_key(r):
        # Rank by distance covered before falling (or full-episode
        # distance if it never fell) -- NOT by never-falling alone.
        # Round 1 showed "never falls" configs were just standing
        # nearly still (~0.4 m over 5 seconds); a config that walks
        # 3+ m before falling at ~2.5 s is a far more useful baseline
        # than one that never falls but never really walks either.
        return -r["distance"]

    results.sort(key=sort_key)

    print("\n" + "=" * 90)
    print("TOP 5 CONFIGURATIONS (best first)")
    print("=" * 90)

    for r in results[:5]:
        fall_step_display = (
            r["fall_step"] if r["fall_step"] is not None else "NEVER"
        )
        print(
            f"hip_amp={r['hip_amplitude']:.2f} | "
            f"kd={r['kd']:.2f} | "
            f"freq={r['frequency']:.2f} | "
            f"distance_at_fall_or_end={r['distance']:.3f} | "
            f"fall_step={fall_step_display} | "
            f"reward={r['reward']:.2f}"
        )


if __name__ == "__main__":
    main()