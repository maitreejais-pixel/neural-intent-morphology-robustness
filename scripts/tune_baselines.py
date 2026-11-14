"""
Tune the Direct and CPG baselines with a DOCUMENTED procedure.

Why: after the control-step timing fix (src/controllers.py, CONTROL_DT =
0.05 s) the old gains no longer apply, and the old tuning scripts scored
Forward walking only. This script scores each configuration on the full
Forward -> Left -> Right -> Stop schedule, the same protocol used for
the final evaluation, at NOMINAL morphology.

Selection criterion (state this in the paper):
    mean TASK reward (tracking reward only; a robot that stands still
    scores about 150) over --episodes episodes (default 10) with tuning
    seeds 200.. (disjoint from the PPO checkpoint-selection seeds
    100+/500+ and from the evaluation seeds 1000+). Three episodes were
    too few: a setting that looked stable on seeds 200-202 fell within
    17 steps on a fresh seed. Falling ends an episode, so
    controllers that fall early score low automatically.

Run from the project root:

    python scripts/tune_baselines.py

Then copy the printed values into the defaults of
DirectTrajectoryController / FixedCPGController in src/controllers.py
and keep results/tuning_direct.csv and results/tuning_cpg.csv as the
record of the search.
"""

import argparse
import itertools
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

import numpy as np
import pandas as pd

from src.controllers import (
    DirectTrajectoryController,
    FixedCPGController,
    ZeroActionController,
)
from src.env import MorphologyAntEnv
from src.protocol import run_episode

TUNING_SEED0 = 200

# Search grids (edit to widen the search).
DIRECT_FREQUENCIES = [0.3, 0.6, 1.0, 1.5, 2.0]
DIRECT_AMPLITUDES = [0.2, 0.3, 0.45, 0.6]

CPG_FREQUENCIES = [0.5, 1.0, 1.5, 2.0]
CPG_HIP_AMPLITUDES = [0.25, 0.35, 0.5, 0.65]
CPG_KDS = [0.2, 0.5]


def _mean_ignoring_nan(values):
    """Mean of the non-NaN values (NaN if there are none), silently."""

    values = [v for v in values if not np.isnan(v)]

    return float(np.mean(values)) if values else float("nan")


def evaluate(env, make_controller, episodes, steps):

    controller = make_controller()

    def policy(obs, intention, step):
        return controller.act(intention, step, obs)

    results = []

    for ep in range(episodes):

        controller.reset()

        results.append(
            run_episode(
                env,
                policy,
                seed=TUNING_SEED0 + ep,
                morphology=np.ones(4, dtype=np.float32),
                steps=steps,
            )
        )

    return {
        "task_reward": float(
            np.mean([r["task_reward"] for r in results])
        ),
        "fall_rate": float(
            np.mean([r["fallen"] for r in results])
        ),
        "steps": float(
            np.mean([r["steps"] for r in results])
        ),
        "fwd_vel_err": _mean_ignoring_nan(
            [r["fwd_vel_err"] for r in results]
        ),
        "yaw_rate_err": _mean_ignoring_nan(
            [r["yaw_rate_err"] for r in results]
        ),
        "distance": float(
            np.mean([r["distance"] for r in results])
        ),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Tune Direct and CPG with the evaluation protocol."
    )

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--outdir", default="results")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    env = MorphologyAntEnv(morphology_observation=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    try:

        # ---------------- Direct ----------------

        rows = []

        for frequency, amplitude in itertools.product(
            DIRECT_FREQUENCIES,
            DIRECT_AMPLITUDES,
        ):
            score = evaluate(
                env,
                lambda: DirectTrajectoryController(
                    amplitude=amplitude,
                    frequency=frequency,
                ),
                args.episodes,
                args.steps,
            )
            rows.append(
                {
                    "frequency": frequency,
                    "amplitude": amplitude,
                    **score,
                }
            )

        direct = pd.DataFrame(rows).sort_values(
            "task_reward",
            ascending=False,
        )

        direct.to_csv(
            os.path.join(args.outdir, "tuning_direct.csv"),
            index=False,
        )

        print("=" * 90)
        print("DIRECT: top 5 by mean task reward")
        print("=" * 90)
        print(direct.head(5).to_string(index=False))

        best = direct.iloc[0]

        print(
            "\nBest Direct: "
            f"amplitude={best['amplitude']}, "
            f"frequency={best['frequency']} "
            f"(task reward {best['task_reward']:.1f}, "
            f"fall rate {best['fall_rate']:.2f})"
        )

        # ---------------- CPG ----------------

        rows = []

        for frequency, hip, kd in itertools.product(
            CPG_FREQUENCIES,
            CPG_HIP_AMPLITUDES,
            CPG_KDS,
        ):
            score = evaluate(
                env,
                lambda: FixedCPGController(
                    frequency=frequency,
                    hip_amplitude=hip,
                    kd=kd,
                ),
                args.episodes,
                args.steps,
            )
            rows.append(
                {
                    "frequency": frequency,
                    "hip_amplitude": hip,
                    "kd": kd,
                    **score,
                }
            )

        cpg = pd.DataFrame(rows).sort_values(
            "task_reward",
            ascending=False,
        )

        cpg.to_csv(
            os.path.join(args.outdir, "tuning_cpg.csv"),
            index=False,
        )

        print()
        print("=" * 90)
        print("CPG: top 5 by mean task reward")
        print("=" * 90)
        print(cpg.head(5).to_string(index=False))

        best = cpg.iloc[0]

        print(
            "\nBest CPG: "
            f"frequency={best['frequency']}, "
            f"hip_amplitude={best['hip_amplitude']}, "
            f"kd={best['kd']} "
            f"(task reward {best['task_reward']:.1f}, "
            f"fall rate {best['fall_rate']:.2f})"
        )

        # Stand-still reference on exactly the same episodes.
        reference = evaluate(
            env,
            ZeroActionController,
            args.episodes,
            args.steps,
        )

        print()
        print("=" * 90)
        print("READING THE RESULT")
        print("=" * 90)
        print(
            "Stand-still reference (same seeds): task reward "
            f"{reference['task_reward']:.1f}, forward-speed error "
            f"{reference['fwd_vel_err']:.2f}, yaw-rate error "
            f"{reference['yaw_rate_err']:.2f}."
        )
        print(
            "A controller only locomotes if its task reward is clearly "
            "ABOVE the stand-still reference and its forward-speed "
            "error is well below 1.0 (error ~1.0 means no forward "
            "motion)."
        )

        for label, table in (("Direct", direct), ("CPG", cpg)):

            top = table.iloc[0]

            walks = (
                top["task_reward"] > reference["task_reward"] + 10.0
                and top["fwd_vel_err"] < 0.8
            )

            print(
                f"  {label}: best task reward {top['task_reward']:.1f} "
                f"(forward-speed error {top['fwd_vel_err']:.2f}) -> "
                + (
                    "locomotes."
                    if walks
                    else "does NOT locomote (at best it stands still "
                    "or falls); report it as such or redesign it."
                )
            )

    finally:
        env.close()


if __name__ == "__main__":
    main()