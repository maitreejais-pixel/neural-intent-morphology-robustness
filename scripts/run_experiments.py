"""
Full-physics MuJoCo morphology robustness experiments.

Run from the project root. Example:

    python scripts/run_experiments.py \
        --controllers direct cpg ppo ppo_nomorph standstill \
        --model models/ppo_seed42.zip \
        --model-nomorph models/ppo_nomorph_seed42.zip \
        --output results/physics_results.csv

    python scripts/analyze_results.py --input results/physics_results.csv

CHANGES vs. the previous version
--------------------------------
1. ALL controllers now use the SAME block of initial-state seeds
   (--seed0, default 1000), so episodes are PAIRED across controllers.
   Before, every controller had its own seed block (Direct 1000-1009,
   CPG 2000-2009, PPO 3000-3009), which made comparisons unpaired.
2. New controller "ppo_nomorph": the domain-randomization-only PPO
   ablation (trained with --no-morph). It does not observe the
   morphology vector, so comparing it with "ppo" isolates the effect of
   conditioning on morphology. Needs --model-nomorph.
3. Extra metrics that separate locomotion from standing still:
       task_reward    tracking reward only (stand-still scores ~150)
       fwd_vel_err    mean |forward speed - target| in Forward segments
       yaw_rate_err   mean |yaw rate - target| in Left/Right segments
       stop_speed     mean residual speed in Stop segments
4. The episode logic itself (intention schedule, observation refresh,
   fall snapshot, metric collection) lives in src/protocol.py and is
   shared with eval_checkpoints.py, so checkpoint selection and the
   final evaluation cannot drift apart.
5. The default controller list now includes the stand-still reference,
   because analyze_results.py compares every controller against it.

Kept from earlier revisions (see src/protocol.py run_episode):

 * Total episode reward is accumulated over the whole episode.
 * distance / forward_distance / yaw_change / energy are snapshotted at
   the moment of the first fall (or taken at the end of the episode if
   the robot never fell), so post-fall tumbling does not contaminate
   them. env.py also ends the episode on a fall.
 * "fallen", "fall_step" and "steps" describe the whole episode.
   ("recovery_time" is always empty: episodes end at the first fall.)
 * The intention schedule is fixed and noise-free:
       Forward -> Left -> Right -> Stop, 150 steps each, repeating.
   It depends only on the step index, not on the episode seed, and it
   does not come from the EEG decoder. Every controller receives the
   same schedule in every episode.
"""

import argparse
import csv
import os
import sys

# Make "from src..." work whether this file is run as
# "python scripts/run_experiments.py" or "python -m scripts.run_experiments".
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from src.env import MorphologyAntEnv
from src.controllers import (
    DirectTrajectoryController,
    FixedCPGController,
    ZeroActionController,
)
from src.protocol import (
    morphology_params,
    run_episode,
)


ALL_CONTROLLERS = (
    "direct",
    "cpg",
    "ppo",
    "ppo_nomorph",
    "standstill",
)

DEFAULT_CONTROLLERS = (
    "direct",
    "cpg",
    "ppo",
    "standstill",
)

# Name written to the CSV for each controller (analyze_results.py and
# make_figures.py use these labels).
CONTROLLER_LABELS = {
    "direct": "direct",
    "cpg": "cpg",
    "ppo": "morphology-aware",
    "ppo_nomorph": "ppo-no-morphology",
    "standstill": "standstill",
}

AXES = (
    "mass",
    "limb_length",
    "joint_strength",
    "friction",
)

FIELDNAMES = [
    "controller",
    "axis",
    "level",
    "episode",
    "seed",
    "distance",
    "forward_distance",
    "yaw_change",
    "energy",
    "reward",
    "task_reward",
    "fwd_vel_err",
    "yaw_rate_err",
    "stop_speed",
    "steps",
    "fallen",
    "fall_step",
    "recovery_time",
]


# ======================================================================
# MORPHOLOGY
# ======================================================================

def params(axis, level):
    """
    Return morphology parameters:

        [mass, limb_length, joint_strength, friction]

    Nominal:
        [1.0, 1.0, 1.0, 1.0]

    Perturbation:
        mass            -> +level
        limb_length     -> +level
        joint_strength  -> -level
        friction        -> -level
    """

    return morphology_params(
        axis,
        level,
    )


# ======================================================================
# RESULT ROWS
# ======================================================================

def make_row(
    label,
    axis,
    level,
    episode,
    seed,
    result,
):
    """
    One CSV row from the dictionary returned by run_episode().
    """

    return {
        "controller": label,
        "axis": axis,
        "level": level,
        "episode": episode,
        "seed": seed,
        "distance": result["distance"],
        "forward_distance": result["forward_distance"],
        "yaw_change": result["yaw_change"],
        "energy": result["energy"],
        "reward": result["reward"],
        "task_reward": result["task_reward"],
        "fwd_vel_err": result["fwd_vel_err"],
        "yaw_rate_err": result["yaw_rate_err"],
        "stop_speed": result["stop_speed"],
        "steps": result["steps"],
        "fallen": result["fallen"],
        "fall_step": result["fall_step"],
        "recovery_time": result["recovery_time"],
    }


# ======================================================================
# EPISODE RUNNER (NON-LEARNING CONTROLLERS)
# ======================================================================

def run_classical(
    name,
    episodes,
    axis,
    level,
    seed0,
    episode_steps,
):
    """
    Evaluate a non-learning controller.

    Controllers:
        direct
        cpg
        standstill   (zero-torque reference)
    """

    if name == "direct":
        controller = DirectTrajectoryController()

    elif name == "cpg":
        controller = FixedCPGController()

    elif name == "standstill":
        controller = ZeroActionController()

    else:
        raise ValueError(
            f"Unknown classical controller: {name}"
        )

    def policy(obs, intention, step):

        return controller.act(
            intention,
            step,
            obs,
        )

    # Classical controllers read only the Ant joint state, so the
    # morphology vector is not part of their observation.
    env = MorphologyAntEnv(
        morphology_observation=False
    )

    morphology = params(
        axis,
        level,
    )

    rows = []

    try:

        for ep in range(episodes):

            seed = seed0 + ep

            controller.reset()

            result = run_episode(
                env,
                policy,
                seed=seed,
                morphology=morphology,
                steps=episode_steps,
            )

            rows.append(
                make_row(
                    CONTROLLER_LABELS[name],
                    axis,
                    level,
                    ep,
                    seed,
                    result,
                )
            )

    finally:
        env.close()

    return rows


# ======================================================================
# PPO EVALUATION
# ======================================================================

def run_ppo(
    model,
    episodes,
    axis,
    level,
    seed0,
    episode_steps,
    morphology_observation=True,
    label="morphology-aware",
):
    """
    Evaluate a PPO controller.

    morphology_observation=True   morphology-aware policy
                                  (113-dimensional observation)
    morphology_observation=False  no-morphology ablation
                                  (109-dimensional observation)

    Important:
        MorphologyAntEnv constructs the complete PPO observation:

            Ant state
            + morphology (only if morphology_observation)
            + intention one-hot

        and run_episode() refreshes it every step with the CURRENT
        intention before the policy is queried, so the policy never
        sees a stale intention after a schedule switch. We therefore do
        not concatenate observation components by hand.
    """

    def policy(obs, intention, step):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        return action

    env = MorphologyAntEnv(
        morphology_observation=morphology_observation
    )

    morphology = params(
        axis,
        level,
    )

    rows = []

    try:

        for ep in range(episodes):

            seed = seed0 + ep

            result = run_episode(
                env,
                policy,
                seed=seed,
                morphology=morphology,
                steps=episode_steps,
            )

            rows.append(
                make_row(
                    label,
                    axis,
                    level,
                    ep,
                    seed,
                    result,
                )
            )

    finally:
        env.close()

    return rows


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(
    rows,
    output_path,
):
    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run full-physics MuJoCo morphology "
            "robustness experiments."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of repeats per condition.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Maximum MuJoCo steps per episode.",
    )

    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ],
        help="Morphology perturbation levels.",
    )

    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=ALL_CONTROLLERS,
        default=list(DEFAULT_CONTROLLERS),
        help=(
            "Which controllers to run. Default: direct cpg ppo "
            "standstill. Add 'ppo_nomorph' for the "
            "no-morphology ablation."
        ),
    )

    parser.add_argument(
        "--model",
        default="models/morphology_ant_ppo.zip",
        help="Path to trained morphology-aware PPO model "
        "(only needed for 'ppo').",
    )

    parser.add_argument(
        "--model-nomorph",
        default=None,
        help="Path to the trained no-morphology PPO model "
        "(only needed for 'ppo_nomorph').",
    )

    parser.add_argument(
        "--seed0",
        type=int,
        default=1000,
        help="First initial-state seed. The SAME seeds are used for "
        "every controller, so episodes are paired.",
    )

    parser.add_argument(
        "--output",
        default="results/physics_results.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Check and load PPO models (only the ones that will be used)
    # --------------------------------------------------------------

    models = {}

    for name, path in (
        ("ppo", args.model),
        ("ppo_nomorph", args.model_nomorph),
    ):

        if name not in args.controllers:
            continue

        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                f"PPO model not found for '{name}': {path}"
            )

    print()
    print("=" * 70)
    print("FULL-PHYSICS MUJOCO MORPHOLOGY EXPERIMENT")
    print("=" * 70)
    print()
    print(
        f"Controllers : {args.controllers}"
    )

    if "ppo" in args.controllers:
        print(
            f"PPO model   : {args.model}"
        )

    if "ppo_nomorph" in args.controllers:
        print(
            f"No-morph PPO: {args.model_nomorph}"
        )

    print(
        f"Episodes    : {args.episodes}"
    )
    print(
        f"Steps       : {args.steps}"
    )
    print(
        f"Levels      : {args.levels}"
    )
    print(
        f"Seeds       : {args.seed0}.."
        f"{args.seed0 + args.episodes - 1} "
        "(shared by all controllers)"
    )
    print(
        f"Output      : {args.output}"
    )
    print()

    # Import Stable-Baselines3 only when a PPO controller is run.
    if "ppo" in args.controllers or "ppo_nomorph" in args.controllers:

        from stable_baselines3 import PPO

        if "ppo" in args.controllers:
            models["ppo"] = PPO.load(
                args.model
            )

        if "ppo_nomorph" in args.controllers:
            models["ppo_nomorph"] = PPO.load(
                args.model_nomorph
            )

    rows = []

    # --------------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------------

    for axis in AXES:

        for level in args.levels:

            print(
                f"\n[{axis}] perturbation = "
                f"{level:.0%}"
            )

            for name in args.controllers:

                print(
                    f"  Running {name}..."
                )

                if name in ("direct", "cpg", "standstill"):

                    rows.extend(
                        run_classical(
                            name=name,
                            episodes=args.episodes,
                            axis=axis,
                            level=level,
                            seed0=args.seed0,
                            episode_steps=args.steps,
                        )
                    )

                elif name == "ppo":

                    rows.extend(
                        run_ppo(
                            model=models["ppo"],
                            episodes=args.episodes,
                            axis=axis,
                            level=level,
                            seed0=args.seed0,
                            episode_steps=args.steps,
                            morphology_observation=True,
                            label=CONTROLLER_LABELS["ppo"],
                        )
                    )

                elif name == "ppo_nomorph":

                    rows.extend(
                        run_ppo(
                            model=models["ppo_nomorph"],
                            episodes=args.episodes,
                            axis=axis,
                            level=level,
                            seed0=args.seed0,
                            episode_steps=args.steps,
                            morphology_observation=False,
                            label=CONTROLLER_LABELS["ppo_nomorph"],
                        )
                    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_results(
        rows,
        args.output,
    )

    print()
    print("=" * 70)
    print(
        f"Wrote {len(rows)} episode results"
    )
    print(
        f"Output: {args.output}"
    )
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()