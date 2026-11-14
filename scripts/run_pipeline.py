"""
Train the final models of the study with ONE identical, documented
protocol, for every seed and for both arms.

Arms:
    aware     the policy observes the morphology vector
    nomorph   ablation: same training, no morphology vector observed

Protocol per seed and arm (the curriculum is the same for both arms):

    Stage 0  Forward-only pretraining       1M steps  (--forward-only)
    Stage 1  four-intention fine-tuning     3M steps  lr 3e-4  (--init-from stage 0)
    Stage 2  continued fine-tuning          4M steps  lr 1e-4  (--init-from stage 1)
    Select   best stage-2 checkpoint by eval_checkpoints.py
             (mean task reward under the intention schedule, 16 selection
              episodes, fall rate <= 0.10) -> models/final_<arm>_s<seed>.zip

All stages use --ent-coef 0.0 and the default reward (Forward penalty
weights 0.30 / 0.20). Every run writes models/<name>_config.json.

Usage (from the project root):

    python scripts/run_pipeline.py --dry-run                  # print the plan
    python scripts/run_pipeline.py --seeds 42                 # one seed, both arms
    python scripts/run_pipeline.py --seeds 43 --arms aware    # one seed, one arm

To use several CPU cores, start one process per seed in separate
terminals (for example --seeds 42, --seeds 43, --seeds 44).

A finished stage is skipped when its output already exists (the pipeline
can be restarted after an interruption); --force redoes everything.
Output of each stage goes to logs/pipeline/<name>.log (--no-log streams
it to the console instead).

Approximate cost per seed and arm at ~1,300 steps/s: stage 0 about 13
minutes, stage 1 about 40, stage 2 about 55, selection a few minutes.
Keep the computer plugged in and awake while this runs.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN = os.path.join(ROOT, "scripts", "train_policy.py")
SELECT = os.path.join(ROOT, "eval_checkpoints.py")


def build_plan(
    seeds,
    arms,
    steps0=1_000_000,
    steps1=3_000_000,
    steps2=4_000_000,
    lr1=3e-4,
    lr2=1e-4,
    select_episodes=16,
    max_fall_rate=0.10,
):
    """
    The full list of steps. Each step is a dict with:
        desc    human-readable description
        name    log file / run name
        argv    command line (starts with the Python interpreter)
        output  file whose existence means the step is done
    """

    plan = []

    for seed in seeds:
        for arm in arms:

            flag = ["--no-morph"] if arm == "nomorph" else []

            n0 = f"fwd_{arm}_s{seed}"
            n1 = f"ft1_{arm}_s{seed}"
            n2 = f"ft2_{arm}_s{seed}"
            final = os.path.join("models", f"final_{arm}_s{seed}.zip")

            tag = f"[{arm}, seed {seed}]"

            plan.append(
                {
                    "desc": f"{tag} stage 0: Forward-only pretraining "
                    f"({steps0:,} steps)",
                    "name": n0,
                    "argv": [
                        sys.executable, TRAIN,
                        "--name", n0,
                        "--seed", str(seed),
                        "--forward-only",
                        "--ent-coef", "0.0",
                        "--timesteps", str(steps0),
                    ] + flag,
                    "output": os.path.join("models", f"{n0}.zip"),
                }
            )

            plan.append(
                {
                    "desc": f"{tag} stage 1: four-intention fine-tuning "
                    f"({steps1:,} steps, lr {lr1})",
                    "name": n1,
                    "argv": [
                        sys.executable, TRAIN,
                        "--name", n1,
                        "--seed", str(seed),
                        "--ent-coef", "0.0",
                        "--init-from", os.path.join("models", f"{n0}.zip"),
                        "--learning-rate", str(lr1),
                        "--timesteps", str(steps1),
                    ] + flag,
                    "output": os.path.join("models", f"{n1}.zip"),
                }
            )

            plan.append(
                {
                    "desc": f"{tag} stage 2: continued fine-tuning "
                    f"({steps2:,} steps, lr {lr2})",
                    "name": n2,
                    "argv": [
                        sys.executable, TRAIN,
                        "--name", n2,
                        "--seed", str(seed),
                        "--ent-coef", "0.0",
                        "--init-from", os.path.join("models", f"{n1}.zip"),
                        "--learning-rate", str(lr2),
                        "--timesteps", str(steps2),
                    ] + flag,
                    "output": os.path.join("models", f"{n2}.zip"),
                }
            )

            plan.append(
                {
                    "desc": f"{tag} checkpoint selection "
                    f"({select_episodes} + {select_episodes} episodes, "
                    f"fall rate <= {max_fall_rate})",
                    "name": f"select_{arm}_s{seed}",
                    "argv": [
                        sys.executable, SELECT,
                        "--name", n2,
                        "--episodes", str(select_episodes),
                        "--max-fall-rate", str(max_fall_rate),
                        "--copy-best-to", final,
                    ] + flag,
                    "output": final,
                }
            )

    return plan


def main():

    parser = argparse.ArgumentParser(
        description="Train the final models with one identical protocol."
    )

    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=["aware", "nomorph"],
        default=["aware", "nomorph"],
    )
    parser.add_argument("--steps0", type=int, default=1_000_000)
    parser.add_argument("--steps1", type=int, default=3_000_000)
    parser.add_argument("--steps2", type=int, default=4_000_000)
    parser.add_argument("--lr1", type=float, default=3e-4)
    parser.add_argument("--lr2", type=float, default=1e-4)
    parser.add_argument("--select-episodes", type=int, default=16)
    parser.add_argument("--max-fall-rate", type=float, default=0.10)
    parser.add_argument("--workdir", default=ROOT)
    parser.add_argument("--log-dir", default=os.path.join("logs", "pipeline"))
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    plan = build_plan(
        args.seeds,
        args.arms,
        args.steps0,
        args.steps1,
        args.steps2,
        args.lr1,
        args.lr2,
        args.select_episodes,
        args.max_fall_rate,
    )

    print("=" * 78)
    print(f"PIPELINE: seeds {args.seeds}, arms {args.arms}, "
          f"{len(plan)} steps")
    print("=" * 78)

    if args.dry_run:

        for i, step in enumerate(plan, 1):
            print(f"\n{i:2d}. {step['desc']}")
            print("    " + " ".join(step["argv"][1:]))

        print("\n(dry run: nothing was executed)")

        return

    os.makedirs(
        os.path.join(args.workdir, args.log_dir),
        exist_ok=True,
    )

    started = time.time()

    for i, step in enumerate(plan, 1):

        output = os.path.join(args.workdir, step["output"])

        if os.path.exists(output) and not args.force:
            print(f"[{i}/{len(plan)}] skip (exists): {step['desc']}")
            continue

        print(f"[{i}/{len(plan)}] start: {step['desc']}", flush=True)

        t0 = time.time()

        if args.no_log:

            result = subprocess.run(step["argv"], cwd=args.workdir)

        else:

            log_path = os.path.join(
                args.workdir, args.log_dir, f"{step['name']}.log"
            )

            with open(log_path, "w") as log:
                result = subprocess.run(
                    step["argv"],
                    cwd=args.workdir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

        minutes = (time.time() - t0) / 60.0

        if result.returncode != 0:

            print(f"FAILED after {minutes:.1f} min (exit code "
                  f"{result.returncode}).")

            if not args.no_log:

                print(f"Last lines of {log_path}:")

                with open(log_path) as log:
                    for line in log.readlines()[-15:]:
                        print("   " + line.rstrip())

            sys.exit(result.returncode)

        print(f"        done in {minutes:.1f} min", flush=True)

    print()
    print("=" * 78)
    print(f"All steps finished in {(time.time() - started) / 3600:.2f} h.")
    print("Final models:")

    for seed in args.seeds:

        for arm in args.arms:
            print(f"  models/final_{arm}_s{seed}.zip")

    print()
    print("Next, per seed (example for seed 42):")
    print("  python scripts/run_experiments.py --controllers direct cpg "
          "ppo ppo_nomorph standstill "
          "--model models/final_aware_s42.zip "
          "--model-nomorph models/final_nomorph_s42.zip "
          "--output results/physics_results_s42.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()