"""
Train a PPO policy on the intention-switching task.

Run from the project root. Examples:

    # morphology-aware policy (morphology vector in the observation)
    python scripts/train_policy.py --name ppo_seed42 --seed 42

    # ablation: identical training (same domain randomization, same
    # intention switching) but the morphology vector is NOT observed
    python scripts/train_policy.py --name ppo_nomorph_seed42 --seed 42 --no-morph

    # more training seeds, for confidence intervals across runs
    python scripts/train_policy.py --name ppo_seed43 --seed 43
    python scripts/train_policy.py --name ppo_seed44 --seed 44

    # reward-shaping variants for the Forward-only test (see eval_forward.py)
    python scripts/train_policy.py --name fwd_lowpen --forward-only --ent-coef 0.0 --timesteps 1000000 --forward-yaw-weight 0.3 --lateral-weight 0.2
    python scripts/train_policy.py --name fwd_smooth --forward-only --ent-coef 0.0 --timesteps 1000000 --penalty-tau 1.0

    # sanity test: can PPO learn to WALK FORWARD at all with this reward?
    # (intention fixed to Forward, no switching)
    python scripts/train_policy.py --name fwd_test --forward-only --ent-coef 0.0 --timesteps 1000000
    python scripts/eval_forward.py --model models/fwd_test.zip

    # no entropy bonus (the first runs used --ent-coef 0.01, the default)
    python scripts/train_policy.py --name ppo_ent0_seed42 --seed 42 --ent-coef 0.0

    # warm start: begin the four-intention run from the Forward-only policy
    # that learned to walk (same observation layout), then fine-tune
    python scripts/train_policy.py --name multi_warm --ent-coef 0.0 --init-from models/fwd_lowpen.zip --timesteps 3000000

    # quick test of the whole pipeline before the long run
    # (needs --timesteps of at least 100000 for a checkpoint to be saved)
    python scripts/train_policy.py --name smoke --timesteps 120000

Outputs:
    models/checkpoints/<name>/   checkpoint every 100k timesteps
    models/<name>.zip            final model
    models/<name>_config.json    every setting of the run (for the paper)

CHANGES vs. the previous version
--------------------------------
1. Intention now SWITCHES during training: a new random intention every
   150 steps (the evaluation schedule switches every 150 steps). It used
   to be fixed for the whole episode, which created a train/evaluation
   mismatch.
2. --no-morph trains the domain-randomization-only ablation, which is
   what isolates the effect of conditioning on the morphology vector.
3. --seed, --name and --timesteps are command-line arguments (the
   timestep budget used to be hardcoded). Checkpoints are saved in a
   folder named after the run, so runs do not overwrite each other.
4. Rewards are normalized with VecNormalize (rewards only; observations
   are left untouched, so no normalization statistics are needed when
   the model is evaluated).
5. The reward in src/env.py changed (baseline-subtracted tracking), so
   every previously trained model is obsolete: retrain from scratch.
6. --ent-coef sets the PPO entropy bonus (default 0.01, as in the first
   runs, so old runs stay reproducible). The first 5M-step runs showed
   the action standard deviation rising from 1.0 to about 1.6, a sign
   that the bonus dominated a weak reward signal.
7. --forward-only trains with the intention fixed to Forward and no
   switching: a sanity test of whether this reward and setup can
   produce forward walking at all. Evaluate it with
   scripts/eval_forward.py (the full-schedule evaluation would score
   a Forward-only policy badly on Left/Right/Stop by construction).
8. Every run writes models/<name>_config.json with all its settings.
9. src/env.py now starts every limited joint inside its range (see its
   docstring, change 5); models trained before that change saw a
   different start pose, so retrain.
10. --init-from starts training from the weights of an existing model
   (warm start / curriculum), e.g. the Forward-only policy that learned
   to walk. The observation layout must match (same --no-morph setting).
   The optimizer state and the reward-normalization statistics start
   fresh. --learning-rate sets the PPO learning rate (default 3e-4).
   Both are saved in models/<name>_config.json.
"""

import argparse
import json
import os
import sys

# Make "from src..." work whether this file is run as
# "python scripts/train_policy.py" or "python -m scripts.train_policy".
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env import MorphologyAntEnv
from src.protocol import SEGMENT_STEPS


# ============================================================
# Morphology sampling
# ============================================================

def sample_morphology(rng):
    """
    Sample randomized physical morphology parameters.

    Parameter order:
        [mass, limb_length, joint_strength, friction]

    Training ranges (these match the perturbation ranges that are
    evaluated: mass and limb length go UP to +50%, joint strength and
    friction go DOWN to -50%):
        mass:
            0.5 - 1.5

        limb length:
            0.5 - 1.5

        joint strength:
            0.5 - 1.0

        friction:
            0.5 - 1.0
    """

    return np.array(
        [
            rng.uniform(0.5, 1.5),
            rng.uniform(0.5, 1.5),
            rng.uniform(0.5, 1.0),
            rng.uniform(0.5, 1.0),
        ],
        dtype=np.float32,
    )


# ============================================================
# Randomized MuJoCo Ant environment
# ============================================================

class RandomizedMorphologyAntEnv(MorphologyAntEnv):
    """
    MuJoCo Ant environment with:

    1. Randomized morphology
    2. Morphology parameters included in the observation
       (unless morphology_observation=False, the ablation)
    3. Intention conditioning with SWITCHING intentions

    The policy therefore receives:

        Ant state
        + current morphology   (morphology-aware policy only)
        + intention

    and learns a motor policy that has to follow a changing intention.

    Intention switching: at reset a random intention is drawn; every
    `switch_every` steps a different random intention replaces it, and
    the observation is refreshed so the policy sees the new intention
    on the very next action.
    """

    def __init__(
        self,
        seed=None,
        morphology_observation=True,
        switch_every=SEGMENT_STEPS,
        forward_only=False,
        **env_kwargs,
    ):

        super().__init__(
            morphology_observation=morphology_observation,
            **env_kwargs,
        )

        self.intention = 0

        self.switch_every = int(
            switch_every
        )

        # Sanity-test mode: the intention is always Forward.
        self.forward_only = bool(
            forward_only
        )

        # Independent random-number generator for
        # reproducible morphology / intention sampling.
        if seed is None:
            seed = 42

        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed=None,
        options=None,
        morphology=None,
        intention=None,
    ):
        """
        Reset the environment.

        If morphology is not explicitly supplied,
        sample a new morphology.

        If intention is not explicitly supplied,
        sample one of the four intentions.
        """

        # Respect Gymnasium's reset seed.
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # ----------------------------------------------------
        # Morphology
        # ----------------------------------------------------

        if morphology is None:
            morphology = sample_morphology(
                self.rng
            )

        # ----------------------------------------------------
        # Intention
        # ----------------------------------------------------

        if intention is None:

            if self.forward_only:
                intention = 0

            else:
                intention = int(
                    self.rng.integers(
                        0,
                        4
                    )
                )

        # ----------------------------------------------------
        # Parent environment reset
        # ----------------------------------------------------

        return super().reset(
            seed=seed,
            options=options,
            morphology=morphology,
            intention=intention,
        )

    def step(self, action):
        """
        Step the environment and switch the intention every
        `switch_every` steps.
        """

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = super().step(
            action
        )

        if (
            not self.forward_only
            and not (terminated or truncated)
            and self.step_count % self.switch_every == 0
        ):

            # Pick a DIFFERENT intention, so every switch is a real
            # change.
            choices = [
                i
                for i in range(4)
                if i != self.intention
            ]

            self.set_intention(
                int(
                    self.rng.choice(
                        choices
                    )
                )
            )

            # The base Ant state is unchanged; only the intention
            # one-hot needs refreshing.
            obs = self.get_current_observation()

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def set_intention(self, intention):

        super().set_intention(
            intention
        )


# ============================================================
# Environment factory
# ============================================================

def make_env(
    rank,
    base_seed,
    morphology_observation,
    forward_only=False,
    env_kwargs=None,
):

    def _init():

        return RandomizedMorphologyAntEnv(
            seed=base_seed + rank,
            morphology_observation=morphology_observation,
            forward_only=forward_only,
            **(env_kwargs or {}),
        )

    return _init


# ============================================================
# Training
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train a PPO policy on the intention-switching task."
    )

    parser.add_argument(
        "--name",
        default="morphology_ant_ppo",
        help="Run name: models/<name>.zip and "
        "models/checkpoints/<name>/.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed (PPO and environment sampling).",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=5_000_000,
        help="Total training timesteps.",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments.",
    )

    parser.add_argument(
        "--no-morph",
        action="store_true",
        help="Ablation: do NOT include the morphology vector in the "
        "observation (domain randomization only).",
    )

    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy bonus. Default 0.01 (the first runs); use "
        "0.0 for no exploration bonus.",
    )

    parser.add_argument(
        "--forward-yaw-weight",
        type=float,
        default=None,
        help="Override the Forward yaw-rate penalty weight "
        "(default 0.30).",
    )

    parser.add_argument(
        "--lateral-weight",
        type=float,
        default=None,
        help="Override the Forward lateral-speed penalty weight "
        "(default 0.20).",
    )

    parser.add_argument(
        "--penalty-tau",
        type=float,
        default=None,
        help="Smooth the Forward yaw and lateral penalties with an "
        "exponential moving average of this time constant in "
        "seconds (default: none = instantaneous).",
    )

    parser.add_argument(
        "--init-from",
        default=None,
        help="Start from the weights of this saved model (warm start). "
        "Its observation layout must match this run.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate (default 3e-4).",
    )

    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Sanity test: intention fixed to Forward, no switching. "
        "Evaluate with scripts/eval_forward.py.",
    )

    args = parser.parse_args()

    morphology_observation = not args.no_morph

    forward_only = args.forward_only

    # Reward-shaping overrides, passed to every environment.
    env_kwargs = {}

    if args.forward_yaw_weight is not None:
        env_kwargs["forward_yaw_weight"] = args.forward_yaw_weight

    if args.lateral_weight is not None:
        env_kwargs["forward_lateral_weight"] = args.lateral_weight

    if args.penalty_tau is not None:
        env_kwargs["penalty_tau"] = args.penalty_tau

    checkpoint_dir = os.path.join(
        "models",
        "checkpoints",
        args.name,
    )

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Vectorized environments
    # --------------------------------------------------------

    n_envs = args.n_envs

    env = DummyVecEnv(
        [
            make_env(
                i,
                args.seed,
                morphology_observation,
                forward_only,
                env_kwargs,
            )
            for i in range(n_envs)
        ]
    )

    # Reward normalization only. Observations are NOT normalized,
    # so evaluation needs no extra statistics.
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        gamma=0.99,
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            100_000 // n_envs,
            1
        ),
        save_path=checkpoint_dir,
        name_prefix=args.name,
    )

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    model = PPO(
        "MlpPolicy",
        env,

        learning_rate=args.learning_rate,

        n_steps=2048,

        batch_size=256,

        gamma=0.99,

        gae_lambda=0.95,

        ent_coef=args.ent_coef,

        verbose=1,

        seed=args.seed,
    )

    # --------------------------------------------------------
    # Warm start: replace the fresh model by an existing one.
    # --------------------------------------------------------

    if args.init_from:

        if not os.path.exists(args.init_from):
            raise FileNotFoundError(
                f"--init-from model not found: {args.init_from}"
            )

        model = PPO.load(
            args.init_from,
            env=env,
        )

        # Apply this run's settings to the loaded model.
        model.ent_coef = args.ent_coef
        model.learning_rate = args.learning_rate
        model._setup_lr_schedule()
        model.set_random_seed(args.seed)

    # --------------------------------------------------------
    # Record every setting of this run (written BEFORE training, so
    # it exists even if the run is interrupted).
    # --------------------------------------------------------

    config = {
        "name": args.name,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "n_envs": n_envs,
        "morphology_observed": morphology_observation,
        "forward_only": forward_only,
        "ent_coef": args.ent_coef,
        "learning_rate": args.learning_rate,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "segment_steps": SEGMENT_STEPS,
        "reward_normalization": "VecNormalize(rewards only)",
        "start_in_joint_range": True,
        "forward_yaw_weight": (
            args.forward_yaw_weight
            if args.forward_yaw_weight is not None
            else MorphologyAntEnv.FORWARD_YAW_WEIGHT
        ),
        "forward_lateral_weight": (
            args.lateral_weight
            if args.lateral_weight is not None
            else MorphologyAntEnv.FORWARD_LATERAL_WEIGHT
        ),
        "penalty_tau": args.penalty_tau,
        "init_from": args.init_from,
        "learning_rate_setting": args.learning_rate,
    }

    with open(
        os.path.join("models", f"{args.name}_config.json"),
        "w",
    ) as f:
        json.dump(config, f, indent=2)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    total_timesteps = args.timesteps

    print()
    print("=" * 70)
    print(f"Starting PPO training: {args.name}")
    print("=" * 70)

    print(
        f"Seed                  : {args.seed}"
    )

    print(
        f"Parallel environments : {n_envs}"
    )

    print(
        f"Total timesteps       : {total_timesteps:,}"
    )

    print(
        f"Morphology observed   : {morphology_observation}"
    )

    print(
        f"Entropy coefficient   : {args.ent_coef}"
    )

    print(
        f"Forward only          : {forward_only}"
    )

    print(
        f"Learning rate         : {args.learning_rate}"
    )

    print(
        f"Warm start from       : {args.init_from or 'none (fresh)'}"
    )

    print(
        "Forward penalties     : "
        f"yaw weight {args.forward_yaw_weight if args.forward_yaw_weight is not None else MorphologyAntEnv.FORWARD_YAW_WEIGHT}, "
        f"lateral weight {args.lateral_weight if args.lateral_weight is not None else MorphologyAntEnv.FORWARD_LATERAL_WEIGHT}, "
        f"smoothing tau {args.penalty_tau if args.penalty_tau is not None else 'none'}"
    )

    print(
        "Morphology range      : "
        "mass [0.5,1.5], "
        "limb length [0.5,1.5], "
        "strength [0.5,1.0], "
        "friction [0.5,1.0]"
    )

    print(
        "Intentions            : "
        + (
            "Forward only (no switching)"
            if forward_only
            else "Forward, Left, Right, Stop "
            f"(switching every {SEGMENT_STEPS} steps)"
        )
    )

    print(
        f"Checkpoints           : {checkpoint_dir} "
        "(every 100,000 timesteps)"
    )

    print("=" * 70)
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model_path = os.path.join(
        "models",
        args.name,
    )

    model.save(
        model_path
    )

    print()
    print("=" * 70)
    print(
        "Training complete."
    )
    print(
        f"Saved: {model_path}.zip"
    )
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()