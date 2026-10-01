import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env import MorphologyAntEnv

# NOTE: No functional changes were needed in this file.
#
# It already called super().__init__(morphology_observation=True) and
# already samples morphology as a 4-element NumPy array in the order
# [mass, limb_length, joint_strength, friction]. Both match env.py's
# corrected parameter name and reset()'s array-accepting logic, so
# this file works unchanged once env.py is replaced with the fixed
# version. Retraining with the fixed env.py is required to pick up
# the corrected Left/Right yaw reward targets, since the previously
# saved model was trained against the swapped targets.


# ============================================================
# Morphology sampling
# ============================================================

def sample_morphology(rng):
    """
    Sample randomized physical morphology parameters.

    Parameter order:
        [mass, limb_length, joint_strength, friction]

    Training ranges:
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
    3. Intention conditioning

    The policy therefore receives:

        Ant state
        + current morphology
        + intention

    and learns a morphology-conditioned motor policy.
    """

    def __init__(self, seed=None):

        super().__init__(
            morphology_observation=True
        )

        self.intention = 0

        # Independent random-number generator for
        # reproducible morphology sampling.
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

    def set_intention(self, intention):

        super().set_intention(
            intention
        )


# ============================================================
# Environment factory
# ============================================================

def make_env(rank):

    def _init():

        return RandomizedMorphologyAntEnv(
            seed=42 + rank
        )

    return _init


# ============================================================
# Training
# ============================================================

def main():

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "models/checkpoints",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Vectorized environments
    # --------------------------------------------------------

    n_envs = 8

    env = DummyVecEnv(
        [
            make_env(i)
            for i in range(n_envs)
        ]
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            100_000 // n_envs,
            1
        ),
        save_path="models/checkpoints",
        name_prefix="morphology_ant_ppo",
    )

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    model = PPO(
        "MlpPolicy",
        env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=256,

        gamma=0.99,

        gae_lambda=0.95,

        ent_coef=0.01,

        verbose=1,

        seed=42,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    total_timesteps = 5_000_000

    print()
    print("=" * 70)
    print("Starting morphology-aware PPO training")
    print("=" * 70)

    print(
        f"Parallel environments : {n_envs}"
    )

    print(
        f"Total timesteps       : {total_timesteps:,}"
    )

    print(
        "Morphology range      : "
        "mass [0.5,1.5], "
        "limb length [0.5,1.5], "
        "strength [0.5,1.0], "
        "friction [0.5,1.0]"
    )

    print(
        "Intentions             : "
        "Forward, Left, Right, Stop"
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

    model_path = (
        "models/morphology_ant_ppo"
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