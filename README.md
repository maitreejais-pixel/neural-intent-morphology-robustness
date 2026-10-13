# Neural Intent → Morphology-Aware Locomotion

Code and data for the paper

> **Decoupling Neural Intent from Body-Specific Motor Control: A Synthetic-EEG Study of Robustness to Physical Variation in Full-Physics Simulated Legged Locomotion**

The project asks whether separating *what a user intends* (a discrete intention such as Forward, Left, Right, Stop) from *how a particular body executes it* makes a legged robot more robust when its physical parameters change. It compares three controllers, plus a zero-torque reference, on the MuJoCo **Ant-v5** quadruped under systematic perturbations of mass, limb length, joint strength, and friction.

## Main findings

Mean over 10 episodes per condition, 0–50% perturbation on each of four axes (21 distinct conditions).

| Controller | Nominal reward | Falls |
|---|---|---|
| (A) Direct, open-loop actuator commands | 650.2 | 1 of 10 episodes at 40% limb length, none elsewhere |
| (B) Fixed-gain CPG + local PD feedback | −13.9 | ≥ 90% of episodes in all but two conditions (60% at 50% joint strength, 50% at 50% limb length), including nominal |
| (C) Morphology-aware PPO | 648.9 | none |
| (R) Zero-torque stand-still reference | 672.1 | none |

- PPO has slightly higher mean reward than the open-loop controller under friction and joint-strength perturbation, is comparable under mass perturbation, and is substantially worse under limb-length perturbation (58.3% vs. 94.9% retained reward at 50%).
- **A controller that does nothing scores higher than both Direct and PPO in all 21 conditions**, so on this reward neither shows sustained locomotion. Read the reward and fall-rate results with that in mind.

## What is implemented

- **Synthetic EEG decoder.** 4-class synthetic motor-imagery EEG (22 channels, 250 Hz, 4 s trials, 576 trials), Welch log band-power features (8–30 Hz), standardization, and linear discriminant analysis, evaluated with 5-fold stratified cross-validation (68.75% accuracy, 68.79% macro-F1). This is a synthetic-data result, not real-EEG performance.
- **Environment.** `MorphologyAntEnv`, a Gymnasium wrapper around Ant-v5 that adds an intention one-hot, four morphology parameters, an intention-specific reward, fall detection with episode termination, and morphology perturbation.
- **Controllers.** Direct open-loop actuator commands, a fixed-gain CPG with local PD feedback, a morphology-conditioned PPO policy (Stable-Baselines3), and a zero-torque stand-still reference.
- **Protocol.** Four perturbation axes × six levels × 10 episodes per controller, logging reward, falls, energy, and displacement.
- **Analysis.** Summary tables, reward-minus-stand-still comparison, and publication figures generated directly from the results CSV.

## Important design notes

- **The decoder is not connected to the controllers.** The controller experiments use a fixed, prescribed intention schedule (Forward → Left → Right → Stop, 150 steps each, repeating over the 1000-step episode). Decoder errors never reach them. Integrating the decoded stream is future work.
- **Simulation details.** Physics timestep 0.01 s, `frame_skip = 5`, so one control step lasts 0.05 s; episodes are 1000 steps (50 s). A fall (torso height < 0.25 m or uprightness < 0.45) ends the episode.
- **Observation.** 105 (Ant) + 4 (morphology) + 4 (intention one-hot) = 113 dimensions for the PPO policy.
- **Morphology vector order.** `[mass, limb_length, joint_strength, friction]`, nominal `1.0`. Perturbation *raises* mass and limb length and *lowers* joint strength and friction.
- **Intentions.** `0` Forward, `1` Left, `2` Right, `3` Stop.

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt
```

The scripts import the `src` package, so run everything **from the repository root** with the root on the Python path:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."
# macOS / Linux
export PYTHONPATH=.
```

If you see `ModuleNotFoundError: No module named 'src'`, this is the cause.

## Quick start

A fast check that the environment and controllers work (no trained model needed):

```bash
python scripts/run_experiments.py --controllers direct cpg standstill --episodes 2 --levels 0 0.5 --output results/smoke.csv
```

## Reproducing the paper

All commands run from the repository root. Reported numbers come from `results/summary.csv`; figures are generated from it, not typed in.

**1. EEG decoder:**

```bash
python scripts/run_eeg.py
```

Prints accuracy, macro-F1, and the confusion matrix (68.75% / 68.79%, Left recall 80.6%).

**2. Train the PPO policy** (5M steps, about 80 minutes on a laptop CPU):

```bash
python scripts/train_policy.py
```

Saves a checkpoint every 100k steps in `models/checkpoints/` and the final model as `models/morphology_ant_ppo.zip`. **Back up `models/` first**: training overwrites both. The training run uses seed 42, 8 vectorized environments, `MlpPolicy` (two hidden layers of 64 units), and samples morphology per episode (mass and limb length from U(0.5, 1.5), joint strength and friction from U(0.5, 1.0)) with one intention per episode.

**3. Choose a checkpoint** (optional):

```bash
python eval_checkpoints.py
```

Scores 10 evenly spaced checkpoints with a heuristic error score at nominal morphology. The score is only a coarse indicator and cannot distinguish standing from locomotion. The policy reported in the paper is the **2.2M-step checkpoint**; to use it:

```bash
# Windows (PowerShell)
Copy-Item models\checkpoints\morphology_ant_ppo_2200000_steps.zip models\morphology_ant_ppo.zip -Force
# macOS / Linux
cp models/checkpoints/morphology_ant_ppo_2200000_steps.zip models/morphology_ant_ppo.zip
```

**4. Run the experiments:**

```bash
python scripts/run_experiments.py --output results/physics_results.csv
python scripts/run_experiments.py --controllers standstill --output results/standstill_results.csv
```

The first command runs Direct, CPG, and PPO (default 10 episodes, levels 0 to 0.5 in steps of 0.1, 720 episode records). The second runs the stand-still reference (240 records). Merge them:

```bash
python -c "import pandas as pd; a=pd.read_csv('results/physics_results.csv'); b=pd.read_csv('results/standstill_results.csv'); pd.concat([a,b]).to_csv('results/physics_results_with_standstill.csv', index=False)"
```

**5. Analyze and plot:**

```bash
python scripts/analyze_results.py --input results/physics_results_with_standstill.csv --outdir results_with_standstill
python scripts/make_figures.py
```

`analyze_results.py` writes `summary.csv`, the reward-minus-stand-still table, per-axis plots, and the best-controller table. `make_figures.py` writes Figures 3–5 (reward, limb-length fall rate, mass-axis energy) to `figures/`.

### Reproducibility notes

- Direct, CPG, and the PPO policy are deterministic. Episodes within a condition differ only through the simulator's small reset noise.
- Each controller uses a fixed block of initial-state seeds (Direct 1000–1009, CPG 2000–2009, PPO 3000–3009; the stand-still reference uses 1000–1009), reused for every axis and level. The 0% level is therefore identical across axes, giving 630 unique episodes for the three controllers (210 per controller).
- Retraining may not reproduce the reported checkpoint exactly (library versions and hardware can change results), so keep the trained model file with your results.

## Repository layout

```
src/
  env.py                  MorphologyAntEnv: reward, falls, morphology perturbation
  controllers.py          Direct, fixed-gain CPG + PD, PPO wrapper, stand-still reference
  eeg.py                  Synthetic EEG generation, log band-power features, LDA decoder
scripts/
  run_eeg.py              Evaluate the synthetic-EEG decoder
  train_policy.py         Train the morphology-conditioned PPO policy
  run_experiments.py      Run the perturbation protocol (--controllers, --episodes, --levels, --output)
  analyze_results.py      Summaries and plots (--input, --outdir)
  make_figures.py         Paper figures 3-5 from summary.csv
  evaluate_intentions.py  Behavior of one policy under each intention
eval_checkpoints.py       Score saved PPO checkpoints
eval_all_intentions.py    Compare CPG gain sets across all four intentions
models/                   Trained policy and checkpoints
results/                  Per-episode results and summaries
```

Development diagnostics (`diagnose_leg_sides.py`, `scripts/test_actuators.py`, `scripts/test_leg_mapping.py`, `scripts/test_yaw.py`, `test_cpg.py`, `test_intentions.py`, `tune_cpg*.py`, `robust_check.py`) are kept for transparency about how the leg mapping, yaw convention, and CPG gains were checked.

## Known limitations

- **Reward calibration.** A zero-torque policy scores above Direct and PPO, so reward near 650 and zero falls reflect standing stability, not locomotion.
- **Controller time step.** Direct and CPG advance their oscillator phase by 0.01 s per control step, whereas a control step lasts 0.05 s, so their gaits run at 0.3 Hz in real time. Correcting this changes their behavior and requires retuning.
- **Limb-length perturbation is partial.** It scales the hip-attachment offsets and the half-length of each leg capsule, but not capsule centers or the hip-to-ankle offsets. Mass scaling leaves body inertia unchanged.
- **Training/evaluation mismatch.** PPO was trained with one intention per episode but evaluated with intention switching every 150 steps.
- **One training run, one seed.** The PPO result may not generalize across seeds. The checkpoint was chosen from 10 candidates of one run.
- **Synthetic EEG only.** No real BCI recordings are used, and there is no hardware validation.

## Citation

If you use this code, please cite the paper. The author list and publication details will be added when available.

```bibtex
@misc{neural_intent_morphology_robustness,
  title        = {Decoupling Neural Intent from Body-Specific Motor Control: A Synthetic-EEG Study of Robustness to Physical Variation in Full-Physics Simulated Legged Locomotion},
  author       = {<Author names>},
  howpublished = {GitHub repository},
  url          = {https://github.com/maitreejais-pixel/neural-intent-morphology-robustness}
}
```
