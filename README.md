# Morphology-Conditioned Locomotion Under Physical Variation

Code and data for the paper

> **Morphology-Conditioned Locomotion Policies Under Physical Variation: A Controlled Simulation Study**

The project asks, in simulation, whether a locomotion policy that observes the robot's own physical parameters performs more robustly under physical variation than a policy trained using the same procedure but without access to those parameters. A prescribed intention schedule (Forward, Left, Right, Stop) drives a quadruped in MuJoCo **Ant-v5** while mass, limb length, joint strength and friction are perturbed by 0–50%. A four-class EEG intention decoder is evaluated separately on **synthetic** data and is **not connected** to the controllers.

## Main findings (5 training seeds per PPO arm)

Mean over the 21 distinct conditions (nominal plus four axes at five levels), 10 episodes per condition.

| Controller | Task reward | Fall rate |
|---|---|---|
| Stand-still reference | 149.8 | 0% |
| Direct (open loop) | 123.4 | 0% |
| CPG-style + PD feedback | 106.1 | 27.6% |
| Morphology-aware PPO | 302.6 ± 52.6 | 14.2% |
| Morphology-blind PPO | 311.5 ± 39.6 | 9.9% |

PPO rows: mean ± SD over five training seeds. Task reward excludes the fall penalty, so fall rate is always reported beside it.

- **Morphology conditioning showed no statistically significant advantage.** Aware minus blind: −8.9 task-reward points (95% Welch CI [−77.8, +59.9], p = 0.770) and +4.3 percentage points in fall rate (95% Welch CI [−3.2, +11.7] pp, p = 0.219). With only five training seeds per arm, the confidence intervals remain wide; these results do not establish equivalence between the policies.
- A zero-torque reference exposed a flaw in the initial reward, which allowed standing still to score too highly. After correcting the reward, both PPO policies learned locomotion, whereas the tested hand-built baselines did not achieve sustained locomotion.
- The EEG decoder reaches 68.75% five-fold cross-validated accuracy (macro-F1 68.79%) on synthetic data only.

The full numbers are in `results/aggregate_report_v2.txt`.

## What is implemented

- **Synthetic EEG decoder.** 576 synthetic motor-imagery-style trials (22 channels, 250 Hz, 4 s), log Welch band-power features (8–30 Hz), standardization and linear discriminant analysis, evaluated with 5-fold stratified cross-validation. This is a synthetic-data result, not real-EEG performance.
- **Environment.** `MorphologyAntEnv`, a Gymnasium wrapper around Ant-v5 that adds an intention one-hot, four morphology parameters, a baseline-subtracted intention-specific reward, fall detection with episode termination, and morphology perturbation.
- **Controllers.** A zero-torque stand-still reference, a Direct open-loop controller, a CPG-style controller (sinusoidal references with joint-level PD feedback), and two PPO policies (Stable-Baselines3) trained with identical procedures: one observes the morphology vector, one does not.
- **Protocol.** Four perturbation axes × six levels × 10 episodes per controller and training seed, five training seeds per PPO arm.
- **Analysis.** Per-seed aggregation, primary unpaired Welch
  comparisons with 95% confidence intervals, leave-one-seed-out
  checks, and figure generation scripts.
  
## Important design notes

- **The decoder is not connected to the controllers.** The controller experiments use a fixed schedule (Forward → Left → Right → Stop, 150 steps each, repeating over the 1000-step episode). Decoder errors never reach them.
- **Simulation.** Physics timestep 0.01 s with `frame_skip = 5`, so one control step lasts 0.05 s; episodes last up to 1000 steps (50 s). A fall (torso height < 0.25 m or uprightness < 0.45) ends the episode.
- **Observation.** 105 (Ant) + 4 (morphology) + 4 (intention one-hot) = 113 dimensions; 109 for the morphology-blind policy.
- **Morphology vector.** `[mass, limb_length, joint_strength, friction]`, nominal `1.0`. Perturbation raises mass and limb length and lowers joint strength and friction.
- **Intentions.** `0` Forward, `1` Left, `2` Right, `3` Stop.

## Installation

Python 3.10 or newer (developed with 3.12).

```bash
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the main packages (Gymnasium 1.4.0, MuJoCo 3.17.0, Stable-Baselines3 2.9.2, PyTorch 2.15.0, NumPy 2.5.4, SciPy, scikit-learn, pandas, matplotlib). `requirements_frozen.txt` lists every installed package. NumPy 2.x is required.

Run everything **from the repository root**. If you see `ModuleNotFoundError: No module named 'src'`, run the script as a module (for example `python -m scripts.run_eeg`) or set the root on the Python path:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "."
# macOS / Linux
export PYTHONPATH=.
```

## Quick start

A fast check that the environment and the hand-built controllers work (no trained model needed):

```bash
python scripts/run_experiments.py --controllers direct cpg standstill --episodes 2 --levels 0 0.5 --output results/smoke.csv
```

## Reproducing the paper

Training takes about 3.5 hours per seed for both arms on a laptop CPU.

**1. Tune the hand-built baselines** (nominal morphology, full schedule, seeds 200–209):

```bash
python scripts/tune_baselines.py
```

Writes `results/tuning_direct.csv` and `results/tuning_cpg.csv`.

**2. Train both arms for each seed.** Stage 0: Forward-only, 1M steps. Stage 1: four intentions, 3M steps, learning rate 3e-4. Stage 2: continued, 4M steps, learning rate 1e-4. Then checkpoint selection (16 nominal and 16 random-morphology episodes, fall-rate limit 0.10) writes `models/final_<arm>_s<seed>.zip`:

```bash
python scripts/run_pipeline.py --seeds 42
python scripts/run_pipeline.py --seeds 43     # likewise 44, 45, 46 (one terminal per seed)
```

**3. Per-intention evaluation** (each intention held fixed, 20 episodes of 300 steps):

```bash
python scripts/evaluate_intentions.py --model models/final_aware_s42.zip --episodes 20 > results/intent_aware_s42.txt
python scripts/evaluate_intentions.py --model models/final_nomorph_s42.zip --episodes 20 --no-morph > results/intent_nomorph_s42.txt
```

**4. Perturbation experiments** (21 conditions, 10 episodes each, evaluation seeds 1000–1009). Seed 42 also runs the three deterministic references; the other seeds need only the two PPO arms:

```bash
python scripts/run_experiments.py --controllers direct cpg ppo ppo_nomorph standstill --model models/final_aware_s42.zip --model-nomorph models/final_nomorph_s42.zip --output results/physics_results_s42.csv
python scripts/run_experiments.py --controllers ppo ppo_nomorph --model models/final_aware_s43.zip --model-nomorph models/final_nomorph_s43.zip --output results/physics_results_s43.csv
```

**5. Aggregate over seeds.** Each distinct condition is counted once (the nominal point is shared by all axes); the report includes Welch tests, 95% confidence intervals and leave-one-seed-out checks:

```bash
python scripts/aggregate_seeds.py > results/aggregate_report_v2.txt
```

**6. Figures:**

```bash
python scripts/make_paper_figures.py
python scripts/make_fig1.py
```

**7. Optional: nominal re-check on fresh start states** (seeds 2000–2019):

```bash
python scripts/nominal_check.py
```

**8. Synthetic EEG decoder:**

```bash
python -m scripts.run_eeg
```

## Repository layout

```
src/        env.py (MorphologyAntEnv), controllers.py, protocol.py (shared schedule and metrics),
            eeg.py (synthetic EEG decoder)
scripts/    run_pipeline.py, train_policy.py, run_experiments.py, evaluate_intentions.py,
            aggregate_seeds.py, tune_baselines.py, nominal_check.py, run_eeg.py,
            make_paper_figures.py, make_fig1.py, analyze_results.py, eval_forward.py
results/    per-seed CSVs, per-intention reports, checkpoint-selection and tuning records, aggregate report
models/     *_config.json for every training run (model weights are not committed)
figures/    paper figures
archive/    superseded scripts, kept for history only
eval_checkpoints.py      checkpoint selection (called by run_pipeline.py)
diagnose_leg_sides.py    checks the leg and actuator layout of Ant-v5
```

## Results files

| File | Content |
|---|---|
| `results/physics_results_s<seed>.csv` | episode-level results for each training seed |
| `results/seed_summary.csv`, `seed_overall.csv` | means, SDs and ranges across seeds |
| `results/morph_vs_nomorph_split.csv`, `leave_one_out.csv` | aware-minus-blind comparisons |
| `results/aggregate_report_v2.txt` | the report cited in the paper |
| `results/intent_*.txt` | per-intention behavior of each model |
| `results/nominal_check.csv` | nominal re-check on fresh start states |
| `results/checkpoint_selection_ft2_*.csv` | scores of the sampled checkpoints per model |
| `results/tuning_direct.csv`, `tuning_cpg.csv` | baseline tuning grids |

## Known limitations

- Trained model weights are not committed. Every run's settings are in `models/*_config.json`; in configs written before a later fix, the `learning_rate` field is a fixed 0.0003 for every stage, and the real value is in `learning_rate_setting`.
- The two PPO arms do not share an initialization, so "matched by seed" means matched by seed index only; the unpaired Welch test is the main test, and p-values are descriptive and uncorrected.
- Perturbation levels lie inside the training range, so the study tests robustness within a randomized distribution, not extrapolation to unseen bodies.
- One model (morphology-aware, seed 44) was selected with the fallback rule because no sampled checkpoint met the fall-rate limit.
- The Forward penalty weights (yaw 0.30, lateral 0.20) were chosen after Forward-only PPO tests, so they are a degree of freedom of the study.
- Simulation only: one quadruped, one perturbed parameter at a time, parameters fixed within an episode, no hardware, and synthetic EEG only.
- The earlier single-seed version of this project is preserved as the git tag `old-single-seed`; its results are superseded.

## Citation

If you use this code, please cite the paper. Publication details will be added when available.

```bibtex
@misc{jaiswal2026morphology,
  title  = {Morphology-Conditioned Locomotion Policies Under Physical Variation: A Controlled Simulation Study},
  author = {Jaiswal, Maitree},
  year   = {2026},
  url    = {https://github.com/maitreejais-pixel/neural-intent-morphology-robustness}
}
```
