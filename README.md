# Neural Intent → Morphology-Aware Locomotion

Physics-based research prototype for the paper:
**Decoupling Neural Intent from Body-Specific Motor Control: A Physics-Based, EEG-Grounded Study of Robustness to Physical Variation in Legged Locomotion**

This project replaces the paper's analytical locomotion proxy with MuJoCo physics while retaining the frozen EEG-intention-stream design.

## Implemented
- Synthetic 4-class EEG
- Welch log-band-power + LDA
- Frozen intention stream
- MuJoCo Walker2d-v5
- Direct trajectory baseline
- Fixed-gain CPG
- Morphology-conditioned PPO policy
- Mass, leg-length, actuator-strength and friction perturbations
- Distance, falls and energy logging
- CSV results and plotting

## Install
Python 3.10+:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python scripts/run_eeg.py
python scripts/train_policy.py
python scripts/run_experiments.py --episodes 50
python scripts/analyze_results.py
```

Quick smoke test:
```bash
python scripts/run_experiments.py --episodes 3 --levels 0 0.5
```

Results are written to `results/physics_results.csv`.

**Do not put numerical results into the paper until the experiments have actually been run.**
