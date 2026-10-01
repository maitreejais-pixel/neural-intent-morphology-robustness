# Updating the paper after running the simulator

Do NOT carry the old values 79.8%, 51.9%, 29.2%, or the old k values into the new paper.

Delete the reduced-order robustness equations:
- sigma_m, sigma_l, sigma_s, sigma_mu
- k_dir, k_cpg, k_ma
- m_eff
- analytical speed/fall/energy equations

Replace the locomotion Methods with the actual MuJoCo setup and report:
- simulator version
- Walker2d model/version
- timestep
- controller parameters
- PPO hyperparameters
- training duration
- domain-randomization ranges
- seeds
- episode length
- fall criterion
- energy definition

The morphology-aware controller must be trained without using evaluation outcomes for tuning.

Report results directly from `results/physics_results.csv` and `results/summary.csv`.

Use energy per distance as the primary energetic metric where possible.

The paper should continue to state that the EEG decoder is synthetic unless real EEG is actually evaluated.
