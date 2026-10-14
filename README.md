# Trying out active MRI


Current goals: 
- Grab acs lines, then predict sampling pattern. 
- This should be different for different anatomies


Future goals 
- [ ] More sequential undersampling (sample some k-space then predcit next to sample *hard* due to kspace reordering)
- [ ] Incorperating reordering into RL k-space sampling
- [ ] Motion compensation with this method? Auto gating? 


## Training

One entry point trains each sampler. Unset flags keep that model's YAML default; trailing `key=value`
arguments are Hydra overrides applied last.

```bash
python train.py --model loupe     --recon varnet --dataset brain_knee_balanced --acceleration 8
python train.py --model conv      --recon unet
python train.py --model reinforce --recon varnet --entropy-weight 0.005
python train.py --model ppo       --recon varnet --acceleration 8
python train.py --model dqn       --recon varnet --acceleration 4
python train.py --model loupe --recon varnet --print-config      # resolved config, no training
```

`--model sequential` runs the same trainer as loupe/conv/reinforce with `sampler.sampler_type` left to
an override (`caipi_2d`, `autosamp_2d`, the 1D samplers). Configs live in `configs/`
(`config.yaml` for loupe/conv/reinforce, `train_ppo.yaml`, `train_dqn.yaml`); the builders that turn
them into trainer configs are `activemri/training/builders.py` and `rl_builders.py`.

## Scripts

Run from anywhere; each script finds `activemri` and its siblings through `scripts/_path.py`.

| Folder | Contents |
| --- | --- |
| `scripts/launch/` | Shell launchers for the experiment runs and GPU queues (call `train.py`) |
| `scripts/eval/` | Re-evaluating checkpoints and comparing runs |
| `scripts/plots/` | Figures from finished runs |
| `scripts/diagnostics/` | Probes of why a learned mask looks the way it does |
| `scripts/oracle/` | CG-SENSE oracle and greedy-search studies |
| `scripts/legacy/` | Superseded trainers (joint, EMA-DDQN, single-slice, tuned REINFORCE) |
