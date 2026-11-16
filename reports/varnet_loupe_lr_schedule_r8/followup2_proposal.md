# Follow-up 2 proposal: does the lr gain hold at 6 cascades? (VarNet + LOUPE, R=8)

**Status: approved by the user 2026-11-16 ("can we instead train a 6 cascade on the full data? Just try the best candidates
so far").**

## Why

At 12 cascades (30.1M parameters) and full data, `lr4e3_step` beats the 1e-3 control by +0.171 dB and `lr2e3_step` by +0.123 dB
(`followup_proposal.md`). Train loss is equal across arms (0.0222-0.0225 over epochs 91-100) while val loss is lower, so the gain
is in generalization. One reading is that a higher lr regularizes an over-parameterized network: 160 training volumes for
30M parameters. `brain_varnet_scaling` found 6 cascades tie 12 at 40-80 brain volumes. If the gain is regularization, it should
shrink at 6 cascades; if it is optimization, it should stay.

## Arms (6 runs)

Everything as `followup_proposal.md` (full local v1 FLAIR + knee, 100 epochs, val every 5, mask lr 1e-3, step at 90, x0.1)
with `recon.num_cascades=6`. The cosine arms are dropped (they lost to step at 12 cascades).

| arm | `training.lr` |
|---|---|
| `lr1e3_step` (**control**) | 1e-3 |
| `lr2e3_step` | 2e-3 |
| `lr4e3_step` | 4e-3 |

Seeds 42 and 43. Output `runs_full_c6/`, run names `vlls_full_c6_<arm>_s<seed>`.

## Scoring

Full local v1 val (80 volumes, 1,938 slices), `last_model.pt`, paired per slice vs the 6-cascade control of the same seed
(`scripts/eval/varnet_loupe_lr_schedule_r8_eval.py --runs-dir .../runs_full_c6`).

## Expected outcomes

- 6-cascade control within 0.1 dB of the 12-cascade control (33.77-33.81 dB pooled); a little lower on knee.
- Regularization reading: `lr4e3_step` gain at 6 cascades below +0.10 dB. Optimization reading: about +0.15 dB or more, as at 12.

## Success criteria (frozen)

- **lr effect at 6 cascades**: the same adopt rule as `followup_proposal.md` (mean >= +0.10 dB over 2 seeds, positive in
  both, no condition below -0.05 dB, SSIM not below -0.002). The largest passing mean wins.
- **Interaction** (reported): `d6 - d12` for `lr4e3_step` and `lr2e3_step`, where d is the seed-mean paired gain vs the control
  of the same cascade count. The gain "shrinks" if `d6 - d12 <= -0.08 dB` with both seeds' d6 below their d12; "holds"
  if |`d6 - d12`| < 0.08 dB; otherwise inconclusive.
- **Cascade count** (reported, unpaired): 6-cascade minus 12-cascade control, pooled and per condition, per seed.

## Compute

CoreHPC, RTX Pro 6000, one GPU per run, `CASCADES=6 sbatch --array=0-5 --time=04:00:00 followup.sbatch`. About half of the
12-cascade epoch time, so about 2 h per run. Crashed runs are rerun in full.
