# Follow-up proposal: lr and schedule at full data, 100 epochs (VarNet + LOUPE, R=8)

**Status: approved by the user 2026-11-13 ("do option 2, full data 100 epochs with 4e-3 and cosine. Remove obvious bad
runs and only run about a quarter of it. Let current runs finish first before trying").**

## Why a follow-up

The first sweep (`proposal.md`, 20% data, 400 epochs) overfits in every arm: trainer val PSNR peaks at epoch 70-90 and
falls about 2.1 dB by epoch 400 while train loss keeps falling (val loss 0.034 -> 0.042). Its frozen criteria score
`last_model.pt` at epoch 400, so they rank overfitting tolerance, not optimization. Its interim trend: higher lr is
better at every schedule (5e-4 < 1e-3 < 2e-3), step > wsd > cosine at epoch 400, arms within about 0.2 dB at their peaks.

Same question, in the regime the main results use: full local v1 FLAIR + knee, 100 epochs (about 25k steps), where
the earlier 100-epoch runs were still improving at the end (`muon_layernorm_100ep`).

## Arms (10 runs, about a third of the first sweep's 33)

Dropped as clearly worse at epoch 400: every 5e-4 arm, every wsd arm, constant, and lr1e3_wcos. Added: 4e-3, and
cosine at the higher lrs (the user's hypothesis: cosine needs a higher peak lr). Mask-logit lr pinned at
`sampler.lr=0.001` as before; the schedule scales it too.

| arm | `training.lr` | schedule |
|---|---|---|
| `lr1e3_step` (**control**, main's default) | 1e-3 | step at 90, x0.1 |
| `lr2e3_step` | 2e-3 | step at 90 |
| `lr4e3_step` | 4e-3 | step at 90 |
| `lr2e3_wcos` | 2e-3 | warmup_cosine, warmup 2 ep (about 500 steps), floor 1% |
| `lr4e3_wcos` | 4e-3 | warmup_cosine |

Seeds 42 and 43. Everything else as `proposal.md`'s control with `datasets.limit_volumes=1`, `training.num_epochs=100`,
`training.val_every=5`.

## Scoring

Full local v1 val, `last_model.pt` (epoch 100), trainer metric path, paired per slice vs the control of the same seed
(`scripts/eval/varnet_loupe_lr_schedule_r8_eval.py`). Training validates on the same full val, so no held-out subset is
needed; `best_model.pt` is not used for the decision.

## Expected outcomes

- lr2e3_step beats the control by +0.05 to +0.15 dB (the 20%-data trend, without the overfitting).
- lr4e3_step: anywhere from a further gain to a loss; Muon at 4e-3 is untested here. A NaN or divergence counts as a loss.
- Cosine at 2e-3 and 4e-3 ties step at the same lr within 0.1 dB; the 1e-3 cosine loss on Adam came from less time
  at high lr, which a higher peak offsets.
- Overfitting check: if val peaks before epoch 80 in any arm, the comparison is again partly about overfitting and
  the report says so.

## Success criteria (frozen)

- **Adopt** an arm when all hold: mean paired PSNR_Masked over the 2 seeds >= +0.10 dB vs control; positive in both
  seeds; no condition (brain, knee-PD, knee-PDFS) worse than -0.05 dB on the seed mean; SSIM_Masked seed mean not
  worse than -0.002. Several passing: the largest mean wins. Two seeds is weaker than the first sweep's three; a
  default flip on main still needs a third seed of the winner and control.
- **Inconclusive**: best arm 0 to +0.10 dB, or positive in only one seed. Control stays.
- **Control confirmed**: every arm's seed mean <= 0.
- **Cosine question** (reported): `lr4e3_wcos - lr4e3_step` and `lr2e3_wcos - lr2e3_step`, same rule with a 0.10 dB bar.

## Compute

CoreHPC, RTX Pro 6000, 1 GPU per run, sbatch array of 10. About 145 s per training epoch and 45 s per validation:
about 4.3 h per run, time limit 7 h. Submitted only after the first sweep's array (3042468) has finished. Crashed runs
are rerun in full.
