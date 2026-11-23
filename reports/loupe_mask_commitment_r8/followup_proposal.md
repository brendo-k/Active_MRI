# Follow-up proposal: does `mix` gain once it trains longer, or once its random-mask batches stop?

**Status: approved and frozen 2026-11-20.** Same branch (`exp/loupe_training__loupe_mask_commitment_r8`), same report folder.
It is a follow-up report, written before the first round's verdict. The first round's frozen result does not change: it
stays inconclusive at 100 epochs (`report.md`).

## Why

In the first round, `mix` was the only arm that kept the right band in both seeds (539 / 527). Its paired gain was
−0.018 / +0.095 dB, short of +0.10 dB. Two things suggest 100 epochs undercounts it:

- **Still rising.** Its trainer val on brain rose +0.09 / +0.07 dB between epochs 90 and 100, while every other arm stayed flat
  or dropped (−0.03 to +0.03).
- **A cost of its own.** For the whole run, 25% of the recon's steps were on random masks it is never tested on, and
  `mix_s42` was the lowest of all 8 runs on brain from epoch 20 to 60.

Knee was flat for every arm over the last 30 epochs, and `mix_s42`'s knee-PD loss (−0.122 dB) shows no sign of shrinking.

The follow-up separates two explanations:

- **(a) `mix` converges more slowly.** Then a longer schedule should lift it past the control.
- **(b) The band helps, but the off-distribution batches cost the recon.** Then dropping the random-mask batches early should
  give the gain at 100 epochs, provided the band survives once they stop.

## Arms (12 cascades × 18 channels, everything else as in `proposal.md`)

| arm | epochs, lr step | change vs its control | control |
|---|---|---|---|
| `control_e150` | 150, step at 135 | none | — |
| `mix_e150` | 150, step at 135 | `sampler.flat_mask_fraction=0.25` (constant, as round 1) | `control_e150` |
| `mixdecay` | 100, step at 90 | `flat_mask_fraction` 0.25 at epoch 0, falling linearly to 0 at epoch 30 (`sampler.flat_mask_decay_epochs=30`), then pure LOUPE for 70 epochs | round-1 `control` s42/s43 + new `control` s44 |

Seeds: 42, 43 and 44. Seed 42's knee-PD decided round 1, so two seeds are too few. That is 10 new runs:

- 3 × `control_e150`;
- 3 × `mix_e150`;
- 3 × `mixdecay`;
- 1 × `control` s44 at 100 epochs.

The round-1 `control` s42/s43 are reused. The only code change is the new decay option, which is off by default; the
constant-fraction path draws the same batches as in round 1, checked in a unit test.

## Scoring

The same as round 1: full local v1 val (80 volumes, 1,938 slices), `last_model.pt`, paired per slice against the same-seed
control with `scripts/eval/varnet_loupe_lr_schedule_r8_eval.py`. Band count comes from
`scripts/plots/loupe_mask_commitment_r8_masks.py`.

## Expected outcomes

- **If (a):** `mix_e150` beats `control_e150` by about +0.10 dB, mostly on brain. `mixdecay` keeps a partial band and gains
  less.
- **If (b):** `mixdecay` keeps the band (>= 200) after 70 epochs without random batches, and gains about +0.10 dB or more,
  including knee-PD.
- **If the band is drained again once the random batches stop:** `mixdecay`'s band falls under 200. That would mean the band
  needs continuing support: the early race is not a one-time event.
- **Side reading:** `control_e150` vs round-1 `control` (the same seeds) shows how much the control itself gains from 50 more
  epochs.

## Success criteria (frozen once approved; the same rule as round 1, now over 3 seeds)

Applied to each arm against its own control:

- **Band kept**: right-band count >= 200 in all 3 seeds.
- **Fix works**: mean paired PSNR gain >= +0.10 dB; positive in every seed; no condition below −0.05 dB in any seed; SSIM
  gain not below −0.002 in any seed.

**Reading:**

- "slow convergence" if `mix_e150` passes both criteria and `mixdecay` does not.
- "band helps once the recon is not taxed" if `mixdecay` passes both criteria, whatever `mix_e150` does.
- "band kept, gap not explained" if both arms keep the band but neither passes "fix works".
- Otherwise inconclusive.

## Compute

Each run uses one CoreHPC RTX Pro 6000:

- 150-epoch runs take about 5.6 h (9 h limit);
- 100-epoch runs take about 3.7 h.

That is about 48 GPU-hours in total, submitted as one array of 10 after a smoke run of the same 10 tasks. The run state (decay option +
`followup.sbatch`) is committed once, before launch. Crashed runs are rerun in full.
