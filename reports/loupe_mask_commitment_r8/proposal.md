# Proposal: why does LOUPE drop the right k-space band, and which fix keeps it? (VarNet + LOUPE, R=8)

**Status: approved and frozen 2026-11-18 (written 2026-11-17).** It starts its own branch
(`exp/loupe_mask_commitment_r8`, report `reports/loupe_mask_commitment_r8/`) from a clean, synced `main`, via the experiment
loop. Numbers below come from `reports/varnet_loupe_lr_schedule_r8/` (`masks.json`, `report.md` Results 6).

## Why

In `varnet_loupe_lr_schedule_r8`, both 12-cascade lr 1e-3 controls at full local v1 left a band of k-space empty. The band is rows
within 12 of the centre, columns more than 72 right of centre. Its expected sample count was 8.4 and 0.2, against 396-475 for
both `lr4e3_step` seeds and 403-469 for every 6-cascade run. Those runs were 0.12-0.17 dB better. In the controls the band's
logits are saturated (median slope·logit −10.3, sigmoid' 3e-5, about 8,000x below peak), and the mask commits by epoch 10-20.
The runs that kept the band had a better VarNet by epoch 5, mostly on knee (31.5 vs 31.0 dB trainer val).

Two readings, not yet separated:

- **Saturation**: the band is pushed past the point where the sigmoid passes gradient, and can't come back.
- **Early race / co-adaptation**: an early VarNet that can't yet use the band teaches the mask it is useless. The budget is
  zero-sum, so the band is drained. A VarNet that has never seen band data then keeps reporting no benefit.

An earlier grad-floor test (`reports/GRAD_FLOOR_R8.md`, 1 seed, older data, slope 5) kept gradient flowing to saturated locations
and did not change the dead fraction (41.9% vs 42.0%). That favours the second reading, but weakly.

This matters beyond lr: any LOUPE result here can hinge on which early commitment a run happens to make, and that inflates the
0.3-0.5 dB seed spread that hides our 0.05-0.2 dB effects.

## Control

`lr1e3_step` at 12 cascades, as in `reports/varnet_loupe_lr_schedule_r8/followup_proposal.md`:

- full local v1 FLAIR + knee, 100 epochs, val every 5
- recon lr 1e-3 (Muon + Adam), mask lr 1e-3, step at epoch 90 (x0.1)
- slope 10, uniform-noise mask start, R=8, seeds 42 and 43

The existing `vlls_full_lr1e3_step_s{42,43}` runs are reused only if the new code is bit-identical with the new options off
(checked with a 1-epoch smoke run). Otherwise both control seeds are rerun at the new run state.

## Arms (one change each, seeds 42 and 43)

| arm | change | tests |
|---|---|---|
| `clamp` | soft-clamp LOUPE logits as `conv_2d` does: \|slope·logit\| <= 6, leak 0.01 beyond | saturation alone |
| `anneal` | slope 2 -> 10 linearly over epochs 0-20, then 10 | delayed commitment |
| `mix` | 25% of training batches use a fresh uniform random mask at the same R; those batches update the recon only | VarNet kept competent everywhere (co-adaptation) |

That is 6 new runs, plus 2 control reruns if needed. `lr4e3_step` s42/s43 is the reference for the target gap (+0.171 dB).

## Scoring

- Full local v1 val (80 volumes, 1,938 slices), `last_model.pt`, paired per slice against the control of the same seed, with
  `scripts/eval/varnet_loupe_lr_schedule_r8_eval.py`.
- Band count and the other mask metrics come from `scripts/plots/varnet_loupe_lr_schedule_r8_masks.py`, with the new runs
  added as a group.

## Expected outcomes

- If saturation is the cause, `clamp` keeps the band and gains about +0.15 dB.
- If the early race is the cause, `clamp` keeps a band count under 50 and gains under 0.05 dB, while `anneal` and/or `mix` keep
  the band and gain +0.10 dB or more.
- `mix` may cost a little on its own: the recon sees off-distribution masks for 25% of its steps.

## Success criteria (frozen once approved)

- **Band kept**: right-band expected count >= 200 in both seeds (the absent runs measured <= 25; the present ones >= 140).
- **Fix works**: mean paired PSNR gain over the control >= +0.10 dB, positive in both seeds, no condition below −0.05 dB, SSIM
  not below −0.002.
- **Reading**:
  - "saturation" if `clamp` passes both criteria.
  - "race / co-adaptation" if `clamp` fails "band kept" and `anneal` or `mix` passes both.
  - Otherwise inconclusive.

## Compute

Each arm needs a small trainer change (a LOUPE logit clamp, a slope schedule, a random-mask batch fraction), committed as one run
state before launch. The runs go on CoreHPC, one RTX Pro 6000 per run, 12 cascades, about 3 h each: an array of 6, or 8 with the
control reruns. Crashed runs are rerun in full.
