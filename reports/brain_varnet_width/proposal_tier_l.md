# Proposal addendum: brain VarNet cascades × channels on tier L (CoreHPC)

**Status: requested by the user on 2026-11-16** ("Can we do the {6, 12} cascade, {18, 26} channel on a larger dataset on
corehpc? Just continue extending it out"). It extends `reports/brain_varnet_width/proposal.md` on the same branch; the
original criteria are unchanged. Written and frozen before any tier-L run starts.

## Question
At 80 training volumes (tier S) no width above 18 channels helped and every model from about 14 M to 32 M parameters
tied (`reports/brain_varnet_width/report.md`, seed 42 + 43). The working explanation is that 80 volumes limit the data,
not the model.
3. With 3.75× the data (tier L), does extra capacity (26 channels and/or 12 cascades) beat 6×18?
4. How much does the data itself add for a fixed model (6×18, tier S → tier L)?

## Design
Same as the tier-S runs except the training set and the platform.
- Brain FLAIR, local v1 manifest **tier L**: 300 training volumes (the 80 tier-S volumes plus 220 more; manifest
  `train_slices` sums to 4,762 slices, about 3.9× the 1,211 of tier S). `datasets.train_tier=L`, roots on CoreHPC
  `~/scratch/data/fastmri/brain/data` (all 340 FLAIR train + val files present, checked 2026-11-16).
- Val: the same 40 FLAIR val volumes. The trainer validates on the same pinned 10-volume subset
  (`val_limit_volumes=0.25`, draw seed 1234; to be checked against the local list in the smoke test).
- Frozen mask S (same checkpoint, copied to CoreHPC, sha256 checked), R=8 of acquired, Muon + LayerNorm, L1 on RSS,
  batch 16, lr 1e-3.
- **Budget: 100 epochs** (about 29,800 steps), lr ×0.1 at epoch 90, val every 5 epochs, early-stop patience 5.
  - Why equal epochs rather than the 7,600 steps of tier S: every tier-S cell early-stopped or plateaued by 5,300–7,600
    steps, so it was trained to convergence. At 3.9× the data, 7,600 steps is 26 epochs and probably not converged,
    and an under-trained comparison would hide any capacity gain. Early stopping still ends each run when the
    trainer val stops improving.
  - Equal-steps readout (exploratory, free): the trainer-val loss at epoch 25 (≈ 7,450 steps).
- `recon.checkpoint_cascades` off for every cell (96 GB cards; it changes speed and memory, not results).

**Runs** (CoreHPC partition `gpu`, all pinned to `nvidia_rtx_pro_6000`, one GPU each):

| Cell | Tier | Seeds | Role |
|---|---|---|---|
| 6×18 | L | 42, 43 | baseline at L |
| 6×26 | L | 42, 43 | width at 6 cascades |
| 12×18 | L | 42, 43 | depth at 18 channels |
| 12×26 | L | 42, 43 | width and depth |
| 6×18 | S | 42, 43 | platform control: CoreHPC (py3.14, torch 2.12, Blackwell bf16) vs the local A6000 runs |

10 runs, about 35 GPU-hours. Crashed runs rerun in full. W&B runs offline on CoreHPC, then `wandb sync`.

## Scoring
Checkpoints are copied back and scored locally with `scripts/eval/brain_varnet_width_eval.py` (data roots reset to the
local copies), with the same recipe as tier S: `best_model.pt` and `last_model.pt` on the full brain val set (40 volumes,
605 slices), common per-slice Bernoulli draws (1234, 4321), the apodized metric mask, per-slice pairing, 95% bootstrap
CIs over volumes. The train − val gap uses the 80 tier-S training volumes, which every tier-L run also trained on.

## Expected outcome
- Data: 6×18 gains +0.4 to +0.8 dB from S to L (smaller than the +1.0–1.1 dB of 20 → 80 volumes).
- Capacity: with less over-fitting, 26 channels or 12 cascades gain 0 to +0.15 dB over 6×18 at L.

## Criteria (full-val PSNR at `best_model.pt`; "both seeds" = paired per slice in s42 and s43)
- **Validity:** every run finishes, brain SSIM ≥ 0.80, no NaNs or skipped steps.
- **Platform control:** CoreHPC 6×18 tier S is within 0.10 dB of the local 6×18 run of the same seed (37.23 s42,
  37.13 s43). If not, data-gain statements use CoreHPC-vs-CoreHPC only, and tier-L vs tier-S cell comparisons are
  reported as confounded by platform.
- **Q3, capacity at L:**
  - capacity pays: the best of 6×26, 12×18, 12×26 beats 6×18 at L by ≥ 0.15 dB in both seeds;
  - still saturated: all three within 0.05 dB of 6×18 in both seeds;
  - otherwise inconclusive.
- **Q4, data gain:** 6×18 L − 6×18 S (CoreHPC) ≥ +0.3 dB in both seeds = data still pays; < +0.1 dB in both = data
  saturated; otherwise in between (report the size).
- **Exploratory:** does the gap to 6×18 grow with data? (X − 6×18) at L minus (X − 6×18) at S for each X, S side from
  the local runs (12×26 at S has seed 42 only).
- **Next step:** the cheapest cell by wall-clock within 0.05 dB of the best two-seed mean at L becomes the tier-L
  reconstructor.
