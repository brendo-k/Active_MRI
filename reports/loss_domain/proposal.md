# Proposal: loss_domain (frozen 2026-10-30)

Branch `exp/loss_domain`, report folder `reports/loss_domain/`. Design: `docs/experiments/loss_domain/design.md` and
`roadmap.md`. This run covers roadmap items 1-3 in one branch, because all the arms share one set of RSS-L1 twins. The
user's decisions of 2026-10-30, made after the pre-launch QC below, change parts of the design. They are listed
under "Deviations from the design".

## Question

Training minimises L1 on the RSS magnitude, which leaves the maps' common phase free. Moving the loss to the complex
image, or to k-space, may change three things against the production RSS-image L1 twin of the same seed: the RSS metrics
(PSNR / NMSE / SSIM under the apodized metric mask), the learned LOUPE mask, and the maps' common phase. Does any of
the losses below change them? In k-space of the RSS magnitude, does the image-space support mask `M` matter?

## Pre-launch QC (val, 1938 slices; RSS-L1 control = `patient_specific_loupe_r8_flair_knee_100ep_seed42`, last_model)

Scripts and JSON: `qc/`.
- Metric mask vs `M` (`|x| > 0` of the cached ESPIRiT image): 0% of metric-mask pixels lie outside `M`. `M` is about
  36% larger than the metric mask. This answers design open question 4.
- `M` is nearly a no-op for the complex and SENSE k-space losses. The learned maps are zeroed outside their
  10x10-centre support (`sense_mask_method=pre_network`), and 99.93% of that support lies inside `M`. The control's
  `recon_complex` holds 0.05% of its L1 mass outside `M` (brain 0.14%, knee <= 0.01%). The mask-on arms of items 1 and 2
  are therefore dropped. In RSS k-space, `M` is a real change: the RSS target outside `M` holds 7.7% of the energy
  inside the metric mask (mean; p95 37%).
- The ESPIRiT magnitude is not the scored target. Median `|x|`/RSS is 0.97 / 0.98 / 0.89 (brain / PD / PDFS), and the
  5th percentile is 0.70 / 0.61 / 0.35 (RSS noise bias plus soft-SENSE weighting). A pseudo-output `|x|` + (control
  output − target) scores −1.19 / −5.12 / −7.13 dB below the control (27.66 vs 32.24 dB overall).
- The phase is smooth, not random. Inside the signal region, the control's `Σ conj(S)·coil` phase relative to ESPIRiT
  jumps by more than π/2 on 0.2% of neighbouring pixel pairs (ESPIRiT itself: 1%). The circular std of the offset is
  0.95 rad (brain) and 0.52 rad (knee).
- Control health: SSIM_Masked is 0.839 overall (brain 0.846, PD 0.901, PDFS 0.770). It depends on field strength
  (PDFS 0.72 at 1.5T vs 0.81 at 3T; brain 0.76 vs 0.88), so it is SNR-limited, not a bug. The design's absolute
  0.88 bar is replaced (below).
- Compute: the control ran 377 s train + 80 s val per epoch, about 12.7 h per 100-epoch run.

## Arms (each trained from scratch at seeds 42, 43, 44)

Shared settings:
- local v1 FLAIR + knee-PD + knee-PDFS, 320², manifest loader;
- shared `loupe_2d`, straight-through, VD start p=4, no anatomy conditioning;
- R=8 of each slice's acquired k-space (`budget_over=acquired`), `coil_combine=sense`;
- E2E VarNet with 12 cascades and `SensetivityModel_mc` maps (pre_network support), `csme=false`;
- batch 16, 100 epochs, lr 1e-3 stepped at epoch 90.

Pairs within a seed share the init, the batch order and the mask-draw RNG stream.

Notation:
- `x̂` = VarNet `recon_complex`.
- `S` = learned maps.
- `E` = the acquired k-space extent.
- `x` = cached ESPIRiT image.
- `rss` = the scored RSS output.
- `t` = RSS target.
- `F` = centred ortho FFT.
- `k̂_c = F(S_c·x̂)·E` = the coil k-space the scored output is built from.
- `x̂_E = Σ conj(S_c)·F⁻¹(k̂_c)`.

Every loss is the plain mean of `|·|` over all elements (no per-example normalisation).

| Arm | Loss | Role |
|---|---|---|
| A | `mean |rss − t|` (production `loss_type=l1`) | twin / control |
| B | `mean |x̂_E − x|` | complex image, ESPIRiT target (item 1) |
| K_sense | `mean |k̂_c − F(S_c.detach()·x)·E|` | coil k-space through the learned maps (item 2, mask off) |
| K_rss | `mean |F(rss) − F(t)|` | RSS magnitude in k-space (item 3 control) |
| K_rss+M | `mean |F(M·rss) − F(M·t)|` | item 3 treatment: the single change is `M` |

## Readout (every run, `last_model.pt`, val, two fixed mask-draw seeds)

- Per-slice PSNR_Masked, NMSE and SSIM_Masked under the trainer's apodized metric mask (the `MaskedMetricAccumulator`
  definition), overall and per condition. Paired differences against the same seed's A, with a 95% CI over volumes.
- LOUPE mask correlation: the Pearson correlation of each arm's learned probability map with its twin's, against the
  three pairwise correlations among the A seeds.
- The maps' common phase against ESPIRiT: the circular std of `angle(Σ conj(S_c)·coil_c · conj(x))`, weighted by `|x|`
  on metric ∩ map support ∩ `M`, plus the share of neighbouring pairs that jump by more than π/2.

## Success criteria (fixed now)

- **Loss bar** (B, K_sense, K_rss, K_rss+M, each against A): arm minus A in PSNR_Masked, averaged over all val
  slices. The mean over the 3 seeds must be >= +0.10 dB, and the difference must be positive in all 3 seeds. Missed =
  NO-GO for that loss as the default.
- **Mask bar** (K_rss+M minus K_rss): |mean over seeds| >= 0.10 dB, with the same sign in all 3 seeds. Missed = the
  background of the RSS target is not load-bearing once the error is an FFT.
- **Health gate** (whether a run's numbers are trusted): per condition, SSIM_Masked no more than 0.01 below its seed's
  A twin. The A twins must land within 0.01 of the control's 0.839 overall. B, and probably K_sense, are expected to
  miss the gate because of the ESPIRiT magnitude. Before a miss is called a bug, it is checked against the
  pseudo-output SSIM (0.837 / 0.873 / 0.669 by condition).
- Mask correlation and phase std are reported and do not gate. A phase or mask change inside ±0.10 dB is a finding,
  not a reason to change the default loss.

## Expected outcomes

- **B:** PSNR well below A (about −1 / −5 / −7 dB by condition, from the target magnitude alone), so it fails the loss
  bar for a reason unrelated to the domain. Phase std falls toward 0.
- **K_sense:** the same target magnitude, so also clearly below A. Phase std falls.
- **K_rss:** within ±0.2 dB of A. An unmasked L1 in k-space is not the image L1, and its weighting may favour low
  frequencies (smoother images, possibly lower SSIM). The phase std stays at A's level; a change there would be a bug.
- **K_rss+M:** within ±0.2 dB of K_rss, sign unknown. `M` removes the noise background from the target.
- **Masks:** the correlation with the twin stays inside the A seed-to-seed range for every arm.

## Compute

15 runs × about 12.7 h ≈ 190 GPU-hours. One run per 48 GB GPU (the control peaked at 42 GB). Out of memory → rerun
with `recon.checkpoint_cascades=true`. Any other crash is rerun once from scratch.
- Local (GPUs 0-1, about 25 h): seed 42 A and B, then seed 42 K_rss and K_rss+M.
- CoreHPC (sbatch): seeds 43 and 44, all five arms, plus seed 42 K_sense. Platform: torch 2.12 on L40S / H100 class
  GPUs; local is torch 2.6 on A6000.
- Pre-launch: a smoke test of every arm with `limit_batches` at one seed (first-batch file names and mask draw equal
  across arms, finite loss), and a check that every train and val slice is in the cache.

## Deviations from the design (user, 2026-10-30)

- Arm C (complex L1 against `t·x/|x|`, which removes the magnitude mismatch) is stashed for now.
- The mask-on arms of items 1 and 2 are dropped (no-op, see QC). Item 3 keeps its pair.
- Plain mean L1 everywhere, replacing design decision 5's per-example normalisation (Adam ignores a global loss scale;
  the normalisation would only reweight slices against the production twin).
- B and K_sense read the image after the extent projection (`x̂_E`, `k̂_c`), the same image the scored output is built
  from.
- Health gate relative to the A twin instead of the absolute SSIM_Masked >= 0.88.
- One branch for items 1-3, with shared A twins.
- Follow-up raised by the user: test the maps' support mask on and off (`sense_mask_method`). With full-support maps,
  `M` would act on the complex and SENSE k-space losses.

## Amendment 2026-11-01: round 2 (user decisions after the seed-42 diagnosis)

Round 1 (seed 42 only, `reports/loss_domain/eval/summary.json`, `diag/`) found three defects, so its numbers are not
reused:
- K_rss compared the full-FOV FFT of the RSS target, whose noise-floor pedestal sits everywhere, against an output that
  is zero outside the learned maps' support. L1 in k-space then removes the pedestal through DC (`diag/floor_*.py`).
- B and K_sense mixed conventions. The learned and ESPIRiT maps differ by a common phase field, and `S_learned · x`
  disagrees with the measured coil data by 136-140% (`diag/kspace_consistency.py`, `diag/complex_phase.py`).
- Loader bug: brain k-space was flipped with `flip(-2)` on the even 320 grid, which moved DC to row 159. That put a
  ~1 cycle/FOV phase ramp on every brain image and shifted the 10x10 centre crops (sensitivity-net input, LOUPE
  centre, normalisation) by one row. The RSS magnitude is unchanged.

Changes:
- The brain flip now reflects rows about the centre (`flip_rows_about_centre`), so DC stays on row 160. Both cache
  versions are bumped (2). The ESPIRiT cache was rebuilt into `/export/scratch/bkadota/fastmri_complex_espirit_v2`:
  10,270 slices with 0 failures, and recomputed slices match exactly. With the rebuilt cache, the brain phase ramp of
  `x_L · conj x_E` is −0.16 (×2π/320 per px), down from 0.80; knee is 0.19-0.24.
- New arm set, trained from scratch on the fixed loader. Runs go to `~/data/seqmri_checkpoints/loss_domain/runs_v2`
  and the eval to `reports/loss_domain/eval_v2`.
- The sensitivity-net support mask Ω is off in every arm (`+sampler.sense_mask_method=null`), at the user's request
  after `diag/mask_overlap.py` and `diag/masks.png`.
  - Ω was the 0.1-max threshold of the 10x10-centre image, with no hole filling. It nested as metric ⊂ Ω ⊂ M, but had
    holes inside the anatomy wherever that blurry image is dark: FLAIR ventricles, PDFS marrow, dark PD tissue.
  - In knee the output was exactly 0 in those holes. They are 1% of the metric mask (10% on the worst PDFS slice) and
    took 7% (PD) / 4% (PDFS) of the scored L1 error.
  - The maps now cover the whole grid, so the output can represent the background as well.

| Arm | Loss | Role |
|---|---|---|
| rss (A) | `mean |rss − t|` (production `loss_type=l1`) | twin / control |
| rss_M | `mean |M·rss − M·t|` (`l1` + `support_mask`) | image RSS with the ESPIRiT support mask |
| kspace_gt | `mean |(F(S_c·x̂) − k_c)·E|` (`kspace_gt_l1`) | coil k-space against the measured, fully sampled k-space; no masking, no maps on the target, so the maps train on it |
| complex (B) | `mean |x̂_E − x|` (`complex_l1`) | complex image against the ESPIRiT image, unchanged |

Success criteria:
- The bars, thresholds and per-condition relative health gate above are unchanged.
- The loss bar applies to rss_M, kspace_gt and complex, each against A.
- The mask bar becomes rss_M minus A.
- The check that the A twin lands within 0.01 of the old control's 0.839 becomes informational only. That control had
  Ω on, so it is a different recipe. The AGENTS.md rule (SSIM < 0.80 = bug) still applies.

Expected outcomes:
- **A:** maybe slightly above round 1 (32.1 dB / 0.836), most likely in knee, where the holes in Ω were forcing 0.
  Brain may also move by about 0.1 dB from the loader fix.
- **rss_M:** within ±0.1 dB of A, sign unknown. With Ω off, `M` is a real change: the output extends over the whole
  grid, and `M` drops the background outside the ESPIRiT crop from the loss. That background holds the RSS noise floor
  and is never scored.
- **kspace_gt:** sign unknown. It fits the coil data linearly, so like B it lacks the RSS noise bias, and |x̂| reads
  darker than `t` in low-SNR tissue (PDFS most). The per-slice gain+offset PSNR (`diag/intensity_fit.py`) is reported
  alongside, so that this bias is not read as a worse reconstruction. Data consistency at the sampled points should
  stay at rss's level (19-50%).
- **complex:** still well below A. The ESPIRiT common phase (BART rotates it to the first principal component, no -P)
  can't be produced by the per-coil learned maps. Knee, which never had the flip, showed the same 118-135% data
  inconsistency in round 1. The health-gate miss is expected.
- **Masks:** correlation with the twin stays inside the A seed-to-seed range.

Compute: 4 arms × about 12.7 h.
- Seed 42 runs locally on GPUs 0-1, about 26 h: rss then rss_M on one GPU, kspace_gt then complex on the other.
- Seeds 43-44 (8 runs) wait for CoreHPC. They are on hold until the user approves; `HANDOFF.md` describes round 1 and
  is superseded.
- Pre-launch: 18 unit tests, and a 1-epoch smoke test of each arm through the real launcher, run with Ω on and again
  with Ω off.
