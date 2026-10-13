# Input-adaptive and deterministic masks: parked design notes (2026-10-05)

Status: design only. Nothing here is implemented or run.

## Question
Why do REINFORCE, LOUPE and AutoSamp learn one pattern per anatomy instead of a patient- or slice-specific one?
Can they be conditioned on the 10x10 calibration block, and do the DCM (Sun et al., IEEE TCI 2024) and
multi-contrast (Yang et al., IEEE TMI 2023) papers help?

## What the code does today
| Sampler | Learned object | Depends on the input? |
|---|---|---|
| LOUPE (straight-through) | `ProbMask.mask`, `nn.Parameter` `[C, H, W]` (`samplers/loupe.py`) | Only through the anatomy label |
| REINFORCE | The same logits, sampled by Plackett-Luce (`Sampler._reinforce_forward`) | Only through the anatomy label |
| AutoSamp | `positions`, `[K, 2]` (`samplers/autosamp.py`) | No |

- The training objective is the dataset-level expected loss, so each of these finds one best mask per class.
  Batch size only changes gradient variance. The batch-averaged gradient is an unbiased estimate of the same
  dataset-level gradient, and every sample's gradient lands on the same parameters.
- REINFORCE with `reinforce_group_size=0` already trains one slice per step (`Trainer._replicate_for_reinforce`)
  and the baseline is per slice, yet the mask is still global. Batch size 1 is therefore not a fix.
- Per-anatomy maps already exist: `num_conditioning_classes` gives one mask per class and `label_key` selects the
  label field. Whether the dataset has a contrast field to use as a label has not been checked.
- The straight-through path binarises with `ThresholdRandomMaskSigmoidV1`, and `Sampler.forward` has no eval-mode
  branch, so validation masks are random draws (noisy checkpoint selection).
- REINFORCE trains on Plackett-Luce draws but evaluates the deterministic top-k of the logits (train/eval mismatch).
- `init_variable_density` is only applied when `mask_train_mode='reinforce'`; straight-through LOUPE starts from a
  uniform-random mask.
- `_reinforce_forward` caps `k` at the tightest example in the batch. If a group mixes padded and unpadded slices the
  acceleration for the unpadded ones is silently lowered.
- The sense support comes from the central 10x10 block of `full_kspace` (`SensetivityModel_mc.mask_center`),
  regardless of `old_mask`. Check that `preselect_num` covers it, otherwise the calibration is free data.

## Support size and the k-space pattern
Two separate effects, not one:
- Spacing: aliases fall at FOV/dk. A smaller object support allows a coarser lattice without alias overlap, and the
  alias shifts should avoid the support's autocorrelation. This is why CAIPI-like lattices work.
- Energy extent: shrinking an object by `s` widens its spectrum by `1/s`, so small objects (end-of-volume slices)
  want a flatter variable-density profile.
The grid extent is fixed by the matrix size, so what adapts is the density profile and lattice spacing, roughly the
same pattern scaled in k-space by the support size.
The recon output is zero outside the sense support, so aliasing outside it costs nothing.

## Proposed design
`score(k) = W_prior(k; p_hat(calib), s(calib)) - W_r[class]`, then deterministic top-M with a softmax-relaxed
straight-through backward.
- `W_prior`: variable-density prior. Decay exponent `p_hat` from the spectrum slope inside the 10x10 block, spatial
  scale `s` from the bounding box or area of `sense_mask`. 1-3 analytic or learned scalars per slice.
- `W_r[class]`: learned per-anatomy map, dense straight-through gradients.
- The reconstructor sees one deterministic mapping per condition.
- Rejected for now: a REINFORCE-trained dense head (signal-to-noise is already marginal, see
  `scripts/reinforce_gradient_snr.py`), and AutoSamp per-slice positions (in-place projection and shared `_snap`
  assume one position set). If AutoSamp is revisited, use a per-slice 2x2 warp of the shared positions, identity at init.

## Experiments, in order
0. Per-slice oracle mask vs best shared mask, split by support size. A small gap means adaptivity won't pay off.
1. New `topk_st` mask mode: hard top-k forward, `k*softmax(score/tau)` backward, reusing the `taken`/`valid_mask`
   exclusion logic. Compare against straight-through + VD init, to separate "prior" from "determinism".
2. Add per-class `W_r`.
3. Add per-slice `p_hat` and `s` from the calibration.
Evaluate on typical slices (or stratified), with seed variance. Per the support audit, tail slices are dominated by
mask-independent unsupported-pixel error.
Normalise REINFORCE advantages per slice if slices are ever mixed within a group.

## Papers
- DCM (Sun et al.): `TopM(W_LPK - W_r)` with a fixed VD prior, softmax-relaxed straight-through backward (tau=5).
  Reported +2.5 to 2.9 dB PSNR over LOUPE on brain DICOM and about 4x faster convergence; single-coil, U-Net/KIKI-Net.
  Their `p` is tuned per acceleration and dimension, and the split used is not stated in the text. Tables were not
  available when this was written, so the numbers are unverified. My reading of eqs. 16-17 (not checked):
  with scores in [0,1] and tau=5 the backward is close to an identity straight-through with the mean subtracted.
- Yang et al. (multi-contrast): prior `r = |F(synth(x_ref)) - y|` averaged over the dataset, plus a learned
  adjustment, deterministic top-k at inference. Global pattern, single-coil only, simulated k-space on the public set,
  and needs a fully sampled reference contrast. Skip unless the data has aligned multi-contrast pairs. The reusable
  idea is the prior; the calibration block plays the role of a very weak reference.

## Open questions
- Does the config used for the audited `loupe_anatomy` runs set `num_conditioning_classes > 0`?
- Does the dataset expose contrast, orientation or slice position as label fields?
- Is `preselect_num` at least 5 (n per side, giving the 10x10 block)?
