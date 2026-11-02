# Idea: route learned masks by the scan header (coil count), not by the ACS

2026-10-21. Status: an idea with some eval-only support. It hasn't been tested as a trained arm.

## The idea

- Train a few learned (LOUPE) masks. Choose one per scan with a lookup on header fields: anatomy, coil count, and
  possibly field strength.
- This is not a learned router and doesn't use the ACS. The mask must be fixed before the scan, and those fields are
  known by then.
- How we got here:
  - It started as ~8 masks routed by the ACS centre.
  - Then it became "route a brain or knee mask for each field strength".
  - The data below say coil count is the variable that matters.

## Evidence so far

Setup: R=8, E2E VarNet, the new metric mask, and the two seeds of the per-anatomy LOUPE run (no new training). The
score is the knee mask minus the brain mask on the same scans, in PSNR, averaged over the two seeds.

**Knee:** the knee mask wins at both 1.5T and 3T, on every validation volume, in both seeds.

**Brain:**

| Brain scans | Volumes | Knee mask vs brain mask |
|---|---|---|
| 1.5T, 4–6 coils (Avanto) | 4 | −0.10 dB; all 4 worse in both seeds |
| 3T, 4 coils (TrioTim) | 1 | −0.05 dB |
| 1.5T, 16–20 coils (Aera) | 3 | about 0 |
| 3T, 14–20 coils | 15 | +0.03 dB (Skyra volumes up to +0.11) |

- **It's a property of the scan.** Each volume's preference correlates 0.96 between the two separately trained seeds,
  and the shared-mask model shows the same pattern.
- **Coil count predicts it best.** Correlation with coil count 0.78, field strength 0.59, base PSNR 0.27. Field
  strength looks predictive mainly because most 1.5T brain scans here are the 4–6-coil Avanto.
- **The physical reading.**
  - The knee mask puts more samples at high frequencies: 31–32% of samples within radius 48 px, vs 33.5–35% for the
    brain mask.
  - Many-coil scans can afford a flatter mask.
  - With few coils, parallel imaging is weaker, so samples are better kept near the centre.
- **Same direction in CG-SENSE.** In the separate CG-SENSE ACS gate, the brain gainers were the same few-coil Avanto
  scans (gain vs coil count r −0.52). See `reports/ACS_MASK_GATE_CGSENSE_R8.md`.
- **Size.** Sending many-coil brain scans to the knee mask gains about +0.02 dB on brain overall.

## Caveats

- The routes were picked on the same 23 brain validation volumes they were scored on.
- The few-coil group is only 5 volumes. Coil count is also confounded with older scanner models (Avanto, TrioTim).
- The pipeline pads or truncates every scan to 16 coils, so 20-coil scans lose 4 coils.
- The effect is far below the 0.3–0.5 dB spread between seeds of joint LOUPE + VarNet training. It needs many seeds or a
  design that removes that spread.
- These are borrowed masks: each was trained for an anatomy, not a coil class. A mask trained per coil class could do
  better, or might not.

## Why not route the mask shape on the ACS (closed on fastMRI)

- **Similar ACS doesn't mean similar best mask.**
  - Adjacent slices have nearly identical ACS (0.97–0.98 similar), but their per-slice best masks correlate only 0.03
    at the pixel scale, where most of the oracle gain is.
  - The slice with the most similar ACS from another volume: −0.001.
- **Per-scan tuning gains almost nothing on new slices.**
  - A mask plus VarNet tuned to each subject with ground truth gives +0.05 dB on that subject's held-out slices.
  - Smooth density families chosen per volume: +0.008 dB (brain) and −0.024 dB (knee).
- **The large oracle numbers are in-sample.** The +0.3–0.5 dB per-volume fit scored −0.20 dB on the other half of the
  same volume.
- **The adaptive conv samplers didn't route even by anatomy.** They saw calibration data that separates brain from
  knee, yet gave both the same mask (r ≥ 0.988). Scan-specific masks were worth ≤0.11 dB.
- **One place the ACS could still help:** an effective-coil-count or g-factor estimate from the ACS. It would have to
  beat the plain header coil count.
- **Still open:** true 3D SPACE (one ky–kz mask per volume) is untested. The in-house data is the place to try it.

Details: `reports/ADAPTIVITY_MECHANISMS_R8.md`, `reports/SUBJECT_COADAPTATION_R8.md`.

## How to test it in the full-fastMRI run

**Arms.** All arms use the same VarNet, the same epochs and the same seeds. Each samples exactly 8,192 points (12.5%),
including the 10×10 centre.

1. Parallel imaging: 2D CAIPI 2×4 (`activemri/samplers/caipi.py`), trimmed to the exact count.
2. Compressed sensing: variable-density Poisson-disc (`bart poisson`), with the density tuned on the tuning split.
3. One shared LOUPE mask.
4. Per-anatomy LOUPE: brain and knee.
5. Coil-class LOUPE: brain ≤8 coils, brain >8 coils, and knee.

**Design rules**

- **Test set.** The official fastMRI `multicoil_val` stays untouched as the test set.
  - Tune on a split of `multicoil_train`, stratified by contrast × field strength × coil class.
  - Pick routes on the tuning split.
- **Mask rule.** Every arm uses one fixed mask per class. For LOUPE, freeze a draw and fine-tune the VarNet on it.
  - A tuned Poisson-disc arm could win; Zibetti et al. 2022 found that.
  - Decide in advance what we claim if it does.
- **Seeds.** Use 3 per arm, or the number the seed pilot calls for. The pilot is two seeds of a fixed-CAIPI VarNet on
  the current data; it shows whether the seed spread comes from mask learning or from VarNet training.
- **Starting mask.** Start every LOUPE row from the same random draw. Check at step 0 that the shared and per-class
  runs start with identical weights.
- **Routing table (eval-only, on top of the trained arms).** For each group (anatomy × contrast × field × coil class),
  pick the best trained mask on the tuning split, then score it on the test set.
- **Decision rule (write it down before any results).** Coil-class beats per-anatomy on few-coil brain with the CI
  excluding 0, with no significant loss elsewhere, and the same sign in every seed.

**Code needed**

- Coil-count grouping in `_build_balanced_datasets` / `_group_files` (`activemri/datasets/fastmri_data.py`).
- A third (test) split; the loader only has train/val.
- CAIPI and Poisson-disc mask files with the exact sample count, loaded by `FileMaskPolicy`
  (`activemri/samplers/fixed_mask.py`). `scripts/diagnostics/make_spacing_masks_r8.py` already does exact counts.
- A same-draw init option in `ProbMask` (`activemri/samplers/loupe.py`).

## Cheap checks first

- Check the routing without training: choose routes on training slices and score them on validation, using the
  existing checkpoints. `scripts/eval/eval_anatomy_pair_ismrm_r8.py` scores validation only, so it needs a train-split
  option.
- Count the coils per volume in full fastMRI brain and knee on CoreHPC. Use the header's `receiverChannels` or the
  k-space coil dimension. This shows how many few-coil scans exist.

## How the numbers above were made

- Per-slice scores: `reports/ismrm2027/anatomy_pair_seed42.pt` and `anatomy_pair_seed43.pt`.
- Score per volume: the mean over slices of PSNR(per-anatomy recon, other anatomy's mask) minus PSNR(per-anatomy
  recon, own mask). For the shared recon, use the same two maps.
- Grouping: from each val file's `ismrmrd_header` (`systemFieldStrength_T`, `systemModel`). Coil count is the number
  of nonzero coils in the h5 k-space.
- The analysis was an inline script and isn't committed.
