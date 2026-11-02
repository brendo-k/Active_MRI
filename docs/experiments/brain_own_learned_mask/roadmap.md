# Brain own learned mask — Roadmap

Generated from [design.md](design.md) on 2026-11-01. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Why item 1 comes first

Both halves of the question come from the legacy data, which is smaller (23 brain / 10 knee val volumes). The two legacy runs were also not a matched pair. The only local v1 VarNet pair gives the opposite sign for brain.

| Run | Data | brain Δ | knee-PD Δ | knee-PDFS Δ |
|---|---|---|---|---|
| VarNet, 2 masks (brain, knee), VD p=4, step lr, conv sens upsample, s42 / s43 | legacy T2 + knee | −0.12 / −0.07 | +0.10 / +0.12 | +0.02 / +0.14 |
| CG-SENSE, 3 masks, random start, constant lr, bilinear sens upsample, s42 | legacy T2 + knee | +0.18 | −0.17 | −0.33 |
| VarNet, 3 masks, VD p=4, R of the grid, full tier S, s42 / s43 | local v1 FLAIR + knee | +0.07 / +0.12 | +0.07 / +0.12 | −0.02 / −0.015 |

Δ = per-class run minus its shared twin, PSNR dB. Sources: `reports/ismrm2027/vdinit_summary.json`, `reports/cgsense_anatcond/metric_apodized/cross_eval_proto100ep.json`, `reports/local_v1_flair_knee/`.

## Shared settings

Every training run:

```
python train.py --model sequential sampler.sampler_type=loupe_2d recon=<varnet|cg_sense> \
  datasets=fastmri_local_v1_flair_knee datasets.limit_volumes=0.5 'misc.resolution=[320,320]' \
  'sampler.accelerations=[8.0]' sampler.budget_over=acquired sampler.loupe_vd_init_power=4 \
  sampler.mask_train_mode=straight_through training.batch_size=16 training.num_epochs=100 \
  training.lr_step_size=90 training.seed=42
```

- Sensitivity maps come from the sense net (`SensetivityModel_mc`, defaults) in both reconstructors. CG-SENSE runs change only `recon=cg_sense`: 10 CG iterations, λ = 0, and it trains just the mask and the sense net.
- `limit_volumes=0.5` keeps half of each condition: 40 FLAIR, 20 PD and 20 PDFS train volumes. It halves val too, so score on the full val split (40 / 20 / 20 volumes, prerequisite 2).
- `budget_over=acquired` puts every class at R=8 of the k-space it acquires. The 10-26 FLAIR gain was under `grid`, where FLAIR ran at about R=5.6.
- True pair: the arms of an item share the seed, the VD start, the reconstructor and sense-net init, and the batch order.
- Score with `scripts/eval/eval_anatomy_pair_ismrm_r8.py`. It uses the trainer's apodized metric mask and common Bernoulli thresholds (seeds 1234, 4321), pairs per slice, and bootstraps over volumes (95%).
  - Δ is per-class mask minus shared mask, per class. PSNR decides; SSIM_Masked is reported. SSIM below 0.80 is a bug.
- Keep item 1's `last_model.pt` files until items 2–4 are done; those items reuse their masks.

## Hypotheses and the item that separates them

| # | Why a brain mask helps CG-SENSE but not VarNet | Item | Signature |
|---|---|---|---|
| H0 | Legacy artifact: the smaller data or the unmatched pair | 1, 1b | interaction CI covers 0 on local v1 |
| H1 | Co-adaptation: joint training leaves the brain row where the co-adapting VarNet led it | 3 | a fresh VarNet on the final masks shows a larger Δ than the joint run |
| H2 | Mask learning: the gradient through VarNet misses the brain structure that CG-SENSE finds | 2, 3 | the CG-SENSE-learned brain mask helps a fresh VarNet; the VarNet-learned one does not |
| H3 | Sharing: one VarNet serving three masks pays for interference and lost transfer | 4 | brain-only VarNet Δ > mixed VarNet Δ |
| H4 | No use: VarNet's learned prior already fixes what a brain mask fixes for CG-SENSE | 3, 4 | Δ ≤ 0 under every VarNet, while CG-SENSE still prefers the brain mask |

These can co-exist. Each item measures one share.

## Prerequisites (code)

1. `_load_fixed_masks` in `activemri/training/trainer.py`: copy the matching row of a 3-mask checkpoint into each class slot. Today it raises on more than one row. A 1-mask (shared) checkpoint fills all three slots, so shared and per-class frozen arms run through the same code path.
2. `scripts/eval/eval_anatomy_pair_ismrm_r8.py` needs two flags:
   - Score on the full val split (`datasets.limit_volumes=1` at score time).
   - Score one pair's maps on another pair's reconstructors (item 2). Check that `sampler.slope` is equal across the crossed runs.
   - Confirm that it scores a `cg_sense` pair: smoke-test with `--limit 8` on a CG-SENSE checkpoint, and extend the script if it fails. It has only scored VarNet pairs so far. The interaction in item 1 needs both pairs scored by the same script, thresholds and metric path, so do not fall back to `anatcond_cgsense_cross_eval_r8.py`, which uses different draws.
3. `scripts/launch/train_loupe2d_varnet_r8_local_v1_flair_knee.sh` and its queue script: a `RECON` variable (`varnet` | `cg_sense`), used in the command and the run name.

## Arms

| Arm | Reconstructor | Mask | What trains | Item |
|---|---|---|---|---|
| V-sh / V-pc | VarNet | shared / per-class, learned | mask, VarNet, sense net | 1 |
| C-sh / C-pc | CG-SENSE | shared / per-class, learned | mask, sense net | 1 |
| F-V-sh / F-V-pc | fresh VarNet | frozen from V-sh / V-pc | VarNet, sense net | 3 |
| F-C-sh / F-C-pc | fresh VarNet | frozen from C-sh / C-pc | VarNet, sense net | 3 |
| B-sh / B-pc | fresh VarNet, brain slices only | frozen, from item 3's masks | VarNet, sense net | 4 |

Frozen arms add `sampler.anatomy_conditioning=true sampler.fixed_mask_checkpoints='{brain: <ckpt>, knee-PD: <ckpt>, knee-PDFS: <ckpt>}'`. The VarNet stays label-blind.

## Next

- [ ] **1. Matched sign-flip test** — branch `exp/brain_mask_recon_flip`
  - Question: On local v1, does a per-class brain mask help CG-SENSE more than VarNet? (H0, the design question)
  - Control: V-sh and C-sh.
  - Treatment: V-pc and C-pc.
    - Within a reconstructor, the only change is `sampler.anatomy_conditioning=true`.
    - Between reconstructors, the only change is `recon`.
  - Readouts:
    - Δ_V and Δ_C per class, and the interaction Δ_C − Δ_V per volume (bootstrapped).
    - Each brain row's mean radius and r ≥ 64 share, against the shared mask.
  - Expected outcome: Δ_C(brain) +0.1 to +0.2 dB. Δ_V(brain) between −0.1 and +0.1 dB; the legacy and local v1 results disagree.
  - Success criterion: the brain interaction CI is above 0.
    - Reproduced: Δ_V(brain) ≤ 0 as well.
    - Partial: Δ_V(brain) > 0. Items 2–4 still run, and the question becomes why VarNet gains less.
  - Failure means: interaction CI covers 0, so the reconstructor does not change the value of a FLAIR mask. Run 1b.
  - Depends on: prerequisites 2 and 3
  - Compute: 4 runs, about 20 GPU-h (about 6.5 h wall if 4 GPUs are free; sense_mask and loss_domain may hold some).
    - VarNet: about 6.3 h per run at 0.5 volumes (450 s/epoch on full tier S).
    - CG-SENSE: time epoch 1. The legacy runs took 2.4 min/epoch.

- [ ] **1b. Same test on T2 + knee (only if 1 fails)** — branch `exp/brain_mask_recon_flip_t2`
  - Question: Is the legacy flip specific to T2, or was it the smaller data or the unmatched pair?
  - Change from 1: `datasets=brain_knee_condition_balanced 'datasets.condition_classes=[brain,knee-PD,knee-PDFS]'`. The manifest has 80 T2, 40 PD and 40 PDFS train volumes, halved by the volume limit.
  - Success criterion: same as 1.
    - Reproduced: items 2–4 run on T2.
    - Not reproduced: add a settled result that the flip came from the legacy data or pair, and stop.
  - Depends on: 1
  - Compute: as 1. Brain val outlier AXT2_201_2010571: pick figure slices by PSNR rank.

- [ ] **2. Cross-reconstructor mask scoring** — branch `exp/brain_mask_cross_recon_eval`
  - Question: Is the VarNet-learned brain mask a worse pattern, or a good one that VarNet does not use? (H2 vs H1, H3, H4)
  - Method (no training): score the maps of V-sh, V-pc, C-sh and C-pc on the C-sh and V-sh reconstructors, with the per-class reconstructors as a check.
    - CG-SENSE barely co-adapts (legacy sense-net effect about −0.01 dB), so it is a near-neutral judge.
    - The VarNet scores are co-adapted to their own masks and are secondary.
  - Readouts:
    - Brain Δ of the VarNet masks on the CG-SENSE reconstructor, Δ^C(V).
    - Brain-row overlap between V-pc and C-pc.
  - Expected outcome: Δ^C(V) between 0 and Δ_C(brain).
  - Routing (no pass/fail):
    - Δ^C(V) CI above 0: VarNet learned a brain pattern it does not use. The cause is on the network side; items 3–4 decide which part.
    - Δ^C(V) ≤ 0: VarNet's brain row has no structure that a linear reconstructor values (H2). Item 3's C arms become the key test.
  - Depends on: 1; prerequisite 2
  - Compute: eval only, about 2 GPU-h.

- [ ] **3. Fresh VarNet on frozen masks** — branch `exp/brain_mask_frozen_varnet`
  - Question: Without joint training, does a per-class brain mask help a VarNet? (open question 2, co-adaptation; H1, H2)
  - Control: F-V-sh and F-C-sh, each with one shared mask copied into all three slots.
  - Treatment: F-V-pc and F-C-pc. The four runs are identical except for the checkpoint paths.
  - Readouts:
    - Δ_fresh(V) and Δ_fresh(C) per class.
    - Co-adaptation cost: Δ_fresh(V) − Δ_joint(V) (item 1), per volume.
  - Expected outcome: Δ_fresh(V) above Δ_joint(V) by 0.05–0.1 dB. Earlier swap costs were mostly reconstructor co-adaptation.
  - Success criterion (either or both):
    - H1: the co-adaptation cost CI is above 0.
    - H2: the Δ_fresh(C) CI is above 0, and the CI of Δ_fresh(C) − Δ_fresh(V) is above 0.
  - Failure means: both Δ_fresh ≤ 0. A brain mask does not help one blind VarNet, wherever it was learned. Item 4 decides between H3 and H4.
  - Depends on: 1; prerequisite 1
  - Compute: 4 runs × about 6.3 h, about 25 GPU-h.

- [ ] **4. Brain-only VarNet on the same frozen masks** — branch `exp/brain_mask_brain_only_varnet`
  - Question: Does serving three masks with one VarNet cost the brain? (open question 2, two patterns; H3 vs H4)
  - Control: item 3's mixed-data runs on the same masks.
  - Treatment: B-sh vs B-pc, with `'datasets.train_anatomies=[brain]'`.
    - 100 epochs, the same brain exposure as item 3. Val still runs on every class.
    - Masks from the V runs. Add the C masks if item 3 found Δ_fresh(C) > 0.
  - Readouts:
    - Δ_brain-only on brain val.
    - Sharing cost: Δ_brain-only − Δ_fresh (item 3), per volume.
  - Expected outcome: sharing cost 0 to +0.1 dB.
  - Success criterion:
    - H3: the sharing-cost CI is above 0.
    - H4: the Δ_brain-only CI is at or below 0.
  - Failure means:
    - H4: per-class brain masks only help a linear reconstructor. Add a settled result.
    - H3: FiLM (see Later) splits the sharing cost into interference and lost transfer.
  - Depends on: 1 for the masks; prerequisite 1. It is compared against 3 and can run alongside it.
  - Compute: 2 runs (4 with the C masks) × about 3 h. The runs train on a third of the slices and run full val every epoch.

- [ ] **5. Seeds for the item-1 pairs** — branch `exp/brain_mask_recon_flip_seeds`
  - Question: Is the knee gain real beyond seed noise, and does item 1's brain result hold? (open question 1)
  - Control / treatment: item 1's four arms at seeds 43 and 44.
  - Success criterion:
    - knee-PD Δ_V has the same sign in all three seeds, and every CI excludes 0.
    - The same rule applies to the brain interaction.
  - Failure means: the knee gain is NULL at this data size. Update the "knee gain depends on the start" settled result.
  - Depends on: 1
  - Compute: 4 VarNet runs (about 25 GPU-h) and 4 CG-SENSE runs.

## Later / unscheduled

- FiLM-conditioned VarNet on frozen per-class masks (`recon.film_conditioning=true`). It splits item 4's sharing cost into interference and lost transfer.
- One VarNet per class: the ceiling with no sharing. Needs code.
- Two-stage training as a method if item 3 finds H2: learn the masks under CG-SENSE, then train a VarNet on them frozen.
- `budget_over=grid` vs `acquired` on FLAIR (old routing item 5). This would explain the 10-26 FLAIR gain.
- The header-routing items (coil count, field strength, PromptMR) stay in the routing roadmap: `git show HEAD:docs/experiments/anatomy_routing/roadmap.md`.

## Done

None yet.
