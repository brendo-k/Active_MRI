# Addendum 2026-11-16 (fscr1): SeqMRI's design on all the FLAIR data, with an input it can read

## Why

kscr3's SeqMRI arm (F) beat one-step LOUPE by +0.367 dB on knee, but none of it was adaptivity (`kscr3/README.md`):
- F ignored its data. Its raw real and imaginary k-space over the ACS peak is about 1e-5 beyond the centre (99th
  percentile 7e-4), next to a 0/1 mask channel, so the U-Net read only the mask. Zeroing its data channels left its
  density unchanged (correlation 1.0000), and its swap cost was 0.000.
- L was short of converging: about 3,400 optimizer steps, against about 17,000 for kscr2's one-step LOUPE.

On 11-16 the user asked for the next round on CoreHPC: FLAIR, tier F (all the data), a 6 × 18 VarNet, 4 steps against
LOUPE, with the signed-log input. In the user's words: "I want to see if more data and longer training helps the
results." From the agent's proposal the user took:
- the data-blind twin (yes);
- the small-data control on tier S, every 5th slice (yes);
- no second seed for now;
- 100 epochs;
- a 10 × 10 centre instead of 40 × 40, "so we might get more adaptability".

FLAIR has one condition, so there is no per-condition LOUPE arm and no routing confound.

## Arms

All six use seed 42. Within a tier the VarNet and the sensitivity net start identical (both are built before the
sampler) and see the same batch order.

| Arm | Sampler | Steps | VarNet passes per forward | Train data |
|---|---|---|---|---|
| F_4_s42_fscr1_tF | `seqmri_2d`, signed-log input | 4 | 5 | tier F |
| Fb_4_s42_fscr1_tF | `seqmri_2d`, data-blind twin | 4 | 1 | tier F |
| L_1_s42_fscr1_tF | static LOUPE from uniform noise, one map | 1 | 1 | tier F |
| F_4_s42_fscr1_tS5 | as F | 4 | 5 | tier S, every 5th slice |
| Fb_4_s42_fscr1_tS5 | as Fb | 4 | 1 | tier S, every 5th slice |
| L_1_s42_fscr1_tS5 | as L | 1 | 1 | tier S, every 5th slice |

- **F** is kscr3's F with one change, `sampler.seqmri_input_scale=signed_log`.
  - Each of the four data channels v (real and imaginary parts of the predicted and the measured virtual-coil k-space,
    over the ACS k-space peak) becomes sign(v) log1p(|v| / 1e-6) / log1p(1e6).
  - The floor 1e-6 is the one the log-magnitude channels already use. 0 stays 0 (unmeasured locations), the peak maps
    to ±1, and a typical 1e-5 value to about 0.17. The mask channel is unchanged.
  - Otherwise it is kscr3's F:
    - a reconstruction before every step, the centre included;
    - SeqMRI's routes: the union is detached between steps, so earlier draws are credited only through the
      reconstructions later steps read;
    - the loss is on the final reconstruction only.
  - The signed log also passes the loss's gradient back through the predicted k-space with a weight of up to
    1 / ((1e-6 + |v|) log1p(1e6)), so the in-loop route into the VarNet is far stronger than in kscr3, where it was
    effectively off. kscr2's S arm read log magnitudes and phases, with the same kind of weighting, and trained.
- **Fb** is the twin: a fixed N(0, 1) tensor (its own generator, seed 0) replaces the four data channels and the mask
  channel stays, so spacing between steps is open to it.
  - It reads no reconstruction, so none runs before its steps.
  - It starts from F's U-Net, bit for bit: the fixed tensor draws nothing from the global random stream.
  - **Caveat, stated now:** under SeqMRI's routes an earlier step's draws are credited only through the
    reconstructions later steps read. Fb reads none, so its steps 0–2 learn only through the weights they share with
    step 3. F − Fb therefore holds that credit route as well as adaptivity. F's VarNet also trains on its in-loop
    reconstructions of partial masks.
- **L** is the user's control, as in kscr3.

## Shared recipe

- **Data:** FLAIR only (`datasets=fastmri_local_v1_flair`, one condition). Tier F is 339 volumes (5,376 slices before
  the energy filter); tier S is 80, and every 5th slice keeps about a fifth of them.
- **Scoring:** the FLAIR dev half, 20 of the 40 val volumes (`datasets.val_half=dev`, the fixed split kscr1–3 used for
  knee). The holdout half is untouched.
- **Sampling:** R = 8 of each slice's acquired k-space (`sampler.budget_over=acquired`) and a 10 × 10 centre. FLAIR
  acquires about 64% of the 320² grid, so the centre is about 1% of a slice's budget (it was 12.5% with kscr3's
  40 × 40 on knee).
- **Reconstructor:** a VarNet with 6 cascades × 18 channels and learned sensitivity maps, with cascade checkpointing
  in every arm (`recon.checkpoint_cascades=true`; it changes memory, not the gradients).
- **Training:** kscr2's and kscr3's recipe.
  - L1 loss; Muon at lr 1e-3, with the sampler and the LOUPE logits in Adam at 1e-3.
  - Batch 16, 100 epochs, lr × 0.1 at epoch 90, no early stopping.
  - At tier F that is about 32,000 optimizer steps; at tier S5, about 1,600.
  - F and Fb also keep `epoch_010.pt` for the early warning below.
- **Platform:** every job on CoreHPC, on an RTX PRO 6000; offline wandb, synced afterwards.
- **Score:** `last_model.pt` with `eval_multistep_adaptivity.py score`, draw seeds 1234 and 4321, as in kscr1–3.

## Readouts

- **Primary:** F_tF − L_tF, paired per slice: pooled PSNR with a 95% volume-cluster bootstrap CI.
- **Adaptivity:** F_tF − Fb_tF.
- **The data question:** the same gaps at tier S5, and their differences:
  - (F_tF − L_tF) − (F_tS5 − L_tS5);
  - (F_tF − Fb_tF) − (F_tS5 − Fb_tS5).
  - These are differences of differences on the same slices and draws, with the same bootstrap.
- **Descriptive only:**
  - SSIM and NMSE.
  - Each arm's data gain (e.g. L_tF − L_tS5).
  - F's swap cost through its own VarNet. The leak check showed that own-VarNet swap costs include co-adaptation.
  - Per-step data-swap movement: the share of a step's draws that move when another slice's data is read.
  - Per-step zero-data movement: the share that moves when F's four data channels are zeros.
- **Early warning:** F_tF's `epoch_010.pt` is scored the same way once it exists.
  - If its data-swap and zero-data movement are at kscr3's level (0.2–0.7% of draws), F has gone data-blind again.
  - The user is told then. The run continues unless the user stops it.
- **Guards:**
  - pooled SSIM ≥ 0.80;
  - realised fraction 0.125 ± 0.002 of the acquired region;
  - no draw outside the acquired region;
  - slices below 20 dB are listed;
  - Fb's swap null is exact.

## How the results will be read (fixed before the runs)

| Contrast, pooled PSNR | At least the threshold, CI above 0 | CI includes 0 | CI below 0 |
|---|---|---|---|
| F_tF − L_tF (primary), +0.10 dB | SeqMRI's design beats a converged LOUPE on all the FLAIR data | no gap at full data | it loses to LOUPE at full data |
| F_tF − Fb_tF, +0.05 dB | reading the scan is worth something: a patient-specific gain, with the credit-route caveat | F's gain over L is data-blind, as in kscr3 | reading the data costs; the twin's static map trains better |
| (F_tF − L_tF) − (F_tS5 − L_tS5), +0.10 dB | more data and longer training widen SeqMRI's lead over LOUPE | no evidence that data changes the gap | more data helps LOUPE more, as an unconverged LOUPE predicts |
| (F_tF − Fb_tF) − (F_tS5 − Fb_tS5), +0.05 dB | more data grows the adaptive gain | no evidence that data changes it | the adaptive gain shrinks with data |

- An F that is data-blind at epoch 100 cannot test the design: data-swap and zero-data movement at kscr3's level, with
  F_tF − Fb_tF's CI including 0. The rows above are then reported, and the reading says so.
- **Scope:** one seed. The experiment's seed criterion needs the same sign in a second seed, so a pass here earns
  seed 43 (the user's call), not a claim. The seed spread of joint mask and VarNet training (0.3–0.5 dB) is not in these
  CIs, which hold only slices and draws.

## Expected outcome

- L_tF converges, at about 32,000 steps against kscr2's 17,000.
- **F_tF − L_tF:** between −0.1 and +0.2 dB.
  - For F: kscr2's 16-step arms beat one-step LOUPE by +0.10, and kscr3's F led an unconverged LOUPE by +0.37.
  - Against F: our full-data one-step U-Net sampler (`conv_2d`) trailed LOUPE by 0.20–0.27.
- **F_tF − Fb_tF:** between 0 and +0.1 dB, the range of every adaptivity estimate so far at R = 8 (kscr2's A − T was
  +0.06).
- **The data question:**
  - At tier S5, LOUPE should again be short of converging (about 1,600 steps), so F − L there should be larger than
    at tier F. That puts the first difference of differences between −0.3 and 0.
  - The adaptivity difference of differences: between 0 and +0.05.
- **The 10 × 10 centre:** step 0's reconstruction comes from 100 samples, so its prediction is crude and adaptivity
  can enter only from step 1. The sampler places about 99% of the budget, against about 87% with kscr3's centre.

## Compute

- **Order:** a validation job runs first, and the six training jobs wait on it (`--dependency=afterok`).
  - It runs every arm for 10% of an epoch and scores 120 dev slices.
  - It checks the twins' nulls, that F's data swap and zeroed data channels both move draws, the swap function's
    self-check, the extent guard and the shipped commit (`validate_fscr1.sbatch`).
  - Each training job scores its run on the same card when it finishes.
- **Wall time:** these are estimates from kscr2's five-pass S arm (312 s an epoch on 2,718 knee slices) and the tier-L
  brain 6 × 18 runs (94 s an epoch on 4,545 slices). The validation job measures them.

  | Arm | Estimate |
  |---|---|
  | F_tF | 18–30 h |
  | Fb_tF | 7–11 h |
  | L_tF | 4–5 h |
  | Each tier-S5 arm | 1–2 h |

  That is about 45 GPU-hours, and 1–1.5 days of wall time if the cards are free.
- **Crashed runs** are rerun in full.

## Code (run state 7)

- `activemri/samplers/seqmri.py`:
  - `input_scale='signed_log'` (sampler key `seqmri_input_scale`; the default `raw` keeps kscr3 reproducible);
  - `data_blind` for `seqmri_2d`.
  - SequentialSampling runs no in-loop reconstruction for the twin.
- **Data:**
  - Tier F from main (ac56a1f: the 39 F-only manifest rows, `datasets.train_tier=F`, `tests/test_train_tiers.py`).
  - `configs/datasets/fastmri_local_v1_flair.yaml`.
  - The manifest split moved into `_manifest_split_rows`, unchanged.
- **Scorer:**
  - the zero-data movement for `seqmri_2d`;
  - the count of final-mask locations outside the acquired region, a guard;
  - `contrast --pair LABEL=(A-B)-(C-D)`.
- **CoreHPC:**
  - the `<F|Fb|L>_<K>_s<s>_fscr1_t<F|S5>` arms in `common.sh`;
  - `validate_fscr1.sbatch`;
  - run.md section 5d.
- **Tests:**
  - the signed log;
  - the twin's fixed input, start and lack of reconstructions;
  - the arm reading its data;
  - the contrast syntax;
  - the zero-data probe;
  - every fscr1 arm's CoreHPC arguments;
  - the FLAIR tier-F and dev-half splits.
