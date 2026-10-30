# Loss domain at R=8: CoreHPC handoff

Written 2026-10-30 for whoever runs the CoreHPC share of this experiment. It covers the question, the design and the
decisions behind it, what runs where, how to run and check the CoreHPC jobs, and how the results come back to be
scored. Everything is on git branch `exp/loss_domain` of SeqMRI. The local worktree is
`~/code/active_mri/SeqMRI-worktrees/exp-loss_domain` on yippee, and the snapshot shipped to CoreHPC is
`~/scratch/loss_domain/code`.

Sources of truth, in order:
1. `reports/loss_domain/proposal.md`: the frozen plan and success criteria. It is not edited after its commit; any
   change is explained in the final report.
2. `docs/experiments/loss_domain/design.md` and `roadmap.md`: the original design. Some of its choices were changed
   after QC; see "Decisions" below.
3. This file: how to run the CoreHPC share.

---

## 1. What we are doing, in one paragraph

SeqMRI learns k-space undersampling masks (LOUPE, R=8) jointly with an end-to-end VarNet reconstructor. Training
minimises L1 on the root-sum-of-squares (RSS) magnitude image, so it never constrains the phase of the learned
coil-sensitivity maps. The question is whether moving the training loss into another domain changes three things
against an otherwise identical RSS-L1 twin trained at the same seed: the RSS reconstruction metrics, the learned
sampling mask, and the maps' common phase. The three other domains are the complex image (with an ESPIRiT reference),
the coil k-space through the learned maps, and k-space of the RSS image (with and without an image-space support mask).
Five arms × three seeds = 15 runs of 100 epochs. Seed 42 trains mostly on yippee; CoreHPC trains seeds 43 and 44 and
one seed-42 arm.

## 2. Background

**The model** (`activemri/models/sequential_sampling.py`, `activemri/reconstructors/VarNet.py`):
- A static, learned LOUPE probability mask (`loupe_2d`) draws a binary mask per slice (straight-through gradient).
  The budget is R=8 of each slice's acquired k-space. A fully sampled 10×10 centre is always included.
- `SensetivityModel_mc` estimates the sensitivity maps `S_c` from that 10×10 centre. The maps are RSS-normalised
  (Σ|S_c|² = 1) and zeroed outside a support (`sense_mask_method=pre_network`: the centre's RSS above 10% of its max).
- The 12-cascade VarNet updates a per-coil k-space estimate and returns one complex image,
  `x̂ = Σ_c conj(S_c)·IFFT(k̂_c)`, called `recon_complex`. Outside the maps' support, `x̂` is exactly 0.
- **Scored output**: `rss = RSS_c( IFFT( FFT(S_c·x̂) · E ) )`, where `E` is the acquired k-space extent. Inside the
  support it is about `|x̂|`. `E` only matters for FLAIR, which acquires about 64% of the 320 grid.
- **Target**: `t` = RSS of the fully sampled coil images.
- **Metrics**: PSNR_Masked, SSIM_Masked and NMSE inside the trainer's apodized per-anatomy metric mask
  (`activemri/utils/metric_mask.py`, `activemri/utils/metrics.py`).

**Why the loss domain matters:** an RSS-magnitude loss is blind to a common phase `θ(r)` on the maps
(`S_c → S_c·e^{iθ}` leaves every magnitude unchanged). A complex-image loss with a fixed-phase reference (ESPIRiT)
pins it. A coil k-space loss through the maps does too, because its reference is ESPIRiT. A k-space loss on the RSS
image is phase-blind, so it is the control for "k-space vs image" without any phase supervision.

**The ESPIRiT reference** is a cache of one ESPIRiT (BART `ecalib`, soft-SENSE with an eigenvalue crop) coil-combined
complex image per slice. It was built from exactly the k-space the loader serves (`scripts/data/build_complex_image_cache.py`),
and the maps were not stored. It holds 480 volumes and 10,270 slices (5.1 GB):
- local copy: `/export/scratch/bkadota/fastmri_complex_espirit`;
- CoreHPC copy: `~/scratch/loss_domain/cache`, already copied and checked at 320 brain + 160 knee files.

`M = (|x| > 0)` is the ESPIRiT support mask.

**Data**: fastMRI "local subset v1", brain AXFLAIR + knee PD + knee PDFS, 320×320 grid, raw h5 read through the
manifest loader (`configs/datasets/fastmri_local_v1/manifest.csv`).
- Train (tier S): brain 80 volumes / 1211 slices, knee-PD 40 / 1369, knee-PDFS 40 / 1349.
- Val: 40 / 605, 20 / 673, 20 / 660.
- All 240 files are on CoreHPC under `~/scratch/data/fastmri/{brain,knee}/data/multicoil_{train,val}`, checked
  2026-10-30 to have the same size as the local copies.

## 3. The arms

Shared recipe:
- data: `datasets=fastmri_local_v1_flair_knee`, `misc.resolution=[320,320]`;
- sampler: shared `loupe_2d`, straight-through, VD start p=4 (`sampler.loupe_vd_init_power=4`), no anatomy
  conditioning, `sampler.budget_over=acquired`, `sampler.coil_combine=sense`;
- reconstructor: E2E VarNet with 12 cascades, `recon.csme=false` (external maps from `SensetivityModel_mc`), bf16
  U-Nets, compiled;
- training: batch 16, 100 epochs, lr 1e-3 stepped ×0.1 at epoch 90.

All of it lives in `scripts/launch/train_loss_domain_r8_local_v1.sh`. Notation as in section 2; `F` is the centred ortho
FFT, and `k̂_c = F(S_c·x̂)·E` is the coil k-space the scored output is built from. Every loss is the plain mean of
`|·|` over all elements (pixels, coils and batch).

| ARM | Loss | Hydra overrides | Role |
|---|---|---|---|
| `rss` | `mean|rss − t|` (production) | `loss.loss_type=l1` | RSS twin; every other arm is compared with it |
| `complex` | `mean|x̂_E − x|`, where `x̂_E = Σ conj(S_c)·F⁻¹(k̂_c)` | `loss.loss_type=complex_l1` + cache | complex image vs ESPIRiT (roadmap item 1) |
| `kspace_sense` | `mean|k̂_c − F(S_c.detach()·x)·E|` | `loss.loss_type=kspace_sense_l1` + cache | coil k-space through the learned maps (item 2) |
| `kspace_rss` | `mean|F(rss) − F(t)|` | `loss.loss_type=kspace_rss_l1` | RSS image in k-space, phase-blind (item 3 control) |
| `kspace_rss_M` | `mean|F(M·rss) − F(M·t)|` | `loss.loss_type=kspace_rss_l1 loss.support_mask=true` + cache | item 3 treatment: the only change is `M` |

"+ cache" = `datasets.complex_cache_dir=<cache>`. The launcher sets it from `COMPLEX_CACHE`.
- The maps are detached on the `kspace_sense` target, so the loss cannot shrink by editing its own reference.
- `complex` and `kspace_sense` read the image after the extent projection (`x̂_E`, `k̂_c`): the same image the scored
  output is built from.

**Pairing.** A seed fixes the network initialisation, the training batch order (a seeded shuffle generator) and the
mask-draw RNG stream. At seed 42 on yippee all five arms were checked to see the same first batch, the same initial
weights and the same first mask (`reports/loss_domain/check_pairing.py`). Pairs are compared only within a seed.

## 4. Pre-launch QC and the decisions it led to

QC on the 1938 val slices and the existing RSS-L1 control checkpoint
(`patient_specific_loupe_r8_flair_knee_100ep_seed42`, same recipe, trained 2026-10-27). Scripts and JSON are in
`reports/loss_domain/qc/`.

| Finding | Number | Decision |
|---|---|---|
| The metric mask lies inside `M` | 0% of metric pixels outside `M`; `M` is about 36% larger | the score never looks where a masked loss is switched off |
| `M` is nearly a no-op for the complex and SENSE k-space losses: the maps' support is already inside `M` | 99.93% of the support inside `M`; 0.05% of `x̂`'s L1 mass outside `M` | mask-on arms of items 1 and 2 **dropped** |
| `M` is a real change for RSS k-space | RSS target energy outside `M` = 7.7% of the in-metric energy (p95 37%) | item 3 keeps its mask pair |
| The ESPIRiT magnitude ≠ the RSS target (RSS noise bias + soft-SENSE weighting) | `|x|`/RSS median 0.97 / 0.98 / 0.89, p05 0.70 / 0.61 / 0.35 (brain / PD / PDFS); a pseudo-output `|x|` + the control's residual scores −1.2 / −5.1 / −7.1 dB | `complex` and `kspace_sense` are expected to lose a lot of RSS PSNR for this reason alone. A magnitude-matched target `t·x/|x|` (arm "C") fixes it and is **stashed for later** by the user's choice |
| The maps' phase relative to ESPIRiT is smooth, not random | 0.2% of neighbouring pairs jump more than π/2 (ESPIRiT itself 1%); circular std 0.95 rad (brain), 0.52 rad (knee) | a phase-pinning loss removes a smooth offset; expect small mask effects |
| `loupe_2d` never reads its input image | — | `coil_combine` cannot affect the mask; only the loss gradient can |
| Control SSIM_Masked is low but SNR-limited | 0.839 overall; PDFS 0.72 at 1.5T vs 0.81 at 3T; brain 0.76 vs 0.88 | the absolute "≥ 0.88" health bar is replaced by a gate relative to the twin |

Other user decisions (2026-10-30):
- plain mean L1 in every arm, replacing the design's per-example normalisation (Adam ignores a global loss scale;
  normalisation would only reweight slices against the production twin);
- extent-projected images for `complex` and `kspace_sense`;
- one branch for roadmap items 1–3 with shared twins;
- 24 h on 2 local GPUs, the rest on CoreHPC, keeping whole seeds on one platform.

A follow-up raised by the user, not part of this run: test the maps' support mask on and off
(`sense_mask_method`). With full-support maps, `M` would act on the complex and SENSE k-space losses.

## 5. Success criteria (frozen in the proposal)

All on PSNR_Masked averaged over all val slices, scored from `last_model.pt` with two fixed mask-draw seeds.

- **Loss bar** (each of `complex`, `kspace_sense`, `kspace_rss`, `kspace_rss_M` vs `rss`): the mean over the 3 seeds of
  (arm − twin) is ≥ +0.10 dB, and the difference is positive in every seed. Missed = NO-GO for that loss as the default.
- **Mask bar** (`kspace_rss_M` − `kspace_rss`): |mean| ≥ 0.10 dB, with the same sign in all 3 seeds.
- **Health gate** (whether a run's numbers are trusted):
  - per condition, SSIM_Masked no more than 0.01 below its seed's `rss` twin;
  - the `rss` twins within 0.01 of 0.839;
  - `complex` and `kspace_sense` are expected to miss it because of the ESPIRiT magnitude. Check a miss against the
    pseudo-output SSIM (0.837 / 0.873 / 0.669 by condition) before calling it a bug.
- Reported, not gating: the LOUPE mask correlation with the twin (against the correlations among the `rss` seeds),
  and the maps' common-phase circular std against ESPIRiT.

Expected:
- `complex` and `kspace_sense` well below `rss` (target magnitude), with the phase std falling toward 0;
- `kspace_rss` and `kspace_rss_M` within ±0.2 dB of `rss`, with the phase std unchanged (a phase change there is a bug);
- masks inside the seed-to-seed range.

## 6. Who runs what

15 runs, about 12.7 h each on an RTX A6000 (the control: 377 s train + 80 s val per epoch). Peak GPU memory is about
42 GiB.

| Where | Runs |
|---|---|
| yippee, GPUs 0–1 (`scripts/launch/queue_loss_domain_local.sh`) | seed 42: `rss` + `complex`, then `kspace_rss` + `kspace_rss_M` |
| **CoreHPC** (`scripts/launch/hpc/loss_domain/run_loss_domain.sbatch`) | seeds 43 and 44 × all five arms (tasks 0–9), plus seed 42 `kspace_sense` (task 10) |

The array task map is in the sbatch (`TASKS=`): 0–4 = seed 43 `rss complex kspace_sense kspace_rss kspace_rss_M`,
5–9 = seed 44 in the same order, 10 = `kspace_sense:42`.

Platforms differ: CoreHPC has torch 2.12 / Python 3.14 on RTX Pro 6000, yippee torch 2.6 / Python 3.13 on A6000.
Whole seeds stay on one platform, except `kspace_sense` seed 42, whose twin is on yippee. Note that caveat when reading
that one pair. All CoreHPC tasks request the same GPU type (`--gres=gpu:nvidia_rtx_pro_6000:1`). Don't mix GPU types
within a seed.

## 7. CoreHPC: setup and running

**Access** (from yippee; `hpc` does not resolve):
```bash
ssh -fN -o ControlMaster=yes -o ControlPersist=8h -o ControlPath=/tmp/claude-2081/cm-bastion \
    bkadota@chpc-ucsf-bastion-vm1.corehpc.ucsf.edu
ssh -fN -o ControlMaster=yes -o ControlPersist=8h -o ControlPath=/tmp/claude-2081/cm-login \
    -o "ProxyCommand=ssh -o ControlPath=/tmp/claude-2081/cm-bastion -W %h:%p bkadota@chpc-ucsf-bastion-vm1.corehpc.ucsf.edu" \
    bkadota@chpc-ucsf-login-vm1.corehpc.ucsf.edu
ssh -o ControlPath=/tmp/claude-2081/cm-login bkadota@chpc-ucsf-login-vm1.corehpc.ucsf.edu '<command>'
```
**Never run compute on the login node**: no python, imports or tests. It is for rsync, `ls`, `sbatch` and `squeue`
only. Environment checks and smoke tests go in SLURM jobs.

**Folder** `~/scratch/loss_domain/`:
```
code/          snapshot of branch exp/loss_domain (code/SNAPSHOT.txt = commit); never edit while jobs run
cache/         ESPIRiT complex-image cache (brain/, knee/), 5.2 GB, copied 2026-10-30
runs/          one folder + .log per (arm, seed); runs/debug/ for smoke tests
slurm_log/     SLURM stdout, one file per task
HANDOFF.md     this file
```
- Env: `source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh; conda activate dl`. The sbatch does
  this and prints the python / torch / hydra versions.
- `WANDB_MODE=offline`.
- Data roots are passed as Hydra overrides:
  `'datasets.roots.hpc="~/scratch/data/fastmri/{anatomy}/data"' datasets.roots.local_batch2=~/scratch/data/fastmri/knee/data`.
  The inner double quotes are needed because of the braces.

**Steps**
```bash
cd ~/scratch/loss_domain

# 1. Smoke test (about 15-30 min): task 10 (kspace_sense, the heaviest loss), 2% of one epoch plus a full val pass.
#    It also writes the loader's per-volume slice-energy caches (.slice_energy/ next to the data, if writable), so the
#    full tasks don't each read all ~100 GB of k-space at start-up. Run it before the full array.
sbatch --array=10 --export=ALL,DEBUG=1 --time=02:00:00 code/scripts/launch/hpc/loss_domain/run_loss_domain.sbatch

# 2. Check the smoke log (slurm_log/loss_domain-<job>_10.log and runs/debug/*.log):
#    - versions printed, GPU = RTX Pro 6000
#    - "Manifest train set (volumes, slices): {'brain': (80, 1211), 'knee-PD': (40, 1369), 'knee-PDFS': (40, 1349)}"
#    - "Manifest val set (volumes, slices): {'brain': (40, 605), 'knee-PD': (20, 673), 'knee-PDFS': (20, 660)}"
#    - "Epoch = [   1/   1] TrainLoss = ... ValLoss = ..." with finite numbers, and no KeyError (cache miss) or
#      out-of-memory error. A "BrokenPipeError ... Exception ignored" after the epoch line is harmless.
#    Yippee reference at seed 42 (2% of an epoch): kspace_sense TrainLoss 0.0259 ValLoss 0.0165, val pass 140 s.
#    Expect small differences (torch version and GPU).
grep -h -E "python|Manifest|Epoch =|Error|out of memory" slurm_log/loss_domain-*_10.log runs/debug/*.log

# 3. Full array, at most 6 at a time.
sbatch --array=0-10%6 code/scripts/launch/hpc/loss_domain/run_loss_domain.sbatch
squeue -u $USER
grep -h "Epoch = " runs/loss_domain_*_seed4[34].log | tail
```
- Time limit: 2 days per task (partition max 4 days). A task that hits the limit has to be rerun.
- **There is no resume**: checkpoints carry no optimizer state. A rerun is the same task from scratch.
- Inside a task, out-of-memory triggers one rerun with `recon.checkpoint_cascades=true`. Any other failure exits 1 and
  keeps `<run>.attemptN.log`. Resubmit it with `sbatch --array=<task>`.
- If `--account=hpc_core` is rejected, delete that line. If RTX Pro 6000s are scarce, switch every task, not just
  some, to `--gres=gpu:nvidia_h100_nvl:1`.
- L40S (about 45 GiB usable) is too tight for the 42 GiB peak; don't use it.

## 8. When runs finish: bring results back and score

Score on yippee, not CoreHPC, so every run is scored on one platform with the same code.
```bash
# on yippee, per finished run (each folder ~350 MB: last_model.pt, best_model.pt, .hydra/, run_info.json, logs, wandb)
rsync -a -e "ssh -o ControlPath=/tmp/claude-2081/cm-login" \
    bkadota@chpc-ucsf-login-vm1.corehpc.ucsf.edu:scratch/loss_domain/runs/loss_domain_\*_seed4\[34\]\* \
    ~/data/seqmri_checkpoints/loss_domain/runs/
rsync -a -e "ssh -o ControlPath=/tmp/claude-2081/cm-login" \
    bkadota@chpc-ucsf-login-vm1.corehpc.ucsf.edu:scratch/loss_domain/runs/loss_domain_kspace_sense_r8_flair_knee_100ep_seed42\* \
    ~/data/seqmri_checkpoints/loss_domain/runs/

# score everything that has a last_model.pt (per-run results are cached, so rerun as runs arrive)
cd ~/code/active_mri/SeqMRI-worktrees/exp-loss_domain
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval/eval_loss_domain.py \
    --runs-root ~/data/seqmri_checkpoints/loss_domain/runs --out reports/loss_domain/eval

# wandb: sync the offline runs when convenient
wandb sync ~/data/seqmri_checkpoints/loss_domain/runs/<run>/wandb/offline-run-*
```
`run_info.json` from CoreHPC has an empty commit (the snapshot has no `.git`). The commit is in `code/SNAPSHOT.txt`;
record it in the report. The eval script writes `reports/loss_domain/eval/summary.json`, which holds:
- per-run metrics by condition;
- paired (arm − twin) differences with volume-bootstrap 95% CIs;
- the loss and mask bars and the health gate;
- mask correlations and phase statistics.

Then write the report with the `experiment-report` skill and the experiment-loop closing steps: the criteria table,
merge or park recommended, and wait for the user's verdict.

## 9. Status at handoff (2026-10-30)

- Branch commits:
  - proposal `73bb128`;
  - loss code `6187b24`;
  - launcher and checks `a7da95c`;
  - queue, sbatch and eval `f4a1e12`;
  - this handoff `1ba1ccf`. That is the code shipped to `~/scratch/loss_domain/code` and the code the local runs use.
- Checks passed on yippee:
  - 12 unit tests (`tests/test_loss_domain.py`);
  - cache coverage: 5,867 slices, 0 missing;
  - seed pairing: same batch, init and mask; finite gradients; 42 GiB peak for every arm.
- 1-epoch smoke tests through the real launcher (seed 42, 2% of the train batches + a full val pass, A6000), all exit 0:

  | ARM | TrainLoss | ValLoss | val PSNR brain / knee (unmasked, after 5 steps) | ValTime |
  |---|---|---|---|---|
  | rss | 0.1688 | 0.1745 | 21.6 / 17.2 | 128 s |
  | complex | 0.2899 | 0.2616 | 15.5 / 11.8 | 136 s |
  | kspace_sense | 0.02592 | 0.01652 | 15.7 / 12.2 | 140 s |
  | kspace_rss | 0.08591 | 0.05857 | 15.8 / 12.5 | 132 s |
  | kspace_rss_M | 0.08548 | 0.05769 | 15.8 / 12.4 | 135 s |

- **Local queue running since 14:31**, from the detached worktree `~/code/active_mri/SeqMRI-worktrees/run-loss_domain`
  (pinned at `1ba1ccf`). Seed-42 `rss` is on GPU 0 and `complex` on GPU 1; `kspace_rss` and `kspace_rss_M` follow at
  about 03:30 on 10-31. The log is `~/data/seqmri_checkpoints/loss_domain/queue_local.log`, and the runs are in
  `~/data/seqmri_checkpoints/loss_domain/runs/`.
- CoreHPC:
  - done: cache copied, data verified, `code/` shipped (`SNAPSHOT.txt` = `1ba1ccf`);
  - **nothing submitted yet**. Next: the DEBUG smoke task (step 1 in section 7), then the array.
