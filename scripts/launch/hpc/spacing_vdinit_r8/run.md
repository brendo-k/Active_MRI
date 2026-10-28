# Spacing test + VD-init LOUPE (R=8, VarNet, brain T2 + knee)

Two questions, one SLURM array:

1. **Does sample spacing matter for VarNet at R=8?** The learned LOUPE mask's pixel spacing matches an independent random draw (nearest-neighbour pair correlation 0.86, vs 0.87 for a random draw; `reports/mask_spacing_r8/mask_spacing_r8.json`). If a Poisson-disc mask with the *same* density, centre and count beats an independent draw, then position-aware mask learners (BASS, AutoSamp) have room to beat LOUPE. If they tie, LOUPE's density is close to the ceiling.
2. **Does starting LOUPE from a variable-density (VD) prior help?** This is the low-frequency prior of Sun et al. (TCI 2024) used as an initialisation: `clamp(c ((D - r)/D)^4, 0.01, 0.99)`, with `c` set so the starting map holds exactly the sample budget. Compared with the default uniform-random start, at the same seed.

## Folder

```
~/scratch/loupe_spacing_vdinit_r8/
  run_spacing_vdinit.sbatch   SLURM array, one task per arm (table below)
  run.md                      this file
  code/                       snapshot of the local SeqMRI working tree (see code/SNAPSHOT.txt);
                              separate from ~/scratch/code/active_MRI, which is untouched
  masks/                      the fixed masks (+ masks_r8.json stats, masks_r8.png)
  data/fastmri_r8split/       symlinks into ~/scratch/data/fastmri/{brain,knee}/data/multicoil_train that
                              reproduce the local data layout file for file (data/README.txt), so the balanced
                              split is the local one
  slurm_log/                  SLURM stdout/stderr, one file per task
  runs/                       created by the jobs: one Hydra/train dir per arm (runs/debug/ for smoke tests)
```

The jobs use your existing setup: `conda activate dl`, `WANDB_MODE=offline`, partition `gpu`, 1 GPU, 16 CPUs, 64 GB, 12 h limit.

**Data.** The default `DATA_ROOT` is `data/fastmri_r8split`. The loader builds the balanced split from the files under the data root: it sorts their paths and then does a seeded shuffle. So the split depends on the file set and layout.
- The local runs use 200 + 38 brain and 23 + 27 knee volumes. Knee is the smallest pool, and it caps each anatomy at 1,350 train and 344 val slices.
- A larger pool would build a split several times bigger. It would differ from the local runs and push 50 epochs well past the 12 h limit.
- The symlinks reproduce the local 288 files and paths exactly, pointing at the original fastMRI files in `~/scratch/data/fastmri/{brain,knee}/data/multicoil_train`. So the split is the local one: 89 + 23 brain and 40 + 10 knee volumes.
- The local copies of some files carry extra locally added datasets (`sense`, `sens_maps`). Training does not read them, and `kspace` is the same array.

**Heads-up: `~/scratch/data/fastmri_staged` is broken.** All 1,687 of its links point at `fastmri/{brain,knee}/multicoil_train/...`, but the files now live one level down, in `fastmri/{brain,knee}/data/multicoil_train/`. Any job that uses it as `DATA_ROOT` will find no readable files, including `~/jobs/active_mri/train_loupe_1step.sh`. `~/scratch/data/fastmri/balanced_pd_pdfs_t2` resolves fine, but it is a different volume subset.

## Run it

Everything goes through `sbatch`; nothing needs to run on the login node.

```bash
cd ~/scratch/loupe_spacing_vdinit_r8

# 1. Smoke test (~10-15 min): one VD-init task and one fixed-mask task, 1 short epoch, outputs in runs/debug/.
#    Also confirms the dl env (python / torch / hydra versions are printed at the top of each log).
sbatch --array=2,4 --export=ALL,DEBUG=1 --time=01:00:00 run_spacing_vdinit.sbatch

# 2. Check both logs show the local split, then a val line and the epoch line:
#      Balanced train set (volumes, slices): {'brain': (89, 1350), 'knee': (40, 1350)}
#      Balanced val set (volumes, slices): {'brain': (23, 344), 'knee': (10, 344)}
#      [brain] ... PSNR=.. SSIM=.. Sparsity=0.125 (n=344) | [knee] ... Sparsity=0.125 (n=344)
#      Epoch = [   1/   1] ... TrainTime = Xs ValTime = Ys
#    A full epoch takes about X / 0.02 + Y seconds (the debug epoch runs 2% of the train batches).
grep -h -E "Balanced|Sparsity=|Epoch =" slurm_log/spacing_vdinit-*_2.log slurm_log/spacing_vdinit-*_4.log

# 3. Full run: all 9 arms, at most 4 at a time.
#    Use --array=0-7%4 to skip the optional top-k arm.
sbatch --array=0-8%4 run_spacing_vdinit.sbatch

squeue -u $USER
tail -f slurm_log/spacing_vdinit-<jobid>_<task>.log
```

Each task takes about 5 h on a 48 GB GPU (L40S class). The local reference run took 4.8 h for 50 epochs on an RTX A6000, on this same split: ~5 min train + 25 s val per epoch. With `%4` the whole array finishes in roughly 12-15 h. If the smoke-test estimate times 50 is over ~10 h, raise `--time`. The `gpu` partition allows up to 4 days.

### Summarise

The summary only parses the training logs. Run it as a tiny CPU job:

```bash
cd ~/scratch/loupe_spacing_vdinit_r8
srun -p cpu -A hpc_core -q cpu -c 1 --mem=2G -t 00:10:00 bash -c \
  'source /mnt/software/ohpc/pub/miniforge3/25.11.0/etc/profile.d/conda.sh && conda activate dl &&
   python code/scripts/eval/summarize_spacing_vdinit.py --runs runs --out summary.json' | tee summary.md
```

Alternatively, rsync `runs/*/train_sequential.log` back and run the same script locally. It prints:
- per-arm final brain / knee PSNR and SSIM;
- the paired differences;
- mean val PSNR at epochs 1, 5, 10, 20, 30, 40, 50 for the four LOUPE arms.

wandb logs are offline. Sync them later with `wandb sync runs/<arm>-<jobid>/wandb/offline-run-*`.

## Arms

| task | arm | what trains | mask | compare with |
|---|---|---|---|---|
| 0 | `random_s1` | LOUPE + VarNet jointly, seed 1 | learned, uniform-random start (default) | `vdinit_s1` |
| 1 | `random_s2` | same, seed 2 | | `vdinit_s2` |
| 2 | `vdinit_s1` | LOUPE + VarNet jointly, seed 1 | learned, VD-prior start (`sampler.loupe_vd_init_power=4`) | `random_s1` |
| 3 | `vdinit_s2` | same, seed 2 | | `random_s2` |
| 4 | `pd_loupe_s0` | VarNet only (seed 42) on a fixed mask | Poisson-disc realization of the learned LOUPE density, mask seed 0 | `white_loupe_s0` |
| 5 | `pd_loupe_s1` | same | mask seed 1 | `white_loupe_s1` |
| 6 | `white_loupe_s0` | VarNet only (seed 42) on a fixed mask | independent draw from the same density, mask seed 0 | `pd_loupe_s0` |
| 7 | `white_loupe_s1` | same | mask seed 1 | `pd_loupe_s1` |
| 8 | `topk_loupe` (optional) | VarNet only on a fixed mask | the 8,092 highest-probability locations, deterministic | `white_loupe_s0` |

All arms share the reference recipe: `--model loupe --recon varnet --dataset brain_knee_balanced_legacy --acceleration 8`, batch 16, 50 epochs, lr 1e-3 with a step at epoch 40, L1 loss, slope 5, grad floors 0.05, a 10 x 10 fully sampled centre, and 8,192 samples (Sparsity 0.125). In the fixed-mask arms the LOUPE sampler is only a placeholder: the saved mask replaces it in training and validation, and only the VarNet (with its sensitivity net) trains.

### The masks

All the masks come from the learned density of the local reference run `checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05` (seed 42), as `scripts/diagnostics/make_spacing_masks_r8.py` makes them. On a real val batch I checked that this density equals the model's own budget-rescaled map (max difference 0.0), and that the centre equals the trainer's pre-selected block. Every mask has the same 10 x 10 centre and exactly 8,192 samples.

| mask | ring-by-ring density error (max / mean) | pair correlation at 1 px, by k-space radius 0-24 / 24-48 / 48-96 / 96+ px |
|---|---|---|
| `pd_loupe_s0` | 5.9% / 1.6% | 0.94 / 0.59 / 0.30 / 0.12 |
| `pd_loupe_s1` | 3.4% / 1.0% | 0.92 / 0.60 / 0.29 / 0.13 |
| `white_loupe_s0` | 5.9% / 2.6% | 1.06 / 1.05 / 0.96 / 1.07 |
| `white_loupe_s1` | 12.5% / 3.7% (outermost small ring) | 0.99 / 0.99 / 1.02 / 1.00 |
| `topk_loupe` | 98% / 31% (moves samples inward) | 1.71 / 1.31 / 0.74 / 0.20 |

A pair correlation below 1 means neighbouring samples are avoided; 1 means independent. How the masks were made:
- **Poisson-disc** (`pd_loupe_*`): a textbook variable-density Poisson-disc of this density under-samples the inner rings by up to 57% and pushes samples outward. That would confound spacing with density. So the input density is iteratively compensated until each ring's count matches the learned density, and the result stays within ~6% per ring.
- **White** (`white_loupe_*`): the 8,092 locations with the smallest `u/p`, with `u` white noise. That is an independent draw at an exact count.
- **Spacing acts only at 1 px.** The learned density is structured at the pixel scale: outside r = 96 px its median probability is 0.004 but its mean is 0.056, so samples sit on scattered high-probability pixels. Poisson-disc spacing therefore can only act between neighbouring pixels, and the pair correlation at 2-3 px is ~1 for both families.
- **A smoothed-density family was tried and dropped.** Poisson-disc on the density blurred by 4 px formed 2 px lattices and could not be density-matched (worst ring off by 11-32%).

## Reading the results

**Spacing** (tasks 4-7): average `pd - white` over the two mask seeds, per anatomy.
- **Within about ±0.05 dB, with the two seeds disagreeing in sign:** the VarNet does not reward 1-px spacing at R=8. Position-aware learners have little room over LOUPE's density.
- **Poisson-disc ahead by 0.1 dB or more, in both seeds and both anatomies:** spacing matters. A deterministic, position-aware learner (BASS-style swaps, AutoSamp-style positions) is worth building.
- For scale: in your seed-controlled runs at R=4, anatomy-specific masks moved PSNR by 0.1-0.26 dB.

**VD-init** (tasks 0-3):
- Compare final PSNR per seed: `vdinit_s1 - random_s1` and `vdinit_s2 - random_s2`.
- Compare the convergence table at epochs 5-20. The paper's claim is mostly speed: its prior started where random-init LOUPE was after ~60k steps.
- A tie at epoch 50 alongside a lead early on means the prior speeds convergence but doesn't change where LOUPE ends up.

**Top-k** (task 8): deterministic, but a coherent dense core with sparse outer pixels. Expect it to trail `white_loupe_s0`. That would show that making the mask deterministic doesn't help by itself.

## Caveats

- **The split matches the local runs** (same files, same seeded split; the summary prints `n=344` per anatomy). The local seed-42 run is therefore a matched third random-init seed, and it is also the source of the masks' density: brain 36.55 / 0.9328, knee 34.26 / 0.9529, at `checkpoints/grad_floor_compare/loupe2d_r8_slope5_floor0.05`.
  - The HPC env (torch 2.12, Python 3.14) and GPUs differ from local (torch 2.6, A6000), which can move numbers by a few hundredths of a dB.
  - So compare HPC arms with each other first, and use the local run as a sanity anchor.
- **The spacing arms use one training seed (42) and are paired by mask seed.** A difference has to agree across both mask seeds before it counts.
- **Spacing is tested at the 1-px scale only** (see the masks section).
- **Top-k is optional:** submit `--array=0-7%4` to skip it.

## What changed in the code snapshot

`code/` is the local working tree (HEAD plus uncommitted changes; `code/SNAPSHOT.txt` lists them). New for this experiment:
- `sampler.fixed_mask_path`: `activemri/samplers/fixed_mask.py`. A saved `[H, W]` mask, centre included, replaces the sampler's mask for every slice; only the reconstructor trains.
- `sampler.loupe_vd_init_power`: `Sampler.init_loupe_variable_density`. Starts a straight-through `loupe_2d` mask at the budget-matched VD prior.
- `scripts/diagnostics/make_spacing_masks_r8.py`: makes `masks/` (run locally; it needs the reference checkpoint).
- `scripts/eval/summarize_spacing_vdinit.py`: the summary above.
- Run names: `fixed-mask-<mask>_...` and `loupe2d_R8_vdinit-p4_...`.

Checked locally before shipping:
- 70 related unit tests pass.
- On a real batch, the VD-init starting map equals the budget exactly (rescale is the identity), and the fixed mask is used unchanged in train and eval, with no gradient to the sampler.
- 1-epoch end-to-end runs of a fixed-mask arm and a VD-init arm completed with Sparsity 0.125.
