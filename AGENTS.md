@RTK.md
# SeqMRI

## Goal

Acquire **patient-specific MRI sequences**: learn undersampling patterns (k-space masks) that are tailored to the
individual scan — anatomy, coil count, field strength, and what the ACS / calibration lines reveal — instead of one
fixed mask for everyone. Reconstruction quality (PSNR/NMSE/SSIM vs. the fully sampled reference) is the yardstick, and
every adaptive mask is compared against a non-adaptive twin.

Roadmap (user's order, 2026-10-19):
1. Anatomy / header routing (brain vs knee, coil count) — in progress.
2. ACS-based budget adaptation (choose acceleration R per volume).
3. Multi-step sequential sampling — blocked until echo-train evolution (SPACE) is modelled.

Main operating point is R=8, E2E VarNet reconstructor, fastMRI brain + knee. See `docs/idea.md`,
`docs/ADAPTIVE_MASK_EXPERIMENT_PLAN.md`, `docs/ISMRM_2027_ABSTRACT.md` for current thinking.

## Layout

- `train.py` — single entry point: `python train.py --model {loupe,conv,reinforce,ppo,dqn,sequential} --recon {varnet,unet} --dataset ... --acceleration 8`. Trailing `key=value` are Hydra overrides; `--print-config` shows the resolved config.
- `activemri/` — `samplers/` (LOUPE, conv, REINFORCE, routers), `reconstructors/`, `models/`, `datasets/`, `envs/`, `training/` (builders + trainer), `utils/` (metrics, evaluate, BART ecalib, mask binarization).
- `configs/` — Hydra configs (`config.yaml`, `datasets/`, `recon/`).
- `scripts/` — `launch/`, `eval/`, `plots/`, `diagnostics/`, `oracle/`, `archive/`; each uses `scripts/_path.py` to find `activemri`.
- `reports/<name>/` — one folder per experiment write-up (use the `experiment-report` skill).
- `docs/` — design notes and the ISMRM 2027 abstract.

## Conventions

- Python env: uv (`pyproject.toml`, Python 3.13); activate the env before running anything.
- Log experiments to **wandb**, not tensorboard.
- Prefer self-documenting code over comments.
- Search with `rg`, find files with `fd`.

## Experiment loop

Follow `.claude/skills/experiment-loop/SKILL.md`: clean, synced `main` -> proposal (control, experiment, expected outcomes,
success criteria) -> branch `exp/<topic>` in a git worktree -> run -> report -> wait for the human verdict. Success merges the
branch; failure commits everything on the branch, removes the worktree, and leaves the report on `main` with the branch commit hash.

## Gotchas

- **Metrics:** trainer scores inside an apodized per-anatomy metric mask (changed 2026-10-20; `SSIM_Masked` fixed 2026-10-23). Don't compare numbers across those dates, and don't use eval scripts that score full images. Healthy SSIM ≈ 0.88–0.93; **< 0.80 means a bug** — debug before reporting.
- **Sampler input:** conj bug fixed 2026-10-27; older conv/adaptive checkpoints see a different input.
- **Effect sizes are tiny** (0.05–0.2 dB) vs. 0.3–0.5 dB seed spread of joint LOUPE+VarNet training — use seed-swap controls and multiple seeds. The main goal is to expand this.
- **Sense maps:** BART ecalib (tuned crop + soft) is the default; no post-hoc smoothing of maps.
- **PromptMR backbone:** diverges at lr 1e-3, use 2e-4. REINFORCE policy lr ≈ 0.01–0.05.
- **Oracle noise 0.02 is ~100× real noise** — kills PI-like masks in CG-SENSE oracle.
- **Brain val outlier:** AXT2_201_2010571 (~25 dB); pick figure slices by PSNR rank.
- **Checkpoints:** optimizer state stripped (no resume); they live in `~/data/seqmri_checkpoints`.
- **Data:** new fastMRI data goes to `/export/scratch/bkadota/fastmri_raw` via the manifest loader (default since 10-27). `~/data` is an NFS symlink, not `/home` (which is nearly full).
- **Compute:** no compute on the HPC login node (rsync/ls only); crashed runs are rerun in full (no resume).
