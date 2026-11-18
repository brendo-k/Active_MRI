# Reports

A report is a folder, `reports/<topic>/`, holding `report.md` (starts with a provenance block), `manifest.json` (code commit, metric mask,
data, runs, commands, result hashes), `run.sh` (regenerates every result file) and the result files it cites. Write one with the
`experiment-report` skill (`.claude/skills/experiment-report/SKILL.md`). Check one with:

    python scripts/report_manifest.py check reports/<topic>

## Conforming reports

| Folder | Question |
|---|---|
| `metric_mask_reeval_r8/` | Does the new metric mask change the ranking of the R=8 / R=4 checkpoints? |
| `varnet_caipi_promptmr_anatcond_r4/` | Does an anatomy label help a PromptMR+-style VarNet on fixed CAIPI R=4 sampling? |
| `sense_mask_varnet/` | Does the body multiply on the sensitivity maps help a 12-cascade VarNet? Exploratory, 1 seed: `null` beat `pre_network` by 1.26 dB; `null` is now the default. |
| `varnet_train_speed/` | How much faster can 1-step LOUPE + VarNet (R=8) train? Exact fast path adopted as the default; Muon and LayerNorm opt-in |
| `muon_layernorm_100ep/` | Do Muon and LayerNorm still help at full data and 100 epochs? Muon + LayerNorm +0.34 dB at 28-37% of the wall-clock, 2 seeds; LayerNorm alone inconclusive |
| `muon_layer_speed/` | How much less wall-clock does Muon + LayerNorm need to reach its 100-epoch quality? Exact step code -12% per epoch, adopted; shorter schedules not faster so far, 1 seed. Muon + LayerNorm is now the default |
| `varnet_loupe_lr_schedule_r8/` | **Parked** (`exp/varnet_loupe_lr_schedule_r8` @ `2e6e37f`). Which recon lr and schedule for joint VarNet + LOUPE (R=8)? Full local v1: lr 4e-3 step +0.171 dB vs main's 1e-3 (2 seeds), +0.117 at 6 cascades; cosine loses to step; 20% data overfits. Mild gain, grid ends at 4e-3; larger lr on more data and a LOUPE mask-commitment proposal stashed |
| `loss_domain/` | **Parked** (`exp/loss_domain` @ `8464558`). Does a complex-image or k-space training loss beat RSS-image L1 for R=8 LOUPE + VarNet? No, seed 42: rss_M +0.07 dB (inconclusive), kspace_gt -2.39 dB (RSS noise bias; ties after an oracle gain+offset fit), complex -8.69 dB (ESPIRiT target phase). Brain-flip fix brought to main as `6ee2565` |
| `final_dc/` | **Parked** (`exp/final_dc` @ `9f55c2e`). Does a final data-consistency step on the scored output help R=8 LOUPE + VarNet at inference? Hard DC -2.0 dB on 8 RSS-trained checkpoints, no scalar gate helps; their coil images drift 0.15-0.45 rad from the data's phase; kspace_gt +0.58 dB, gap to RSS not recovered |
| `brain_varnet_scaling/` | Does a smaller VarNet beat 12 cascades on 20 brain volumes, and does the best size grow with data (R=8, frozen mask, cascades 2-12 x 20/40/80 volumes)? Q1 inconclusive (seeds flip), no shift; 6 cascades tie 12 at 40-80 volumes at half the parameters; 20 -> 80 volumes +1.1 dB. Adds `datasets.val_anatomies` and `training.early_stop_patience` |

## Planning reports

| Folder | Scope |
|---|---|
| [future_work/](future_work/report.md) | Future Work: simple ACS CNN routing, adaptive mask optimization, sequential acquisition, and RL. Synthesis and proposals; historical evaluation snapshots, no new training results. |

## Earlier flat reports

The other `reports/*.md` files predate the convention. Each experiment report carries a banner saying so and that its metric mask is not
recorded. Their result files are in the `reports/*_r8/` folders. Design and literature notes
(`ADAPTIVE_MASK_CONDITIONING`, `FUTURE_DIRECTIONS`, `LITERATURE_ADAPTIVE_SAMPLING_R8`, `PPO_SAMPLER`, `REINFORCE_STABILITY_FIXES`) report no results of
ours and have no banner. Convert a report by moving it into a folder and following the skill's steps; rescore its checkpoints under the
current metric mask if its numbers are to be compared with new runs.

## Former `analysis/` folders

`analysis/` was merged into `reports/` (2026-10-21). Logs, launchers and result files from the old `analysis/<topic>/` now sit beside the report they
support: `acs_gate_cgsense_r8`, `adaptivity_mechanisms_r8`, `adaptivity_next_steps_r8`, `cgsense_anatcond` (was `anatcond_cgsense`),
`cgsense_mask_variants_r8`, `conv_vs_loupe_r8`, `label_gonogo_r8`, `mask_spacing_r8`, `sens_upsample_r8`, `subject_coadaptation_r8` (was `subject_coadapt`).
Launcher-only and scratch folders keep their names: `clamp_launch`, `mask_draws_launch`, `promptmr_ahain`, `varnet_caipi_anatcond_launch`, `mnet_ref`, `mnet_smoke`, `oct7_scan`.
| `brain_varnet_width/` | Do wider U-Nets help a brain VarNet, and is depth or width the better use of parameters (R=8, frozen mask, cascades 3/6/12 x chans 12-36, 80 volumes; plus a tier-L addendum at 300 volumes on CoreHPC)? Width above 18 never helps; depth beats width at 3 cascades only; 6x18 stays. Tier L: data +0.45 dB, capacity <= +0.06 dB, platform control failed in seed 42. Adds the CoreHPC sbatch and a pathlib._local unpickle alias in the trainer |
