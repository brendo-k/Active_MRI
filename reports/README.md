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
