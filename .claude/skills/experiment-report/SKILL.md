---
name: experiment-report
description: Write a concise, self-contained, reproducible experiment report in reports/<name>/ for SeqMRI analyses (question, runs, methods, results, conclusions, caveats, plus a manifest, run.sh and a provenance block). Use when the user asks to write up, summarise or document an experiment, investigation or analysis, or to turn a finished analysis into a report.
---

# Experiment report

A report is a folder, `reports/<topic_snake_case>/`, that a reader who saw none of the session can follow and rerun:

```
reports/<topic>/
  report.md        the write-up (headings below), starting with the provenance block
  manifest.json    written by scripts/report_manifest.py: code commit, metric mask, data, runs, commands, result hashes
  run.sh           the commands that regenerate every result file in this folder
  *.json, *.npz, *.md   result files the report cites (flat, or in results/); *.png in figures/
```

Reference example: `reports/metric_mask_reeval_r8/`. Older flat reports (`reports/FOO.md`) predate this convention; do not
copy their layout. Never save a new report as a loose `.md` in `reports/`.

Reports for new experiments are written inside the experiment's worktree as part of the loop in
`.claude/skills/experiment-loop/SKILL.md`; add its **Branch**, **Proposal vs outcome** and **Recommendation** items under the provenance block.

## Before writing
1. Every number must come from a script in `scripts/` that exists in the repo and is committed. If an analysis was
   only run inline, save it as a script and rerun it, then confirm the numbers match what you intend to report.
2. Commit the scripts first, then produce the results. The manifest records the commit the results came from; if the
   scripts changed after the run, rerun, or pass `--produced-at <commit>` with the commit that really produced them.
3. Write results into the report folder (give the script an `--out` flag), not into `/tmp` or a sibling folder.
4. Re-read the final outputs (json, logs, figures) instead of quoting from memory.
5. Separate what was measured from what is a hypothesis.

## Steps
1. Create `reports/<topic>/` and generate the results into it.
2. Write `run.sh`: `set -euo pipefail`, `cd "$(dirname "$0")/../.."`, then the exact command lines (one per result
   producer), and say in a comment what to delete to force a recompute.
3. Write the manifest and get the provenance block:
   `python scripts/report_manifest.py write reports/<topic> --metric-mask apodized|legacy --runs <names or globs>
   --script <each producing script> --command "<each run.sh command>" --produced-at <commit> --data-note "<split, volumes, slices>"`
   Use `--hash-checkpoints` when the report ranks checkpoints.
4. Paste the printed block directly under the title of `report.md`. Do not retype it.
5. Write the sections below.
6. Run `python scripts/report_manifest.py check reports/<topic>` and fix every problem it prints. A report is not done
   until this prints `conforms`. Rerun `write` after any result file changes, then re-paste the block.

## Structure (keep these headings, in this order)
- **Title**: the question as a title, with the date (YYYY-MM-DD).
- **Provenance block** (from `write`): commit, metric mask, runs, results, how to reproduce, code drift and dirty-tree warnings.
- **Question**: 2-4 sentences. What was asked and why. State conventions the reader needs (axes, units).
- **Runs compared**: a table of groups, checkpoint paths, key settings (cascades, acceleration, epochs, seeds),
  plus headline validation metrics so the reader knows the models are healthy.
- **Methods**: numbered, one short item per experiment. Say what was computed, on what data, with which script.
  Define any statistic the first time (e.g. "z-score = value minus mean, over standard deviation").
- **Results**: lead with tables. One sentence of reading per table. Report effect sizes with the comparison
  baseline (the control, the noise floor, the other anatomy) next to them.
- **Conclusions**: bullets, each tied to the evidence above. Mark anything untested as "Hypothesis, untested".
- **Caveats**: post-hoc choices (windows, thresholds picked after looking), single seeds, out-of-distribution
  probes, missing controls, assumptions not checked, untracked inputs (checkpoints), uncommitted code.
- **Figures**: paths inside the folder with a few words on each, or "None".
- **Next steps**: 2-4 concrete follow-ups.
- **Reproduce**: `bash reports/<topic>/run.sh`, what it needs (checkpoints, GPU), and the `check` command.

## Style
- Concise but standalone: no references to "earlier in the session" or "as discussed".
- Plain numbers with units (dB, columns, seeds, n). Give ranges over seeds, not just means.
- Do not state a mechanism as a finding unless a test supports it.
- Prefer a table to prose for anything with more than three numbers.
- Cite other reports by folder path (`reports/<topic>/report.md`).

## Project gotchas to check before quoting metrics
- State the metric mask in the provenance block. Since 2026-10-20 the trainer scores under the k-space-apodized
  per-anatomy mask (`activemri/utils/metric_mask.py`: `metric_mask_from_real_kspace` + `_apply_signal_mask`); runs
  logged before that used `target > 0.1 * max` ("legacy"). The two are not comparable: never put numbers from both
  in one comparison, and give a legacy-mask report a banner saying so.
- Use the trainer's background-masked metrics with the data_range = batch max. The older `eval_*` scripts that score the
  full image give SSIM around 0.5 instead of about 0.93. Healthy SSIM is about 0.93; below 0.80 means a bug, so debug
  before reporting.
- Mask columns are the phase-encode axis (fastMRI brain k-space is 768 readout x 396 phase-encode); the plots
  label them "kx".
- Learned mask p(k) = sigmoid(slope * logits), before the sampler's budget rescaling.
- Checkpoints are untracked; the manifest records each run's overrides, config hash, wandb id and (for runs started
  after `run_info.json` was added to `train.py`) the training commit. Older runs show `trained_at_commit: null`; say so.
- Log experiments to wandb, not tensorboard.
- Run Python through `.venv` (`source .venv/bin/activate`) and use `rg`/`fd`, not grep/find.

## Output
Save the folder, run `check`, and tell the user the folder path, the scripts it depends on, and whether `check` passed.
