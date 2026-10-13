---
name: experiment-report
description: Write a concise, self-contained experiment report in reports/ for SeqMRI analyses (question, runs, methods, results, conclusions, caveats). Use when the user asks to write up, summarise or document an experiment, investigation or analysis, or to turn a finished analysis into a report.
---

# Experiment report

Write one Markdown file, `reports/<TOPIC_IN_CAPS>.md`, that a reader who saw none of the session can follow.
Reference example: `reports/LOUPE_COLUMN_STREAKS.md`.

## Before writing
1. Every number must come from a script in `scripts/` that exists in the repo. If an analysis was only run
   inline, save it as a script and rerun it, then confirm the numbers match what you intend to report.
2. Re-read the final outputs (json, logs, figures) instead of quoting from memory.
3. Separate what was measured from what is a hypothesis.

## Structure (keep these headings, in this order)
- **Title**: the question as a title, with the date (YYYY-MM-DD).
- **Question**: 2-4 sentences. What was asked and why. State conventions the reader needs (axes, units).
- **Runs compared**: a table of groups, checkpoint paths, key settings (cascades, acceleration, epochs, seeds),
  plus headline validation metrics so the reader knows the models are healthy.
- **Methods**: numbered, one short item per experiment. Say what was computed, on what data, with which script.
  Define any statistic the first time (e.g. "z-score = value minus mean, over standard deviation").
- **Results**: lead with tables. One sentence of reading per table. Report effect sizes with the comparison
  baseline (the control, the noise floor, the other anatomy) next to them.
- **Conclusions**: bullets, each tied to the evidence above. Mark anything untested as "Hypothesis, untested".
- **Caveats**: post-hoc choices (windows, thresholds picked after looking), single seeds, out-of-distribution
  probes, missing controls, assumptions not checked.
- **Figures**: paths with a few words on each.
- **Next steps**: 2-4 concrete follow-ups.

## Style
- Concise but standalone: no references to "earlier in the session" or "as discussed".
- Plain numbers with units (dB, columns, seeds, n). Give ranges over seeds, not just means.
- Do not state a mechanism as a finding unless a test supports it.
- Prefer a table to prose for anything with more than three numbers.

## Project gotchas to check before quoting metrics
- Use the trainer's background-masked metrics (`_apply_signal_mask`, data_range = batch max). The older
  `eval_*` scripts score the full image and give SSIM around 0.5 instead of about 0.93. Healthy SSIM is about
  0.93; below 0.80 means a bug, so debug before reporting.
- Mask columns are the phase-encode axis (fastMRI brain k-space is 768 readout x 396 phase-encode); the plots
  label them "kx".
- Learned mask p(k) = sigmoid(slope * logits), before the sampler's budget rescaling.
- Log experiments to wandb, not tensorboard.
- Run Python through `.venv` (`source .venv/bin/activate`) and use `rg`/`fd`, not grep/find.

## Output
Save the report, list the scripts and figure paths it depends on, and tell the user the file path.
