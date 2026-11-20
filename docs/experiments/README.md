# Experiment lines

One folder per research line. `design.md` holds the question, fixed setup and settled results. `roadmap.md` lists the
line's experiments, each with its branch.

Open experiments are named after their line, so a branch says where it belongs:

- branch `exp/<line>__<topic>`, worktree `SeqMRI-worktrees/exp-<line>__<topic>`;
- a report-only branch is `report/<line>__<topic>`;
- a detached run snapshot is `SeqMRI-worktrees/<topic>__run-<hash>`.

Merged and parked branches keep the names they had, because commits and reports on `main` cite them. The
multistep_adaptivity, loupe_training and loss_domain roadmaps list theirs under Done. Older branches are not assigned to a
line yet: `exp/final_dc`, `exp/brain_varnet_scaling`, `exp/brain_varnet_width`, `exp/muon_layernorm_100ep`,
`exp/muon_layer_speed`, `exp/flair_tier_f`, and the branches from before the experiment loop.

| Line | Open branch | State (2026-11-20) |
|---|---|---|
| [multistep_adaptivity](multistep_adaptivity/) | `exp/multistep_adaptivity__seqmri_lr_bias_r8` | in flight: run state 3 (49be699) queued on CoreHPC |
| [loss_domain](loss_domain/) | `exp/loss_domain__loupe_loss_partial_fourier` | in flight: run state 2 (24db7c7) queued on CoreHPC |
| [loupe_training](loupe_training/) | `exp/loupe_training__loupe_mask_commitment_r8` | awaiting verdict |
| [brain_own_learned_mask](brain_own_learned_mask/) | `exp/brain_own_learned_mask__brain_static_vs_loupe` | awaiting verdict |
| [adaptivity_levers](adaptivity_levers/) | `report/adaptivity_levers__adaptivity_levers_r8` | awaiting verdict |
| [rss_extent](rss_extent/) | none | draft |
| [sense_mask](sense_mask/) | none | draft |

Old snapshot paths that reproduction scripts on `main` hard-code are symlinks to the renamed folders:
`run-ml100-91932da` (used by `reports/final_dc/run.sh`), and `run-mls-0920f8b` and `run-mls-e61a166` (used by
`reports/muon_layer_speed/`).
