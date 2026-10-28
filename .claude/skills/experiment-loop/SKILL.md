---
name: experiment-loop
description: The SeqMRI experiment loop - propose an experiment with control, treatment and expected outcomes, run it on its own git branch in a worktree, write a report, then wait for the human verdict (merge on success, park the branch on failure and record its commit hash in the report on main). Use whenever the user wants to propose, start, run, close out or abandon an experiment.
---

# Experiment loop

One experiment = one branch = one worktree = one report. `main` only ever receives (a) merged successful experiments and
(b) reports of parked ones. Never run experiments in the main checkout.

```
 main clean + synced -> propose (human OKs) -> branch + worktree -> freeze proposal commit -> run -> report -> WAIT
                                                                                                            |
                          success: merge --no-ff into main, drop worktree  <-------- human verdict --------+
                          failure: commit all on branch, drop worktree, report on main with branch hash <---+
```

## 0. Preconditions (stop and tell the user if any fails; never fix by committing or pushing on your own)
1. `git status --short` is empty on `main`. Uncommitted or untracked work: list it and ask the user what to do.
2. `git fetch origin` then `git rev-list --left-right --count main...origin/main` prints `0 0`. If behind, ask to
   `git pull --ff-only`; if ahead, ask before pushing.
3. No leftover worktree for this topic (`git worktree list`).

## 1. Propose (in chat, before any branch exists)
Write the proposal and wait for the user to approve or edit it. It must contain:
- **Topic / branch**: `<topic_snake_case>`, branch `exp/<topic>`, report folder `reports/<topic>/`.
- **Question** and why it matters for the roadmap step in `AGENTS.md`.
- **Control**: exact config (model, recon, dataset, R, seeds, metric mask) and which existing run or checkpoint is reused.
- **Experiment**: the single thing that differs from control. Anything else that differs is a confound; remove it.
- **Noise floor**: seed spread or seed-swap control used to judge the difference. Effect sizes here are 0.05-0.2 dB against
  0.3-0.5 dB seed spread, so a one-seed comparison cannot succeed.
- **Expected outcomes**: predicted direction and size for each metric (PSNR, NMSE, SSIM under the trainer's apodized
  metric mask), and what each alternative result would mean.
- **Success criteria**: numeric and decided now, e.g. "mean dB gain over control >= X with the same sign in all N seeds".
  Also the **failure / inconclusive** criteria.
- **Compute plan**: runs, GPUs, rough wall time. Crashed runs are rerun in full.

## 2. Branch and worktree
```bash
git switch main
git worktree add ../SeqMRI-worktrees/exp-<topic> -b exp/<topic>
cd ../SeqMRI-worktrees/exp-<topic>
ln -s /home/bkadota/code/active_mri/SeqMRI/.venv .venv   # untracked env; activate it before running anything
```
Work only inside the worktree from here on. Checkpoints go to `~/data/seqmri_checkpoints` and logs to wandb as usual,
so nothing large lives in the worktree except the report folder.

Freeze the plan: save the approved proposal as `reports/<topic>/proposal.md` and commit it as the first commit on the
branch (`proposal: <topic>`). The success criteria are not edited after this commit; if they must change, say so in the
report and why.

## 3. Run
- Commit code changes in small commits on the branch before launching, so each run's recorded commit is real.
- Launch control and experiment with the same seeds, the same data and the same budget. Log to wandb.
- Apply the metric gotchas in `AGENTS.md` (SSIM < 0.80 means a bug: debug before reporting).
- Resubmit crashed runs in full; do not report partial runs as results.

## 4. Report, then wait
Write the report with the `experiment-report` skill, inside the worktree at `reports/<topic>/`. Add to it, under the
provenance block:
- **Branch**: `exp/<topic>`, and the commit hash that produced the results.
- **Proposal vs outcome**: the frozen success criteria, each marked met / not met with the number beside it.
- **Recommendation**: merge or park, as a recommendation only.

Then stop. Show the user the report path and the criteria table, and wait. Do not merge, delete or move anything
until the user gives the verdict.

## 5a. Verdict: success -> merge
```bash
cd <worktree>; git add -A; git commit -m "<topic>: results and report"
cd /home/bkadota/code/active_mri/SeqMRI; git switch main
git merge --no-ff exp/<topic>
git worktree remove ../SeqMRI-worktrees/exp-<topic>
git branch -d exp/<topic>
```
Add the report to the "Conforming reports" table in `reports/README.md` and commit. Ask before pushing.

## 5b. Verdict: failure or inconclusive -> park the branch
```bash
cd <worktree>; git add -A; git commit -m "<topic>: parked results and report"   # everything, incl. scratch and partial runs
BRANCH_HASH=$(git rev-parse HEAD)                                              # final; never amend or rebase afterwards
cd /home/bkadota/code/active_mri/SeqMRI
git worktree remove ../SeqMRI-worktrees/exp-<topic>      # --force only after the commit above succeeded
git switch main
git checkout exp/<topic> -- reports/<topic>
```
Edit `reports/<topic>/report.md` on main: set **Status: parked**, add `Branch: exp/<topic> @ $BRANCH_HASH` and
`Return to the code with: git worktree add ../SeqMRI-worktrees/exp-<topic> exp/<topic>` under the provenance block, and
fill in the verdict and why. Add a row to `reports/README.md` marked parked, commit on main
(`<topic>: parked report, branch exp/<topic> @ <short hash>`). Keep the branch; never delete it.

If a parked experiment is resumed, add a new worktree on the same branch, and write a follow-up report rather than
rewriting the old one.

## Rules
- The human decides success; the agent only compares against the frozen criteria and recommends.
- One question per branch. A second idea starts at step 0 on a new branch.
- Hash in the main report is the branch tip after the final commit; check with `git rev-parse exp/<topic>`.
- `main` is never left dirty at the end of a loop.
