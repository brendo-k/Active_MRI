# Long schedule: does loupe_uniform catch static with twice the training? (2026-11-12)

Frozen before launch, approved by the user on 2026-11-12 ("yes, freeze it and launch"). The criteria of `proposal.md`
and `followup_proposal.md` are unchanged; this file adds runs and criteria on the same branch, data, budget and
scoring.

## Why
- loupe_uniform lost to static by −0.164 dB in the main experiment (−0.047, −0.035, −0.409; 3/3 negative).
- The user's hypothesis: 65 epochs were too few for a mask learned from a uniform start. Its sigmoid slows learning
  until pixels flip.
- At epoch 64 the loupe_uniform masks are still soft and still moving:
  - 69% of pixels have p in 0.1–0.9, and the mean σ′ is 0.147 (max 0.25);
  - each mask still correlates 0.60 with its random start;
  - on the trainer's 150-slice val, loupe_uniform gained +0.18, +0.39 and +0.10 dB over epochs 49–64, against
    +0.10, +0.14 and +0.12 for static.
- Doubling the schedule tests whether the deficit closes or reverses.

## Runs
New runs go in the same queue, `~/data/seqmri_checkpoints/brain_static_vs_loupe/`. Each has the job line of the run it
copies except for `training.num_epochs=130 training.lr_step_size=116`.

| Run | Copies | Mask |
|---|---|---|
| `long_static_s{42,43,44}` | static_s{seed} | S, frozen |
| `long_uniform_s{42,43,44}` | loupe_uniform_s{seed} | learned from the uniform start |

- The lr drops 10× at epoch 116 of 130, the same 0.9 point of the schedule as 58 of 65.
- A seed gives both arms the same VarNet initialisation and data order, so a pair differs only in the mask.
- Training code: the run-state commit after this proposal. It adds only the queue script to `4009a72`, the follow-up's
  training code. static_repeat_s42 showed that code leaves static unchanged (−0.030 dB against static_s42).

## Scoring
- `scripts/eval/brain_static_vs_loupe_swap.py`, as in the follow-up: `last_model.pt`, the full brain val set
  (40 volumes, 605 slices), common per-slice Bernoulli draws (seeds 1234, 4321) and the trainer's apodized metric mask.
- Each new VarNet is scored with its own mask and with the other new run's mask of the same seed, on val and on the
  301 training slices.

## Criteria (PSNR, paired per slice, per seed)
**Validity:** 6/6 runs finish all 130 epochs, brain SSIM ≥ 0.80.

**Primary: long_uniform − long_static**, each with its own mask.
- Uniform wins: mean ≥ +0.10 dB, positive in 3/3 seeds, and paired t over seeds ≥ 2.92 (one-sided 95%, 2 df).
- No effect: |mean| < 0.05 dB.
- Static wins: mean ≤ −0.10 dB and negative in 3/3 seeds.
- Otherwise inconclusive.

**Secondary: catch-up** = (long_uniform − long_static) − (loupe_uniform − static), per seed. Same thresholds as the
primary. Positive means the longer schedule helps uniform more than static. Training is not bit-deterministic, so each
term carries repeat-run noise of about 0.03 dB.

**Reported without a threshold:**
- each long_uniform mask's correlation with its random start (0.60 at 65 epochs) and its share of pixels with p in
  0.1–0.9 (69% at 65 epochs);
- own-mask train − val gap per run;
- trainer val gain over the last 15 epochs;
- mask effect and VarNet effect of the (long_uniform, long_static) pair from the swap.

## What each result means
- **Uniform wins:** the main result's static ≥ loupe_uniform was a schedule artefact. A brain mask learned from scratch
  beats the shared mask given enough steps, and the report's conclusion changes.
- **No effect or inconclusive, with catch-up > 0:** a longer schedule narrows the gap but does not reverse it.
- **Static wins:** the deficit is not about schedule length.

Prediction: catch-up +0.05 to +0.10 dB, with the primary between −0.10 and 0 (no effect or inconclusive). Uniform could
pull ahead in s42 and s43, where it was within 0.05 dB of static.

## Compute
6 runs × about 72 min on GPUs 0–1 (3 rounds, about 3.6 h), plus about 30 min of scoring. Crashed runs are rerun in full.
