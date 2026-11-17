# kscr3: SeqMRI's design on multi-coil knee (lab server, run state 6 = 0329845)

The design and the reading rule are in `../addendum_2026-11-16.md`, written before the runs.

| Arm | Sampler | Ran on | Wall time | wandb (train, score) |
|---|---|---|---|---|
| F_4_s42_kscr3 | `seqmri_2d`: SeqMRI's Sampler2D, softplus, 4 steps, a VarNet reconstruction before every step | GPU 0 | 5.4 h | 2l8m63ni, rpcl7rt0 |
| L_1_s42_kscr3 | static LOUPE, one map for PD and PDFS | GPU 1 | 1.6 h | hoax9yj4, rass0xke |
| Lc_1_s42_kscr3 | static LOUPE, one map per contrast (added by the agent) | GPU 1, after L | 1.5 h | yo7rd22x, u3jr8z0c |

Shared settings:
- a 4-cascade VarNet with cascade checkpointing, at 320×320;
- a 40×40 fully sampled centre, which is 1/8 of the R=8 budget;
- every 5th training slice of the 80 local volumes: 544 of 2,718 slices;
- kscr2's recipe (L1, Muon at lr 1e-3, 100 epochs, ×0.1 at epoch 90, batch 16), seed 42.

Scoring used the dev half (669 slices) with draw seeds 1234 and 4321, as in kscr1 and kscr2. Training and scoring both
ran at 0329845 with a clean tree. Files:
- `scores/`;
- `contrasts.{json,txt}`;
- `launcher.log`;
- `analysis/`: the post-hoc probe scripts. Their figures are in `~/seqmri_figures/` (PNG, not tracked).

## Results

| Arm | knee-PD | knee-PDFS | pooled PSNR | SSIM pooled (PD, PDFS) | NMSE pooled |
|---|---|---|---|---|---|
| F_4 | 35.372 | 32.852 | 34.121 | 0.856 (0.915, 0.797) | 0.00523 |
| L_1 | 34.863 | 32.629 | 33.754 | 0.846 (0.904, 0.787) | 0.00567 |
| Lc_1 | 35.016 | 32.491 | 33.763 | 0.845 (0.906, 0.783) | 0.00575 |

| Paired contrast, dB [95% CI] | knee-PD | knee-PDFS | pooled |
|---|---|---|---|
| F − L (primary) | +0.510 [+0.404, +0.618] | +0.223 [+0.163, +0.286] | +0.367 [+0.306, +0.433] |
| F − Lc | +0.357 [+0.257, +0.457] | +0.360 [+0.283, +0.441] | +0.358 [+0.294, +0.422] |
| Lc − L (routing) | +0.153 [+0.118, +0.189] | −0.137 [−0.158, −0.117] | +0.009 [−0.014, +0.033] |

- **F's data dependence:**
  - swap cost −0.000 [−0.000, +0.000];
  - share of a step's draws that move when another slice's data is swapped in: 0.007, 0.002, 0.002 and 0.002 for steps 0–3.
- **Guards:**
  - pooled SSIM is 0.845–0.856 and the realised fraction 0.125;
  - the knee-PDFS SSIM is 0.78–0.80 in all three arms. That trips AGENTS.md's "below 0.80 means a bug" line for this condition, though the pooled guard passes. It is equal across arms, so it does not move F − L. It matches the known open PDFS issue: PDFS SSIM was already the lowest condition in kscr2 (0.80–0.81), and this reconstructor is weaker.
  - L and Lc each have one slice below 20 dB: file1001715 slice 24, an end slice where every arm scores about 20.7 dB. One of L's two draws collapsed to 14.2 dB. That moves L's pooled mean by about 0.003 dB.

## Reading, by the rule fixed before the runs

- **F − L's CI is above 0**, so SeqMRI's design beats one-step LOUPE here. The pilot criterion is met: at least +0.10,
  with the CI above 0 and the same sign on knee-PD (+0.51) and knee-PDFS (+0.22).
- **It is not contrast routing.** F − Lc is also above 0, and Lc − L nets to 0: routing helps PD by +0.15 and costs
  PDFS −0.14.
- **The rule's caveat** was that without a twin the gain cannot be split into adaptivity and the rest.

## What the post-hoc probes add (not pre-registered; they change the reading)

- **F ignores its data.**
  - Its swap cost is 0.000.
  - Under a data swap, 0.2–0.7% of its draws move.
  - Zeroing its four data channels leaves its density unchanged: correlation 1.0000, mean change 0.002 at epoch 100,
    and 0.001 at epoch 23.
  - Its masks do not rotate with a slice rotated 90° about the k-space centre; the density for the rotated slice
    correlates 1.00 with the original's. This was checked at epoch 25; the rotation and per-scan figures show F
    mid-training, at epochs 25 and 34.
  - The cause is SeqMRI's raw real and imaginary k-space, divided by the ACS peak. Outside the centre it is about 1e-5
    of the peak (99th percentile about 7e-4), next to a mask channel that is 0 or 1. So the U-Net reads only the mask.
- **So none of F − L is adaptivity.**
  - F is in effect a data-blind 4-step mask: a smooth, contiguous low-frequency block that is the same on every slice.
  - It beats a LOUPE map that had about 3,400 optimizer steps, against about 17,000 for kscr2's one-step LOUPE.
- **L is short of converging:**

  | | L | kscr2's LOUPE |
  |---|---|---|
  | median \|logit\| | 3.9 | 6.2–6.5 |
  | share of pixels switched off (p < 0.01) | 35% | 66–72% |
  | checkerboard interleaving, side/diagonal neighbour correlation | +0.06/+0.26 (half formed) | −0.49/+0.39 on the PD map |

  The checkerboard is the speckle; 9×9 smoothing hides it.
- **Several causes could explain the gap; none is separated:**
  - **The regime may favour a block.** At full data our one-step `conv_2d` U-Net sampler trailed LOUPE by 0.20–0.27
    dB. Here a smooth U-Net mask leads by 0.37. That fits a regime of small data, 4 cascades and few steps favouring a
    contiguous block over an unconverged per-pixel map.
  - **F's VarNet sees nearly the same mask every batch.** Much of F's mask is sampled with probability near 1, so its
    VarNet trains on an almost fixed mask. L's VarNet sees fresh draws from a half-converged map. A reconstructor tuned
    to one mask is a known advantage.
  - **Four steps space the samples.** That may add a little; in kscr2 the 16-step twin gained +0.044 over one step.
  - The +0.37 is also outside the addendum's guessed range of ±0.15, which fits a gain from something other than
    SeqMRI's mechanism.
- **Adaptivity elsewhere:**
  - kscr2's adaptive arm reads log-magnitude inputs. Under the same rotation it changes only slightly (correlation
    0.97–0.98 with its unrotated density).
  - Its final-mask probability is smooth: its draws vary, so the checkerboard in its logits averages out.

## What would test SeqMRI's design

1. **An F whose sampler can see the data:** the same sampler with the real and imaginary parts passed through a signed
   log, so the k-space beyond the centre reaches the network. Compare it against today's F, which is in effect its own
   data-blind twin, or against a twin built for the purpose.
2. **A converged LOUPE baseline:** enough optimizer steps (or the full data) for LOUPE to finish converging.

## Caveats

- One seed, the dev half only; the holdout is untouched.
- The lab server's A6000s.
- The 40×40 centre, the unconditioned samplers, 4 cascades and 20% of the slices all differ from kscr1 and kscr2, so
  these numbers do not compare across rounds.
