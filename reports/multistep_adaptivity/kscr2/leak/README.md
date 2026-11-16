# kscr2 leak check: C's masks through a VarNet that cannot read them

Design and reading rule written 2026-11-16, before the run.

## Question

C_16_s42_kscr2 draws its mask from the fully sampled k-space, so the pattern of sampled locations can tell C's own VarNet
about k-space it never measured. Is C's +0.247 dB over A (each through its own VarNet, `../contrasts.txt`) a better
choice of measurements, or C's VarNet reading the mask as a code? A's and S's masks depend only on measured data, so
they carry no such code.

## Design

- **Masks:** the final masks that C_16, A_16, S_16 and T_16 drew in their own scoring. That means the same 669 dev-half
  slices, draw seeds 1234 and 4321, and thresholds keyed by slice and step. Each run also draws on each slice's donor
  (another volume's slice with the same condition, acquired-location mask and coil count) with the slice's thresholds,
  as the swap cost does.
- **Decoders:** each decoder brings its own sensitivity net and VarNet.
  - T_16_s42_kscr2, primary. It trained on data-blind 16-step masks, so it cannot have learned a data-dependent mask
    code.
  - T_1_s42_kscr2, secondary. It trained on one-step LOUPE masks.
  - Each arm's own, the diagonal. It must reproduce the CoreHPC scores.
- **Platform:** the lab server's RTX A6000 (GPU 0), not the pinned RTX PRO 6000. Every number below comes from this one
  job, and nothing is paired with the CoreHPC score files.
- **Gate,** before reading anything:
  - the slice hash (720f599c…), draw seeds and donors equal the CoreHPC scores';
  - each arm decoded by its own VarNet is within 0.01 dB pooled PSNR of its CoreHPC score.
  - If the gate fails, the result is not read and the check moves to CoreHPC.
- **Data:** the dev half only (669 slices, 20 volumes). The holdout half stays untouched.

## Readouts and how they will be read (fixed before the run)

- **Primary 1, the neutral swap cost:** C's swap cost through T_16's VarNet minus A's, paired per slice.
  - An arm's own mask and its donor mask come from the same sampler, so the decoder's mismatch with that sampler's masks
    cancels, and T_16 can read neither mask as a code.
  - What remains is the value of the mask matching this slice.
- **Primary 2, the neutral gap:** C's masks through T_16's VarNet minus A's. This reads in one direction only.
  - C's masks are the furthest from T_16's training masks, so they pay the largest mismatch.
  - A gap that survives is real; one that vanishes is ambiguous.

| Primary 1 (swap cost, C − A) | Primary 2 (gap, C − A) | Reading | Next |
|---|---|---|---|
| CI above 0 | any | C's choices fit each slice in a way any VarNet can use: the information exists, partial data does not reveal it (outcome-map row 2 stands) | what the early steps acquire |
| otherwise | CI above 0 | C's masks are better on average, not better matched: a better pattern, not adaptivity | score C's average mask as a static mask |
| otherwise | otherwise | C's ceiling cannot be told apart from the mask code and co-adaptation: everything near +0.1 dB (row 4) | ACS budget adaptation |

- **Secondary:** the same two readouts through T_1's VarNet.
- **Reported only:**
  - the full mask × decoder matrix;
  - S as a second leak-free arm;
  - the share of C's own-VarNet gap (+0.247) that survives through T_16.

## Commands

```bash
CK=~/data/seqmri_checkpoints/multistep_kscr2   # runs/<arm>/{last_model.pt,run_info.json,.hydra} copied from CoreHPC 11-16
K=reports/multistep_adaptivity/kscr2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval/eval_multistep_adaptivity.py cross --name leak_kscr2 \
    --masks C_16_s42_kscr2=$CK/C_16_s42_kscr2 --masks A_16_s42_kscr2=$CK/A_16_s42_kscr2 \
    --masks S_16_s42_kscr2=$CK/S_16_s42_kscr2 --masks T_16_s42_kscr2=$CK/T_16_s42_kscr2 \
    --decoders T_16_s42_kscr2=$CK/T_16_s42_kscr2 --decoders T_1_s42_kscr2=$CK/T_1_s42_kscr2 \
    --decoders C_16_s42_kscr2=$CK/C_16_s42_kscr2 --decoders A_16_s42_kscr2=$CK/A_16_s42_kscr2 \
    --decoders S_16_s42_kscr2=$CK/S_16_s42_kscr2 \
    --reference $K/scores --data-root 'hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}' \
    --data-root local_batch2=/export/scratch/bkadota/fastmri_raw/knee --out $K/leak/scores
```

The contrasts come from the same folder. `v C T_16` stands for `C_16_s42_kscr2__via__T_16_s42_kscr2`.

```bash
v() { echo "${1}_16_s42_kscr2__via__${2}_s42_kscr2"; }
E="scripts/eval/eval_multistep_adaptivity.py contrast --scores $K/leak/scores"
.venv/bin/python $E --quantity swap_cost --pair swap_C_minus_A_via_T16="$(v C T_16)-$(v A T_16)" \
    --pair swap_C_minus_A_via_T1="$(v C T_1)-$(v A T_1)" --pair swap_C_minus_A_own="$(v C C_16)-$(v A A_16)" \
    --pair swap_S_minus_A_via_T16="$(v S T_16)-$(v A T_16)" --pair swap_S_minus_A_own="$(v S S_16)-$(v A A_16)" \
    --out $K/leak/contrasts_swap.json
.venv/bin/python $E --pair gap_C_minus_A_via_T16="$(v C T_16)-$(v A T_16)" --pair gap_C_minus_A_via_T1="$(v C T_1)-$(v A T_1)" \
    --pair gap_C_minus_A_own="$(v C C_16)-$(v A A_16)" --pair gap_S_minus_A_via_T16="$(v S T_16)-$(v A T_16)" \
    --pair gap_S_minus_A_own="$(v S S_16)-$(v A A_16)" --pair G_C_via_T16="$(v C T_16)-$(v T T_16)" \
    --pair G_A_via_T16="$(v A T_16)-$(v T T_16)" --out $K/leak/contrasts_gap.json
```

## Code

Run state 5, commit b8fff95. The check ran from b21c6b2 with exactly this diff uncommitted, so its scores record
`eval_code` b21c6b2 with these three files modified:
- `scripts/eval/eval_multistep_adaptivity.py`: `cross`, and `contrast --quantity swap_cost`;
- `activemri/utils/multistep.py`: `cross_decode`;
- `tests/test_multistep_eval.py`: decoding by the drawing run is its output, and another decoder changes it.

## Results

Run 2026-11-16 at 11:30 on the 669 dev-half slices. wandb run `d8thextx`. Files:
- `scores/<mask>__via__<decoder>.{json,npz}`;
- `scores/leak_kscr2_summary.json`;
- `leak_kscr2.log`;
- `contrasts_{swap,gap}.{json,txt}`.

### Gate: passed

- The slice hash (720f599c…), the draw seeds (1234, 4321) and the donors equal the CoreHPC scores'.
- Each arm decoded by its own VarNet reproduces its CoreHPC pooled PSNR to within 0.01 dB. The differences are C −0.0001, A −0.0007, S −0.0014 and T −0.0006 dB.
- Single slices can differ more: up to 0.84 dB for S, and 0.08–0.23 dB for the others.
  - A draw close to its threshold can flip between GPUs, and that changes the rest of the trajectory.
  - So these per-slice files must never be paired with the CoreHPC ones. Every contrast below comes from this one job.

### Primary readouts

Pooled dB, 95% CI:

| Readout, C − A | through T_16 (primary) | through T_1 (secondary) | through each arm's own VarNet (reference) |
|---|---|---|---|
| Swap cost (primary 1) | +0.184 [+0.078, +0.272] | +0.064 [−0.081, +0.183] | +0.877 [+0.772, +0.974] |
| Gap (primary 2) | −0.083 [−0.150, −0.033] | −0.238 [−0.343, −0.161] | +0.247 [+0.209, +0.286] |

By condition, which was not pre-registered:
- the swap cost C − A through T_16 is +0.364 [+0.164, +0.512] on knee-PD and +0.001 [−0.062, +0.064] on knee-PDFS;
- through T_1 it is +0.323 [+0.057, +0.513] and −0.199 [−0.308, −0.092].

So the pooled effect comes from PD.

### Reading, by the rule fixed before the run

- **Primary 1's CI is above 0, so outcome-map row 2 stands.** C's choices are slice-matched in a way that T_16's VarNet,
  which cannot read C's masks as a code, still uses.
- **The secondary decoder does not confirm it.** Through T_1 the CI includes 0, and knee-PDFS reverses.
- **Primary 2 is below 0.** It answers one way only, so it is not read.

### What this corrects

- **The +0.25 dB "ceiling" was mostly C's own VarNet.**
  - Through T_16's VarNet, none of C's +0.247 over A survives: the gap is −0.083.
  - C's swap cost falls from +0.99 to +0.26 through T_16; A's falls from +0.11 to +0.08.
- **What survives:** masks chosen from the full k-space are better matched to their slice than A's, by +0.18 dB through a
  neutral VarNet. Added after the results: that is 21% of the own-VarNet swap-cost difference (+0.877).
- **How much a better partial-data sampler could gain over A is unknown,** and well under +0.25 dB.

### Reported only: the full mask × decoder matrix

Pooled PSNR_Masked (dB):

| masks \ decoder | T_16 | T_1 | C_16 | A_16 | S_16 |
|---|---|---|---|---|---|
| C_16 | 34.642 | 34.483 | **35.020** | 34.757 | 34.789 |
| A_16 | 34.725 | 34.721 | 34.402 | **34.773** | 34.767 |
| S_16 | 34.683 | 34.646 | 34.381 | 34.741 | **34.773** |
| T_16 | **34.714** | 34.679 | 34.309 | 34.705 | 34.704 |

Swap cost, own mask minus donor mask, pooled PSNR (dB) [95% CI]:

| masks \ decoder | T_16 | T_1 | C_16 | A_16 | S_16 |
|---|---|---|---|---|---|
| C_16 | +0.264 [+0.141, +0.371] | +0.160 [−0.006, +0.302] | **+0.989** [+0.868, +1.103] | +0.350 [+0.267, +0.438] | +0.361 [+0.286, +0.436] |
| A_16 | +0.080 [+0.058, +0.102] | +0.096 [+0.070, +0.122] | +0.132 [+0.099, +0.162] | **+0.112** [+0.087, +0.136] | +0.114 [+0.090, +0.136] |
| S_16 | +0.132 [+0.101, +0.163] | +0.149 [+0.122, +0.177] | +0.211 [+0.167, +0.249] | +0.177 [+0.143, +0.207] | **+0.188** [+0.155, +0.217] |
| T_16 | 0 | 0 | 0 | 0 | 0 |

Bold cells are each arm through its own VarNet. They match `../scores`.

Observations, not readouts:
- **C's VarNet is specialised to C's masks.** It loses 0.37–0.41 dB on every other arm's masks.
- **A's and S's VarNets never saw C's masks either.** Through them, C's masks are about level with the home masks: −0.016 dB against A's masks through A's VarNet, and +0.022 against A's through S's.
- **S's extra slice-matching mostly survives a neutral VarNet.** Swap cost S − A is +0.052 [+0.033, +0.070] through T_16, against +0.076 through own VarNets. Its gap S − A is −0.041 [−0.067, −0.015] through T_16, against 0.000 through own.
- **A's own advantage mostly disappears.** A's masks minus T's, both through T_16's VarNet, is +0.011 [−0.002, +0.024], against G_A = +0.059 through own VarNets. T's VarNet has home advantage in that comparison.

### Caveats

- One seed and the dev half only; the holdout is untouched.
- An A6000 instead of the pinned card. The scorer ran uncommitted and was committed afterwards, unchanged, as b8fff95.
- Decoding through a VarNet that never trained on a mask also costs a mismatch. The swap cost cancels it within an arm, but the gaps carry it.
