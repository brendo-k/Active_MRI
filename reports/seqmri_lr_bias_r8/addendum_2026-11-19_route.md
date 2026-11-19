# Addendum 2026-11-19: the per-step maps under a gradient route that credits every step

This changes arm B of `proposal.md` before any of it ran. Arms R and S, their rows, the guards and the noise-floor rule
are unchanged.

## Why

- The frozen arm B added one learned logit map per step to F and to its twin, under SeqMRI's routes.
- SeqMRI detaches the mask between steps. An earlier step's draws reach the loss only through the in-loop
  reconstructions that later steps read. The twin reads none.
- One training forward and backward pass of fscr1's F_tF and Fb_tF with zero maps added, on two FLAIR dev slices
  (`scripts/diagnostics/seqmri_step_bias_gradients.py`), gave these gradient norms of each step's map:
  - F: 1.8e-6, 3.0e-6, 4.1e-6 and 7.0e-6 for steps 0–3;
  - the twin: exactly 0 for steps 0–2, and 5.5e-6 for step 3.
- So the twin's first three maps would never move. B2 (F_bias − Fb_bias) and B3 (Fb_bias − L) would compare F with four
  learned maps against a twin with one.
- **The user's decision (2026-11-19):** keep per-step maps, and give earlier draws a gradient route. "This is more
  principled." Both F and the twin then train differently from fscr1, so they get new controls.

## The route

- `sampler.seqmri_union_route=true`: the union of the draws keeps every step's straight-through increment, as the loops
  of the other samplers do. The loss on the final mask then credits each step's draws directly.
- F's in-loop reconstructions stay attached, as in SeqMRI.
- The forward pass is unchanged. With the route on, the draws, masks and output are bit-identical to those without it
  (`tests/test_seqmri_step_bias.py`); only the gradients change.
- With the route, the same check gives:
  - F: 7.3e-6, 7.7e-6, 7.1e-6 and 7.0e-6;
  - the twin: 4.2e-6, 4.6e-6, 4.6e-6 and 5.5e-6.
- The unit tests confirm that every step's map gets gradient, and that without the route the twin's earlier maps get
  none.

## Arms (replacing B)

| Arm | Change from fscr1's tier-F arm | Role |
|---|---|---|
| F_4_s42_route_tF, Fb_4_s42_route_tF | the route | controls of the maps |
| F_4_s42_routebias_tF, Fb_4_s42_routebias_tF | the route and the per-step maps (`sampler.seqmri_step_bias=true`) | the maps |

## Rows (replacing B1–B3; paired per slice, pooled PSNR, 95% volume bootstrap, seed-42 screen)

| Row | Contrast | Pass | CI includes 0 | CI below 0 |
|---|---|---|---|---|
| B1 | F_routebias − F_route | ≥ +0.10 dB, CI above 0: per-pixel per-step maps help F | no effect | they hurt |
| B2 | F_routebias − Fb_routebias | ≥ +0.10, CI above 0: reading the data pays over a twin with the same maps, both credited at every step | data-blind | reading the data costs |
| B3 | Fb_routebias − L_tF | CI above or including 0: the twin no longer trails LOUPE | | still trails |

- Reported, not ruled: the route's own effect, F_route − F_tF and Fb_route − Fb_tF. The routes differ, and there is one
  seed.
- Expected:
  - the route's effect: within ±0.1 dB;
  - B1: +0.1 to +0.3;
  - B2: 0 to +0.1;
  - B3: about 0.

## Code and compute (run state 2)

- `sampler.seqmri_union_route`, with tests. The launcher arms `route` and `routebias`; the frozen `bias` arms are
  refused.
- The four runs take about 51 GPU-h on CoreHPC: F about 19 h each, Fb about 6.5 h each.
- Run state 1's jobs read `~/scratch/seqmri_lr_bias_r8/code`. So run state 2 ships to its own experiment folder,
  `~/scratch/seqmri_lr_bias_r8_rs2`, and its validation job runs with `RESUME_ARMS=''` and the four route arms.
