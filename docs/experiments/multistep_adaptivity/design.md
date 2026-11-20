# Multi-step adaptive sampling — Design

Status: active
Last verdict: 2026-11-19, merged (`reports/multistep_adaptivity/report.md`)

## Question

Does a sampler that reads the k-space it has already measured, over several steps, pick better samples for that scan
than its data-blind twin with the same budget? This is roadmap step 3 in `AGENTS.md`.

## Fixed setup

- Dataset: fastMRI FLAIR tier F (all 339 usable volumes) since fscr1; the earlier kscr runs used knee PD + PDFS
- Acceleration: R=8
- Reconstructor: E2E VarNet, 6 cascades × 18 channels (fscr1)
- Metric path: trainer apodized metric mask; PSNR / NMSE / SSIM_Masked
- Controls: the data-blind twin (`sampler.data_blind=true`) and one-step LOUPE, seed 42
- Single seed is fine for now

## Decisions

| # | Decision | Why | Evidence |
|---|----------|-----|----------|
| 1 | No claim about SPACE (echo-train) sampling yet | These runs are order-free; the echo-train evolution is not modelled | `AGENTS.md`, roadmap step 3 |

## Settled results (do not relitigate)

- **Order-free multi-step adaptivity at R=8** — NULL: no multi-step sampler beat its matched data-blind twin by the
  frozen bars (fine-tuned G ≤ +0.018 dB; from scratch +0.06 to +0.08 dB, knee, one seed). On FLAIR tier F, SeqMRI's
  design minus one-step LOUPE is +0.11 dB with a CI that includes 0. Evidence: `reports/multistep_adaptivity/report.md`.

## Open questions

- Is F under-converged at epoch 100, or was its sampler lr (Adam 1e-3) too high until the step at 90? — roadmap item 1
- Is F's fscr1 lead scan-type routing for the 1.5 T narrow-k-space scans? A header-routed LOUPE is the test. — roadmap item 2
- Do the one-sided masks use conjugate symmetry (partial Fourier)? — run under the loss_domain line

## Success rule

Set per experiment in its `reports/<topic>/proposal.md`. The multi-step bars (`reports/multistep_adaptivity/proposal.md`):
an adaptive gain over the twin of at least +0.10 dB passes, and below +0.05 dB fails.

## Constraints

- Every adaptive sampler is compared with its data-blind twin, trained the same way (`AGENTS.md`).

## Links

- Roadmap: [roadmap.md](roadmap.md)
- Reports: `reports/multistep_adaptivity/`
- Related: [loss_domain](../loss_domain/) (the partial-Fourier follow-up of fscr1), [adaptivity_levers](../adaptivity_levers/)
