# <Feature name> — Design

Status: draft | active | blocked (<reason>) | done
Last verdict: <YYYY-MM-DD, report link>

## Question

<One or two sentences. What per-patient information do we exploit, and what should it change in the acquisition?>

## Fixed setup

- Dataset: <e.g. fastMRI local v1, FLAIR + knee>
- Acceleration: <R=8>
- Reconstructor: <E2E VarNet, backbone, lr>
- Metric path: <trainer apodized metric mask; PSNR / NMSE / SSIM_Masked>
- Controls: <non-adaptive twin; seed-swap control; seeds>
- Single seed is fine for now
- Do not mention noise floor

## Decisions

| # | Decision | Why | Evidence |
|---|----------|-----|----------|
| 1 | <decision> | <reason> | <reports/... or wandb run> |

## Settled results (do not relitigate)

- **<Result name>** — <NO-GO | PASS | NULL>: <one line with numbers>. Evidence: <link>. Reopen only if: <condition>.

## Open questions

- <Question> — blocks roadmap item <n>

## Success rule

<Numeric, pre-registered. e.g. own-prediction minus condition-mean > X dB over N seeds, with the seed-swap control not explaining it.>

## Constraints

- <Hard constraints that may not be in Agents.md, Gotchas that apply here>

## Links

- Roadmap: [roadmap.md](roadmap.md)
- Related docs: <docs/...>
- Reports: <reports/...>
