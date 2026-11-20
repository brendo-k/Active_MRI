# Adaptivity levers — Design

Status: draft
Last verdict: none yet (`report/adaptivity_levers__adaptivity_levers_r8` awaits one)

## Question

Which adaptive levers can still give a larger benefit than the 0.05–0.2 dB measured so far, what do the papers that
report adaptive gains have in common, and what should be run next?

## Fixed setup

- Dataset: fastMRI local v1 val (FLAIR, T2, knee PD and PDFS)
- Acceleration: R=8, with the census also at R=4
- Reconstructor: none trained; a CPU census of linear-SENSE noise amplification (g-factor)
- Metric path: the trainer's apodized metric mask enters only as the object support
- Single seed is fine for now

## Decisions

None yet.

## Settled results (do not relitigate)

None yet. The report's findings, awaiting a verdict (`reports/adaptivity_levers_r8/report.md`, on the branch):

- Mask shape at a fixed budget is a flat direction with many coils: at most 0.07 dB per-patient noise gain in the census.
- Encoding difficulty spans about 15 dB at R=8, and object fraction and coil count explain 86% of it, so the budget
  (R per scan) is the lever with the most between-patient spread. Water-filling R per volume gives +0.39 dB mean
  under the linear model.
- A stable mask learner is a prerequisite for any further adaptive comparison.

## Open questions

- The four experiments the report proposes — roadmap Next

## Success rule

Set per experiment in its `reports/<topic>/proposal.md`.

## Constraints

- The census is linear SENSE, not VarNet PSNR.

## Links

- Roadmap: [roadmap.md](roadmap.md)
- Report: `reports/adaptivity_levers_r8/report.md` on branch `report/adaptivity_levers__adaptivity_levers_r8`
- Related: [multistep_adaptivity](../multistep_adaptivity/)
