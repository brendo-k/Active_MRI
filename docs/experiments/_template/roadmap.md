# <Feature name> — Roadmap

Generated from [design.md](design.md) on <YYYY-MM-DD>. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Next

- [ ] **1. <Experiment name>** — branch `exp/<topic>`
  - Question: <what this answers, tied to a design.md open question or decision>
  - Control: <what is held fixed / the baseline>
  - Treatment: <the single change>
  - Expected outcome: <direction and rough size in dB>
  - Success criterion: <numeric, per design.md success rule>
  - Failure means: <what we conclude and which design.md entry to add>
  - Depends on: <none | item n>
  - Compute: <GPUs, runs, seeds, rough hours>

- [ ] **2. <Experiment name>** — branch `exp/<topic>`
  - Question:
  - Control:
  - Treatment:
  - Expected outcome:
  - Success criterion:
  - Failure means:
  - Depends on: 1
  - Compute:

## Later / unscheduled

- <Idea, not yet specified enough to run>

## Done

- [x] **0. <Experiment name>** — verdict: PASS | FAIL | NULL (<YYYY-MM-DD>)
  - Report: reports/<name>/
  - Branch: `exp/<topic>` @ <commit hash> (merged | parked)
  - Design change: <design.md decision or settled result added, or none>
