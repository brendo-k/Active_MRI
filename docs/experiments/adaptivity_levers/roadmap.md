# Adaptivity levers — Roadmap

Generated from [design.md](design.md) on 2026-11-20. Edit freely; the loop runs the first unchecked item whose dependencies are done.

## Awaiting verdict

- [ ] **0. Encoding census and literature synthesis** — branch `report/adaptivity_levers__adaptivity_levers_r8`
  - Renamed from `report/adaptivity_levers_r8` on 2026-11-20; its report still names the old branch.
  - Report: `reports/adaptivity_levers_r8/report.md`, on the branch. No GPU run, no checkpoint.

## Next (proposed in that report, not yet approved)

- [ ] **1. Mask-learner stability benchmark** — several seeds each of LOUPE, an annealed exact-count top-k and a swap
  search that accepts only measured improvements, through one frozen VarNet; score seed-to-seed mask overlap and PSNR spread.
- [ ] **2. Rate-distortion census (eval only)** — per-volume PSNR at R = 4, 6, 8, 12, 16 through CG-SENSE with ecalib
  maps and one multi-R VarNet.
- [ ] **3. 2 × 2 specialisation test on tier L** — {shared mask, per-class frozen masks} × {shared VarNet, trunk +
  per-class adapters}, with a shuffled class-to-adapter twin.
- [ ] **4. Regime sweep** — the shared-versus-adaptive comparison with 1D lines and with 1, 2, 4, 8, 16 virtual coils,
  plus a same-patient PD → PDFS oracle-correlation check.

## Done

None yet.
