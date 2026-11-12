# Phase 2: the sweep, the floor pair and the ablations (CoreHPC, run state 2 = 44c3fba)

Submitted 11-10 at about 22:30 PST, once every pilot gate had passed (`../p1/README.md`), as run.md section 4 lists. Each
job has a 1-day limit on the RTX PRO 6000. `train.sbatch` scores each final checkpoint when its job ends. A watcher
copies each job's log and scores here as it finishes.

| Arm | Job | Role |
|-----|-----|------|
| A_1_s42, T_1_s42 | 3001463, 3001464 | K=1 pair, base 42: the null anchor G_1 |
| A_4_s42, T_4_s42 | 3001465, 3001466 | K=4 pair, base 42 |
| A_1_s43, T_1_s43 | 3001467, 3001468 | K=1 pair, base 43 |
| A_4_s43, T_4_s43 | 3001469, 3001470 | K=4 pair, base 43 |
| A_16_s43, T_16_s43 | 3001471, 3001472 | K=16 pair, base 43. The base-42 pair is the pilot, jobs 3000520 and 3000521 |
| T_1_s42_f142, T_16_s42_f142 | 3001473, 3001474 | twin-vs-twin noise floor F_1 and F_16 (fine-tune seed 142) |
| A_4_s42_rec | 3001475 | recon-input ablation, one of the FAIL checks |
| A_16_s42_img | 3001476 | k-space channel ablation, reported only |

- **Where the masks come from.** Every arm fine-tunes from B_42's or B_43's learned mask and VarNet. Those bases learned
  their masks from the VD p=4 prior (`../p1/README.md`). `feedback_no_vd_start` allows reusing such masks if the report
  says where they came from, so the report has to.
- **Compute estimate: about 30 GPU-hours.** A K=16 run takes about 2.9 h, from the pilot's epoch times. A K=1 run takes
  about 1.5 h, from the bases. K=4 has no full-epoch timing yet; about 2 h is assumed, between the two.
- **All 14 completed 0:0** between 11-11 00:16 and 05:24, about 38 GPU-hours in all (K=1 ~1.6 h, K=4 ~1.8 h, K=16
  ~3 h, A_4_s42_rec 4.1 h). Logs and scores are in this folder; the contrasts are in `contrasts.md`.
- **P2b as frozen is replaced.** On 11-11 the user chose a uniform-noise start and a knee-only from-scratch subproblem
  first (`../addendum_2026-11-11.md`).
