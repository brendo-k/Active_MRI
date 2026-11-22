#!/bin/bash
# Regenerates every result file in this folder from the score files in scores/. No GPU.
#
# scores/*.{json,npz} are the scorer's output: the nine new runs were scored on CoreHPC by train.sbatch right after
# training, on the same card, at run state 3 (49be699):
#   python scripts/eval/eval_multistep_adaptivity.py score --run <run dir> --name <run> --out <scores> --gradients
# and the three fscr1 controls are copies of reports/multistep_adaptivity/fscr1/scores (d58087f). Rescoring needs the
# checkpoints (CoreHPC ~/scratch/multistep_adaptivity__seqmri_lr_bias_r8{,_rs2}/runs) and a GPU.
# Delete contrasts*.json, summary.*, by_scan_type.*, curves.json and figures/{curves,contrasts,data_swap}.png to force
# a recompute; every command below overwrites its outputs anyway.
set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity__seqmri_lr_bias_r8/scores --out reports/multistep_adaptivity__seqmri_lr_bias_r8/contrasts.json --pair 'R1=(F_4_s42_fscr1_tF_r150-L_1_s42_fscr1_tF_r150)-(F_4_s42_fscr1_tF-L_1_s42_fscr1_tF)' --pair S1=F_4_s42_lrs_tF-F_4_s42_fscr1_tF --pair B1=F_4_s42_routebias_tF-F_4_s42_route_tF --pair B2=F_4_s42_routebias_tF-Fb_4_s42_routebias_tF --pair B3=Fb_4_s42_routebias_tF-L_1_s42_fscr1_tF
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity__seqmri_lr_bias_r8/scores --out reports/multistep_adaptivity__seqmri_lr_bias_r8/contrasts_secondary.json --pair F150_minus_F100=F_4_s42_fscr1_tF_r150-F_4_s42_fscr1_tF --pair Fb150_minus_Fb100=Fb_4_s42_fscr1_tF_r150-Fb_4_s42_fscr1_tF --pair L150_minus_L100=L_1_s42_fscr1_tF_r150-L_1_s42_fscr1_tF --pair F150_minus_L150=F_4_s42_fscr1_tF_r150-L_1_s42_fscr1_tF_r150 --pair Flrs_minus_L=F_4_s42_lrs_tF-L_1_s42_fscr1_tF --pair Froutebias_minus_L=F_4_s42_routebias_tF-L_1_s42_fscr1_tF --pair Fblrs_minus_Fb=Fb_4_s42_lrs_tF-Fb_4_s42_fscr1_tF --pair Flrs_minus_Fblrs=F_4_s42_lrs_tF-Fb_4_s42_lrs_tF --pair Fblrs_minus_L=Fb_4_s42_lrs_tF-L_1_s42_fscr1_tF --pair route_F=F_4_s42_route_tF-F_4_s42_fscr1_tF --pair route_Fb=Fb_4_s42_route_tF-Fb_4_s42_fscr1_tF --pair maps_Fb=Fb_4_s42_routebias_tF-Fb_4_s42_route_tF --pair Froute_minus_Fbroute=F_4_s42_route_tF-Fb_4_s42_route_tF
python reports/multistep_adaptivity__seqmri_lr_bias_r8/analysis/summarise.py all
