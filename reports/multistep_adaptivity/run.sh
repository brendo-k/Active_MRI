#!/usr/bin/env bash
# Regenerates the result files of reports/multistep_adaptivity/, the whole experiment from P0 to fscr1.
# Section 1 trains and scores on CoreHPC (SLURM; see scripts/launch/hpc/multistep_adaptivity/run.md), at each phase's
# run state; it only runs with RUN_COREHPC=1, from a CoreHPC shell holding the shipped code. Sections 2-5 run on the lab
# server from the score files in this folder (section 5's render stage needs GPU 0 and fscr1's six last_model.pt).
# Every result is overwritten in place; delete a file to see it recomputed.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=reports/multistep_adaptivity

# 1. CoreHPC: validation, training, and train.sbatch's scoring of each final checkpoint into scores/.
#    Ship the code at the run state named in each comment first (run.md section 1).
if [[ "${RUN_COREHPC:-0}" == 1 ]]; then
    J='--gres=gpu:nvidia_rtx_pro_6000:1'; C=code/scripts/launch/hpc/multistep_adaptivity; G='RTX PRO 6000'
    # P1 bases and pilot, then the P2 sweep, floor pair and ablations (run state 2 = 44c3fba)
    for s in 42 43; do sbatch $J --time=1-00:00:00 --export=ALL,ARM=B_$s,GPU_NAME="$G" $C/train.sbatch; done
    for arm in A_16_s42 T_16_s42; do sbatch $J --export=ALL,ARM=$arm,CHECKPOINT=epoch_010.pt,GPU_NAME="$G" $C/eval.sbatch; done
    for arm in A_1_s42 T_1_s42 A_4_s42 T_4_s42 A_16_s42 T_16_s42 A_1_s43 T_1_s43 A_4_s43 T_4_s43 A_16_s43 T_16_s43 \
               T_1_s42_f142 T_16_s42_f142 A_4_s42_rec A_16_s42_img; do
        sbatch $J --time=1-00:00:00 --export=ALL,ARM=$arm,GPU_NAME="$G" $C/train.sbatch
    done
    # kscr1 (run state 3 = 707a5c9)
    for arm in A_16_s42_kscr1 T_16_s42_kscr1 T_16_s142_kscr1; do
        sbatch $J --time=1-00:00:00 --export=ALL,ARM=$arm,GPU_NAME="$G" $C/train.sbatch
    done
    # kscr2 (run state 4 = b21c6b2)
    V=$(sbatch --parsable $J --export=ALL,GPU_NAME="$G",EXPECTED_COMMIT=b21c6b2 $C/validate_kscr2.sbatch)
    for arm in T_16_s42_kscr2 A_16_s42_kscr2 C_16_s42_kscr2 T_1_s42_kscr2; do
        sbatch $J --dependency=afterok:$V --time=1-00:00:00 --export=ALL,ARM=$arm,GPU_NAME="$G" $C/train.sbatch
    done
    sbatch $J --dependency=afterok:$V --time=2-00:00:00 --export=ALL,ARM=S_16_s42_kscr2,GPU_NAME="$G" $C/train.sbatch
    # fscr1 (run state 7 = d58087f)
    V=$(sbatch --parsable $J --export=ALL,GPU_NAME="$G",EXPECTED_COMMIT=d58087f $C/validate_fscr1.sbatch)
    for arm in F_4_s42_fscr1_tF Fb_4_s42_fscr1_tF L_1_s42_fscr1_tF; do
        sbatch $J --dependency=afterok:$V --time=2-00:00:00 --export=ALL,ARM=$arm,GPU_NAME="$G" $C/train.sbatch
    done
    for arm in F_4_s42_fscr1_tS5 Fb_4_s42_fscr1_tS5 L_1_s42_fscr1_tS5; do
        sbatch $J --dependency=afterok:$V --time=1-00:00:00 --export=ALL,ARM=$arm,GPU_NAME="$G" $C/train.sbatch
    done
    sbatch $J --export=ALL,ARM=F_4_s42_fscr1_tF,CHECKPOINT=epoch_010.pt,GPU_NAME="$G" $C/eval.sbatch
    exit 0
fi
# kscr3 trained and scored on the lab server (run state 6 = 0329845, GPUs 0-1), into kscr3/scores; opt in with RUN_KSCR3=1.
if [[ "${RUN_KSCR3:-0}" == 1 ]]; then
    bash scripts/launch/train_multistep_kscr3_local.sh
fi

# 2. Contrasts. P2's were computed over P1's and P2's scores together; kscr2's cascade comparison over kscr1's and kscr2's.
P1P2_SCORES=$(mktemp -d); CASCADE_SCORES=$(mktemp -d)
ln -s "$PWD/$R"/p1/scores/* "$PWD/$R"/p2/scores/* "$P1P2_SCORES"/
ln -s "$PWD/$R"/kscr1/scores/* "$PWD/$R"/kscr2/scores/* "$CASCADE_SCORES"/
python scripts/eval/eval_multistep_adaptivity.py contrast --scores $P1P2_SCORES \
    --pair 'G_1_s42=A_1_s42-T_1_s42' --pair 'G_4_s42=A_4_s42-T_4_s42' --pair 'G_16_s42=A_16_s42-T_16_s42' --pair 'spacing_4_s42=T_4_s42-T_1_s42' --pair 'U_4_s42=A_4_s42-T_1_s42' --pair 'spacing_16_s42=T_16_s42-T_1_s42' --pair 'U_16_s42=A_16_s42-T_1_s42' --pair 'G_1_s43=A_1_s43-T_1_s43' --pair 'G_4_s43=A_4_s43-T_4_s43' --pair 'G_16_s43=A_16_s43-T_16_s43' --pair 'spacing_4_s43=T_4_s43-T_1_s43' --pair 'U_4_s43=A_4_s43-T_1_s43' --pair 'spacing_16_s43=T_16_s43-T_1_s43' --pair 'U_16_s43=A_16_s43-T_1_s43' --pair 'F_1=T_1_s42-T_1_s42_f142' --pair 'F_16=T_16_s42-T_16_s42_f142' --pair 'G_rec_4=A_4_s42_rec-T_4_s42' --pair 'chan_16=A_16_s42-A_16_s42_img' --pair 'G_img_16=A_16_s42_img-T_16_s42' \
    --out reports/multistep_adaptivity/contrasts.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/fscr1/scores \
    --pair 'U_tF=F_4_s42_fscr1_tF-L_1_s42_fscr1_tF' --pair 'G_tF=F_4_s42_fscr1_tF-Fb_4_s42_fscr1_tF' --pair 'U_tS5=F_4_s42_fscr1_tS5-L_1_s42_fscr1_tS5' --pair 'G_tS5=F_4_s42_fscr1_tS5-Fb_4_s42_fscr1_tS5' --pair 'U_data=(F_4_s42_fscr1_tF-L_1_s42_fscr1_tF)-(F_4_s42_fscr1_tS5-L_1_s42_fscr1_tS5)' --pair 'G_data=(F_4_s42_fscr1_tF-Fb_4_s42_fscr1_tF)-(F_4_s42_fscr1_tS5-Fb_4_s42_fscr1_tS5)' \
    --out reports/multistep_adaptivity/fscr1/contrasts.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/fscr1/scores \
    --pair 'B_tF=Fb_4_s42_fscr1_tF-L_1_s42_fscr1_tF' --pair 'B_tS5=Fb_4_s42_fscr1_tS5-L_1_s42_fscr1_tS5' --pair 'B_data=(Fb_4_s42_fscr1_tF-L_1_s42_fscr1_tF)-(Fb_4_s42_fscr1_tS5-L_1_s42_fscr1_tS5)' --pair 'data_F=F_4_s42_fscr1_tF-F_4_s42_fscr1_tS5' --pair 'data_Fb=Fb_4_s42_fscr1_tF-Fb_4_s42_fscr1_tS5' --pair 'data_L=L_1_s42_fscr1_tF-L_1_s42_fscr1_tS5' --pair 'F_ep100_minus_ep10=F_4_s42_fscr1_tF-F_4_s42_fscr1_tF_epoch_010' \
    --out reports/multistep_adaptivity/fscr1/contrasts_descriptive.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/kscr1/scores \
    --pair 'G_16_kscr1=A_16_s42_kscr1-T_16_s42_kscr1' --pair 'F_16_kscr1=T_16_s42_kscr1-T_16_s142_kscr1' \
    --out reports/multistep_adaptivity/kscr1/contrasts.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/kscr2/scores \
    --pair 'G_16_A_kscr2=A_16_s42_kscr2-T_16_s42_kscr2' --pair 'G_16_C_kscr2=C_16_s42_kscr2-T_16_s42_kscr2' --pair 'G_16_S_kscr2=S_16_s42_kscr2-T_16_s42_kscr2' --pair 'S_minus_A_kscr2=S_16_s42_kscr2-A_16_s42_kscr2' --pair 'C_minus_A_kscr2=C_16_s42_kscr2-A_16_s42_kscr2' --pair 'U_16_kscr2=A_16_s42_kscr2-T_1_s42_kscr2' --pair 'T16_minus_T1_kscr2=T_16_s42_kscr2-T_1_s42_kscr2' --pair 'S_minus_T1_kscr2=S_16_s42_kscr2-T_1_s42_kscr2' --pair 'C_minus_T1_kscr2=C_16_s42_kscr2-T_1_s42_kscr2' \
    --out reports/multistep_adaptivity/kscr2/contrasts.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/kscr2/scores \
    --pair 'C_minus_S_kscr2=C_16_s42_kscr2-S_16_s42_kscr2' \
    --out reports/multistep_adaptivity/kscr2/contrasts_c_minus_s.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores $CASCADE_SCORES \
    --pair 'A6_minus_A12=A_16_s42_kscr2-A_16_s42_kscr1' --pair 'T6_minus_T12=T_16_s42_kscr2-T_16_s42_kscr1' \
    --out reports/multistep_adaptivity/kscr2/contrasts_cascades.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/kscr2/leak/scores \
    --pair 'gap_C_minus_A_via_T16=C_16_s42_kscr2__via__T_16_s42_kscr2-A_16_s42_kscr2__via__T_16_s42_kscr2' --pair 'gap_C_minus_A_via_T1=C_16_s42_kscr2__via__T_1_s42_kscr2-A_16_s42_kscr2__via__T_1_s42_kscr2' --pair 'gap_C_minus_A_own=C_16_s42_kscr2__via__C_16_s42_kscr2-A_16_s42_kscr2__via__A_16_s42_kscr2' --pair 'gap_S_minus_A_via_T16=S_16_s42_kscr2__via__T_16_s42_kscr2-A_16_s42_kscr2__via__T_16_s42_kscr2' --pair 'gap_S_minus_A_own=S_16_s42_kscr2__via__S_16_s42_kscr2-A_16_s42_kscr2__via__A_16_s42_kscr2' --pair 'G_C_via_T16=C_16_s42_kscr2__via__T_16_s42_kscr2-T_16_s42_kscr2__via__T_16_s42_kscr2' --pair 'G_A_via_T16=A_16_s42_kscr2__via__T_16_s42_kscr2-T_16_s42_kscr2__via__T_16_s42_kscr2' \
    --out reports/multistep_adaptivity/kscr2/leak/contrasts_gap.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/kscr2/leak/scores --quantity swap_cost \
    --pair 'swap_C_minus_A_via_T16=C_16_s42_kscr2__via__T_16_s42_kscr2-A_16_s42_kscr2__via__T_16_s42_kscr2' --pair 'swap_C_minus_A_via_T1=C_16_s42_kscr2__via__T_1_s42_kscr2-A_16_s42_kscr2__via__T_1_s42_kscr2' --pair 'swap_C_minus_A_own=C_16_s42_kscr2__via__C_16_s42_kscr2-A_16_s42_kscr2__via__A_16_s42_kscr2' --pair 'swap_S_minus_A_via_T16=S_16_s42_kscr2__via__T_16_s42_kscr2-A_16_s42_kscr2__via__T_16_s42_kscr2' --pair 'swap_S_minus_A_own=S_16_s42_kscr2__via__S_16_s42_kscr2-A_16_s42_kscr2__via__A_16_s42_kscr2' \
    --out reports/multistep_adaptivity/kscr2/leak/contrasts_swap.json
python scripts/eval/eval_multistep_adaptivity.py contrast --scores reports/multistep_adaptivity/kscr3/scores \
    --pair 'F_minus_L=F_4_s42_kscr3-L_1_s42_kscr3' --pair 'F_minus_Lc=F_4_s42_kscr3-Lc_1_s42_kscr3' --pair 'Lc_minus_L=Lc_1_s42_kscr3-L_1_s42_kscr3' \
    --out reports/multistep_adaptivity/kscr3/contrasts.json

# 3. kscr2's leak check: every arm's masks through every VarNet (checkpoints copied from CoreHPC to CK on 11-16).
CK=~/data/seqmri_checkpoints/multistep_kscr2
K=$R/kscr2
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_multistep_adaptivity.py cross --name leak_kscr2 \
    --masks C_16_s42_kscr2=$CK/C_16_s42_kscr2 --masks A_16_s42_kscr2=$CK/A_16_s42_kscr2 \
    --masks S_16_s42_kscr2=$CK/S_16_s42_kscr2 --masks T_16_s42_kscr2=$CK/T_16_s42_kscr2 \
    --decoders T_16_s42_kscr2=$CK/T_16_s42_kscr2 --decoders T_1_s42_kscr2=$CK/T_1_s42_kscr2 \
    --decoders C_16_s42_kscr2=$CK/C_16_s42_kscr2 --decoders A_16_s42_kscr2=$CK/A_16_s42_kscr2 \
    --decoders S_16_s42_kscr2=$CK/S_16_s42_kscr2 \
    --reference $K/scores --data-root 'hpc=/export/scratch/bkadota/fastmri_raw/{anatomy}' \
    --data-root local_batch2=/export/scratch/bkadota/fastmri_raw/knee --out $K/leak/scores
v() { echo "${1}_16_s42_kscr2__via__${2}_s42_kscr2"; }
E="scripts/eval/eval_multistep_adaptivity.py contrast --scores $K/leak/scores"
python $E --quantity swap_cost --pair swap_C_minus_A_via_T16="$(v C T_16)-$(v A T_16)" \
    --pair swap_C_minus_A_via_T1="$(v C T_1)-$(v A T_1)" --pair swap_C_minus_A_own="$(v C C_16)-$(v A A_16)" \
    --pair swap_S_minus_A_via_T16="$(v S T_16)-$(v A T_16)" --pair swap_S_minus_A_own="$(v S S_16)-$(v A A_16)" \
    --out $K/leak/contrasts_swap.json
python $E --pair gap_C_minus_A_via_T16="$(v C T_16)-$(v A T_16)" --pair gap_C_minus_A_via_T1="$(v C T_1)-$(v A T_1)" \
    --pair gap_C_minus_A_own="$(v C C_16)-$(v A A_16)" --pair gap_S_minus_A_via_T16="$(v S T_16)-$(v A T_16)" \
    --pair gap_S_minus_A_own="$(v S S_16)-$(v A A_16)" --pair G_C_via_T16="$(v C T_16)-$(v T T_16)" \
    --pair G_A_via_T16="$(v A T_16)-$(v T T_16)" --out $K/leak/contrasts_gap.json

# 4. Post-hoc analyses (not pre-registered). kscr3's probes write PNGs outside the repository.
python reports/multistep_adaptivity/fscr1/analysis/by_scan_type.py
mkdir -p ~/seqmri_figures
CUDA_VISIBLE_DEVICES=0 python reports/multistep_adaptivity/kscr3/analysis/blind_probe.py
CUDA_VISIBLE_DEVICES=0 python reports/multistep_adaptivity/kscr3/analysis/rotate_probe.py ~/seqmri_figures/kscr3_rotate_probe.png
python reports/multistep_adaptivity/kscr3/analysis/loupe_speckle.py ~/seqmri_figures/kscr3_loupe_speckle.png
python reports/multistep_adaptivity/kscr3/analysis/per_scan_speckle.py ~/seqmri_figures/kscr3_per_scan_speckle.png
python reports/multistep_adaptivity/kscr3/analysis/render_mask_compare.py ~/seqmri_figures/kscr3_mask_compare.png

# 5. fscr1 figures: panels from the six final checkpoints (GPU), then the PNGs and the training curves (CPU).
CUDA_VISIBLE_DEVICES=0 python reports/multistep_adaptivity/fscr1/analysis/render_figures.py render
python reports/multistep_adaptivity/fscr1/analysis/render_figures.py plot
python reports/multistep_adaptivity/fscr1/analysis/render_figures.py curves
