#!/usr/bin/env bash
# Toy grid (CPU), all samplers through the conv sampler's clamped scores_to_prob.
#   A  frozen generalist reconstructor           B  joint from scratch
#   C  lock-in: joint from B-static's co-adapted reconstructor and mask (and frozen co-adapted reconstructor)
#   D  the same start mask with the generalist reconstructor (joint, and frozen)
#   E  weaker class difference (purity 0.8)      R  can the net represent the table's maps (direct regression)
cd "$(dirname "$0")/../.."
source .venv/bin/activate
TOY="python scripts/diagnostics/toy_adaptive_sampling.py --threads 4"
OUT=/tmp/bkadota_toy
GEN=$OUT/generalist
run() { name=$1; shift; $TOY "$@" --out $OUT/$name > $OUT/$name.log 2>&1; }
for seed in 0 1; do
  for s in static table; do
    run A_frozen_gen_${s}_s$seed --stage fit --sampler $s --recon-init $GEN --freeze-recon --steps 3000 --seed $seed &
    run B_joint_scratch_${s}_s$seed --stage fit --sampler $s --steps 6000 --seed $seed &
  done
  for input in raw label; do
    run A_frozen_gen_net-${input}_s$seed --stage fit --sampler net --net-input $input --recon-init $GEN --freeze-recon --steps 3000 --seed $seed &
    run B_joint_scratch_net-${input}_s$seed --stage fit --sampler net --net-input $input --steps 6000 --seed $seed &
  done
done
for s in table static; do
  run E_frozen_gen_${s}_p0.8 --stage fit --sampler $s --purity 0.8 --recon-init $GEN --freeze-recon --steps 3000 &
  run E_joint_scratch_${s}_p0.8 --stage fit --sampler $s --purity 0.8 --steps 6000 &
done
run E_joint_scratch_net-raw_p0.8 --stage fit --sampler net --net-input raw --purity 0.8 --steps 6000 &
run E_frozen_gen_net-raw_p0.8 --stage fit --sampler net --net-input raw --purity 0.8 --recon-init $GEN --freeze-recon --steps 3000 &
wait
for seed in 0 1; do
  START=$OUT/B_joint_scratch_static_s$seed
  for s in table net; do
    run C_lockin_coadapted_${s}_s$seed --stage fit --sampler $s --recon-init $START --sampler-init $START --steps 6000 --seed $seed &
    run C_lockin_coadapted_frozen_${s}_s$seed --stage fit --sampler $s --recon-init $START --sampler-init $START --freeze-recon --steps 3000 --seed $seed &
    run D_from_static_generalist_joint_${s}_s$seed --stage fit --sampler $s --recon-init $GEN --sampler-init $START --steps 6000 --seed $seed &
    run D_from_static_generalist_frozen_${s}_s$seed --stage fit --sampler $s --recon-init $GEN --sampler-init $START --freeze-recon --steps 3000 --seed $seed &
  done
done
run R_regress_table_s0 --stage regress --sampler-init $OUT/B_joint_scratch_table_s0 --steps 2000 --batch 64 &
wait
echo "toy done $(date)" >> $OUT/launcher.log
