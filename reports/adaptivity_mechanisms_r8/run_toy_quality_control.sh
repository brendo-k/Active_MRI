#!/usr/bin/env bash
# Toy control for the C-vs-D frozen contrast: a well-fitted reconstructor co-adapted to class-specific masks (from the
# joint table run) instead of the static mask. Does the raw-input net adapt from the static mask through it?
cd "$(dirname "$0")/../.."
source .venv/bin/activate
TOY="python scripts/diagnostics/toy_adaptive_sampling.py --threads 8"
OUT=/tmp/bkadota_toy
for seed in 0 1; do
  RECON=$OUT/B_joint_scratch_table_s$seed
  START=$OUT/B_joint_scratch_static_s$seed
  $TOY --stage fit --sampler net --net-input raw --recon-init $RECON --sampler-init $START --freeze-recon --steps 3000 --seed $seed --out $OUT/Q_frozen_tablerecon_from_static_net_s$seed > $OUT/Q_frozen_tablerecon_from_static_net_s$seed.log 2>&1 &
  $TOY --stage gradients --recon-init $RECON --sampler-init $START --seed $seed --out $OUT/G_gradients_tablerecon_s$seed > $OUT/G_gradients_tablerecon_s$seed.log 2>&1 &
done
wait
echo "toy quality control done $(date)" >> $OUT/launcher.log
