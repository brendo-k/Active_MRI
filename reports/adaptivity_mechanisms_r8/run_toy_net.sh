#!/usr/bin/env bash
# Toy net-sampler arms of run_toy.sh, rerun after restricting the net's input to the acquired 4x4 centre.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
TOY="python scripts/diagnostics/toy_adaptive_sampling.py --threads 8"
OUT=/tmp/bkadota_toy
GEN=$OUT/generalist
for seed in 0 1; do
  $TOY --stage fit --sampler net --recon-init $GEN --freeze-recon --steps 3000 --seed $seed --out $OUT/A_frozen_gen_net_s$seed > $OUT/A_frozen_gen_net_s$seed.log 2>&1 &
  $TOY --stage fit --sampler net --steps 6000 --seed $seed --out $OUT/B_joint_scratch_net_s$seed > $OUT/B_joint_scratch_net_s$seed.log 2>&1 &
done
for purity in 0.8 0.65; do
  $TOY --stage fit --sampler net --steps 6000 --purity $purity --out $OUT/E_joint_scratch_net_p$purity > $OUT/E_joint_scratch_net_p$purity.log 2>&1 &
done
wait
echo "toy net done $(date)" >> $OUT/launcher.log
