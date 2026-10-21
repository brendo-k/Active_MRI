#!/usr/bin/env bash
# Toy net variants: how decodable the class is from the sampler input (raw centre, its magnitude, the label) and
# the head initialisation, under the frozen generalist (A) and joint from scratch (B). Plus the decodability probe.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
TOY="python scripts/diagnostics/toy_adaptive_sampling.py --threads 8"
OUT=/tmp/bkadota_toy
GEN=$OUT/generalist
for purity in 1.0 0.8 0.65; do
  $TOY --stage probe --purity $purity --out $OUT/probe_p$purity > $OUT/probe_p$purity.log 2>&1 &
done
for variant in "magnitude zero" "label zero" "raw default"; do
  set -- $variant
  $TOY --stage fit --sampler net --net-input $1 --net-init $2 --recon-init $GEN --freeze-recon --steps 3000 --out $OUT/A_frozen_gen_net-$1-$2_s0 > $OUT/A_frozen_gen_net-$1-$2_s0.log 2>&1 &
  $TOY --stage fit --sampler net --net-input $1 --net-init $2 --steps 6000 --out $OUT/B_joint_scratch_net-$1-$2_s0 > $OUT/B_joint_scratch_net-$1-$2_s0.log 2>&1 &
done
wait
echo "toy variants done $(date)" >> $OUT/launcher.log
