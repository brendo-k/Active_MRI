#!/usr/bin/env bash
# Toy race test: does the per-class table stay adaptive when its logits move as fast as the net's (lr 5e-2), and does
# the label-input net become adaptive when its population bias moves slowly (lr 1e-3)?
cd "$(dirname "$0")/../.."
source .venv/bin/activate
TOY="python scripts/diagnostics/toy_adaptive_sampling.py --threads 8"
OUT=/tmp/bkadota_toy
GEN=$OUT/generalist
$TOY --stage fit --sampler table --logit-lr 5e-2 --steps 6000 --out $OUT/B_joint_scratch_table-fastlogit_s0 > $OUT/B_joint_scratch_table-fastlogit_s0.log 2>&1 &
$TOY --stage fit --sampler table --logit-lr 5e-2 --recon-init $GEN --freeze-recon --steps 3000 --out $OUT/A_frozen_gen_table-fastlogit_s0 > $OUT/A_frozen_gen_table-fastlogit_s0.log 2>&1 &
$TOY --stage fit --sampler net --net-input label --logit-lr 1e-3 --steps 6000 --out $OUT/B_joint_scratch_net-label-slowbias_s0 > $OUT/B_joint_scratch_net-label-slowbias_s0.log 2>&1 &
$TOY --stage fit --sampler net --net-input label --logit-lr 1e-3 --recon-init $GEN --freeze-recon --steps 3000 --out $OUT/A_frozen_gen_net-label-slowbias_s0 > $OUT/A_frozen_gen_net-label-slowbias_s0.log 2>&1 &
$TOY --stage fit --sampler static --logit-lr 5e-2 --steps 6000 --out $OUT/B_joint_scratch_static-fastlogit_s0 > $OUT/B_joint_scratch_static-fastlogit_s0.log 2>&1 &
wait
echo "toy race done $(date)" >> $OUT/launcher.log
