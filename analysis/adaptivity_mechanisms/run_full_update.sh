#!/usr/bin/env bash
# Full-batch update alignment (common / anatomy / slice-specific parts) for the conv sampler at epoch 4 and 50.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=${GPU:-1}
LOG=analysis/adaptivity_mechanisms
until grep -q "done controls conv_ep4" $LOG/launcher.log; do sleep 30; done
for args in "conv_last" "conv_ep4 --sampler-ckpt resume_from_epoch4.pt"; do
  set -- $args; tag=$1; shift
  echo "start full_update $tag $(date)" >> $LOG/launcher.log
  python scripts/diagnostics/mask_gradient_anatomy_r8.py --tag $tag --from-saved --full-update-only "$@" > $LOG/gradients_${tag}_full_update.log 2>&1
  echo "done full_update $tag $(date) rc=$?" >> $LOG/launcher.log
done
