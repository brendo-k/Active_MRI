#!/usr/bin/env bash
# Regenerates every result file in reports/varnet_train_speed/ from the finished runs in
# ~/data/seqmri_checkpoints/varnet_train_speed (untracked; names, overrides and wandb ids in manifest.json).
# Needs one idle GPU (bench, equivalence) and wandb access (curves). The eval reuses eval_cache.per_slice.npz:
# delete it to rescore every checkpoint. The training runs themselves came from the queue in
# scripts/launch/varnet_train_speed_worker.sh (job lines: ~/data/seqmri_checkpoints/varnet_train_speed/jobs.txt).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=reports/varnet_train_speed
EQ=${TMPDIR:-/tmp}/varnet_train_speed_equivalence

# Exactness of the throughput changes
python scripts/diagnostics/varnet_train_speed_equivalence.py train "$(pwd)" $EQ/a.pt
python scripts/diagnostics/varnet_train_speed_equivalence.py train "$(pwd)" $EQ/b.pt training.train_metrics_every=10 training.visualize_every=5
python scripts/diagnostics/varnet_train_speed_equivalence.py compare $EQ/a.pt $EQ/b.pt $OUT/equivalence_switches.json
python scripts/diagnostics/varnet_train_speed_fused_check.py $OUT/fused_check.json

# Step time per arm on an idle GPU (kernel_profile_contended.json was measured next to another job; rerun on an idle GPU)
python scripts/diagnostics/varnet_train_speed_bench.py --out $OUT/bench.json
python scripts/diagnostics/varnet_train_speed_profile_kernels.py "$(pwd)" $OUT/kernel_profile_contended.json

# Validation curves and epoch times (wandb + run logs), final per-condition scores and paired differences
python scripts/eval/varnet_train_speed_curves.py --out $OUT/curves.json
for control in control control_short full_control; do
    name=$([ "$control" = control ] && echo eval || echo "eval_vs_${control#control_}")
    [ "$control" = full_control ] && name=eval_vs_full
    python scripts/eval/varnet_train_speed_eval.py --control $control --out $OUT/$name.json --cache $OUT/eval_cache.per_slice.npz
done
python scripts/eval/varnet_train_speed_eval.py --error-split 'control_s4?' 'norm_*' 'full_*' 'muon_s4?' --out $OUT/error_split.json
python scripts/diagnostics/varnet_train_speed_masks.py $OUT/masks.json

# Tables and figures
python scripts/eval/varnet_train_speed_summary.py $OUT
python scripts/plots/varnet_train_speed_curves.py $OUT/curves.json $OUT/figures/curves_20pct.png control,muon,norm_layer,norm_none,norm_chln
python scripts/plots/varnet_train_speed_curves.py $OUT/curves.json $OUT/figures/curves_full_data.png full_control,full_muon,full_norm_layer
