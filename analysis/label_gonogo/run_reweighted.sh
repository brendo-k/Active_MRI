#!/usr/bin/env bash
# Reweighted go/no-go on GPU 0: screen predictor objectives by label predictability (fast, no PSNR evaluation),
# then the full go/no-go with the objective whose validation predictability is highest.
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES=0
DIR=analysis/label_gonogo/reweighted
mkdir -p $DIR
SCREENS=(
    "gain_lr1e-4_dev0 --weight gain --lr 1e-4"
    "gain_lr1e-3_dev0 --weight gain --lr 1e-3"
    "gain_lr1e-3_dev1 --weight gain --lr 1e-3 --deviation-weight 1"
    "none_lr1e-3_dev1 --weight none --lr 1e-3 --deviation-weight 1"
)
for screen in "${SCREENS[@]}"; do
    read -r name options <<< "$screen"
    python scripts/diagnostics/label_predictor_gonogo_r8.py --screen $options --out $DIR/screen_$name.json \
        > $DIR/screen_$name.log 2>&1
    echo "screen $name done $(date)" >> analysis/label_gonogo/launcher.log
done
best=$(python - "$DIR" <<'EOF'
import json, sys
from pathlib import Path
scores = {p.stem.removeprefix('screen_'): json.loads(p.read_text())['predictability']['val']['predictor']
          for p in Path(sys.argv[1]).glob('screen_*.json')}
print(max(scores, key=scores.get))
EOF
)
for screen in "${SCREENS[@]}"; do
    read -r name options <<< "$screen"
    [ "$name" = "$best" ] && break
done
echo "full go/no-go with $best $(date)" >> analysis/label_gonogo/launcher.log
python scripts/diagnostics/label_predictor_gonogo_r8.py $options --out reports/conv_vs_loupe_r8/label_gonogo_$best.json \
    > $DIR/gonogo_$best.log 2>&1
echo "reweighted gonogo done $(date)" >> analysis/label_gonogo/launcher.log
