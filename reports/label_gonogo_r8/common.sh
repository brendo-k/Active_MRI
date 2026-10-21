cd "$(dirname "$0")/../.."
source .venv/bin/activate
LABELS=/tmp/bkadota_oracle_labels_r8
free_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 20000 {print $1; exit}'; }
wait_gpu() { until [ -n "$(free_gpu)" ]; do sleep 60; done; free_gpu; }
