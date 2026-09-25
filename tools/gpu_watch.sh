#!/bin/bash
# Poll `cnode mars` for connectable nodes; on each, pick the GPU with the most free
# memory and, if >= MIN_FREE MiB, run the pointops check there. Stops at first success.
#
# Usage:  bash tools/gpu_watch.sh                    (checks every 5 min)
#         INTERVAL=120 MIN_FREE=4096 bash tools/gpu_watch.sh
INTERVAL=${INTERVAL:-300}
MIN_FREE=${MIN_FREE:-2048}
SELF=$(hostname -s)

read -r -d '' REMOTE <<'EOF'
cd ~/data/git/DeformingPointTransformer 2>/dev/null || cd ~/Data/git/DeformingPointTransformer || exit 3
best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
gpu=${best%%,*}; free=${best##*, }
echo "$(hostname -s): best GPU $gpu with ${free} MiB free"
[ "$free" -ge MIN_FREE_PLACEHOLDER ] || exit 2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bodyseg
source /opt/rh/gcc-toolset-12/enable
export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
export CUDA_VISIBLE_DEVICES=$gpu
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
python - <<'PY'
import sys, torch
sys.path.insert(0, "lib")
from pointops.functions import pointops
x = torch.rand(1000, 3, device="cuda"); o = torch.tensor([1000], device="cuda").int()
print("knnquery", tuple(pointops.knnquery(8, x, x, o, o)[0].shape))
print("furthestsampling", tuple(pointops.furthestsampling(x, o, torch.tensor([100], device="cuda").int()).shape))
print("device", torch.cuda.get_device_name(0), "| GPUs on node:", torch.cuda.device_count())
print("POINTOPS_OK")
PY
EOF
REMOTE=${REMOTE//MIN_FREE_PLACEHOLDER/$MIN_FREE}

while true; do
  echo "=== $(date '+%F %T') ==="
  nodes=$(cnode mars 2>/dev/null | awk '$3=="yes"{print $1}' | grep -v "^$SELF$")
  echo "connectable: $(echo $nodes)"
  for n in $nodes; do
    out=$(timeout 600 ssh -o BatchMode=yes -o ConnectTimeout=15 "$n" "bash -s" <<<"$REMOTE" 2>&1)
    echo "$out" | tail -12
    if echo "$out" | grep -q POINTOPS_OK; then
      echo "SUCCESS on $n"
      exit 0
    fi
  done
  sleep "$INTERVAL"
done
