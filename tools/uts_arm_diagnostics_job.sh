#!/bin/bash -l
# CPU audit, or GPU train/validation/test re-evaluation of the existing fold_00 checkpoint.
set -eo pipefail
REPO="${PBS_O_WORKDIR:?}"
cd "$REPO"
source cluster_env_uts.sh
export PYTHONUNBUFFERED=1
if [ "${DIAG_MODE:-audit}" = audit ]; then
    python -u tools/audit_arm_results.py
    exit
fi
WORK="/scratch/$USER/dpt_diagnostic_${PBS_JOBID:?}"
mkdir -p "$WORK/repo" "$REPO/results/diagnostics_arm/fold_00"
rsync -a --exclude=.git --exclude=.codex --exclude=.agents --exclude=.aws --exclude=data --exclude=results --exclude=figures --exclude=vis "$REPO/" "$WORK/repo/"
cd "$WORK/repo"
mkdir -p data/cactus160_loo_val
rsync -a "$REPO/data/cactus160_loo_val/" data/cactus160_loo_val/
mkdir -p results/diagnostics_arm/fold_00
finish() {
    local status=$?
    trap - EXIT
    rsync -a results/diagnostics_arm/fold_00/ "$REPO/results/diagnostics_arm/fold_00/" || status=1
    exit "$status"
}
trap finish EXIT
PREP=data/cactus160_loo_val
FOLD="$PREP/folds/fold_00"
for split in train val test; do
    SAVE="results/diagnostics_arm/fold_00/$split"
    mkdir -p "$SAVE/models"
    cp "$REPO/results/check160/fold_00/models/model_weights_best.pth" "$SAVE/models/"
    echo "Evaluating existing best checkpoint on $split; no optimizer updates"
    python -u main.py --training_list_file_path "$FOLD/train.txt" \
        --testing_list_file_path "$FOLD/$split.txt" \
        --init_mean_shape_path "$FOLD/mean_shape.npz" \
        --label_json_file_path "$PREP/labels" --which_label arm4 \
        --data_root "$PREP/data.npz" --conversion_path "$PREP/affines.npz" \
        --save_path "$SAVE" --batch_size 4 --residual --seed 0 --eval_only \
        > "$SAVE/eval.log" 2>&1
    python tools/volume_eval.py --prepared "$PREP" --split_dir "$FOLD" --results "$SAVE" \
        > "$SAVE/volume_eval.log" 2>&1
done
python - <<'PY'
from pathlib import Path
import pandas as pd
root=Path('results/diagnostics_arm/fold_00')
rows=[]
for split in ('train','val','test'):
    for pred,filename in [('model','trained_vs_target_arm4.csv'),('template','init_vs_target_arm4.csv')]:
        metrics=pd.read_csv(root/split/'csv'/filename,index_col=0)
        volumes=pd.read_csv(root/split/'csv/volumes.csv')
        for organ,r in metrics.iterrows():
            v=volumes[(volumes.organ==organ)&(volumes.pred==pred)]
            rows.append(dict(split=split,organ=organ,pred=pred,cd_mm=r['cd'],hd95_mm=r['hd95'],
                             signed_volume_error_ml=v.err_ml.mean(),abs_volume_error_pct=v.abs_err_pct.mean()))
df=pd.DataFrame(rows)
df.to_csv(root/'split_comparison.csv',index=False)
print(df.round(3).to_string(index=False))
PY
