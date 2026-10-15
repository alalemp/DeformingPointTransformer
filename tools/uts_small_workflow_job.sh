#!/bin/bash -l
# Isolated small-dataset GUIDE steps 1–8. Submitted from repository root.
set -eo pipefail
REPO="${PBS_O_WORKDIR:?}"
cd "$REPO"
source cluster_env_uts.sh
export PYTHONUNBUFFERED=1
FRAME="${FRAME:?}"
[[ "$FRAME" == 160 || "$FRAME" == 220 ]] || exit 2
STAGE="${STAGE:?}"
PREP="small_cactus${FRAME}_loo_val"
RUN="$PREP"
RAW="$HOME/datasets/houdini/cactus_small_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/datasets_cactus pose frames_ply"
META="$REPO/results/small_workflow"
mkdir -p "$META/logs"
exec > >(tee "$META/logs/${PBS_JOBID}.log") 2>&1
WORK="/scratch/$USER/small_dpt_${PBS_JOBID:?}/repo"
mkdir -p "$WORK"
rsync -a --exclude=.git --exclude=.codex --exclude=.agents --exclude=.aws --exclude=data --exclude=results --exclude=figures --exclude=vis "$REPO/" "$WORK/"
cd "$WORK"
mkdir -p data results
finish() {
    local status=$?
    trap - EXIT
    if [ "$STAGE" = prepare ] && [ -d "data/$PREP" ]; then
        mkdir -p "$REPO/data/$PREP"
        rsync -a "data/$PREP/" "$REPO/data/$PREP/" || status=1
    elif [ "$STAGE" = train ] && [ -d "results/$RUN" ]; then
        mkdir -p "$REPO/results/$RUN"
        # Frames/chunks have disjoint fold writers; summary is generated only at the end.
        for fold in "${FOLDS[@]}"; do
            if [ -d "results/$RUN/$fold" ]; then
                mkdir -p "$REPO/results/$RUN/$fold"
                rsync -a "results/$RUN/$fold/" "$REPO/results/$RUN/$fold/" || status=1
            fi
        done
    elif [ "$STAGE" = post ] || [ "$STAGE" = combined ]; then
        for top in results figures vis; do
            if [ -d "$top" ]; then
                mkdir -p "$REPO/$top"
                rsync -a "$top/" "$REPO/$top/" || status=1
            fi
        done
    fi
    echo "SMALL_WORKFLOW_EXIT=$status; scratch=$WORK"
    exit "$status"
}
if [ "$STAGE" = prepare ]; then
    trap finish EXIT
    mkdir -p data/small_houdini_raw
    rsync -a "$RAW/" data/small_houdini_raw/
    python -u tools/prepare_cactus.py --data data/small_houdini_raw --out "data/$PREP" \
        --frame "$FRAME" --leave_out 1 --val_frac 0.1 --seed 0 --workers 4
    python - "$PREP" <<'PY'
import json, sys
from pathlib import Path
root=Path('data')/sys.argv[1]
rows=[]
for fold in sorted((root/'folds').glob('fold_*')):
    ids={s:(fold/f'{s}.txt').read_text().splitlines() for s in ('train','val','test')}
    groups={s:{i.rsplit('_T_',1)[0] for i in v} for s,v in ids.items()}
    assert len(ids['train'])>=4
    assert not (groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    rows.append(dict(fold=fold.name,**{s:len(v) for s,v in ids.items()}))
assert len(rows)==49, f'Expected 49 valid groups, got {len(rows)}'
(root/'small_split_counts.json').write_text(json.dumps(rows,indent=2))
print('SMALL_PREPARATION_OK',len(rows),'folds')
PY
elif [ "$STAGE" = train ]; then
    mapfile -t FOLDS < <(python - "${START:?}" "${END:?}" <<'PY'
import sys
for k in range(int(sys.argv[1]),int(sys.argv[2])+1):
    print(f'fold_{k:02d}')
PY
)
    trap finish EXIT
    mkdir -p "data/$PREP" "results/$RUN"
    rsync -a "$REPO/data/$PREP/" "data/$PREP/"
    for fold in "${FOLDS[@]}"; do
        if [ -d "$REPO/results/$RUN/$fold" ]; then
            rsync -a "$REPO/results/$RUN/$fold" "results/$RUN/"
        fi
    done
    python -m pip freeze > "$META/logs/${PBS_JOBID}.pip_freeze.txt"
    for fold in "${FOLDS[@]}"; do
        python -u tools/run_folds.py --prepared "data/$PREP" --label_set arm4 --save "results/$RUN" \
            --folds "$fold" -- --batch_size 4 --epochs 120 --residual --patience 15 --seed 0
        test -s "results/$RUN/$fold/csv/trained_vs_target_arm4.csv"
        mkdir -p "$REPO/results/$RUN/$fold"
        rsync -a "results/$RUN/$fold/" "$REPO/results/$RUN/$fold/"
    done
    python - "$RUN" "${FOLDS[@]}" <<'PY'
import sys
from pathlib import Path
for fold in sys.argv[2:]:
    root=Path('results')/sys.argv[1]/fold
    for filename in ('csv/trained_vs_target_arm4.csv','csv/init_vs_target_arm4.csv','predictions.npz','models/model_weights_best.pth'):
        assert (root/filename).is_file(), f'Missing {root/filename}; inspect train.log'
print('SMALL_TRAINING_OK',sys.argv[2:])
PY
elif [ "$STAGE" = post ] || [ "$STAGE" = combined ]; then
    trap finish EXIT
    # Post-processing each frame can run independently; use separate prefixed output folders.
    mkdir -p "data/$PREP" "results/$RUN"
    rsync -a "$REPO/data/$PREP/" "data/$PREP/"
    rsync -a --exclude=models "$REPO/results/$RUN/" "results/$RUN/"
    python -u tools/run_folds.py --prepared "data/$PREP" --label_set arm4 --save "results/$RUN" --summary_only
    if [ "$STAGE" = combined ]; then
        mkdir -p data/small_cactus220_loo_val results/small_cactus220_loo_val
        rsync -a "$REPO/data/small_cactus220_loo_val/" data/small_cactus220_loo_val/
        rsync -a --exclude=models "$REPO/results/small_cactus220_loo_val/" results/small_cactus220_loo_val/
    fi
    mapfile -t SAMPLES < <(python - "$PREP" <<'PY'
import sys
from pathlib import Path
root=Path('data')/sys.argv[1]/'folds'
ids=[s for p in sorted(root.glob('fold_*/test.txt')) for s in p.read_text().splitlines()]
for group in ('BB_-0.6_TRL_-0.6_TRM_-0.6_TRLN_-0.6','BB_0_TRL_0_TRM_0_TRLN_0','BB_0.6_TRL_0.6_TRM_0.6_TRLN_0.6'):
    choices=[s for s in ids if s.startswith(group+'_T_')]
    if choices:
        preferred=group+'_T_1_R'
        print(preferred if preferred in choices else sorted(choices)[0])
PY
)
    test "${#SAMPLES[@]}" -gt 0
    for sample in "${SAMPLES[@]}"; do
        out="figures/small_paper/frame${FRAME}/$sample"
        run_args=(--run "Small frame $FRAME=data/$PREP:results/$RUN:$FRAME")
        if [ "$STAGE" = combined ]; then
            out="figures/small_paper_comparison160_220/$sample"
            run_args+=(--run "Small frame 220=data/small_cactus220_loo_val:results/small_cactus220_loo_val:220")
        fi
        python -u tools/paper_figures.py --data "$RAW" "${run_args[@]}" \
            --sample "$sample" --out "$out" --error_max_mm 10
        python -u tools/visualize_cactus.py --data "$RAW" --frame "$FRAME" --id "$sample" \
            --prepared "data/$PREP" --results "results/$RUN" --out "vis/small_interactive/frame${FRAME}/$sample"
        test -s "vis/small_interactive/frame${FRAME}/${sample}.html"
        for name in figS1_full_body fig1_data_overview fig2_training_curves fig3_volume_scatter fig4_error_surface; do
            test -s "$out/$name.png"
            test -s "$out/$name.pdf"
        done
        test -s "$out/table_metrics.csv"
    done
    echo "SMALL_STEPS_6_7_8_OK frame=$FRAME; samples=${SAMPLES[*]}"
else
    echo "Unknown stage: $STAGE" >&2; exit 2
fi
