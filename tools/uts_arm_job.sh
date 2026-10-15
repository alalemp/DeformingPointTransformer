#!/bin/bash -l
# Submit from the repository root with qsub; see UTS_ARMS.md.
# MODE=check | prepare | train; FRAME defaults to 160; FOLD=all runs all folds.
set -eo pipefail
MODE="${MODE:-check}"
FRAME="${FRAME:-160}"
FOLD="${FOLD:-fold_00}"
REPO="${PBS_O_WORKDIR:?Submit this script using qsub from the repository root}"
test -f "$REPO/main.py" || { echo "Submit from the repository root" >&2; exit 1; }
[[ "$FRAME" =~ ^[0-9]+$ ]] || { echo "FRAME must be an integer" >&2; exit 1; }
case "$MODE" in check|prepare|train) ;; *) echo "Unknown MODE: $MODE" >&2; exit 1 ;; esac
if [ "$FOLD" != all ] && ! [[ "$FOLD" =~ ^fold_[0-9]+$ ]]; then
    echo "FOLD must be fold_XX or all" >&2; exit 1
fi
PREPARED="cactus${FRAME}_loo_val"
RUN_NAME="${RUN_NAME:-check${FRAME}}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || { echo "Invalid RUN_NAME" >&2; exit 1; }
source "$REPO/cluster_env_uts.sh"
DATASET_CODE="${DATASET_CODE:-cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220}"
[[ "$DATASET_CODE" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid DATASET_CODE" >&2; exit 1; }
RAW_DATA="${RAW_DATA:-$HOME/datasets/houdini/$DATASET_CODE}"
if [ "$MODE" = prepare ]; then
    test -d "$RAW_DATA/final skin" && test -d "$RAW_DATA/separated muscles" || {
        echo "Expected 'final skin' and 'separated muscles' inside $RAW_DATA" >&2; exit 1;
    }
elif [ "$MODE" = train ]; then
    test -f "$REPO/data/$PREPARED/data.npz" || {
        echo "Prepare $PREPARED first" >&2; exit 1;
    }
fi

WORK="/scratch/$USER/dpt_${PBS_JOBID:?}"
mkdir -p "$WORK/repo"
rsync -a --exclude=.git --exclude=.codex --exclude=.agents --exclude=.aws \
    --exclude=data --exclude=results --exclude=vis --exclude=figures/paper \
    "$REPO/" "$WORK/repo/"
cd "$WORK/repo"
mkdir -p data results

# Copy completed and partial outputs back on normal exit or an error.
# Scratch is retained so it can be recovered if a hard job termination prevents this.
finish() {
    local status=$?
    trap - EXIT
    if [ "$MODE" = prepare ] && [ -d "data/$PREPARED" ]; then
        mkdir -p "$REPO/data/$PREPARED"
        rsync -a "data/$PREPARED/" "$REPO/data/$PREPARED/" || status=1
    fi
    if [ -d results ]; then
        mkdir -p "$REPO/results"
        rsync -a results/ "$REPO/results/" || status=1
    fi
    echo "Job exit $status; scratch retained at $WORK"
    exit "$status"
}
trap finish EXIT

if [ "$MODE" = check ]; then
    mkdir -p results/uts_check
    python -m pip install ./lib/pointops --no-build-isolation
    python tools/check_uts_cuda.py | tee results/uts_check/cuda_check.txt
    python -m pip freeze > results/uts_check/pip_freeze.txt
elif [ "$MODE" = prepare ]; then
    mkdir -p data/houdini_raw
    rsync -a "$RAW_DATA/" data/houdini_raw/
    python tools/prepare_cactus.py --data data/houdini_raw --out "data/$PREPARED" \
        --frame "$FRAME" --leave_out 1 --val_frac 0.1 --seed 0 --workers 4
else
    mkdir -p "data/$PREPARED" "results/$RUN_NAME"
    rsync -a "$REPO/data/$PREPARED/" "data/$PREPARED/"
    if [ -d "$REPO/results/$RUN_NAME" ]; then
        rsync -a "$REPO/results/$RUN_NAME/" "results/$RUN_NAME/"
    fi
    fold_args=()
    if [ "$FOLD" != all ]; then fold_args=(--folds "$FOLD"); fi
    python - "$PREPARED" "$FOLD" <<'PY'
import sys
from pathlib import Path
root = Path('data') / sys.argv[1] / 'folds'
folds = sorted(root.glob('fold_*')) if sys.argv[2] == 'all' else [root / sys.argv[2]]
if not folds:
    raise SystemExit('No prepared folds')
for fold in folds:
    ids = [s for s in (fold / 'train.txt').read_text().splitlines() if s.strip()]
    if len(ids) < 4:
        raise SystemExit(f'{fold}: batch size 4 needs at least 4 training arms; got {len(ids)}')
PY
    # Compute nodes may not provide Git; read the shared checkout metadata instead.
    python - "$REPO" > "results/$RUN_NAME/source_commit.txt" <<'PY'
import sys
from pathlib import Path
gitdir = Path(sys.argv[1]) / '.git'
try:
    head = (gitdir / 'HEAD').read_text().strip()
    if head.startswith('ref: '):
        ref = head[5:]
        loose = gitdir / ref
        if loose.is_file():
            head = loose.read_text().strip()
        else:
            head = next(line.split()[0] for line in
                        (gitdir / 'packed-refs').read_text().splitlines()
                        if line and not line.startswith(('#', '^'))
                        and line.split()[1] == ref)
    print(head)
except (OSError, StopIteration):
    print('unknown')
    print('Warning: could not read source commit metadata', file=sys.stderr)
PY
    python -m pip freeze > "results/$RUN_NAME/pip_freeze.txt"
    python tools/run_folds.py --prepared "data/$PREPARED" --label_set arm4 \
        --save "results/$RUN_NAME" "${fold_args[@]}" \
        -- --batch_size 4 --epochs 120 --residual --patience 15 --seed 0
    # run_folds.py logs individual failures but does not return a failing exit code.
    python - "$PREPARED" "$RUN_NAME" "$FOLD" <<'PY'
import sys
from pathlib import Path
root = Path('data') / sys.argv[1] / 'folds'
folds = sorted(root.glob('fold_*')) if sys.argv[3] == 'all' else [root / sys.argv[3]]
missing = [f.name for f in folds if not
           (Path('results') / sys.argv[2] / f.name / 'csv' / 'trained_vs_target_arm4.csv').is_file()]
if missing:
    raise SystemExit(f'Folds without completed metrics: {missing}; inspect their train.log files')
PY
fi
