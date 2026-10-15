#!/bin/bash -l
# Match the held-out muscle configuration, rather than assuming fold numbers.
set -eo pipefail
cd "${PBS_O_WORKDIR:?}"
source cluster_env_uts.sh
FRAME="${FRAME:-220}"
case "${CONFIG:?}" in
    neg) GROUP=BB_-0.6_TRL_-0.6_TRM_-0.6_TRLN_-0.6 ;;
    zero) GROUP=BB_0_TRL_0_TRM_0_TRLN_0 ;;
    pos) GROUP=BB_0.6_TRL_0.6_TRM_0.6_TRLN_0.6 ;;
    *) echo 'CONFIG must be neg, zero, or pos' >&2; exit 1 ;;
esac
FOLD=$(python - "$FRAME" "$GROUP" <<'PY'
import sys
from pathlib import Path
root = Path('data') / f'cactus{int(sys.argv[1])}_loo_val' / 'folds'
matches = []
for fold in sorted(root.glob('fold_*')):
    ids = (fold / 'test.txt').read_text().splitlines()
    if ids and all(s.startswith(sys.argv[2] + '_T_') for s in ids):
        matches.append(fold.name)
if len(matches) != 1:
    raise SystemExit(f'Expected one matching fold for {sys.argv[2]}, got {matches}')
print(matches[0])
PY
)
export MODE=train FRAME FOLD RUN_NAME="compare${FRAME}_${GROUP}"
echo "Matched $GROUP to $FOLD, frame $FRAME"
exec bash tools/uts_arm_job.sh
