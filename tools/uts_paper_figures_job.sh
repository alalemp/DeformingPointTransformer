#!/bin/bash -l
# Complete GUIDE Step 7 for one matched configuration, comparing frames 160 and 220.
set -eo pipefail
cd "${PBS_O_WORKDIR:?}"
source cluster_env_uts.sh
export PYTHONUNBUFFERED=1
case "${CONFIG:?Specify CONFIG=neg,zero,or pos}" in
    neg) GROUP=BB_-0.6_TRL_-0.6_TRM_-0.6_TRLN_-0.6 ;;
    zero) GROUP=BB_0_TRL_0_TRM_0_TRLN_0 ;;
    pos) GROUP=BB_0.6_TRL_0.6_TRM_0.6_TRLN_0.6 ;;
    *) echo 'Unknown CONFIG' >&2; exit 1 ;;
esac
RAW="$HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220"
OUT="figures/paper_comparison160_220/$GROUP/T_1_R"
python -u tools/paper_figures.py --out "$OUT" --data "$RAW" \
    --run "Frame 160=data/cactus160_loo_val:results/compare160_${GROUP}:160" \
    --run "Frame 220=data/cactus220_loo_val:results/compare220_${GROUP}:220" \
    --sample "${GROUP}_T_1_R" --error_max_mm 10
python - "$OUT" <<'PY'
import sys
from pathlib import Path
out=Path(sys.argv[1])
figures=['figS1_full_body','fig1_data_overview','fig2_training_curves','fig3_volume_scatter','fig4_error_surface']
expected=[out/f'{name}.{ext}' for name in figures for ext in ('png','pdf')]
expected += [out/'table_metrics.csv',out/'table_metrics.tex']
missing=[str(p) for p in expected if not p.is_file() or p.stat().st_size==0]
if missing:
    raise SystemExit(f'Missing or empty Step 7 outputs: {missing}')
(out/'README.md').write_text('''# Complete Step 7 figures for two frames\n\nThis is a configuration comparison, not an all-81-fold result. Each frame includes one completed held-out configuration fold (6 test arms).\n\n- S1, Figure 1, and Figure 4 show the specified T=1 right-arm example.\n- Figure 2 shows the training and validation history of each frame’s model. Validation uses the original monitoring procedure; it is not an additional test evaluation.\n- Figure 3 and the metrics table include all test arms of the available fold for each frame, including other T values and the left arm. The parent T_1_R folder identifies the illustrated sample, not the scope of these aggregate outputs.\n- Figure 4 uses a common 0–10 mm colour scale and displays distance to the target, not signed expansion/shrinkage.\n- Volume estimates deform the template mesh from predicted point displacements; they are not direct segmentations.\n- PDF and PNG versions of all five figures and CSV/LaTeX versions of the table are included.\n''')
print('STEP7_OUTPUTS_OK',out)
PY
