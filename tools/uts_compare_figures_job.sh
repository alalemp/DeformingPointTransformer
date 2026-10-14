#!/bin/bash -l
set -eo pipefail
cd "${PBS_O_WORKDIR:?}"
source cluster_env_uts.sh
FRAME="${FRAME:-160}"
RAW="$HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220"
for config in neg zero pos; do
    case "$config" in
        neg) group=BB_-0.6_TRL_-0.6_TRM_-0.6_TRLN_-0.6 ;;
        zero) group=BB_0_TRL_0_TRM_0_TRLN_0 ;;
        pos) group=BB_0.6_TRL_0.6_TRM_0.6_TRLN_0.6 ;;
    esac
    python -u tools/paper_figures.py \
        --data "$RAW" \
        --run "${config}=data/cactus${FRAME}_loo_val:results/compare${FRAME}_${config}:${FRAME}" \
        --sample "${group}_T_1_R" \
        --out "figures/parameter_comparison${FRAME}/${config}" \
        --only error --error_max_mm 10
done
# Include every completed comparison, preserving signed volume errors.
python - <<'PY'
from pathlib import Path
import pandas as pd
rows = []
for frame in (160, 220):
    for config in ('neg', 'zero', 'pos'):
        root = Path(f'results/compare{frame}_{config}')
        if not (root / 'summary_arm4.csv').is_file():
            continue
        summary = pd.read_csv(root / 'summary_arm4.csv')
        for fold in sorted(root.glob('fold_*')):
            volume_path = fold / 'csv/volumes.csv'
            if not volume_path.is_file():
                continue
            volumes = pd.read_csv(volume_path)
            for organ in summary.organ.unique():
                row = dict(frame=frame, configuration=config, fold=fold.name, organ=organ)
                for pred in ('model', 'template'):
                    for metric in ('cd', 'hd95', 'vol_abs_err_pct'):
                        values = summary[(summary.organ == organ) &
                                         (summary.pred == pred) & (summary.metric == metric)]
                        row[f'{pred}_{metric}'] = values['mean'].iloc[0]
                    row[f'{pred}_signed_volume_error_ml'] = volumes[
                        (volumes.organ == organ) & (volumes.pred == pred)].err_ml.mean()
                row['cd_improvement_pct'] = 100 * (row['template_cd'] - row['model_cd']) / row['template_cd']
                rows.append(row)
out = Path('results/parameter_comparison160_220')
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / 'comparison.csv', index=False)
print(df.round(3).to_string(index=False))
PY
