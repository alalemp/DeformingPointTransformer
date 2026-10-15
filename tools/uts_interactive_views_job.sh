#!/bin/bash -l
# GUIDE Step 8 for each evaluated configuration, both frames; no training.
set -eo pipefail
cd "${PBS_O_WORKDIR:?}"
source cluster_env_uts.sh
export PYTHONUNBUFFERED=1
RAW="$HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220"
for frame in 160 220; do
    for group in BB_-0.6_TRL_-0.6_TRM_-0.6_TRLN_-0.6 BB_0_TRL_0_TRM_0_TRLN_0 BB_0.6_TRL_0.6_TRM_0.6_TRLN_0.6; do
        sample="${group}_T_1_R"
        out="vis/diagnostic_comparison/frame${frame}/${sample}"
        python -u tools/visualize_cactus.py --data "$RAW" --frame "$frame" \
            --id "$sample" --prepared "data/cactus${frame}_loo_val" \
            --results "results/compare${frame}_${group}" --out "$out"
        test -s "${out}.html"
    done
done
echo 'STEP8_OUTPUTS_OK: six HTML views in vis/diagnostic_comparison/'
