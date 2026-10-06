# Per-shell environment for this project on the iHPC cluster. Source it, do not run it:
#
#   source cluster_env.sh
#
# Activates the conda env and sets the compiler / CUDA variables needed to build or load the
# CUDA extensions (pointops, Chamfer3D). Override with CONDA_DIR=... ENV_NAME=... before sourcing.

ENV_NAME="${ENV_NAME:-bodyseg}"
if [ -z "${CONDA_DIR:-}" ]; then
    # the Miniconda that holds the env (several installs may exist), else the first one found
    for d in "/data/$USER/miniconda3" "$HOME/miniconda3"; do
        if [ -d "$d/envs/$ENV_NAME" ]; then CONDA_DIR="$d"; break; fi
    done
    if [ -z "${CONDA_DIR:-}" ]; then
        for d in "/data/$USER/miniconda3" "$HOME/miniconda3"; do
            if [ -x "$d/bin/conda" ]; then CONDA_DIR="$d"; break; fi
        done
    fi
fi

source "$CONDA_DIR/etc/profile.d/conda.sh" && conda activate "$ENV_NAME" || return 1
source /opt/rh/gcc-toolset-12/enable                     # system gcc 8.5 is too old for torch 2.5
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
# TotalSegmentator weights / licence live in ~/.totalsegmentator unless a /data copy exists
if [ -z "${TOTALSEG_HOME_DIR:-}" ] && [ -d "/data/$USER/.totalsegmentator" ]; then
    export TOTALSEG_HOME_DIR="/data/$USER/.totalsegmentator"
fi
echo "env $ENV_NAME ($CONDA_DIR), gcc $(gcc -dumpversion), CUDA $(nvcc --version | grep -o 'release [0-9.]*')"
