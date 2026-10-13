# Source from a UTS shell: source cluster_env_uts.sh
# Uses the compiler installed in bodyseg, rather than the system GCC 8.
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-bodyseg}"
if [ ! -r "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    echo "Miniconda not found at $CONDA_DIR" >&2
    return 1
fi
source "$CONDA_DIR/etc/profile.d/conda.sh" || return 1
conda activate "$ENV_NAME" || return 1
if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh || return 1
fi
module load cuda/12.1 || return 1
export CUDA_HOME=/shared/apps/cuda-12.1
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/dpt_torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLBACKEND=Agg
for executable in "$CC" "$CXX" "$CUDACXX"; do
    if [ ! -x "$executable" ]; then
        echo "Missing compiler: $executable" >&2
        return 1
    fi
done
echo "env $ENV_NAME, Python $(python --version), compiler $($CXX -dumpversion), CUDA $CUDA_HOME"
