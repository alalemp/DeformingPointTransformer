#!/bin/bash
# Install everything for this project on the iHPC cluster (see CLUSTER.md):
#   Miniconda (if missing) -> conda env from environment.yml -> pointops CUDA extension ->
#   Chamfer3D pre-compile -> TotalSegmentator licence + weights -> VIBESegmentator clone -> checks
#
# Run from the repository root on a cluster node (a GPU node is not required to install):
#   bash install_cluster.sh
#   TOTALSEG_LICENSE=aca_XXXX bash install_cluster.sh            # also set the TotalSegmentator licence
#   bash install_cluster.sh --conda_dir ~/miniconda3 --env bodyseg  # use an existing Miniconda / env name
#
# Options:
#   --conda_dir DIR     Miniconda location (default /data/$USER/miniconda3; installed there if missing)
#   --env NAME          conda env name (default bodyseg)
#   --vibe_dir DIR      where to clone VIBESegmentator (default /data/$USER/git/VIBESegmentator)
#   --weights "T1 T2"   TotalSegmentator tasks to download (default "total_mr thigh_shoulder_muscles_mr")
#   --recreate          delete and recreate the conda env
#   --skip_weights      do not download TotalSegmentator weights
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="/data/$USER/miniconda3"
ENV_NAME="bodyseg"
VIBE_DIR="/data/$USER/git/VIBESegmentator"
WEIGHTS="total_mr thigh_shoulder_muscles_mr"
RECREATE=0
SKIP_WEIGHTS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --conda_dir) CONDA_DIR="$2"; shift 2 ;;
        --env) ENV_NAME="$2"; shift 2 ;;
        --vibe_dir) VIBE_DIR="$2"; shift 2 ;;
        --weights) WEIGHTS="$2"; shift 2 ;;
        --recreate) RECREATE=1; shift ;;
        --skip_weights) SKIP_WEIGHTS=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done
step() { echo; echo "=== $* ==="; }

step "0. checks"
[ -f /opt/rh/gcc-toolset-12/enable ] || { echo "gcc-toolset-12 not found on $(hostname)"; exit 1; }
[ -x /usr/local/cuda-12.1/bin/nvcc ] || { echo "CUDA 12.1 not found at /usr/local/cuda-12.1 on $(hostname)"; exit 1; }
home_free_gb=$(df -Pk "$HOME" | awk 'NR==2 {print int($4/1024/1024)}')
echo "host $(hostname), repo $REPO, free in \$HOME: ${home_free_gb} GB"
if [ "$home_free_gb" -lt 5 ]; then
    echo "WARNING: \$HOME is nearly full - keeping pip cache and TotalSegmentator files under /data/$USER"
    export PIP_CACHE_DIR="/data/$USER/.cache/pip"
    mkdir -p "$PIP_CACHE_DIR"
    if [ ! -d "$HOME/.totalsegmentator" ]; then
        export TOTALSEG_HOME_DIR="/data/$USER/.totalsegmentator"
        mkdir -p "$TOTALSEG_HOME_DIR"
    fi
fi

step "1. Miniconda ($CONDA_DIR)"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    installer="$(mktemp -d)/miniconda.sh"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$installer"
    bash "$installer" -b -p "$CONDA_DIR"      # -b: batch mode, accepts the Miniconda licence
    rm -f "$installer"
    echo "installed Miniconda; to have 'conda' in every shell run once: $CONDA_DIR/bin/conda init bash"
else
    echo "found $("$CONDA_DIR/bin/conda" --version)"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

step "2. conda env $ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "$RECREATE" = 1 ]; then
        conda env remove -n "$ENV_NAME" -y
        conda env create -n "$ENV_NAME" -f "$REPO/environment.yml"
    else
        echo "env exists - updating packages from environment.yml (use --recreate for a clean env)"
        conda env update -n "$ENV_NAME" -f "$REPO/environment.yml"
    fi
else
    conda env create -n "$ENV_NAME" -f "$REPO/environment.yml"
fi
CONDA_DIR="$CONDA_DIR" ENV_NAME="$ENV_NAME" source "$REPO/cluster_env.sh"

step "3. pointops CUDA extension"
# lib/pointops/src/*/*_cuda.cpp already include ATen/cuda/CUDAContext.h (THC/THC.h was removed in torch 1.11)
(cd "$REPO/lib/pointops" && python -m pip install . --no-build-isolation)

step "4. Chamfer3D (compiled on first import, cached in ~/.cache/torch_extensions)"
(cd "$REPO" && python -c "from Chamfer3D.dist_chamfer_3D import chamfer_3DDist" ) \
    || echo "WARNING: Chamfer3D did not compile here; it will be compiled on the first training run"

step "5. TotalSegmentator"
if [ -n "${TOTALSEG_LICENSE:-}" ]; then
    totalseg_set_license -l "$TOTALSEG_LICENSE"
else
    echo "no TOTALSEG_LICENSE given - set it once with: totalseg_set_license -l <your key>"
    echo "(needed for licensed tasks such as thigh_shoulder_muscles_mr)"
fi
if [ "$SKIP_WEIGHTS" = 0 ]; then
    for t in $WEIGHTS; do
        totalseg_download_weights -t "$t" || echo "WARNING: weights for $t not downloaded (licence set?)"
    done
fi

step "6. VIBESegmentator ($VIBE_DIR)"
if [ ! -d "$VIBE_DIR/.git" ]; then
    mkdir -p "$(dirname "$VIBE_DIR")"
    git clone https://github.com/robert-graf/VIBESegmentator.git "$VIBE_DIR"
else
    echo "already cloned ($(git -C "$VIBE_DIR" log -1 --format='%h %cs'))"
fi
echo "its nnU-Net weights download automatically on the first run, into $VIBE_DIR/nnUNet/nnUNet_results"

step "7. checks"
(cd "$REPO" && python - <<'EOF'
import importlib, sys
import torch
print(f"python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"GPU visible: {torch.cuda.is_available()}")
for mod in ("pointops_cuda", "totalsegmentator", "nnunetv2", "TPTBox", "nibabel", "skimage", "trimesh"):
    try:
        importlib.import_module(mod)
        print(f"  ok   {mod}")
    except Exception as e:
        print(f"  FAIL {mod}: {e}")
if torch.cuda.is_available():
    sys.path.insert(0, "lib")
    from pointops.functions import pointops
    x = torch.rand(1000, 3, device="cuda"); o = torch.tensor([1000], device="cuda").int()
    print("  ok   pointops on GPU, knnquery", tuple(pointops.knnquery(8, x, x, o, o)[0].shape))
else:
    print("  (no GPU on this node: run on a GPU node to test pointops on the GPU)")
EOF
)
echo
echo "done. In every new shell:  cd $REPO && source cluster_env.sh"
