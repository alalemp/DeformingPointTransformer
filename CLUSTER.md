# Installing on the iHPC cluster

This sets up one conda environment (`bodyseg`) with everything this project uses:

- **DeformingPointTransformer:** this repository, plus its two CUDA extensions, `pointops` and `Chamfer3D`.
- **TotalSegmentator 2.18:** MRI segmentation. Includes the `thigh_shoulder_muscles_mr` task used for the triceps.
- **VIBESegmentator:** VIBE / Dixon MRI segmentation, run from its own git checkout.

It has been tested on the mars nodes: RHEL 8, NVIDIA L4 GPUs, CUDA 12.1 at `/usr/local/cuda-12.1`, gcc-toolset-12 at `/opt/rh/gcc-toolset-12`. How to *use* the code afterwards is in [GUIDE.md](GUIDE.md).

| File | Purpose |
|---|---|
| [install_cluster.sh](install_cluster.sh) | Does the whole installation (section 3). |
| [environment.yml](environment.yml) | The conda environment, with the tested package versions. |
| [cluster_env.sh](cluster_env.sh) | Run `source cluster_env.sh` in every new shell (section 5). |

## Contents

1. [Before you start](#1-before-you-start)
2. [Install Miniconda](#2-install-miniconda)
3. [Get the code and run the installer](#3-get-the-code-and-run-the-installer)
4. [Manual installation (what the installer does)](#4-manual-installation-what-the-installer-does)
5. [Every session](#5-every-session)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Before you start

- **Disk space: install under `/data/$USER`, not in your home directory.**
  - `/home` has a 64 GB quota, and a full home directory breaks installs and runs in confusing ways.
  - The environment alone takes about 11 GB. TotalSegmentator weights take about 0.3 GB per task, and the VIBESegmentator weights a few GB more.
  - Check your free space with `df -h ~ /data/$USER`.
- **Node:** any mars node with Connect = yes in `cnode mars`. Installing needs no GPU; testing the CUDA extensions on a GPU does.
- **TotalSegmentator licence:** some tasks, including `thigh_shoulder_muscles_mr`, need a free academic licence key from the TotalSegmentator website. Keep the key private: don't commit it, and don't paste it into shared files.
- **GitHub access:** to clone this repository over SSH, add your cluster SSH key (`~/.ssh/id_ed25519.pub`) to GitHub, or use the HTTPS URL.

## 2. Install Miniconda

Skip this if you already have one (`ls /data/$USER/miniconda3/bin/conda ~/miniconda3/bin/conda`). The installer script also does this step itself if Miniconda is missing.

```bash
cd /data/$USER
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /data/$USER/miniconda3     # -b: non-interactive
rm Miniconda3-latest-Linux-x86_64.sh
/data/$USER/miniconda3/bin/conda init bash      # optional: `conda` in every new shell (edits ~/.bashrc)
```

`environment.yml` installs from `conda-forge`, not Anaconda's default channels. Those defaults now require you to accept a terms-of-service prompt, and `conda-forge` avoids it.

## 3. Get the code and run the installer

```bash
mkdir -p /data/$USER/git && cd /data/$USER/git
git clone git@github.com:alalemp/DeformingPointTransformer.git     # or https://github.com/alalemp/DeformingPointTransformer.git
cd DeformingPointTransformer
TOTALSEG_LICENSE=<your key> bash install_cluster.sh                 # takes 20-40 min, mostly pip downloads
```

The script runs these steps in order, and stops at the first one that fails:

| Step | What happens |
|---|---|
| 0 | Checks gcc-toolset-12 and CUDA 12.1. If `$HOME` has less than 5 GB free, it keeps the pip cache and the TotalSegmentator files under `/data/$USER`. |
| 1 | Installs Miniconda at `--conda_dir`, if it isn't there yet. |
| 2 | Creates the conda env from `environment.yml`. If the env already exists, it is updated instead. |
| 3 | Builds and installs `lib/pointops`. |
| 4 | Pre-compiles `Chamfer3D`, which is cached in `~/.cache/torch_extensions`. |
| 5 | Sets the TotalSegmentator licence (when `TOTALSEG_LICENSE` is given) and downloads the weights for `--weights`. |
| 6 | Clones VIBESegmentator to `--vibe_dir`. |
| 7 | Checks that every package imports. On a GPU node it also runs `pointops` on the GPU. |

Options:

| Option | Default | Meaning |
|---|---|---|
| `--conda_dir DIR` | `/data/$USER/miniconda3` | Miniconda to use; installed there if missing. Use `~/miniconda3` for an existing install. |
| `--env NAME` | `bodyseg` | Name of the conda env. |
| `--vibe_dir DIR` | `/data/$USER/git/VIBESegmentator` | Where to clone VIBESegmentator. |
| `--weights "T1 T2"` | `"total_mr thigh_shoulder_muscles_mr"` | TotalSegmentator tasks to download weights for (`all` for every task). |
| `--recreate` | | Delete the env and build it again from scratch. |
| `--skip_weights` | | Don't download TotalSegmentator weights. |

Running the script again is safe: it updates the env, rebuilds `pointops` and skips anything that is already in place.

## 4. Manual installation (what the installer does)

Use these steps if the script fails part-way, or if you want to install only some parts.

### 4.1 Conda environment

```bash
source /data/$USER/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml          # Python 3.11, torch 2.5.1 + CUDA 12.1, TotalSegmentator, nnU-Net v2, TPTBox, ...
source cluster_env.sh                        # activate + compiler/CUDA variables
```

`environment.yml` pins the versions that were tested together. To set up the env by hand without it, the equivalent is:

```bash
conda create -n bodyseg python=3.11 -y && conda activate bodyseg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install TotalSegmentator==2.18.0 nnunetv2==2.8.1 TPTBox==1.0.0
pip install ninja nibabel scikit-image pandas matplotlib tqdm trimesh h5py
```

### 4.2 DeformingPointTransformer CUDA extensions

**`pointops`** must be built with gcc 12 and CUDA 12.1. The system gcc is 8.5, which torch 2.5 rejects, and CUDA 12.1 matches torch's build. `cluster_env.sh` sets both up.

```bash
source cluster_env.sh
cd lib/pointops && pip install . --no-build-isolation && cd ../..
```

- The sources in this repository are already patched. The original code included `THC/THC.h`, which was removed in PyTorch 1.11; it now includes `ATen/cuda/CUDAContext.h`.
- `TORCH_CUDA_ARCH_LIST` (set in `cluster_env.sh`) builds the extension for GPU generations from compute capability 7.5 to 9.0, covering the L4 (8.9) and most other cluster GPUs.
- The `build/` and `dist/` folders already in `lib/pointops` are an old Python 3.9 build from the original authors. They are not used.

**`Chamfer3D`** compiles itself the first time it is imported. That needs `ninja` (in the env) and the same `cluster_env.sh` settings. To compile it now rather than at the start of the first training run:

```bash
python -c "from Chamfer3D.dist_chamfer_3D import chamfer_3DDist"      # prints "Loaded JIT 3D CUDA chamfer distance"
```

### 4.3 TotalSegmentator

```bash
totalseg_set_license -l <your key>                       # once; stored in ~/.totalsegmentator/config.json
totalseg_download_weights -t total_mr                    # organs / muscles / bones (MR)
totalseg_download_weights -t thigh_shoulder_muscles_mr   # includes triceps_brachii (needs the licence)
TotalSegmentator -i mri.nii.gz -o segmentations --task thigh_shoulder_muscles_mr    # test
```

- Weights and the licence are kept in `~/.totalsegmentator`.
- To keep them on `/data` instead, run `export TOTALSEG_HOME_DIR=/data/$USER/.totalsegmentator` before setting the licence. `cluster_env.sh` exports that path automatically whenever the folder exists.

### 4.4 VIBESegmentator

VIBESegmentator is not a pip package. It runs from its own checkout, using `TPTBox` and `nnunetv2` from the env.

```bash
git clone https://github.com/robert-graf/VIBESegmentator.git /data/$USER/git/VIBESegmentator
cd /data/$USER/git/VIBESegmentator
wget https://github.com/robert-graf/VIBESegmentator/releases/download/example/example_mri.zip && unzip example_mri.zip
python run_VIBESegmentator.py --img example/mri1.nii.gz --out_path example/seg1.nii.gz     # test (GPU node)
```

The first run downloads the nnU-Net weights to `<VIBESegmentator>/nnUNet/nnUNet_results/`. If a newer TPTBox breaks it, VIBESegmentator's README suggests `pip install TPTBox==0.8.0`.

### 4.5 Check

```bash
python -c "import torch, pointops_cuda, totalsegmentator, nnunetv2, TPTBox; print(torch.__version__, torch.cuda.is_available())"
bash tools/gpu_watch.sh          # finds a node with a free GPU and runs the pointops GPU test there
```

## 5. Every session

```bash
cd /data/$USER/git/DeformingPointTransformer
source cluster_env.sh            # conda env + gcc 12 + CUDA 12.1 (+ TOTALSEG_HOME_DIR if on /data)
nvidia-smi --query-gpu=index,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=0    # the GPU with the most free memory
```

- `cluster_env.sh` finds whichever Miniconda (under `/data/$USER` or `~`) contains the env. Override this with `CONDA_DIR=... ENV_NAME=... source cluster_env.sh`.
- Finding a free GPU, starting long jobs with `nohup`, and running the pipelines are covered in [GUIDE.md](GUIDE.md).

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `You're trying to build PyTorch with a too old version of GCC` | `source cluster_env.sh` (or `source /opt/rh/gcc-toolset-12/enable`) before building. |
| `fatal error: THC/THC.h: No such file or directory` | An unpatched copy of `pointops`. Replace `#include <THC/THC.h>` with `#include <ATen/cuda/CUDAContext.h>` in `lib/pointops/src/*/*_cuda.cpp`. |
| `Ninja is required to load C++ extensions` | `pip install ninja` in the env. |
| `No module named 'pointops_cuda'` | `pointops` isn't installed in the active env. Repeat 4.2. |
| `CUDA error: no kernel image is available` | The extension wasn't built for this GPU. Add its compute capability to `TORCH_CUDA_ARCH_LIST`, then rebuild `pointops` and delete `~/.cache/torch_extensions/*/chamfer_3D`. |
| Chamfer3D fails to compile, or keeps recompiling | Run `source cluster_env.sh`. Clear a half-finished build with `rm -rf ~/.cache/torch_extensions/*/chamfer_3D`. |
| `CUDA error: out of memory` | Someone else is using the GPU. Pick another one (`nvidia-smi`) or reduce `--batch_size` (`python tools/mem_probe.py 2 4 8`). |
| `EnvironmentNameNotFound: bodyseg` | The env is in a different Miniconda. Use `CONDA_DIR=~/miniconda3 source cluster_env.sh`. |
| TotalSegmentator: licence / weights errors | `totalseg_set_license -l <key>`, then `totalseg_download_weights -t <task>`. |
| Installs fail with "No space left on device" | `$HOME` is full. Install under `/data/$USER`, set `PIP_CACHE_DIR=/data/$USER/.cache/pip`, and clear `~/.cache/pip` and `~/miniconda3/pkgs` (`conda clean -a`). |
| `ssh marsN` closes immediately | That node isn't accepting logins right now, even if `cnode` lists it. Try another node. |
| `cp` waits for a yes/no answer in scripts | `cp` is aliased to `cp -i` on the cluster. Use `command cp -f` in scripts. |
