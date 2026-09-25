## What the training script does

- **Model:** `pt_repro` from [model/pointtransformer/pointtransformerlayer.py](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/model/pointtransformer/pointtransformerlayer.py). It has two encoders, one for the body surface and one for the template mean shape, feeding one decoder.
- **Loss:** a Chamfer distance computed per organ ([util/chamfer_loss.py](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/util/chamfer_loss.py)).
- **Training settings:** Adam with learning rate 1e-3, reduced by a factor of 0.3 at epochs 25, 50 and 75. Batch size 16, up to 100 epochs, and training stops early after 6 epochs without improvement.
- **Early stopping caveat:** the "best" checkpoint is chosen on the training loss. The script never uses a validation set.
- **Outputs:** after training it evaluates on the test set (Chamfer distance, HD95 and per-axis errors in mm). It writes CSVs, figures and model weights under `--save_path`.

## Data required

The data comes from the **NAKO 10k** cohort (German National Cohort whole-body MRI). The class is named `NAKO_10k_All_Dataset`, and the image path is hardcoded to `.../nako_10k/images_mri_stitched/`. That data isn't public; you have to apply to NAKO for access. You need these inputs:

| Argument                                                 | Format                                             | Contents                                                     |
| -------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| `--data_root`                                            | a single `.npz` file (despite the argument's name) | Keys are named `"<patient_id>__<field>"`. For each patient: `input_points` holds the body-surface point cloud, shape **(16384, 3)**. Each organ has its own key holding the target point cloud, shape **(4096, 3)**. |
| `--training_list_file_path` / `--testing_list_file_path` | text file                                          | One patient ID per line, matching the IDs in the npz.        |
| `--label_json_file_path`                                 | a **directory**                                    | Must contain `label_organs_<which_label>.json`, which maps label IDs to organ names, e.g. `{"1": "liver", ...}`. The organ names must match the npz field names. |
| `--which_label`                                          | string                                             | Chooses which of those label JSON files to load.             |
| `--init_mean_shape_path`                                 | `.npz`                                             | Contains `mean_pc_np` (template points) and `mean_label_np` (a label for each point). After filtering to the chosen organs it should have **20480 points (5 × 4096)**, so it lines up with the target. |
| `--conversion_path`                                      | `.npz`/`.npy` loaded with pickle allowed           | Keyed by patient ID, each entry a **4×4 voxel-to-world affine**. It's only used to convert results to mm during evaluation, but the dataset raises an error if a patient is missing. |

## Other requirements the code assumes

- **Coordinates normalized to [-1, 1]:** all points are expected in the voxel grid of a **319 × 259 × 315** volume, with axes 0 and 1 flipped ([util/evaluation_functions.py:17-33](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/util/evaluation_functions.py#L17-L33)). If your volumes are a different size, you'll need to change those numbers.
- **Exactly 5 organs:** `N_classes = 5` is hardcoded ([main.py:112](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/main.py#L112)), so the label JSON must list exactly 5 organs.
- **Sizes fixed by the encoder strides:** the body strides (1,4,4,4,4,4) reduce 16384 points to 16, and the mean-shape strides (1,5,4,4,4,4) reduce 20480 to 16. If you change the point counts, update the strides to match.
- **Environment:** requires a CUDA GPU, plus building the `lib/pointops` and `Chamfer3D` CUDA extensions. `nibabel`, `tqdm` and `pandas` are imported, but only pandas is in `requirements.txt`, so install the other two yourself.

A typical run looks like this:



```bash
python main.py --training_list_file_path train.txt --testing_list_file_path test.txt \
  --label_json_file_path labels/ --which_label set1 --data_root data.npz \
  --init_mean_shape_path mean_shape.npz --conversion_path affines.npz --save_path results/
```

The code that turns MRI segmentations into these point clouds isn't in the repo. That includes surface extraction, sampling 16384 or 4096 points, normalizing, and building the mean shape. You'd have to write it yourself or ask the authors for it.



---------------

Right now nothing in the code checks the counts. [main.py:126](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/main.py#L126) and [main.py:129](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/main.py#L129) copy each array into a fixed-size tensor, so if a patient has a different number of points, the script crashes with a shape-mismatch error. It won't pad or crop for you, so equal counts have to be enforced when you build the npz.

## Why the counts must be fixed

- **Batching:** the data is preallocated as `(num_patients, 16384, 3)` and `(num_patients, 5*4096, 3)` tensors.
- **Loss and evaluation:** [util/chamfer_loss.py](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/util/chamfer_loss.py) and [util/evaluation_functions.py](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/util/evaluation_functions.py) split the batch with `view(batch_size, N, 3)`, which assumes every cloud is the same size.
- **Output layout:** the output has one point per template point, and the template has exactly 4096 points per organ. Each target organ must have the same count so the per-organ Chamfer comparison lines up.
- **Encoder strides:** the stride settings are tuned so 16384 and 20480 points both reduce to 16.

The model layers themselves take PointTransformer's `offset` tensors and could handle variable sizes. It's the rest of the pipeline that couldn't.

## How to guarantee the counts: resample during preprocessing

Resample every cloud to the fixed size before saving:



```python
import numpy as np

def resample(points, n, rng=np.random.default_rng(0)):
    """Return exactly n points: subsample if too many, sample with replacement if too few."""
    m = len(points)
    if m >= n:
        idx = rng.choice(m, n, replace=False)
    else:
        idx = np.concatenate([np.arange(m), rng.choice(m, n - m, replace=True)])
    return points[idx].astype(np.float32)

out = {}
for pid, pdata in patients.items():
    out[f"{pid}__input_points"] = resample(pdata["body"], 16384)
    for organ in organs:
        out[f"{pid}__{organ}"] = resample(pdata[organ], 4096)
np.savez(“data.npz”, **out)
```

Better options than random subsampling:

- **Farthest point sampling (FPS)** spreads points evenly over the surface; random sampling tends to leave clumps and gaps. `lib/pointops` already has a CUDA `furthestsampling` you can use.
- **Sampling from the surface mesh:** build a mesh from the segmentation mask (`skimage.measure.marching_cubes`), then sample points from it, for example with `trimesh.sample.sample_surface_even(mesh, n)` or area-weighted sampling. This gives exactly `n` points no matter how big the organ is. That avoids duplicate points, which padding by repetition creates for small organs.
- **Small organs:** if an organ has fewer surface points than you need, sampling from the mesh is better than duplicating points. Duplicates bias the Chamfer loss toward those points.

## Add a check

It's also worth failing early with a clear message instead of a cryptic tensor error. For example, before [main.py:126](vscode-webview://0lgod7a96jdn97ahfq54dvectgdjvkbm2vfofr64eokbvriul7jg/main.py#L126):

```python
assert patient_data['input_points'].shape == (N_in, 3), f"{patient_id}: input has {patient_data['input_points'].shape}"
assert patient_data[organ].shape == (N_points, 3), f"{patient_id}/{organ}: has {patient_data[organ].shape}"
```

The same rule applies to the mean shape: after filtering, each organ in `mean_label_np` must have exactly 4096 points.

I can add these checks to `main.py`, or write a preprocessing script that goes from segmentation masks to the npz format, if that would help.


----------------------

# Using TotalSegmentor Dataset

I've written the preprocessing script, and it has already run: **103 subjects** are ready for `main.py` (82 train, 21 test). I also extended the existing visualiser so it loads a subject by ID. Outputs are in [data/tsmri/](vscode-webview://0kcvvin2c41i995us2tvtv98lca0mgd1ia1o8aegcnk75127ek9u/data/tsmri/).

**Checks I ran:**

- All 103 subjects load the way `main.py` loads them, with the expected point counts. The mean shape has 20,480 points after the label filter.
- Organ points converted back to mm with the repo's own `convert_to_mm` land within one voxel of the original TotalSegmentator mask surfaces. So the [-1, 1] normalisation and the affines match what the evaluation code expects.
- I checked the body masks and point clouds by eye on about half a dozen subjects.
- I haven't run `main.py` itself, because the CUDA extensions aren't built yet (see below).

**How subjects were chosen.** The official `meta.csv` test split would have left only 7 test subjects, so the train/test split is a seeded random 80/20 instead.

| Step                                                         | Subjects left |
| ------------------------------------------------------------ | ------------- |
| All subjects                                                 | 1,232         |
| Scan covers the crop window, and all 5 organs present and uncut | 149           |
| Pass QC                                                      | **103**       |

- **Crop window:** 40 mm below to 360 mm above the top of the sacrum. It's anchored on bone rather than the organs, so the crop doesn't give away where the organs are. Each subject is resampled onto the same 319 × 259 × 315 grid, the size hardcoded in the evaluation code.
- **QC rejections:** 43 because the torso's front or back was outside the scan (mostly thin coronal whole-body scans), and 3 because organs fell outside the body mask. The reason for every excluded subject is in `selection.csv` and `qc.csv`.
- **Mean shape:** built from the 82 training subjects only. For each organ it keeps the most commonly occupied voxels, up to that organ's median volume.

**Body mask.** The original single-threshold mask failed on scans stitched from several stations with different brightness, leaving up to 27% of organ tissue outside the body. I replaced it in [tools/visualize_seg.py](vscode-webview://0kcvvin2c41i995us2tvtv98lca0mgd1ia1o8aegcnk75127ek9u/tools/visualize_seg.py) with a threshold set per axial slice plus hole filling and closing. The preprocessing script uses the same function, so the visualiser shows exactly the mask the data was built from.

**Visualiser.** You can now pass a subject ID:



```bash
conda activate bodyseg
# all 50 structures + body mask
python tools/visualize_seg.py --id s0852 --out vis/s0852
# just the 5 organs, plus the prepared point clouds (toggle them in the HTML legend)
python tools/visualize_seg.py --id s0175 --organs liver spleen kidney_left kidney_right pancreas \
    --prepared data/tsmri --out vis/s0175
```

It writes an interactive 3D HTML and a PNG of three slices, and prints how much of each organ lies outside the body. The old `--img`/`--seg` mode still works. Volumes are now reoriented to standard RAS orientation, so the per-slice hole filling is always axial.

**Re-running the preprocessing:**



```bash
python tools/prepare_totalseg_mri.py --out data/tsmri --workers 16
```

The window size, QC thresholds and split method are all command-line options. The survey stage takes a few minutes and the build about 3.

**Before training:**

- The CUDA extensions aren't built in `bodyseg`. `pointops` isn't installed, and Chamfer3D compiles on import but needs `ninja` (`pip install ninja`). Importing `util/evaluation_functions.py` currently fails for that reason.

- Then run:

  

  ```bash
  python main.py --training_list_file_path data/tsmri/train.txt --testing_list_file_path data/tsmri/test.txt \
    --label_json_file_path data/tsmri/labels --which_label 5organs --data_root data/tsmri/data.npz \
    --init_mean_shape_path data/tsmri/mean_shape.npz --conversion_path data/tsmri/affines.npz --save_path results/tsmri
  ```

- With batch size 16 and `drop_last=True`, the 21 test subjects give only one test batch, and 5 of them are never evaluated.

- 82 training subjects is small compared with the NAKO 10k cohort the paper used, so expect a lot of overfitting.

`data/tsmri/` sits inside the repo and isn't in `.gitignore`, so you may want to add it.

I can build the two CUDA extensions in `bodyseg` next if you like.

# Training

pointops now builds and installs. The smoke test ran out of GPU memory, though, so I'm checking what's on the GPUs.



**Bash** Check GPU memory usage and owners

IN

```
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv; ps -o user,pid,etime,cmd -p $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' | sed 's/,$//') 2>/dev/null | cut -c1-150
```



OUT

```
index, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, 22346 MiB, 23034 MiB, 99 %
1, 22442 MiB, 23034 MiB, 72 %
pid, used_gpu_memory [MiB], process_name
1707325, 21334 MiB, /data/swang7/miniconda3/envs/apple/bin/python
1804026, 904 MiB, /data/swang7/miniconda3/envs/zsvg_clean/bin/python
1795148, 22420 MiB, /data/swang7/miniconda3/envs/apple/bin/python
USER         PID     ELAPSED CMD
swang7   1707325    03:12:13 /data/swang7/miniconda3/envs/apple/bin/python -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=30, 
swang7   1795148    02:22:28 /data/swang7/miniconda3/envs/apple/bin/python -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=30, 
swang7   1804026    02:19:14 /data/swang7/miniconda3/envs/zsvg_clean/bin/python llm_grounding_scanrefer_test.py --prog_path data/scanrefer_programs.js
```

pointops is now built and installed in `bodyseg`, but I haven't tested it on a GPU. Both L4s are full: another user, swang7, has about 22 GB in use on each, so my small test ran out of memory.

**What I ran:**



```bash
cd lib/pointops
# 1. patch the THC include, which was removed in PyTorch 1.11 (done once, already applied)
sed -i 's|#include <THC/THC.h>|#include <ATen/cuda/CUDAContext.h>|' src/*/*_cuda.cpp

# 2. build with a newer compiler and CUDA matching torch (12.1)
source /opt/rh/gcc-toolset-12/enable            # the system gcc 8.5 is too old for torch 2.5
export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9"               # the L4 GPUs
conda activate bodyseg
pip install . --no-build-isolation
```

Two things were stopping the build:

- **Compiler:** the system GCC is 8.5, and torch 2.5 needs GCC 9 or newer. GCC 12 is the newest that CUDA 12.1 supports, and `gcc-toolset-12` is installed.
- **Source:** the `.cpp` files included `THC/THC.h`. They only included it and never called anything from it, so replacing it with the ATen CUDA header is safe. The patch changes 6 files under `lib/pointops/src/`.

The old `build/` and `dist/` folders are a stale Python 3.9 build from the original authors, and nothing uses them.

**Chamfer3D** compiles itself when it's first imported, so it needs the same environment:



```bash
source /opt/rh/gcc-toolset-12/enable
export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH TORCH_CUDA_ARCH_LIST="8.9"
```

Run those in the shell before running `main.py`. The first run compiles Chamfer3D and later runs reuse the cached build.

Once a GPU frees up, this confirms pointops works:



```bash
python -c "import torch,sys; sys.path.insert(0,'lib'); from pointops.functions import pointops; \
x=torch.rand(1000,3,device='cuda'); o=torch.tensor([1000],device='cuda').int(); \
print(pointops.knnquery(8,x,x,o,o)[0].shape)"
```

Even with free GPUs, training at batch size 16 with a 20,480-point template is likely to need most of a 23 GB L4.

## Preliminary training

Training finished without errors. The model roughly halves the organ surface error compared with just placing the mean template.

**Test-set error in mm** ("template" = the mean shape placed with no model):

| Organ        | CD, template | CD, trained | HD95, template | HD95, trained |
| ------------ | ------------ | ----------- | -------------- | ------------- |
| liver        | 19.0         | **11.3**    | 54.1           | **33.3**      |
| spleen       | 15.9         | **8.7**     | 40.4           | **23.0**      |
| kidney_left  | 10.5         | **5.4**     | 29.7           | **17.6**      |
| kidney_right | 10.2         | **5.3**     | 27.5           | **15.6**      |
| pancreas     | 18.8         | **9.5**     | 53.6           | **28.9**      |

Chamfer distance (CD) falls 40–50% for every organ, and HD95 (95th-percentile surface distance) drops by about the same.

**How the run went:**

- Early stopping ended training at epoch 50. The best training loss was 0.00431, at epoch 44.
- Most of the learning happened in the first ~10 epochs, and after that the loss only crept down.
- The loss plot drops to 0 after epoch 50 only because the unused epochs are plotted as zeros. It isn't a real drop.
- In the sample figures, the predicted organs sit in the right places with roughly the right shapes, but their surfaces are noticeably fuzzier than the targets.

**Caveats on these numbers:**

- **Only 16 of the 21 test subjects were evaluated.** The test loader drops the last incomplete batch, and at batch size 8 that leaves 2 batches. The mean-template numbers also differ from the trial run because a different set of subjects was evaluated.
- **The checkpoint is picked on training loss**, since `main.py` has no validation set. These results are fair because the test set played no part in choosing the checkpoint. But with only 82 training subjects, the model may be overfitting in ways this doesn't show.
- **The per-axis columns** in the CSV (`max_width`, `min_depth` and so on) did not all improve. For example, the liver's `max_width` went from 29.8 to 39.8 mm. Those are extreme single-axis errors, so the CD and HD95 figures give a better overall picture.
- **This isn't comparable with the paper's NAKO results.** The dataset is about 100× smaller, and the scans vary a lot in sequence and FOV.

Results are in `results/tsmri_b8/`. If you want to go further, the next step I'd take is to evaluate all 21 test subjects and add a validation split so the checkpoint is chosen properly. Both are small changes to `main.py`.