# Running DeformingPointTransformer on our data

This guide covers two pipelines built on the original code:

- **A. TotalSegmentator MRI:** body surface → liver, spleen, both kidneys, pancreas.
- **B. Houdini cactus-pose arms:** arm skin → biceps and the three triceps heads.

Both write the same file formats, and both are trained and evaluated with `main.py`.
All commands run from the repository root (`~/data/git/DeformingPointTransformer`).

---

## 1. Environment (once per shell)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bodyseg
source /opt/rh/gcc-toolset-12/enable                 # system gcc 8.5 is too old for torch 2.5
export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9                      # L4 GPUs
```

- **pointops:** already built and installed in `bodyseg`. The legacy `THC/THC.h` include in `lib/pointops/src/*/*_cuda.cpp` was replaced by `ATen/cuda/CUDAContext.h`. To rebuild it, run `cd lib/pointops && pip install . --no-build-isolation`.
- **Chamfer3D:** compiled automatically the first time it is imported (`Jitting Chamfer 3D`) and cached in `~/.cache/torch_extensions`. It needs `ninja` and the exports above.
- **Preprocessing** (`tools/prepare_*.py`, `tools/visualize_seg.py`): runs on the CPU and needs no GPU.

## 2. Getting a GPU on the cluster

All nodes share the same filesystem, so data prepared on one node can be used on another.

```bash
cnode mars                        # nodes with Connect = yes can be ssh'd into; %GPU is node-wide
ssh mars12
nvidia-smi --query-gpu=index,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=1     # the GPU with the most free memory
```

- **`tools/gpu_watch.sh`:** polls the connectable nodes and runs a pointops check on the first GPU with enough free memory. Use `INTERVAL=120 MIN_FREE=4096 bash tools/gpu_watch.sh`.
- **`python tools/mem_probe.py 2 4 8`:** measures the peak memory of one training step at each batch size. On an L4 with 8 GB free, batch size 8 fits (6.8 GiB) and 12 does not.
- **Long runs:** start them with `nohup ... > run.log 2>&1 &` so they survive logging out.
- **Multi-GPU:** `main.py` uses one GPU. Using both GPUs of a node would need `DistributedDataParallel`, which is not implemented.

## 3. `main.py` reference

```bash
python main.py \
  --training_list_file_path <train.txt> --testing_list_file_path <test.txt> \
  --label_json_file_path <labels dir> --which_label <label set> \
  --data_root <data.npz> --init_mean_shape_path <mean_shape.npz> \
  --conversion_path <affines.npz> --save_path <results dir> \
  [--batch_size 16] [--epochs 100] [--residual [--residual_scale 0.02]] [--no-bn_recalibrate] [--eval_only]
```

| Option | Meaning |
|---|---|
| `--batch_size`, `--epochs` | Defaults 16 and 100. Training stops early after 6 epochs without improvement in training loss. |
| `--residual` | Predicts *template + residual_scale × correction* instead of absolute coordinates. The last layer is zero-initialised, so the untrained model returns the template exactly. Point *i* of the output stays template point *i*, which is what makes volumes measurable. |
| `--residual_scale` | Size of one unit of correction, in normalised coordinates. For the arms, 1 unit ≈ 256 mm, so 0.02 ≈ 5 mm. |
| `--bn_recalibrate` | On by default. Before evaluating, recomputes the BatchNorm running statistics with one pass over the training set, using the final weights. With small batches the running averages kept during training can drift far from the statistics the weights were trained with. In two residual arm folds this moved eval-mode predictions by 40–80 mm, even on training arms. Turn it off with `--no-bn_recalibrate`. |
| `--eval_only` | Skips training and evaluates `<save_path>/models/model_weights_best.pth`, with recalibration. |

The organ count, point counts and encoder strides are read from the data. Every test subject is evaluated.

Outputs in `--save_path`:

| File | Contents |
|---|---|
| `models/model_weights_{best,last}.pth` | Model weights. |
| `csv/trained_vs_target_<label>.csv` | Model errors (CD, HD95 and per-axis extent errors, in mm). |
| `csv/init_vs_target_<label>.csv` | The same errors for the template alone, the baseline to beat. |
| `predictions.npz` | Predicted points for each test sample, in normalised coordinates. |
| `figures/` | Loss curve and sample plots. |

**Caveat:** the best checkpoint is chosen on *training* loss. There is no validation set.

---

## A. TotalSegmentator MRI pipeline

Data: `/home/aalempij/Data/data/TotalsegmentatorMRI_dataset_v300`.

### A1. Prepare

```bash
python tools/prepare_totalseg_mri.py --out data/tsmri --workers 16
```

The script runs three stages. Pick stages with `--stages survey select build`; the survey is cached in `survey.csv`.

1. **survey:** organ presence, organ extents and whether the scan's FOV cuts them, for all 1,232 subjects.
2. **select:** keeps scans that cover a fixed window of 40 mm below to 360 mm above the top of the sacrum, and contain all five organs uncut.
3. **build:**
   - Resamples each scan to a common 319×259×315 RAS grid (1.7 mm in-plane).
   - Makes a body mask from the MRI with a per-slice threshold.
   - Rejects scans whose torso is cut front or back by the FOV, or whose organs fall outside the body mask.
   - Samples 16,384 body-surface points and 4,096 points per organ.
   - Builds the mean-shape template from the training subjects only.

Result: **103 subjects**, with a random 80/20 split (82 train, 21 test). Use `--split meta` for the `meta.csv` split, which leaves only 7 test subjects.

Outputs: `data.npz`, `mean_shape.npz`, `affines.npz`, `labels/label_organs_5organs.json`, `train.txt`, `test.txt`, and the reports `survey.csv`, `selection.csv` and `qc.csv`.

### A2. Look at a subject

```bash
python tools/visualize_seg.py --id s0175 --organs liver spleen kidney_left kidney_right pancreas \
    --prepared data/tsmri --out vis/s0175
```

This writes `vis/s0175.html`, an interactive 3D view of the body surface and organ meshes. The prepared point clouds can be switched on from the legend. It also writes `vis/s0175.png`, axial, coronal and sagittal slices with the body contour.

- Leave out `--organs` to show all 50 structures.
- The older multi-label mode still works: `--img mri.nii.gz --seg seg.nii.gz`.

### A3. Train and evaluate

```bash
nohup python main.py \
  --training_list_file_path data/tsmri/train.txt --testing_list_file_path data/tsmri/test.txt \
  --label_json_file_path data/tsmri/labels --which_label 5organs \
  --data_root data/tsmri/data.npz --init_mean_shape_path data/tsmri/mean_shape.npz \
  --conversion_path data/tsmri/affines.npz --save_path results/tsmri_b8 \
  --batch_size 8 > results_tsmri_b8.log 2>&1 &
# add --residual to train the residual variant
```

To get train and test errors per subject from a trained model:

```bash
python tools/eval_splits.py --weights results/tsmri_b8/models/model_weights_best.pth \
    --out results/tsmri_b8/csv/per_subject.csv
```

This script is specific to this 5-organ dataset.

**Results so far** (absolute coordinates, batch size 8, early stop at epoch 50). CD in mm, on test and train subjects:

| | liver | spleen | kidney L | kidney R | pancreas |
|---|---|---|---|---|---|
| template (test) | 13.7 | 11.5 | 8.1 | 8.2 | 13.8 |
| model (test) | 11.1 | 8.3 | 5.5 | 5.5 | 9.1 |
| model (train) | 11.3 | 7.8 | 5.7 | 5.9 | 8.4 |

Train error equals test error, so the model is underfitting rather than overfitting. Longer training, `--residual` and a cleaner input surface (dropping the arms) are the next things to try.

---

## B. Houdini cactus-pose arm pipeline

Data: `/data/aalempij/data/datasets_cactus_poses`, as written by `export_cactus_pose_ply_mirrored_tree.py`:

```
final skins for 45th and 56th frames/<subject>/final_skin.<frame:05d>.ply
entire muscles for 45th and 56th frames/<subject>/entire_muscles.<frame:05d>.ply
bones for 45th and 56th frames/bones/v1/bones_v1.<frame:04d>.ply
```

Subject folder names carry the simulation parameters, e.g. `BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1`.

### B1. Prepare

```bash
# single train/test split (20% of subjects held out)
python tools/prepare_cactus.py --out data/cactus45
# leave-one-subject-out folds (N subjects per fold with --leave_out N)
python tools/prepare_cactus.py --out data/cactus45_loo --leave_out 1
```

What it does:

- **Target muscles** (`--pieces`): chosen by the `piece` attribute of the muscle mesh, given as right:left IDs. Biceps brachii is 5:104, lateral triceps 94:193, medial triceps 96:195 and long triceps 95:194.
  - The script **stops** if a muscle lies more than 5 cm from its reference position (`--max_shift_mm`), which catches a new export where the pieces were renumbered.
  - If you can, export a muscle *name* attribute from Houdini so the IDs cannot silently change.
- **Arms** (`--arms both`, the default): the left arm is mirrored onto the right, so each subject gives two samples, `<subject>_R` and `<subject>_L`. Both arms of a subject are always in the same split.
- **Input:** skin within `--crop_mm` (default 100 mm) of a fixed reference position of the target muscles. The crop runs from armpit and shoulder to past the elbow, and keeps 75–93% of the skin signal; 150 mm keeps about 99% but includes much more torso. 16,384 points are sampled from it.
- **Targets:** 4,096 points on each muscle. The exact volume of every muscle is stored in `subjects.csv` as `<muscle>_mesh_ml`.
- **Normalisation:** one isotropic scaling to [-1, 1] shared by all samples. `affines.npz` makes `main.py` report errors in mm.
- **Template:** built from the training samples only. Voxels are filled exactly from the closed muscle meshes, and the template is sized to the mean muscle volume. The template mesh is stored with it for the volume evaluation.
- **`--align bone`:** before cropping, maps each arm onto the reference humerus with the best-fitting rotation, shift and single scale factor, matched vertex to vertex. This removes differences in bone size and position, so the task becomes muscle shape relative to the skeleton.
  - Errors are still reported in each subject's own mm.
  - With the current data the bones are identical, so this has no effect.
  - `--bone_file` accepts `{subject}` and `{frame}` placeholders for per-subject skeletons.
  - All bone meshes must share one topology.
- **Other options:** `--frame 56`, `--test_subjects <names>`, `--seed`, `--n_body`, `--n_organ`, `--reference <subject>`.
- **QC:** check `qc.png` (a sample crop with its muscles, and the template) and `subjects.csv`.

### B2. Train: single split

```bash
python main.py \
  --training_list_file_path data/cactus45/train.txt --testing_list_file_path data/cactus45/test.txt \
  --label_json_file_path data/cactus45/labels --which_label arm4 \
  --data_root data/cactus45/data.npz --init_mean_shape_path data/cactus45/mean_shape.npz \
  --conversion_path data/cactus45/affines.npz --save_path results/cactus_res \
  --batch_size 4 --epochs 100 --residual
python tools/volume_eval.py --prepared data/cactus45 --results results/cactus_res
```

### B3. Train: leave-N-out

```bash
nohup python tools/run_folds.py --prepared data/cactus45_loo --label_set arm4 \
    --save results/cactus_loo_res -- --batch_size 4 --epochs 100 --residual > loo_res.log 2>&1 &
```

- Arguments after `--` are passed to `main.py` unchanged.
- Folds run one after another. Each fold logs to `<save>/fold_XX/train.log`.
- Folds that already have results are skipped, so to resume after an interruption, run the same command again.
- `--folds fold_00 fold_03` runs only those folds.
- `--eval_only` re-evaluates existing fold models without training, for example to add predictions and volumes to an older run.
- `--summary_only` rebuilds the summary without training.

The summary is printed and saved to `<save>/summary_arm4.csv`. For each muscle it gives CD and HD95 in mm, and volume error in ml and %, for both the model and the template. Each value is the mean over all held-out samples, with the standard deviation across folds.

**About the volume metric:** it deforms the template mesh with the predicted points, so it needs point *i* to stay template point *i*.

- This holds for `--residual` models.
- It does not hold for the default absolute-coordinate model, whose points drift about 84 mm from their template points. Its volume is reported as NaN, with a warning.
- Estimating volume from the bare point cloud is not reliable for these thin muscles: it is 35–96% off even on the true point clouds.

**Results so far** (leave one subject out; 11 subjects, 22 arms):

| | biceps | lat. triceps | long triceps | med. triceps |
|---|---|---|---|---|
| template CD (mm) | 3.4 | 2.8 | 2.8 | 2.4 |
| absolute model CD (mm) | 6.8 | 5.3 | 6.0 | 5.1 |
| template volume error (%) | 23 | 28 | 25 | 27 |

- **The absolute model:** it underfits, reaching about 6–9 mm RMS even on its training arms. The residual model starts at the template's accuracy by construction; a full leave-one-out run with `--residual` is the next comparison.
- **The CD noise floor:** two samplings of the *same* mesh with 4,096 points already differ by 1.3–1.7 mm CD. Much of the template's 2–3 mm is sampling noise, and volume error is the more sensitive measure here.
- **Volumes:** each parameter switches its muscle between two volumes, 50–70% apart. For example, the biceps is 318 or 531 ml.
- **Left and right arms:** they agree to within about 1 ml, so there are effectively 11 independent samples, not 22.
- **More simulations:** new Houdini exports with continuous parameter values will do more than anything else. Keep the folder layout, then rerun B1.

---

## Viewing subjects and predictions in 3D (HTML)

Both viewers write one self-contained `.html` file (3D plotly, about 3–12 MB). Open it in a browser, or right-click it in VS Code and choose *Open with Live Server / Preview*. Each layer can be switched on and off by clicking its legend entry. Layers that are hidden by default are greyed out in the legend.

**Arms (cactus):** [tools/visualize_cactus.py](tools/visualize_cactus.py)

```bash
# a subject, both arms: arm skin, the four target muscles, humerus
python tools/visualize_cactus.py --subject BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1 --out vis/s1
# one arm sample + its prepared points (input skin points, target muscle points)
python tools/visualize_cactus.py --id BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1_R --prepared data/cactus45 --out vis/s1_R
# + the model prediction and its template (for leave-N-out, the fold holding the sample is found)
python tools/visualize_cactus.py --id BB_-0.6_TRL_0.6_TRM_0.6_TRLN_0.6_T_1_L \
    --prepared data/cactus45_loo_v2 --results results/cactus_loo_res --out vis/s2_L_pred
```

- **Skin:** `--skin full` shows the whole body, `--skin none` hides it, and `--skin_mm` sets the arm crop (default 150 mm).
- **Other layers:** `--all_muscles` adds every other muscle piece in grey, and `--no_bone` hides the humerus.
- **Left arms:** `_L` samples are mirrored back onto the left arm.

**MRI (TotalSegmentator):** [tools/visualize_seg.py](tools/visualize_seg.py) also writes a PNG of three slices.

```bash
python tools/visualize_seg.py --id s0175 --organs liver spleen kidney_left kidney_right pancreas \
    --prepared data/tsmri --results results/tsmri_b8 --out vis/s0175_pred
```

**Predictions:**
- Predictions only exist for *test* samples, in `<results>/predictions.npz`, which `main.py` writes during evaluation.
- Runs made before that file existed (e.g. `results/tsmri_b8`) can be re-evaluated from their saved weights without retraining. Add `--eval_only` to the original `main.py` command.
