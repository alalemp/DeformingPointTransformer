# Running DeformingPointTransformer on our data

This guide covers two pipelines built on the original code:

- **A. TotalSegmentator MRI:** body surface → liver, spleen, both kidneys, pancreas.
- **B. Houdini cactus-pose arms:** arm skin → biceps and the three triceps heads.

Both write the same file formats, and both are trained and evaluated with `main.py`.
Section C adds a triceps_brachii segmentation of the TotalSegmentator MRI subjects, for arm-muscle work on real scans.
All commands run from the repository root (`~/data/git/DeformingPointTransformer`).

## Contents

0. [Reproducing the arm results (step by step)](#0-reproducing-the-arm-results-step-by-step)
1. [Environment (once per shell)](#1-environment-once-per-shell)
2. [Getting a GPU on the cluster](#2-getting-a-gpu-on-the-cluster)
3. [`main.py` reference](#3-mainpy-reference)
4. [A. TotalSegmentator MRI pipeline](#a-totalsegmentator-mri-pipeline)
   - [A1. Prepare](#a1-prepare)
   - [A2. Look at a subject](#a2-look-at-a-subject)
   - [A3. Train and evaluate](#a3-train-and-evaluate)
5. [B. Houdini cactus-pose arm pipeline](#b-houdini-cactus-pose-arm-pipeline)
   - [B1. Prepare](#b1-prepare)
   - [B2. Train: single split](#b2-train-single-split)
   - [B3. Check one fold first](#b3-check-one-fold-first)
   - [B4. Train: leave-N-out](#b4-train-leave-n-out)
6. [Viewing subjects and predictions in 3D (HTML)](#viewing-subjects-and-predictions-in-3d-html)
   - [Paper figures](#paper-figures)
7. [C. Triceps segmentation with TotalSegmentator](#c-triceps-segmentation-with-totalsegmentator)
   - [C1. Run](#c1-run)
   - [C2. What the script does](#c2-what-the-script-does)
   - [C3. Outputs](#c3-outputs)
   - [C4. Rescore or rerun](#c4-rescore-or-rerun)

---

## 0. Reproducing the arm results (step by step)

This is the complete sequence for the cactus-pose arm experiments: frames 160 and 220, leave one muscle group out with validation-based checkpoints, then the paper figures. Every step is a command you can copy. The sections below explain the options in detail.

**Fixed settings** (all defaults unless shown in the commands):

| Setting | Value |
|---|---|
| Data | `/data/aalempij/data/datasets_cactus_pose`, frames 160 and 220 (108 skins in 49 muscle groups) |
| Folds | leave one group out (49 folds), `prepare_cactus.py --seed 0`, identical in both frames |
| Per fold | test 1 group (2–6 arms), validation 5 groups (`--val_frac 0.1`), training 43 groups |
| Input / targets | skin within 100 mm of the muscles, 16,384 points; 4,096 points per muscle; both arms (left mirrored) |
| Model | residual (`--residual`, scale 0.02), batch size 4, `main.py --seed 0` |
| Optimiser | Adam, learning rate 1e-3, × 0.3 at epochs 25, 50 and 75, at most 120 epochs |
| Checkpoint | lowest validation loss; early stop after 15 epochs without improvement (`--patience 15`) |
| Evaluation | BatchNorm statistics recalibrated on the training set (default); CD, HD95, volume error vs the template |

**Step 1. Install once, then set up the shell.** See [CLUSTER.md](CLUSTER.md) for installation. Then, in every new shell:

```bash
cd ~/data/git/DeformingPointTransformer
source cluster_env.sh
```

**Step 2. Prepare the folds for both frames.** This takes about a minute each and needs no GPU.

```bash
python tools/prepare_cactus.py --out data/cactus160_loo_val --leave_out 1 --val_frac 0.1
python tools/prepare_cactus.py --out data/cactus220_loo_val --leave_out 1 --val_frac 0.1 --frame 220
```

Check `data/cactus160_loo_val/qc.png` (a sample crop with its muscles, and the template) and `subjects.csv`. Each fold folder (`folds/fold_00` … `fold_48`) holds `train.txt`, `val.txt`, `test.txt` and its own `mean_shape.npz`.

**Step 3 (optional). Check one fold learns** (about 40 minutes):

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 --save results/check160 \
    --folds fold_00 -- --batch_size 4 --epochs 120 --residual --patience 15
grep "^Epoch" results/check160/fold_00/train.log | tail
```

Validation CD should fall below the template's within the first 10–20 epochs.

**Step 4. Train all folds of both frames.** On a node with two free GPUs, run one frame per GPU:

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv          # about 7 GB free per GPU is enough
CUDA_VISIBLE_DEVICES=0 nohup python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 \
    --save results/cactus160_loo_val -- --batch_size 4 --epochs 120 --residual --patience 15 \
    > results/cactus160_loo_val.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python tools/run_folds.py --prepared data/cactus220_loo_val --label_set arm4 \
    --save results/cactus220_loo_val -- --batch_size 4 --epochs 120 --residual --patience 15 \
    > results/cactus220_loo_val.log 2>&1 &
```

- **Time:** about 25 s per epoch on an RTX A5500 and about 1 minute on an L4. With early stopping, a fold takes 40–60 minutes on an A5500, so **49 folds take about 1–1.5 days** per frame.
- **More GPUs or nodes:** run the same command several times with different `--folds` lists, e.g. `--folds $(printf "fold_%02d " $(seq 0 24))` and `$(printf "fold_%02d " $(seq 25 48))`. Every process writes to the same `--save` folder, and the filesystem is shared.
- **Interrupted runs:** run the same command again. Finished folds are skipped.
- **Quicker run:** `--leave_out 5` in step 2 gives 10 folds, about 5x faster, with every group still tested once.

**Step 5. Monitor.**

```bash
tail -2 results/cactus160_loo_val.log results/cactus220_loo_val.log                    # current fold
ls -d results/cactus160_loo_val/fold_*/csv/trained_vs_target_arm4.csv | wc -l          # folds done (of 49)
grep -h "Best checkpoint" results/cactus160_loo_val/fold_*/train.log | tail             # selected epochs
```

**Step 6. Summarise each frame.** This works on partial results too.

```bash
python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 --save results/cactus160_loo_val --summary_only
python tools/run_folds.py --prepared data/cactus220_loo_val --label_set arm4 --save results/cactus220_loo_val --summary_only
```

This writes `results/<run>/summary_arm4.csv`: CD, HD95 and volume error per muscle, for the model and the template, as the mean over all held-out arms and the standard deviation across folds. Each fold also has `fold_XX/csv/volumes.csv`, with predicted and true volume per arm and muscle.

**Step 7. Paper figures and table.** Any held-out arm can be the example in figures 1, 4 and S1; the fold that tested it is found automatically.

```bash
python tools/paper_figures.py --out figures/paper \
  --run "Frame 160=data/cactus160_loo_val:results/cactus160_loo_val:160" \
  --run "Frame 220=data/cactus220_loo_val:results/cactus220_loo_val:220" \
  --sample BB_0_TRL_-0.6_TRM_0_TRLN_0.6_T_1_R
```

The output is listed under [Paper figures](#paper-figures). The figures combine all finished folds, so they can be made before every fold is done. The table reports how many folds and test arms it includes.

**Step 8 (optional). Interactive 3D view of any held-out arm:**

```bash
python tools/visualize_cactus.py --id BB_0_TRL_-0.6_TRM_0_TRLN_0.6_T_1_R --prepared data/cactus160_loo_val \
    --results results/cactus160_loo_val --out vis/BB0_R_pred
```

**Reproducibility notes:**
- Fold and validation assignments are fixed by `prepare_cactus.py --seed`, and model initialisation and batch order by `main.py --seed`.
- GPU kernels, pointops especially, are not bit-exact, so reruns can differ in the third decimal place.
- Record the commit (`git rev-parse HEAD`) with the results.

---

## 1. Environment (once per shell)

Installation (Miniconda, the conda env, the CUDA extensions, TotalSegmentator and VIBESegmentator) is covered in [CLUSTER.md](CLUSTER.md). Once installed, each new shell needs one line:

```bash
source cluster_env.sh
```

It does the following:

```bash
source <miniconda>/etc/profile.d/conda.sh && conda activate bodyseg   # finds the Miniconda that holds the env
source /opt/rh/gcc-toolset-12/enable                 # system gcc 8.5 is too old for torch 2.5
export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"    # build CUDA extensions for all cluster GPUs
```

- **GPUs:** the cluster has NVIDIA L4 (mars nodes, compute 8.9) and RTX A5500 (venus nodes, compute 8.6) GPUs. The CUDA extensions are built for every GPU generation from 7.5 to 9.0, so the same environment runs on both.
- **pointops:** built and installed in `bodyseg`. The legacy `THC/THC.h` include in `lib/pointops/src/*/*_cuda.cpp` was replaced by `ATen/cuda/CUDAContext.h`. To rebuild it after `source cluster_env.sh`, run `cd lib/pointops && pip install . --no-build-isolation`. A "no kernel image is available" error means it was built for different GPUs; rebuild it, and delete `~/.cache/torch_extensions/*/chamfer_3D`.
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
  [--batch_size 16] [--epochs 100] [--lr 1e-3] [--milestones 25,50,75] [--gamma 0.3] [--patience 6] \
  [--monitor_test] [--residual [--residual_scale 0.02]] [--no-bn_recalibrate] [--eval_only]
```

| Option | Meaning |
|---|---|
| `--batch_size`, `--epochs` | Defaults 16 and 100. |
| `--lr`, `--milestones`, `--gamma` | Adam learning rate (default 1e-3), multiplied by `--gamma` (default 0.3) at each milestone epoch (default `25,50,75`; empty for a constant rate). |
| `--validation_list_file_path` | Optional validation list. When given, the saved checkpoint is the epoch with the lowest **validation** loss, and early stopping also watches validation loss. Each epoch then reports validation loss and validation CD against the template, and the chosen epoch is written to `csv/best_epoch.txt`. Without it, both use training loss. `run_folds.py` passes a fold's `val.txt` automatically. |
| `--patience` | Stop after this many epochs without a new best loss: validation loss if a validation set is given, otherwise training loss (default 6; `0` = never stop early). |
| `--monitor_test` | After every epoch, also report the test loss and the test CD in mm for both the model and the template, so you can see whether the model beats the template. It costs a few seconds per epoch. |
| `--residual` | Predicts *template + residual_scale × correction* instead of absolute coordinates. The last layer is zero-initialised, so the untrained model returns the template exactly. Point *i* of the output stays template point *i*, which is what makes volumes measurable. |
| `--residual_scale` | Size of one unit of correction, in normalised coordinates. For the arms, 1 unit ≈ 256 mm, so 0.02 ≈ 5 mm. |
| `--bn_recalibrate` | On by default. Before evaluating, recomputes the BatchNorm running statistics with one pass over the training set, using the final weights. With small batches the running averages kept during training can drift far from the statistics the weights were trained with. In two residual arm folds this moved eval-mode predictions by 40–80 mm, even on training arms. Turn it off with `--no-bn_recalibrate`. |
| `--seed` | Random seed for Python, NumPy and torch (default 0). |
| `--eval_only` | Skips training and evaluates `<save_path>/models/model_weights_best.pth`, with recalibration. |

The organ count, point counts and encoder strides are read from the data. Every test subject is evaluated.

Outputs in `--save_path`:

| File | Contents |
|---|---|
| `models/model_weights_{best,last}.pth` | Model weights. |
| `csv/trained_vs_target_<label>.csv` | Model errors (CD, HD95 and per-axis extent errors, in mm). |
| `csv/init_vs_target_<label>.csv` | The same errors for the template alone, the baseline to beat. |
| `predictions.npz` | Predicted points for each test sample, in normalised coordinates. |
| `csv/loss_history.csv` | Per epoch: learning rate, training loss and, with `--monitor_test`, test loss, test CD and template CD. Rewritten every epoch, so it can be checked while training runs. |
| `figures/` | Loss curves (training and test; with `--monitor_test` also test CD against the template) and sample plots. |

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

**Data:** `/data/aalempij/data/datasets_cactus_pose` (current export, frames **160** and **220**):

```
final skin/<subject>/final_skin.<frame:05d>.ply
separated muscles/<Muscle>/<group>/<Muscle>.<frame:05d>.ply    # bicepsBrachialis, tricepsLateralis,
                                                               # tricepsMedialis, tricepsLongus
entire muscles/<group>/entire_muscles.<frame:05d>.ply          # all muscles (only used by the viewer)
bones/bones.<frame:05d>.ply                                    # one skeleton shared by all subjects
```

**Subjects and groups:**
- Skin folder names carry the simulation parameters, e.g. `BB_0_TRL_0.6_TRM_-0.6_TRLN_0_T_1.25`.
  - BB, TRL, TRM and TRLN set the four muscles.
  - T takes the values 0.75, 1 and 1.25 and changes the skin; it does not affect the muscles.
- The muscle folders are named by the four muscle parameters only. This muscle configuration is the subject's **group** (the name without `_T_…`).
- Counts: 108 skins come from **49 groups** (1–3 skins each), giving 216 arm samples.
- Parameter coverage:
  - BB takes only −0.6 and 0.
  - TRL, TRM and TRLN take −0.6, 0 and 0.6.
  - T is 0.75 (11 skins), 1 (49) or 1.25 (48).
- **Splits are always by group.** All skins and both arms of a group stay on the same side, so a test muscle shape never appears in training through a different skin.

**Frames:** frames 160 and 220 differ mainly in the raised forearms and hands. In the upper-arm crop the skin barely moves (median 0.15 mm, at most 28 mm near the elbow), and the muscles change only slightly. 160 is the default; use `--frame 220` for the other pose.

The first export (`/data/aalempij/data/OLD_datasets_cactus_poses`, frames 45 and 56, muscles chosen by piece number from `entire_muscles`) still works. The scripts detect the layout from `--data`; pass `--data /data/aalempij/data/OLD_datasets_cactus_poses --frame 45`.

### B1. Prepare

```bash
# leave-one-group-out: 49 folds, each holding out one muscle configuration (2-6 arm samples)
python tools/prepare_cactus.py --out data/cactus160_loo --leave_out 1
# fewer, larger folds: 5 groups held out per fold -> 10 folds
python tools/prepare_cactus.py --out data/cactus160_l5o --leave_out 5
# single train/test split (20% of groups held out)
python tools/prepare_cactus.py --out data/cactus160
# the other pose
python tools/prepare_cactus.py --out data/cactus220_loo --leave_out 1 --frame 220
```

Each run takes under a minute. `data/cactus160` and `data/cactus160_loo` have already been built.

What it does:

- **Target muscles** (`--muscles`): read from `separated muscles/<Muscle>/`. Each file is a closed mesh holding both arms, split at x = 0 (the right arm is x > 0). The default maps `biceps_brachii:bicepsBrachialis`, `triceps_lateral:tricepsLateralis`, `triceps_medial:tricepsMedialis` and `triceps_long:tricepsLongus`.
  - The old export instead selects muscles by piece number (`--pieces`, right:left IDs 5:104, 94:193, 96:195 and 95:194).
  - The script **stops** if a muscle lies more than 5 cm from its reference position (`--max_shift_mm`), which catches a wrong muscle mapping.
- **Arms** (`--arms both`, the default): the left arm is mirrored onto the right, so each skin gives two samples, `<subject>_R` and `<subject>_L`. Their muscles agree to within 0.5 ml.
- **Input:** skin within `--crop_mm` (default 100 mm) of a fixed reference position of the target muscles. The crop runs from armpit and shoulder to past the elbow, and keeps 75–93% of the skin signal; 150 mm keeps about 99% but includes much more torso. 16,384 points are sampled from it.
- **Targets:** 4,096 points on each muscle. The exact volume of every muscle is stored in `subjects.csv` as `<muscle>_mesh_ml`, next to the subject's parameters, its group and its split or fold.
- **Normalisation:** one isotropic scaling to [-1, 1] shared by all samples. `affines.npz` makes `main.py` report errors in mm.
- **Template:** built from the training samples only. Voxels are filled exactly from the closed muscle meshes, and the template is sized to the mean muscle volume. The template mesh is stored with it for the volume evaluation.
- **`--align bone`:** before cropping, maps each arm onto the reference humerus with the best-fitting rotation, shift and single scale factor, matched vertex to vertex. This removes differences in bone size and position, so the task becomes muscle shape relative to the skeleton.
  - Errors are still reported in each subject's own mm.
  - With the current data every subject shares one skeleton, so this has no effect.
  - `--bone_file` accepts `{subject}` and `{frame}` placeholders for per-subject skeletons.
  - All bone meshes must share one topology.
- **Other options:** `--frame 220`, `--test_subjects <subjects or groups>` (their whole groups are held out), `--test_frac`, `--seed`, `--n_body`, `--n_organ`, `--reference <subject>`.
- **QC:** check `qc.png` (a sample crop with its muscles, and the template) and `subjects.csv`.

### B2. Train: single split

```bash
python main.py \
  --training_list_file_path data/cactus160/train.txt --testing_list_file_path data/cactus160/test.txt \
  --label_json_file_path data/cactus160/labels --which_label arm4 \
  --data_root data/cactus160/data.npz --init_mean_shape_path data/cactus160/mean_shape.npz \
  --conversion_path data/cactus160/affines.npz --save_path results/cactus160_res \
  --batch_size 4 --epochs 100 --residual
python tools/volume_eval.py --prepared data/cactus160 --results results/cactus160_res
```

### B3. Check one fold first

Before running all folds, train a single fold with per-epoch test reporting to see whether the model learns at all:

```bash
source cluster_env.sh
export CUDA_VISIBLE_DEVICES=0
python tools/run_folds.py --prepared data/cactus160_loo --label_set arm4 --save results/try1 --folds fold_00 \
    -- --batch_size 4 --epochs 100 --residual --monitor_test
grep "^Epoch" results/try1/fold_00/train.log                 # the per-epoch lines
```

Or run `main.py` directly, to see the progress live:

```bash
F=data/cactus160_loo/folds/fold_00
python main.py --training_list_file_path $F/train.txt --testing_list_file_path $F/test.txt \
  --label_json_file_path data/cactus160_loo/labels --which_label arm4 \
  --data_root data/cactus160_loo/data.npz --init_mean_shape_path $F/mean_shape.npz \
  --conversion_path data/cactus160_loo/affines.npz --save_path results/try1_main \
  --batch_size 4 --epochs 100 --residual --monitor_test
```

Each epoch prints one line:

```
Epoch 12/100 | lr 1.00e-03 | Avg Loss - 0.000201 | test loss 0.000170 | test CD 2.051 mm (template 2.118 mm) | New best result
```

How to read it:
- **Learning:** test CD drops below the template and keeps falling.
- **Not learning:** both losses stay flat, and test CD stays at the template's value. Try a larger `--residual_scale` (0.05), a larger or smaller `--lr` (3e-3 / 3e-4), or `--patience 0 --epochs 200` so early stopping does not end it.
- **Overfitting:** training loss keeps falling while test loss and test CD rise again.

One fold is a single held-out muscle configuration (2–6 arm samples), so its test numbers are noisy. Compare settings on the same fold, and then check two or three more folds before committing to all 49.

### B4. Train: leave-N-out

**Recommended: leave one group out with a validation set.** `--val_frac 0.1` holds out about 10% of each fold's non-test groups for checkpoint selection, drawn separately for every fold. Train, validation and test never share a muscle group, and the template is built from the training groups only.

| Per fold | Groups | Arm samples | Used for |
|---|---|---|---|
| Test | 1 | 2–6 | the reported result; every group is tested exactly once over 49 folds |
| Validation | 5 | about 20 | choosing the checkpoint (and early stopping) |
| Training | 43 | about 190 | fitting the model and building the template |

```bash
python tools/prepare_cactus.py --out data/cactus160_loo_val --leave_out 1 --val_frac 0.1
python tools/prepare_cactus.py --out data/cactus220_loo_val --leave_out 1 --val_frac 0.1 --frame 220
CUDA_VISIBLE_DEVICES=0 nohup python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 \
    --save results/cactus160_loo_val -- --batch_size 4 --epochs 120 --residual --patience 15 > results/cactus160_loo_val.log 2>&1 &
```

- **Folds are identical in both frames** (same seed), so the two frames can be compared fold by fold.
- **Time:** about 25 s per epoch on an RTX A5500, and about 40–60 min per fold with early stopping. That's roughly 1–1.5 days for 49 folds on one GPU per frame.
- **Faster alternative:** `--leave_out 5` gives 10 folds with every group still tested exactly once, in about 5x less time.
- **Without validation:** omit `--val_frac` (no `val.txt`). The checkpoint is then chosen on training loss, which tends to pick a late, overfitted epoch.

`run_folds.py` behaviour:
- Folds can be split across GPUs or nodes with `--folds`. Every process writes to the same `--save` directory.
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

**Results on the current export, 10/90 split** (`data/cactus160_s10` and `data/cactus220_s10`; 5 test groups, 24 arms). These used checkpoint selection on training loss with `--patience 0`, 120 epochs. The leave-one-group-out runs above replace them.

| | Frame 160 | Frame 220 | Template |
|---|---|---|---|
| Biceps volume error | 4.5% (r 0.98) | 1.6% (r 1.00) | 13.3% |
| Biceps CD | 1.89 mm | 1.85 mm | 2.40 mm |
| Triceps volume error (lat. / med. / long) | 24.0 / 21.6 / 14.9% | 24.1 / 21.6 / 14.5% | 24.9 / 21.6 / 12.9% |

The biceps volume is learned from the skin. The triceps volumes are not: the model predicts the template volume whatever the true size (r −0.2 to 0.2).

**Results on the first export** (`OLD_datasets_cactus_poses`, frame 45; leave one subject out; 11 subjects, 22 arms). No results on the current export yet:

| | biceps | lat. triceps | long triceps | med. triceps |
|---|---|---|---|---|
| template CD (mm) | 3.4 | 2.8 | 2.8 | 2.4 |
| absolute model CD (mm) | 6.8 | 5.3 | 6.0 | 5.1 |
| residual model CD (mm) | 3.7 | 3.0 | 3.0 | 2.8 |
| template volume error (%) | 23 | 28 | 25 | 27 |
| residual model volume error (%) | 24 | 28 | 25 | 27 |

The residual row is from `results/cactus_loo_res`, re-evaluated with `--bn_recalibrate`. Its CD is pulled up by fold_04, where the right arm is 27 mm off. Without that fold, the model and the template agree: CD 2.79 vs 2.84 mm, volume error 26.1 vs 26.2%. The model beats the template on 47 of 88 muscle samples, about a coin flip. With about 10 training subjects per fold, it has not learned a usable skin-to-muscle mapping.

- **The absolute model:** it underfits, reaching about 6–9 mm RMS even on its training arms. The residual model starts at the template's accuracy by construction; a full leave-one-out run with `--residual` is the next comparison.
- **The CD noise floor:** two samplings of the *same* mesh with 4,096 points already differ by 1.3–1.7 mm CD. Much of the template's 2–3 mm is sampling noise, and volume error is the more sensitive measure here.
- **Volumes:** each parameter switches its muscle between two volumes, 50–70% apart. For example, the biceps is 318 or 531 ml.
- **Left and right arms:** they agree to within about 1 ml, so there are effectively 11 independent samples, not 22.
- **More simulations:** the current export (49 muscle configurations, 108 skins) is the first step. Continuous parameter values would help more than further ±0.6 / 0 grid points.

---

## Viewing subjects and predictions in 3D (HTML)

Both viewers write one self-contained `.html` file (3D plotly, about 3–12 MB). Open it in a browser, or right-click it in VS Code and choose *Open with Live Server / Preview*. Each layer can be switched on and off by clicking its legend entry. Layers that are hidden by default are greyed out in the legend.

**Arms (cactus):** [tools/visualize_cactus.py](tools/visualize_cactus.py)

```bash
# a subject, both arms: arm skin, the four target muscles, humerus (frame 160; --frame 220)
python tools/visualize_cactus.py --subject BB_0_TRL_0.6_TRM_-0.6_TRLN_0_T_1 --out vis/s1
# one arm sample + its prepared points (input skin points, target muscle points)
python tools/visualize_cactus.py --id BB_0_TRL_0.6_TRM_-0.6_TRLN_0_T_1_R --prepared data/cactus160 --out vis/s1_R
# + the model prediction and its template (for leave-N-out, the fold holding the sample is found)
python tools/visualize_cactus.py --id BB_0_TRL_0.6_TRM_-0.6_TRLN_0_T_1_L \
    --prepared data/cactus160_loo --results results/cactus160_loo_res --out vis/s1_L_pred
# the first export
python tools/visualize_cactus.py --data /data/aalempij/data/OLD_datasets_cactus_poses --frame 45 \
    --subject BB_0.6_TRL_-0.6_TRM_0.6_TRLN_-0.6_T_1 --out vis/old_s1
```

- **Skin:** `--skin full` shows the whole body, `--skin none` hides it, and `--skin_mm` sets the arm crop (default 150 mm).
- **Other layers:** `--all_muscles` adds all muscles (the `entire_muscles` mesh) in grey, and `--no_bone` hides the humerus.
- **Left arms:** `_L` samples are mirrored back onto the left arm.

**MRI (TotalSegmentator):** [tools/visualize_seg.py](tools/visualize_seg.py) also writes a PNG of three slices.

```bash
python tools/visualize_seg.py --id s0175 --organs liver spleen kidney_left kidney_right pancreas \
    --prepared data/tsmri --results results/tsmri_b8 --out vis/s0175_pred
```

**Predictions:**
- Predictions only exist for *test* samples, in `<results>/predictions.npz`, which `main.py` writes during evaluation.
- Runs made before that file existed (e.g. `results/tsmri_b8`) can be re-evaluated from their saved weights without retraining. Add `--eval_only` to the original `main.py` command.

### Paper figures

[tools/paper_figures.py](tools/paper_figures.py) writes publication figures, as vector PDF plus 300-dpi PNG, and a metrics table. Each `--run` is `"label=prepared_dir:results_dir:frame"`. The script accepts single-split results (`main.py --save_path`) and leave-N-out results (`run_folds.py --save`). For leave-N-out it combines all finished folds.

| Output | Content |
|---|---|
| `figS1_full_body` | The whole body per frame (skin, skeleton, target muscles), with the measured skin motion between frames. |
| `fig1_data_overview` | One arm, front and back view per frame: skin crop (the input), the four muscles, the humerus. |
| `fig2_training_curves` | Training and validation (or test) loss, and held-out CD of the model vs the template. Leave-N-out shows the median over folds with the interquartile range, and the median selected epoch as a dotted line. |
| `fig3_volume_scatter` | Predicted vs true volume of every held-out arm, per muscle and frame, for the model and the template, with r and mean error. |
| `fig4_error_surface` | One held-out arm: the template surface and the predicted surface, coloured by distance to the true muscle surface (turbo colour map: blue small, red large), front and back. |
| `table_metrics.csv` / `.tex` | Per frame and muscle: folds, test arms, CD, HD95, volume error (model and template) and volume correlation. |

| Option | Meaning |
|---|---|
| `--sample` | The arm shown in figures 1, 4 and S1 (any held-out arm). |
| `--only` | A subset: `fullbody overview curves volumes error table`. |
| `--error_cmap` | Colour map for figure 4: `turbo` (default) or `jet`, `viridis`, … |
| `--error_max_mm` | Top of the figure 4 colour scale, in mm (default 6). |

Design notes:
- Colours come from a validated colourblind-safe palette, and model vs template is always blue circles vs orange squares.
- Muscles are always named in the legend, never identified by colour alone.
- The 3D panels are rendered separately and cropped, so they are never clipped.
- The figure 4 template surface is smoothed for display; volumes use the original mesh.

---

## C. Triceps segmentation with TotalSegmentator

The TotalSegmentator MRI dataset (`/home/aalempij/Data/data/TotalsegmentatorMRI_dataset_v300/Totalsegmentator_dataset_v300`) has no triceps in its `segmentations/` folders. [tools/segment_triceps.py](tools/segment_triceps.py) runs TotalSegmentator's `thigh_shoulder_muscles_mr` task on each subject and records which subjects have a triceps_brachii large enough to use.

### C1. Run

It needs a GPU node and the `bodyseg` env. The TotalSegmentator licence must already be set. That is a one-off step with `totalseg_set_license -l <key>`; the key is stored in `~/.totalsegmentator/config.json`, and this script does not need it.

```bash
cd ~/data/git/DeformingPointTransformer
source ~/miniconda3/etc/profile.d/conda.sh && conda activate bodyseg
nvidia-smi --query-gpu=index,memory.free --format=csv     # pick a GPU with several GB free
nohup python tools/segment_triceps.py --device gpu:0 > triceps.log 2>&1 &
tail -f triceps.log                                       # Ctrl+C only stops tail
```

| Option | Meaning |
|---|---|
| `--limit 3` | Process at most 3 subjects, for a quick test. |
| `--subjects s0852 s0925` | Process only these subjects. |
| `--device gpu:1` / `cpu` | Device for TotalSegmentator. |
| `--min_ml 100` | Minimum triceps volume per arm, in ml. |
| `--need any` / `both` | Usable arms a subject needs: one (default) or both. |
| `--all` | Also run subjects with no humerus in the ground truth. |
| `--out_name segmentations` | Output folder inside each subject. |
| `--redo` | Rerun TotalSegmentator even where output exists. |
| `--report_dir data/triceps` | Where the report is written. |

**Time:** about 40 s per subject on an L4, so roughly 5–6 hours for the 490 subjects that have a humerus.

### C2. What the script does

1. **Skips subjects without an upper arm.** If neither `humerus_left` nor `humerus_right` is in the existing ground truth, the arm isn't in the scan. That applies to 625 of the 1,115 subjects; 490 have a humerus (429 on both sides, 61 on one).
2. **Runs** `TotalSegmentator -i mri.nii.gz -o segmentations --task thigh_shoulder_muscles_mr -d <device>`. The task's 18 class names don't clash with the dataset's own masks, so nothing existing gets overwritten. If `triceps_brachii.nii.gz` already exists, the run is skipped.
3. **Scores each arm.** The triceps mask holds both arms. It's split into connected components of at least 1 ml, and each is assigned to the left or right arm by the nearer humerus. An arm is **usable** if its triceps is at least `--min_ml` and doesn't touch the edge of the scan volume; a triceps touching the edge has been cut by the field of view.
4. **Saves progress.** The report is rewritten after every subject, so an interrupted run resumes from where it stopped when started again with the same command.

**Test (s0175, s0692, s0852):**

| Subject | Left triceps | Right triceps | Usable |
|---|---|---|---|
| s0175 | 51 ml (left arm cut by the FOV) | 115 ml | yes, right arm |
| s0692 | 288 ml | 242 ml | yes, both arms |
| s0852 | 104 ml | 127 ml | yes, both arms |

Triceps in these scans are about 100–300 ml. To check a subject visually:

```bash
python tools/visualize_seg.py --id s0175 --organs triceps_brachii humerus_left humerus_right deltoid --out vis/s0175_triceps
```

### C3. Outputs

In `data/triceps/`:

| File | Contents |
|---|---|
| `triceps_survey.csv` | One row per subject: status (segmented / skipped / failed), humerus and triceps volumes, per-arm volume, cut flag and usability, overall `usable`, and the reason when it isn't usable. |
| `usable_subjects.txt` | The subjects that pass, one per line. |
| `failures.log` | The last lines of TotalSegmentator's output for any subject that failed. |

TotalSegmentator's own outputs, 18 muscle masks including `triceps_brachii.nii.gz`, are written into each subject's `segmentations/` folder.

### C4. Rescore or rerun

```bash
# change the usability rule without re-segmenting
python tools/segment_triceps.py --measure_only --min_ml 150 --need both
# re-segment specific subjects
python tools/segment_triceps.py --subjects s0175 --redo
```
source /home/aalempij/miniconda3/etc/profile.d/conda.sh
conda activate fsleyes
cd /home/aalempij/Data/data/TotalsegmentatorMRI_dataset_v300/s0001
fsleyes mri.nii.gz segmentations/triceps_brachii.nii.gz -ot mask -mc 1 0 0 -t 0.5 1 -a 50   segmentations/heart.nii.gz  -ot mask -mc 0 1 0 -t 0.5 1 -a 50
