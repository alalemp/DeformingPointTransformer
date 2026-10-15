# Arm experiment workflow — concise English version

Repository: [DeformingPointTransformer — ranxi-dev](https://github.com/alalemp/DeformingPointTransformer/tree/ranxi-dev)

Prepared on 2026-10-15. The eight steps and original commands are retained, with Cetus adaptations noted separately. Status statements describe this snapshot, not live monitoring. **Computational commands must run through PBS on Cetus, not directly on a login node.**

## 1. Repository capabilities and current task

| Task | Input → output | Documentation |
|---|---|---|
| Houdini arm prediction | Skin point cloud + template → four muscle shapes | GUIDE Section 0 and B |
| MRI-derived abdominal-organ prediction | Body-surface points + template → liver, spleen, kidneys and pancreas | GUIDE Section A |
| MRI triceps segmentation and screening | MRI → whole-triceps segmentation, volume and completeness reports | GUIDE Section C |

Other tools cover preprocessing, cross-validation, re-evaluation, volume/length statistics, interactive 3D views, publication figures, GPU-memory checks, PBS and uploads. VIBESegmentator is a separate external repository; the complete Houdini simulation-generation workflow is not included here.

The current task is the first: train DeformingPointTransformer on Houdini PLY data. Skin points and a training-derived mean muscle template are inputs; the sample’s ground-truth muscle points provide supervision. This is not MRI segmentation and does not require TotalSegmentator weights or a licence.

Targets: `biceps_brachii`, `triceps_lateral`, `triceps_medial`, `triceps_long`. The first label maps to the exported `bicepsBrachialis` folder; its anatomical contents should be verified against the Houdini asset.

Frames are trained separately. Each frame contains 81 configurations × 3 T values × 2 arms = **486 samples**: **432 train, 48 validation, 6 test** per fold. The frames are paired poses of the same configurations, not independent people. Selected folds are complete; all 81 folds have not been trained.

## 2. GUIDE Section 0: Reproducing the arm results

### Step 1. Install once, then set up the shell.

```bash
cd ~/data/git/DeformingPointTransformer
source cluster_env.sh
```

Install once following `CLUSTER.md`, then initialise each new shell with the commands above.

Enter the repository, activate `bodyseg`, and configure GCC, CUDA and build variables. This does not reinstall the environment or start training.

**Cetus equivalent:**

```bash
cd /shared/homes/u24615736/git/DeformingPointTransformer
source cluster_env_uts.sh
```

Each PBS job explicitly activates the same environment stored in shared home.

### Step 2. Prepare the folds for both frames.

```bash
python tools/prepare_cactus.py --out data/cactus160_loo_val --leave_out 1 --val_frac 0.1
python tools/prepare_cactus.py --out data/cactus220_loo_val --leave_out 1 --val_frac 0.1 --frame 220
```

**Purpose:** Read PLY meshes, crop the skin, sample and normalise points, and split by muscle configuration. No training or GPU is required.

- The default frame is 160; the second command selects 220.
- `--leave_out 1`: hold out one configuration group per fold.
- `--val_frac 0.1`: reserve approximately 10% of non-test groups for validation.
- All T variants and both arms of a group remain in the same split.
- Each template uses only that fold’s training samples.

Check `qc.png`, `subjects.csv`, and each fold’s `train.txt`, `val.txt`, `test.txt` and `mean_shape.npz`. These lists specify parameter-update, model-selection and final-evaluation samples respectively.

The supervisor’s dataset has 49 folds; the current dataset has **81**. The guide’s approximately one-minute estimate depends on data and hardware. On Cetus, use a PBS CPU job and supply the actual `--data` path.

### Step 3 (optional). Check one fold learns.

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 --save results/check160 \
    --folds fold_00 -- --batch_size 4 --epochs 120 --residual --patience 15
grep "^Epoch" results/check160/fold_00/train.log | tail
```

**Purpose:** Test `fold_00` before committing to all folds.

| Argument | Meaning |
|---|---|
| `--prepared` | Prepared dataset directory |
| `--label_set arm4` | Four arm-muscle labels |
| `--save` | Output directory |
| `--folds fold_00` | Run only this fold |
| Standalone `--` | Pass subsequent arguments to `main.py` |
| `--batch_size 4` | Four arm samples per batch |
| `--epochs 120` | At most 120 epochs |
| `--residual` | Predict corrections to the template |
| `--patience 15` | Stop after 15 epochs without improved validation loss |

Residual prediction is `prediction = template + residual_scale × model correction`. It does not guarantee improvement or preserve volume.

The guide expects validation CD to beat the template within 10–20 epochs. The completed `fold_00` reached **2.379 mm**, versus **2.694 mm** for the template. This passes the basic learning check, but does not guarantee improved test performance.

On Cetus, use a PBS GPU job and its allocated GPU rather than selecting a GPU directly on the login node.

### Step 4. Train all folds of both frames.

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv
CUDA_VISIBLE_DEVICES=0 nohup python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 \
    --save results/cactus160_loo_val -- --batch_size 4 --epochs 120 --residual --patience 15 \
    > results/cactus160_loo_val.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python tools/run_folds.py --prepared data/cactus220_loo_val --label_set arm4 \
    --save results/cactus220_loo_val -- --batch_size 4 --epochs 120 --residual --patience 15 \
    > results/cactus220_loo_val.log 2>&1 &
```

**Purpose:** Train all folds sequentially within each frame, so every configuration is tested once. The guide runs one frame on each of two free GPUs, with independently trained models.

- Guide estimates: approximately 25 seconds per epoch on A5500 and one minute on L4; 40–60 minutes per fold on A5500, or 1–1.5 days for 49 folds per frame. These are not guarantees for the current hardware.
- The current dataset requires **162 independent training runs**: 81 folds per frame.
- Additional GPUs can process disjoint `--folds` lists, such as the guide’s `fold_00`–`fold_24` and `fold_25`–`fold_48`. Our lists must extend through `fold_80`. Avoid concurrent writes to the same fold and summarise afterwards.
- Completed folds are skipped on rerun; unfinished folds restart rather than resume the interrupted epoch.
- Using `--leave_out 5` in Step 2 gives 10 folds for the supervisor’s data and **17 for ours**, reducing computation while changing the split design.

`nohup` keeps a process running after disconnection, `> ... 2>&1` redirects output and errors, and `&` runs it in the background.

**Cetus adaptation:** Submit one PBS GPU job per frame; PBS allocates GPUs, so `nohup` and manual GPU selection are unnecessary. Run the following only when full cross-validation is intended, and avoid duplicate submissions to the same output directory.

```bash
qsub -N dpt_all160 -q small_gpuq \
  -l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=48:00:00 \
  -v MODE=train,FRAME=160,FOLD=all,RUN_NAME=cactus160_loo_val \
  tools/uts_arm_job.sh

qsub -N dpt_all220 -q small_gpuq \
  -l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=48:00:00 \
  -v MODE=train,FRAME=220,FOLD=all,RUN_NAME=cactus220_loo_val \
  tools/uts_arm_job.sh
```

48 hours may not cover all 81 folds. After a timeout, verify that outputs were copied back before resubmitting; a hard termination can prevent copy-back. Full cross-validation has not yet been started.

### Step 5. Monitor.

```bash
tail -2 results/cactus160_loo_val.log results/cactus220_loo_val.log
ls -d results/cactus160_loo_val/fold_*/csv/trained_vs_target_arm4.csv | wc -l
grep -h "Best checkpoint" results/cactus160_loo_val/fold_*/train.log | tail
```

**Purpose:** Inspect the current fold, count completed evaluations, and view selected checkpoint epochs. These commands do not change training. Use `tail -n 2` if `tail -2` is rejected.

**On Cetus, first inspect PBS:**

```bash
qstat -u "$USER"
qstat -fx JOB_ID
```

`Q` means queued, `R` running, and `H` held for dependencies. `F` with `Exit_status=0` means successful completion.

Live results are under `/scratch/$USER/dpt_JOB_ID/repo/results/ACTUAL_RUN` on the assigned compute node and are copied back when the job exits. Shared-home results may therefore lag behind actual progress. Our configuration runs use different directories from the guide’s full runs.

On the assigned node, set `RUN` to the actual result directory and use:

```bash
find "$RUN" -type f -name train.log -exec tail -n 2 -v {} +
find "$RUN" -type f -path '*/csv/trained_vs_target_arm4.csv' | wc -l
grep -H "Best checkpoint" "$RUN"/fold_*/train.log
```

`-H` retains filenames to identify folds. `tail -f` follows a live log; Ctrl+C stops monitoring without stopping the PBS job.

### Step 6. Summarise each frame.

```bash
python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 --save results/cactus160_loo_val --summary_only
python tools/run_folds.py --prepared data/cactus220_loo_val --label_set arm4 --save results/cactus220_loo_val --summary_only
```

**Purpose:** Use `--summary_only` to aggregate existing results without training. Partial results can be summarised before all folds finish.

- `summary_arm4.csv`: per-muscle model/template CD, HD95 and absolute volume errors, sample-weighted means, standard deviations across fold means, and fold counts.
- `fold_XX/csv/volumes.csv`: ground-truth and estimated volumes for each test arm.
- `err_ml = pred_ml − true_ml`: positive means overestimation; negative means underestimation. Here, `true_ml` comes from the original closed Houdini mesh.

Lower distances and errors are better. Partial summaries are not complete cross-validation; a zero standard deviation from one fold does not demonstrate stability.

Current result directories use explicit configurations, for example `results/compare160_BB_-0.6_TRL_-0.6_TRM_-0.6_TRLN_-0.6`; replace `--save` accordingly. The combined comparison is `results/parameter_comparison160_220/comparison.csv`.

### Step 7. Paper figures and table.

```bash
python tools/paper_figures.py --out figures/paper \
  --run "Frame 160=data/cactus160_loo_val:results/cactus160_loo_val:160" \
  --run "Frame 220=data/cactus220_loo_val:results/cactus220_loo_val:220" \
  --sample BB_0_TRL_-0.6_TRM_0_TRLN_0.6_T_1_R
```

**Purpose:** Generate publication outputs from existing predictions without training. The `--run` format is `label=prepared_directory:result_directory:frame`. `--sample` selects a held-out arm with saved predictions; its fold is located automatically.

| Output | Contents |
|---|---|
| S1 | Whole-body anatomy and pose comparison |
| Figure 1 | Arm skin, muscles, humerus and data overview |
| Figure 2 | Training and validation curves; distributions when multiple folds exist |
| Figure 3 | Estimated versus ground-truth volumes, model and template |
| Figure 4 | Template and prediction surface-error maps |
| Metrics table | CD, HD95, volume errors, fold and test-arm counts |

Figures are PNG/PDF; tables are CSV/LaTeX. Example figures use the selected arm; statistical plots and tables use test arms from completed folds.

Complete two-frame figure sets for the three configurations are under `figures/paper_comparison160_220/`. Each configuration currently includes one fold per frame, not all 81 folds.

### Step 8 (optional). Interactive 3D view of any held-out arm.

```bash
python tools/visualize_cactus.py --id BB_0_TRL_-0.6_TRM_0_TRLN_0.6_T_1_R --prepared data/cactus160_loo_val \
    --results results/cactus160_loo_val --out vis/BB0_R_pred
```

**Purpose:** Generate `vis/BB0_R_pred.html` to rotate, zoom and toggle skin, skeleton, ground-truth muscles, templates and predictions. This inspects individual samples; it does not train a model.

On Cetus, supply the actual raw-data and result paths. For the all-zero configuration:

```bash
python tools/visualize_cactus.py \
  --data "$HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220" \
  --frame 160 --id BB_0_TRL_0_TRM_0_TRLN_0_T_1_R \
  --prepared data/cactus160_loo_val \
  --results results/compare160_BB_0_TRL_0_TRM_0_TRLN_0 \
  --out vis/frame160/BB_0_TRL_0_TRM_0_TRLN_0_T_1_R
```

Generate it through a PBS CPU job, then download it for a local browser. The browser needs access to the online Plotly script.

## 3. Reproducibility

- `prepare_cactus.py --seed` fixes sampling and splits; `main.py --seed` controls initialisation, batch order and other randomness.
- GPU operations may not be bit-exact; small numerical differences do not explain large performance gaps.
- Record `git rev-parse HEAD`, uncommitted changes, dependencies and parameters.
- Current runs use `seed=0` and retain `source_commit.txt`, `pip_freeze.txt`, lists, templates and logs.

## 4. Output locations and next steps

| Location | Contents |
|---|---|
| `data/` | Prepared points, split lists and templates |
| `results/compare160_BB_…/`, `results/compare220_BB_…/` | Models, predictions, logs and metrics for selected configurations |
| `results/parameter_comparison160_220/comparison.csv` | Combined comparison of both frames and three configurations |
| `figures/paper_comparison160_220/` | Publication figures and tables |
| `vis/` | Interactive 3D HTML |
| `results/pbs_logs/` | Categorised PBS logs |

Training also saves basic plots in each fold’s `figures/` subdirectory. The top-level `figures/` contains publication outputs. This is a repository convention, not a Cetus requirement.

The model learns, but test improvements are inconsistent and volume errors remain substantial. Complete consistent train/validation/test re-evaluation and Step 8 first, then compare CD-only training with CD plus per-muscle volume loss, before deciding on full cross-validation. Volume loss has not yet been added. A complete real-MRI surface-to-whole-triceps training pipeline also still requires adaptation.

Documentation: `README.md` describes the model; `GUIDE.md` describes experiments; `CLUSTER.md` covers iHPC installation; `UTS_ARMS.md` covers Cetus/PBS. Some information in `INFO.md` is outdated. A Chinese line-by-line preprocessing explanation is in [prepare_cactus_line_by_line_zh.md](prepare_cactus_line_by_line_zh.md).
