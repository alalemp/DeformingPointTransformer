# 手臂实验流程精简说明（中文版）

仓库：[DeformingPointTransformer — ranxi-dev](https://github.com/alalemp/DeformingPointTransformer/tree/ranxi-dev)

说明整理日期：2026-10-15。以下保留老师 GUIDE 的八步和原始命令，并单列 Cetus 适配。状态描述是本次整理时的记录，不是实时作业监控。**老师的直接计算命令不能直接在 Cetus 登录节点运行，应通过 PBS 申请计算资源。**

## 1. 仓库功能与当前任务

| 任务 | 输入 → 输出 | 文档 |
|---|---|---|
| Houdini 手臂预测 | 皮肤点云＋平均模板 → 四块肌肉形状 | GUIDE 第 0 节、B 节 |
| MRI 腹部器官预测 | 身体表面点云＋模板 → 肝、脾、左右肾、胰腺 | GUIDE A 节 |
| MRI 肱三头肌分割与筛选 | MRI → 整体肱三头肌分割、体积和完整性报告 | GUIDE C 节 |

其他功能包括预处理、交叉验证、重新评估、体积与长度统计、交互三维查看、论文图表、显存检查、PBS 与上传辅助。VIBESegmentator 是外部独立仓库；本仓库不包含完整 Houdini 仿真生成流程。

当前任务是第一类：用 Houdini PLY 训练 DeformingPointTransformer。输入是皮肤点云和训练集平均肌肉模板，监督目标是当前样本的真实肌肉点云；不是 MRI 分割，不需要 TotalSegmentator 权重或许可证。

四个目标：`biceps_brachii`、`triceps_lateral`、`triceps_medial`、`triceps_long`。其中第一个标签映射到导出文件夹 `bicepsBrachialis`，其解剖内容需按 Houdini 资产确认。

两帧分别训练；每帧：81 个肌肉配置 × 3 个 T 值 × 左右两臂 = **486 个样本**。每折 **432 训练、48 验证、6 测试**。两帧是相同配置的不同姿态，不是独立人体。当前完成选定配置折，尚未训练全部 81 折。

## 2. GUIDE 第 0 节：复现手臂实验

### Step 1. Install once, then set up the shell.

```bash
cd ~/data/git/DeformingPointTransformer
source cluster_env.sh
```

按照 `CLUSTER.md` 安装一次；之后每个新 shell 执行上面的环境设置。

进入仓库，激活 `bodyseg`，设置 GCC、CUDA 和编译变量。它不会重新安装环境或启动训练。

**Cetus 对应命令：**

```bash
cd /shared/homes/u24615736/git/DeformingPointTransformer
source cluster_env_uts.sh
```

PBS 作业内部也会激活同一个共享 Conda 环境。

### Step 2. Prepare the folds for both frames.

```bash
python tools/prepare_cactus.py --out data/cactus160_loo_val --leave_out 1 --val_frac 0.1
python tools/prepare_cactus.py --out data/cactus220_loo_val --leave_out 1 --val_frac 0.1 --frame 220
```

**作用：**读取 PLY、裁剪皮肤、采样和归一化，并按肌肉配置组划分数据；不训练，也不需要 GPU。

- 默认帧为 160；第二条命令处理 220。
- `--leave_out 1`：每折留一个配置组测试。
- `--val_frac 0.1`：从非测试组抽约 10% 验证。
- 同组全部 T 变体和左右臂保持在同一划分中。
- 每折模板只使用该折训练样本构建。

检查 `qc.png`、`subjects.csv`，以及各折的 `train.txt`、`val.txt`、`test.txt`、`mean_shape.npz`。训练名单用于参数更新，验证名单用于选模，测试名单用于最终评估。

老师数据有 49 折；当前数据有 **81 折**。老师的约一分钟耗时不是保证，取决于数据和机器。Cetus 上通过 PBS CPU 作业执行，并显式指定实际 `--data` 路径。

### Step 3 (optional). Check one fold learns.

```bash
export CUDA_VISIBLE_DEVICES=0
python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 --save results/check160 \
    --folds fold_00 -- --batch_size 4 --epochs 120 --residual --patience 15
grep "^Epoch" results/check160/fold_00/train.log | tail
```

**作用：**先试跑 `fold_00`，确认模型能学习，再考虑全部折。

| 参数 | 含义 |
|---|---|
| `--prepared` | 预处理数据目录 |
| `--label_set arm4` | 四块手臂肌肉标签 |
| `--save` | 结果目录 |
| `--folds fold_00` | 仅运行此折 |
| 单独的 `--` | 后续参数传给 `main.py` |
| `--batch_size 4` | 每批 4 个手臂样本 |
| `--epochs 120` | 最多 120 轮 |
| `--residual` | 预测模板的位移修正 |
| `--patience 15` | 连续 15 轮验证损失不改善就停止 |

残差预测：`预测坐标 = 模板坐标 + residual_scale × 模型修正量`。它不保证修改一定有益，也不强制保持体积。

老师建议：前 10–20 轮内验证 CD 应低于模板。已完成的 `fold_00` 最佳验证 CD 为 **2.379 mm**，模板为 **2.694 mm**，通过基本学习检查；但不代表测试集一定更好。

Cetus 上通过 PBS GPU 作业执行，使用 PBS 分配的 GPU，不直接照搬登录节点上的 GPU 选择。

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

**作用：**每帧依次训练全部折，使每个配置组轮流作为测试组。老师命令在两块空闲 GPU 上各运行一帧，两帧训练独立模型。

- 老师估计 A5500 每轮约 25 秒、L4 每轮约一分钟；A5500 每折约 40–60 分钟，49 折每帧约 1–1.5 天。这些不是当前环境的保证。
- 当前数据两帧各 81 折，共 **162 次独立训练**。
- 多 GPU 可分配互不重叠的 `--folds` 名单，例如老师的 `fold_00`–`fold_24` 和 `fold_25`–`fold_48`；当前数据需覆盖至 `fold_80`。不要同时写同一折，最终统一汇总。
- 重跑跳过已完成折；未完成折重新训练，不是从中断 epoch 自动续训。
- Step 2 改用 `--leave_out 5` 后，老师数据得到 10 折，当前数据得到 **17 折**；计算减少，但训练划分也改变。

`nohup` 用于终端断开后继续运行，`> ... 2>&1` 保存正常和错误输出，`&` 表示后台运行。

**Cetus：**分别提交两个 PBS GPU 作业，每个申请一块 GPU，无需 `nohup` 或手动挑 GPU。下面是全部折提交方式，只有决定开展完整实验后才运行；不要重复提交同名结果任务。

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

48 小时可能不足以完成 81 折。超时后先确认结果回传，再重提；硬性终止可能来不及回传。当前尚未开展全部折训练。

### Step 5. Monitor.

```bash
tail -2 results/cactus160_loo_val.log results/cactus220_loo_val.log
ls -d results/cactus160_loo_val/fold_*/csv/trained_vs_target_arm4.csv | wc -l
grep -h "Best checkpoint" results/cactus160_loo_val/fold_*/train.log | tail
```

**作用：**分别查看当前折、已完成折数和最佳模型轮次，不会修改训练。若 `tail -2` 报错，改为 `tail -n 2`。

**Cetus 上先查作业：**

```bash
qstat -u "$USER"
qstat -fx JOB_ID
```

`Q` 排队，`R` 运行，`H` 等待依赖；`F` 且 `Exit_status=0` 表示成功结束。

实时训练文件位于计算节点 `/scratch/$USER/dpt_JOB_ID/repo/results/实际实验目录`，作业退出时才回传仓库。登录节点结果数量不一定反映实时进度。当前配置比较目录也不同于老师的全部折目录。

连接分配节点后，设置 `RUN` 为实际结果目录，可执行：

```bash
find "$RUN" -type f -name train.log -exec tail -n 2 -v {} +
find "$RUN" -type f -path '*/csv/trained_vs_target_arm4.csv' | wc -l
grep -H "Best checkpoint" "$RUN"/fold_*/train.log
```

`-H` 保留文件名，便于辨认折。`tail -f` 可持续监视日志，Ctrl+C 只退出监视，不停止 PBS 作业。

### Step 6. Summarise each frame.

```bash
python tools/run_folds.py --prepared data/cactus160_loo_val --label_set arm4 --save results/cactus160_loo_val --summary_only
python tools/run_folds.py --prepared data/cactus220_loo_val --label_set arm4 --save results/cactus220_loo_val --summary_only
```

**作用：**使用 `--summary_only` 汇总已有结果，不训练；支持未完成全部折时先汇总部分结果。

- `summary_arm4.csv`：逐肌肉比较模型/模板的 CD、HD95、体积绝对误差，报告按测试样本数加权的均值、折均值间标准差及纳入折数。
- `fold_XX/csv/volumes.csv`：逐测试臂的真实体积、估计体积及误差。
- `err_ml = pred_ml − true_ml`：正值偏大，负值偏小；`true_ml` 在当前实验中来自 Houdini 闭合网格。

距离与误差越小越好。部分折汇总不等于完整交叉验证；单折标准差为 0 也不代表稳定。

当前结果目录采用完整配置名，例如 `results/compare160_BB_-0.6_TRL_-0.6_TRM_-0.6_TRLN_-0.6`，运行时应替换 `--save`。两帧统一比较表是 `results/parameter_comparison160_220/comparison.csv`。

### Step 7. Paper figures and table.

```bash
python tools/paper_figures.py --out figures/paper \
  --run "Frame 160=data/cactus160_loo_val:results/cactus160_loo_val:160" \
  --run "Frame 220=data/cactus220_loo_val:results/cactus220_loo_val:220" \
  --sample BB_0_TRL_-0.6_TRM_0_TRLN_0.6_T_1_R
```

**作用：**从已有结果生成论文图表，不训练。`--run` 格式为 `显示名=预处理目录:结果目录:帧号`；`--sample` 指定具有已保存预测的测试手臂，脚本自动寻找对应折。

| 输出 | 内容 |
|---|---|
| S1 | 全身与两帧姿态展示 |
| Figure 1 | 手臂皮肤、肌肉、肱骨和数据概览 |
| Figure 2 | 训练与验证曲线，多折时汇总分布 |
| Figure 3 | 模型、模板的估计体积与真实体积散点图 |
| Figure 4 | 模板和预测的表面误差着色图 |
| 指标表 | CD、HD95、体积误差、纳入折数与测试臂数 |

图片为 PNG/PDF，表格为 CSV/LaTeX。示例图使用指定手臂；统计图和表覆盖结果目录中已完成折的测试臂。

当前三个配置的两帧整套图已生成到 `figures/paper_comparison160_220/`，每帧每配置仍只有一个折，不代表全部 81 折。

### Step 8 (optional). Interactive 3D view of any held-out arm.

```bash
python tools/visualize_cactus.py --id BB_0_TRL_-0.6_TRM_0_TRLN_0.6_T_1_R --prepared data/cactus160_loo_val \
    --results results/cactus160_loo_val --out vis/BB0_R_pred
```

**作用：**生成 `vis/BB0_R_pred.html`，可旋转、缩放，并切换皮肤、骨骼、真实肌肉、模板和预测。用于具体样本检查，不训练。

Cetus 上需指定实际原始数据和已有结果目录。例如全零配置：

```bash
python tools/visualize_cactus.py \
  --data "$HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220" \
  --frame 160 --id BB_0_TRL_0_TRM_0_TRLN_0_T_1_R \
  --prepared data/cactus160_loo_val \
  --results results/compare160_BB_0_TRL_0_TRM_0_TRLN_0 \
  --out vis/frame160/BB_0_TRL_0_TRM_0_TRLN_0_T_1_R
```

建议通过 PBS CPU 作业生成，下载到本地浏览器查看。视图需要浏览器能加载在线 Plotly 脚本。

## 3. 可复现性

- `prepare_cactus.py --seed` 固定采样和划分；`main.py --seed` 固定模型初始化、批次顺序等随机过程。
- GPU 运算不保证逐位相同；小数值差异不能解释明显性能差距。
- 用 `git rev-parse HEAD` 保存版本，同时记录未提交修改、依赖和参数。
- 当前采用 `seed=0`，保存 `source_commit.txt`、`pip_freeze.txt`、名单、模板和日志。

## 4. 文件位置与后续安排

| 位置 | 内容 |
|---|---|
| `data/` | 预处理点云、每折名单及模板 |
| `results/compare160_BB_…/`、`results/compare220_BB_…/` | 选定配置的模型、预测、日志和指标 |
| `results/parameter_comparison160_220/comparison.csv` | 两帧三个配置的统一比较 |
| `figures/paper_comparison160_220/` | 整套论文图和表 |
| `vis/` | 交互三维 HTML |
| `results/pbs_logs/` | 分类归档的 PBS 日志 |

训练程序也会在每折的 `figures/` 子目录生成基础图；顶层 `figures/` 放整理后的论文图。这沿用仓库习惯，不是 Cetus 要求。

当前模型能学习，但测试收益不稳定、体积误差明显。先完成训练/验证/测试统一重评估和 Step 8，再研究“仅 CD”与“CD＋逐肌肉体积损失”，最后决定全部折训练。体积损失尚未加入；真实 MRI 到整体肱三头肌的表面预测训练流程也仍需适配。

文档：`README.md` 介绍模型；`GUIDE.md` 介绍实验；`CLUSTER.md` 面向 iHPC 安装；`UTS_ARMS.md` 面向 Cetus/PBS；`INFO.md` 部分内容较旧。预处理逐行解释见 [prepare_cactus_line_by_line_zh.md](prepare_cactus_line_by_line_zh.md)。
