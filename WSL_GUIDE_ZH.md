本文记录当前仓库代码的流程，以及本机已经执行的操作。状态核对日期：2026-10-13。

用户补充：导师说明使用五个 Houdini mesh，即骨骼加四块肌肉。下面的“皮肤预测四块肌肉”是当前脚本和已完成实验的实际行为，尚未确认是否与导师要求的五结构流程一致。当前预处理没有把骨骼保存为网络输入或第五个预测目标。文中的后续训练命令是当前流程的参考；下一步应先确认导师指定的预测输入、输出和运行命令，再决定是否沿用。

这个项目研究的问题是：给定身体外表面的三维形状，能否预测内部器官或肌肉的形状？当前手臂实验的输入是皮肤表面，预测目标是二头肌及三头肌的外侧头、内侧头、长头。

```text
Houdini 原始皮肤和肌肉网格
          ↓ prepare_cactus.py
皮肤点云 + 真实肌肉点云 + 数据划分 + 平均肌肉模板
          ↓ run_folds.py 调用 main.py
独立训练每个 fold 的 DPT 模型，按验证损失保存最佳权重
          ↓ main.py 自动评估测试集
预测点云、误差 CSV、训练和预测图片
          ↓ 汇总与可视化
比较模型与平均模板，检查模型是否从皮肤学到了有效信息
```

**仓库中的几条流程分别做什么。**

| 流程 | 原始输入 | 主要操作 | 与你当前工作的关系 |
|---|---|---|---|
| GUIDE 的 A：MRI 五器官实验 | MRI 图像及器官分割 | 准备身体和器官点云，再训练 DPT | 当前没有执行 |
| GUIDE 的 B / 第 0 节：Houdini 手臂实验 | 导出的皮肤和肌肉 `.ply` | 准备手臂点云，再训练 DPT | 当前正在做 |
| GUIDE 的 C：真实 MRI 三头肌分割 | MRI 图像 | 用 TotalSegmentator 的预训练模型得到肌肉分割并筛选样本 | 当前没有执行；这一步本身不是 DPT 训练 |
| VIBESegmentator | MRI 图像 | 使用另一套预训练分割模型 | 安装说明中的附加工具，当前手臂实验没有调用 |

`TOTALSEG_LICENSE` 是 TotalSegmentator 的许可证。`install_cluster.sh` 下载的分割权重属于 TotalSegmentator；当前 DPT 的权重由你运行训练命令后产生。Houdini 已经提供真实肌肉网格，因此可以直接把它们用作训练目标。

**先认识需要关注的文件。**

| 文件或目录 | 作用 |
|---|---|
| `README.md` | 介绍论文问题和模型结构 |
| `GUIDE.md` | 实验操作说明；当前主要看第 0 节与 B 节 |
| `CLUSTER.md` | 老师的集群环境安装说明 |
| `install_cluster.sh` | 执行集群安装，包含多个工具与分割权重下载 |
| `cluster_env.sh` | 为集群终端激活环境并设置编译器、CUDA |
| `tools/prepare_cactus.py` | 把 Houdini 数据转成 DPT 所需格式 |
| `tools/run_folds.py` | 调度各 fold 的训练，自动传入数据路径并汇总结果 |
| `main.py` | 真正执行模型训练和测试 |
| `model/pointtransformer/pointtransformerlayer.py` | 定义模型结构 |
| `lib/pointops/` | GPU 点云采样、邻域查询等运算；是算法扩展 |
| `Chamfer3D/` | GPU 点云距离计算；是算法扩展 |
| `tools/volume_eval.py` | 根据预测对模板网格的变形计算肌肉体积误差 |
| `tools/visualize_cactus.py` | 生成可旋转、可切换图层的 3D HTML |
| `tools/paper_figures.py` | 汇总已有结果，生成论文图和表 |

`.md` 是说明文档，阅读其中的命令；`.sh` 是 shell 脚本；`.py` 是 Python 程序。克隆 GitHub 仓库只取得代码，激活 Conda 只是选择 Python 环境，安装依赖、准备数据和训练分别是不同的操作。

**第 1 步：准备运行环境，当前已完成。**

本机是 Ubuntu/WSL2，显卡 RTX 3080（10 GB）。使用已有 Miniforge 的 `deforming_pt` 环境，PyTorch 是 `2.8.0+cu128`，本机 CUDA Toolkit 是 12.8。已补齐 `scikit-image` 并验证 pointops 的 GPU 运算、Chamfer3D 前向和反向计算。

老师的集群脚本使用 RHEL 的 `/opt/rh/gcc-toolset-12/enable` 和 CUDA 12.1；当前本机设置对应下面的命令。每次打开用于运行 Python 的新终端，执行：

```bash
cd ~/git/alen_DPT/DeformingPointTransformer
source ~/miniforge3/etc/profile.d/conda.sh
conda activate deforming_pt

export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export CUDA_VISIBLE_DEVICES=0
```

| 命令 | 含义 |
|---|---|
| `cd ...` | 进入项目根目录，让相对路径对应正确文件 |
| `conda activate deforming_pt` | 选择装好依赖的 Python 环境 |
| `PYTHONNOUSERSITE=1` | 隔离 `~/.local` 中另一套 Python 包，避免混入不同 PyTorch |
| `CUDA_HOME` 和 `PATH` | 指定本机 CUDA Toolkit 的位置 |
| `TORCH_CUDA_ARCH_LIST="8.6"` | 编译 CUDA 扩展时使用 RTX 3080 对应的架构 |
| `CUDA_VISIBLE_DEVICES=0` | 让程序使用本机第 0 张 GPU |

这些设置影响当前终端及其子进程。查看日志的终端只运行 `tail`，可以保持在 `(base)`。

**第 2 步：认识原始数据，当前路径已确认。**

Windows 路径：

```text
D:\Houdini_Models\Ranxi_Models\practice\datasets\datasets_20260926\datasets_cactus pose frames_ply
```

同一文件夹在 WSL 中的路径：

```text
/mnt/d/Houdini_Models/Ranxi_Models/practice/datasets/datasets_20260926/datasets_cactus pose frames_ply
```

| 子目录 | 内容 |
|---|---|
| `final skin/` | 不同样本的皮肤网格，是输入数据来源 |
| `separated muscles/` | 分开的四块目标肌肉网格，是真实目标来源 |
| `entire muscles/` | 整体肌肉网格，可用于查看 |
| `bones/` | 骨骼网格，可用于查看和可选的骨对齐 |

文件名中的 160、220 是 Houdini 导出的两个动画帧；它们对应两个姿态。模型名称中的 BB、TRL、TRM、TRLN 是肌肉模拟参数，T 是皮肤模拟参数。

相同 BB/TRL/TRM/TRLN、不同 T 的皮肤共享同一肌肉配置。训练、验证、测试按这个“肌肉配置组”划分，保证一种测试肌肉配置不会通过其他皮肤样本出现在训练组中。

你的数据有 243 个皮肤模型、81 个肌肉配置组。左右手臂分别成为样本，因此每个 frame 有 486 个手臂样本。GUIDE 记录的老师数据是 108 个皮肤模型、49 个组；当前应使用本机实际统计。

**第 3 步：把原始网格转成训练数据，两个 frame 均已完成。**

`prepare_cactus.py` 做以下操作：

1. 从完整皮肤中裁剪手臂附近区域。
2. 从裁剪后皮肤表面采样 16,384 个点。
3. 从四块肌肉表面各采样 4,096 个点。
4. 将左臂镜像到右臂方向，并统一坐标尺度。
5. 按肌肉配置组生成训练、验证、测试划分。
6. 为每个 fold 仅用其训练样本构建平均肌肉模板。
7. 保存数据和 `qc.png` 质量检查图。

点云就是一组 `(x, y, z)` 坐标。将表面网格采样为固定数量的点，方便模型按统一格式处理不同样本。

已使用的预处理命令如下，供理解和记录；现有数据已经生成：

```bash
CACTUS_DATA='/mnt/d/Houdini_Models/Ranxi_Models/practice/datasets/datasets_20260926/datasets_cactus pose frames_ply'

python tools/prepare_cactus.py --data "$CACTUS_DATA" \
  --out data/cactus160_loo_val --frame 160 \
  --leave_out 1 --val_frac 0.1 --seed 0 --workers 4

python tools/prepare_cactus.py --data "$CACTUS_DATA" \
  --out data/cactus220_loo_val --frame 220 \
  --leave_out 1 --val_frac 0.1 --seed 0 --workers 4
```

| 参数 | 含义 |
|---|---|
| `--data` | 原始数据文件夹；本机必须传入实际路径 |
| `--out` | 预处理输出文件夹 |
| `--frame` | 选择 160 或 220 |
| `--leave_out 1` | 每个 fold 留出一个肌肉配置组测试，共生成 81 folds |
| `--val_frac 0.1` | 从非测试组中抽约 10% 用于验证 |
| `--seed 0` | 固定随机划分，使两个 frame 的分组一致 |
| `--workers 4` | 使用四个 CPU 工作进程 |

更改数据或划分设置时，使用新的 `--out` 和训练 `--save` 目录，使数据、模板和模型权重保持对应。

生成的文件：

```text
data/cactus160_loo_val/
├── data.npz                所有样本的皮肤点云和真实肌肉点云
├── affines.npz             将归一化坐标转换回毫米的变换
├── subjects.csv            样本、参数、分组和真实体积记录
├── qc.png                  一个训练样本与模板的检查图
├── labels/
│   └── label_organs_arm4.json
└── folds/
    ├── fold_00/
    │   ├── train.txt       432 个训练手臂，72 个组
    │   ├── val.txt         48 个验证手臂，8 个组
    │   ├── test.txt        6 个测试手臂，1 个组
    │   └── mean_shape.npz  当前 fold 的平均肌肉模板与模板网格
    └── fold_01/ ... fold_80/
```

`cactus220_loo_val` 的结构相同。`.npz` 是 NumPy 数组文件，通常由 Python 读取。

`label_organs_arm4.json` 规定四个编号对应哪块肌肉：1 二头肌、2 三头肌外侧头、3 内侧头、4 长头。它是标签映射。

`qc.png` 上排显示一个训练样本：灰色是皮肤、彩色是实际肌肉；下排显示当前 fold 的平均肌肉模板。左右两列是正面和俯视投影。检查裁剪、坐标和肌肉位置是否合理。

**第 4 步：理解 DPT 怎样训练，当前已完成 frame 160 的 fold_00。**

模型有两路输入：皮肤点云和平均肌肉模板。它分别提取两者的形状信息，合并信息后预测每个模板点的三维修正量。

当前使用 `--residual`：

```text
预测肌肉点 = 模板点 + 0.02 × 模型输出的修正量
```

训练时，真实肌肉点云用于计算预测误差并更新参数。测试时，模型根据皮肤和模板产生预测，真实肌肉只用于检查结果。

训练集用于更新权重；验证集用于选择最佳轮次；测试集用于最终报告。每个 fold 独立初始化和训练一个模型，使用它自己的模板。因此完整 81-fold 实验会产生 81 套模型权重。

`run_folds.py` 负责找到指定 fold 的文件，再调用 `main.py` 训练。它会自动传入 `train.txt`、`val.txt`、`test.txt`、点云、标签和模板路径。

你已执行的命令是：

```bash
python tools/run_folds.py \
  --prepared data/cactus160_loo_val \
  --label_set arm4 \
  --save results/cactus160_loo_val \
  --folds fold_00 \
  -- --batch_size 4 --epochs 120 --residual --patience 15 --seed 0
```

| 参数 | 含义 |
|---|---|
| `--prepared` | 读取哪份预处理数据 |
| `--label_set arm4` | 使用四块手臂肌肉的标签定义 |
| `--save` | 训练结果根目录 |
| `--folds fold_00` | 只训练这个 fold |
| 单独的 `--` | 后面的参数交给 `main.py` |
| `--batch_size 4` | 每次更新使用四个手臂样本 |
| `--epochs 120` | 最多遍历训练数据 120 轮 |
| `--residual` | 学习对模板的修正 |
| `--patience 15` | 验证损失连续 15 轮没有改善就停止 |
| `--seed 0` | 固定初始化、训练顺序等随机过程 |

这次第 50 轮验证损失最低，第 65 轮提前停止。最佳模型是第 50 轮的权重。权重是网络学习到的参数；模板是肌肉几何形状，两者分别保存。

**第 5 步：看日志和最终结果。**

训练输出会写到日志中，因此训练终端可能停留在 `training ...`。另一个终端查看：

```bash
tail -f ~/git/alen_DPT/DeformingPointTransformer/results/cactus160_loo_val/fold_00/train.log
```

每完成一轮，日志打印损失和验证 CD。查看日志终端的 Ctrl+C 只退出查看；训练终端的 Ctrl+C 会中断训练。

```text
results/cactus160_loo_val/
├── summary_arm4.csv
└── fold_00/
    ├── train.log
    ├── models/
    │   ├── model_weights_best.pth
    │   └── model_weights_last.pth
    ├── predictions.npz
    ├── csv/
    │   ├── loss_history.csv
    │   ├── best_epoch.txt
    │   ├── trained_vs_target_arm4.csv
    │   ├── init_vs_target_arm4.csv
    │   ├── volumes.csv
    │   └── per_sample_surface_metrics.csv
    └── figures/
        ├── Running Loss.png
        ├── Output Results - 1.png
        ├── metrics_comparison.png
        └── metrics_comparison.pdf
```

训练脚本生成权重、日志、预测、损失曲线和原始预测图。`run_folds.py` 生成汇总和体积 CSV。`metrics_comparison.*` 与 `per_sample_surface_metrics.csv` 是此次分析额外生成的文件，其中逐样本 CPU 复核的未扰动模板标为 `template_unjittered`。

| 指标 | 含义 | 判断方式 |
|---|---|---|
| CD，mm | 表面点云的平均最近距离 | 越小越好 |
| HD95，mm | 更关注较大误差区域的表面距离 | 越小越好 |
| 体积误差，ml / % | 预测体积与真实体积的差异 | 越小越好 |

模型必须与 template 基线比较，才能判断输入皮肤是否带来了额外收益。当前只有 1/81 folds；fold_00 的验证集改善，但测试组总体没有超过模板，四块肌肉体积都被低估。`std_across_folds=0` 来自仅有一个 fold。

代码还在验证和测试时对模板加随机扰动；正式比较细小差异前，应统一评估设置。当前代码未因这次说明而更改。

**第 6 步：先通过 3D 对照理解模型输出。**

已为 fold_00 的一个真实测试样本生成：

```text
vis/cactus160_fold00_example.html
```

在浏览器中打开，可以旋转、缩放并点击图例切换图层。查看原始肌肉、输入皮肤点、目标点云、模板和预测，理解模型具体修改了哪里。HTML 使用在线 Plotly 脚本，浏览器需要能访问其 CDN。

对应的生成命令是：

```bash
CACTUS_DATA='/mnt/d/Houdini_Models/Ranxi_Models/practice/datasets/datasets_20260926/datasets_cactus pose frames_ply'
python tools/visualize_cactus.py --data "$CACTUS_DATA" --frame 160 \
  --id BB_0_TRL_0.6_TRM_0.6_TRLN_0.6_T_0.75_L \
  --prepared data/cactus160_loo_val \
  --results results/cactus160_loo_val \
  --out vis/cactus160_fold00_example
```

`--id` 使用测试集中的样本编号。已有模型的预测保存在 `predictions.npz`，该文件当前只含当前 fold 的测试样本。

**第 7 步：保持训练设置，检查更多不同肌肉配置。**

在完成第 1 步的终端设置后，先运行两个额外 folds：

```bash
python tools/run_folds.py \
  --prepared data/cactus160_loo_val --label_set arm4 \
  --save results/cactus160_loo_val --folds fold_01 fold_02 \
  -- --batch_size 4 --epochs 120 --residual --patience 15 --seed 0
```

检查它们的验证曲线与模型/模板误差，判断 fold_00 的现象是否重复。训练设置的选择应依靠验证数据，最终测试数据用于报告。

**第 8 步：完整训练两个 frame，再汇总。**

运行全部 frame 160 folds 时，省略 `--folds`：

```bash
python tools/run_folds.py \
  --prepared data/cactus160_loo_val --label_set arm4 \
  --save results/cactus160_loo_val \
  -- --batch_size 4 --epochs 120 --residual --patience 15 --seed 0
```

已生成测试 CSV 的 folds 会被跳过。中断而未完成评估的 fold 会重新训练；脚本当前没有从最后训练轮次恢复的逻辑。

frame 160 完成后，在同一张 GPU 上顺序运行 frame 220：

```bash
python tools/run_folds.py \
  --prepared data/cactus220_loo_val --label_set arm4 \
  --save results/cactus220_loo_val \
  -- --batch_size 4 --epochs 120 --residual --patience 15 --seed 0
```

仅重新生成汇总表：

```bash
python tools/run_folds.py --prepared data/cactus160_loo_val \
  --label_set arm4 --save results/cactus160_loo_val --summary_only
python tools/run_folds.py --prepared data/cactus220_loo_val \
  --label_set arm4 --save results/cactus220_loo_val --summary_only
```

两帧结果都有后，可以生成论文图和表；下面示例选择当前 fold_00 测试的样本：

```bash
CACTUS_DATA='/mnt/d/Houdini_Models/Ranxi_Models/practice/datasets/datasets_20260926/datasets_cactus pose frames_ply'
python tools/paper_figures.py --data "$CACTUS_DATA" --out figures/paper \
  --run 'Frame 160=data/cactus160_loo_val:results/cactus160_loo_val:160' \
  --run 'Frame 220=data/cactus220_loo_val:results/cactus220_loo_val:220' \
  --sample BB_0_TRL_0.6_TRM_0.6_TRLN_0.6_T_0.75_L
```

完整实验最后要回答的是：各肌肉的模型误差是否低于模板、不同肌肉配置上的表现是否一致、体积是否能跟随真实大小变化，以及 frame 160 和 220 的结果有什么差异。

当前已完成环境、两帧预处理、frame 160 的 fold_00 训练及结果分析；本机当前已有 fold_01、fold_02 的评估 CSV；完整 folds 和 frame 220 的实验状态应另行检查。下一步先将导师所说的“骨骼加四块肌肉”五个 mesh 对应到具体脚本及预测输入/输出，确认后再决定是否执行第 7 步。

**将完整原始数据上传到 Cetus：用户已确认的范围和路径。**

上传范围为整个 `datasets_cactus pose frames_ply` 文件夹，包含皮肤、分开的肌肉、整体肌肉、骨骼以及目录里的转换脚本和日志。检查时共有 1,301 个文件，24,696,948,796 字节，约 23.00 GiB；`du -sh` 显示约 24G。

这组数据的目录名称按参数配置编码为 `cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220`。

| 编码 | 含义 |
|---|---|
| `cactus` | cactus 姿态数据 |
| `BB_TRL_TRM_TRLN` | 明确列出这一批数据的四个肌肉参数 |
| `n060p060s060` | 所列肌肉参数共用最小值 -0.60、最大值 +0.60、步长 0.60；数字按实际值 ×100 编码 |
| `T_075100125` | 组织厚度比例 T 的取值为 0.75、1.00、1.25 |
| `F_160_220` | 包含 frames 160 和 220 |

已检查完整参数组合，共 81 个肌肉配置组、每组 3 个 T 值。详细参数记录保存在 `data/upload_configs/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/parameters.json`。以后上传其他肌肉组合时，替换名称中的肌肉参数缩写列表；共享的参数范围代码继续使用 `n060p060s060`。如果各肌肉范围不同，应为各自范围使用单独编码。

用户确认尚未开始上传，因此直接使用这个新目标目录。远端目标为：

```text
/shared/homes/u24615736/git/DeformingPointTransformer/data/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/
```

已准备本机上传脚本 `tools/upload_cetus.sh`。脚本创建远端目录、检查空间、上传原始数据和 `parameters.json`，并检查是否仍有待同步文件。直接在本机 WSL 运行时，会提示输入已有 Cetus 私钥口令。也可先运行 `bash tools/cetus_upload_auth.sh`，在本机窗口输入口令，建立后台 SSH 连接；随后运行 `CETUS_UPLOAD_CONTROL_PATH=/tmp/cetus-upload-rxliu/control bash tools/upload_cetus.sh`，沿用已解锁的连接。私钥口令只输入本机终端，不输入聊天框。

本机日志和状态记录为：

```text
results/cetus_upload/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/upload.log
results/cetus_upload/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/status.json
```

2026-10-13 23:13（悉尼时间）已恢复后台上传。远端目录已创建，`parameters.json` 已上传，原始 mesh 正在同步。后台任务独立于日志查看窗口；需保持本机、WSL 和 UTS VPN 在线，避免电脑睡眠。整批完成后脚本会自动执行同步检查并将状态写为 `completed`。此段记录的是启动状态，最新情况应查看 `status.json` 和 `upload.log`。

`status.json` 的 `phase` 依次为 `waiting_for_authentication`、`transferring`、`verifying`、`completed`；错误时为 `failed`。只有 `completed` 才表示整个脚本成功结束。若上传失败或中断，在本机 WSL 重跑即可同步尚未完成的文件：

```bash
bash ~/git/alen_DPT/DeformingPointTransformer/tools/upload_cetus.sh
```

本机已配置 `ssh cetus`，连接用户为 `u24615736`，主机为 `hpc-login01.hpc.uts.edu.au`。排查初期两个登录节点连接超时；随后用户已在自己的 WSL 终端通过 `ssh -o ConnectTimeout=15 cetus` 成功登录，并输入了加密私钥的口令。WSL 的 SSH 别名配置有效。VS Code Remote-SSH 使用的配置是另一个问题，WSL 中的文件传输可以独立执行。实际上传状态以以上日志及 `status.json` 为准。

在 WSL 终端测试：

```bash
ssh -o ConnectTimeout=15 cetus
```

若 Windows 已能连接、WSL 仍超时，需要检查 WSL 与 Windows/VPN 的网络路由。若二者都不能连接，先排查 UTS 网络接入。连接超时发生在账号密码或密钥认证之前。

SSH 恢复后，在本机 WSL 终端执行以下命令。传文件使用系统的 `ssh` 和 `rsync`，不需要激活训练 Conda 环境或设置 TotalSegmentator 许可证。

```bash
CACTUS_DATA='/mnt/d/Houdini_Models/Ranxi_Models/practice/datasets/datasets_20260926/datasets_cactus pose frames_ply'
CETUS_DATA_DIR='/shared/homes/u24615736/git/DeformingPointTransformer/data/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220'

# 建立目标目录并查看文件系统剩余空间；还应确认个人存储配额可容纳数据。
ssh -o ConnectTimeout=15 cetus "mkdir -p -- '$CETUS_DATA_DIR' && df -h '$CETUS_DATA_DIR'"

# 上传整个原始目录的内容。
rsync -rtvh --partial-dir=.rsync-partial --info=progress2 \
  -e 'ssh -o ConnectTimeout=15' \
  "$CACTUS_DATA/" "cetus:$CETUS_DATA_DIR/"
```

源路径结尾的 `/` 表示复制文件夹内的内容，因此远端参数配置目录下直接是 `bones/`、`final skin/`、`separated muscles/` 等。`--partial-dir` 保留中断时的部分数据，恢复连接后重跑同一条 rsync 命令即可继续同步；命令不删除本机数据。

原始文件上传后，可一并上传参数记录，便于以后区分其他数据批次：

```bash
rsync -tv -e 'ssh -o ConnectTimeout=15' \
  "$HOME/git/alen_DPT/DeformingPointTransformer/data/upload_configs/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/parameters.json" \
  "cetus:$CETUS_DATA_DIR/"
```

参数记录是额外的一个文件；原始数据仍为 1,301 个文件。本机原始数据目录继续使用既有 D 盘路径。

传输成功后，检查是否还有待同步文件：

```bash
rsync -rtni -e 'ssh -o ConnectTimeout=15' \
  "$CACTUS_DATA/" "cetus:$CETUS_DATA_DIR/"
```

该命令仅检查，不实际传输。没有列出待更新项表示 rsync 按文件大小和修改时间判断数据已同步。上传后使用数据的脚本应将 `--data` 指向上述远端目录；原始 mesh 上传本身不决定后续模型的输入和输出。

**VS Code 使用 Remote-SSH 连接 Cetus。**

已确认 Windows 的 VS Code 安装了 Remote-SSH 扩展。Windows 用户是 `liura`，WSL 用户是 `rxliu`，两边的 SSH 配置文件各自独立。已经为 Windows 配好：

```text
C:\Users\liura\.ssh\config
```

对应主机条目：

```sshconfig
Host cetus
    HostName hpc-login01.hpc.uts.edu.au
    User u24615736
    Port 22
    IdentityFile C:/Users/liura/.ssh/id_ed25519_cetus
    IdentitiesOnly yes
    ConnectTimeout 15
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

将 WSL 里现有带口令的 Cetus 密钥复制到了同一电脑的 Windows `.ssh` 目录，并设置私钥仅由当前 Windows 用户访问。VS Code 用户设置已指定 Windows SSH 客户端和配置文件，并启用登录终端显示：

```json
"remote.SSH.configFile": "C:/Users/liura/.ssh/config",
"remote.SSH.path": "C:/Windows/System32/OpenSSH/ssh.exe",
"remote.SSH.showLoginTerminal": true,
"remote.SSH.useLocalServer": false
```

原 Windows SSH 配置、VS Code 用户设置已保存为同目录下带 `.backup-` 时间戳的文件。

连接时保持 UTS VPN 可用。在 Windows VS Code 中按 `Ctrl+Shift+P`，选择 `Remote-SSH: Connect to Host...`，再选择 `cetus`。输入密钥口令，若询问平台则选 Linux。首次连接会安装 VS Code Server，等待连接完成后左下角应显示 `SSH: cetus`。

本次已从 Remote-SSH 日志确认连接成功：远端服务器开始监听，SSH 隧道建立，`ssh-remote+cetus` 已解析，远端 exec server 已创建并可复用。用户也已确认窗口左下角显示 `SSH: cetus`。

打开远端项目目录：

```text
/shared/homes/u24615736/git/DeformingPointTransformer
```

这时 VS Code 的文件浏览器和终端操作远端 Cetus；当前 `WSL: Ubuntu` 窗口仍对应本机文件。上传原始数据时，rsync 命令仍应在本机 WSL 终端执行。

**代码同步与大文件放置：2026-10-14 确认的安排。**

5060 笔记本通过 Remote-SSH 使用 Cetus 的同一份仓库。因此实际需要合并代码的是 3080 的 WSL 仓库和 Cetus 仓库，统一远端为老师的 `alalemp/DeformingPointTransformer`，分支为 `ranxi-dev`。目前先整理文件，暂不提交或推送。

只同步源码、环境配置、说明文档和小型参数配置。原始 PLY、预处理数据、训练权重、预测文件、生成图片和运行日志不纳入普通 Git 提交。`.gitignore` 排除生成目录并不会删除其中现有的文件，也不会中断训练或上传。

上传完成并通过脚本同步检查后，计划把 Cetus 原始数据目录移到：

```text
/shared/homes/u24615736/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/
```

这是用户选定的计划路径，本段不代表迁移已经执行。在上传状态为 `completed`、上传任务已结束之前，继续使用现有上传目标，不移动目录。迁移时保留整个参数命名目录及 `parameters.json`；若目标已存在，先检查内容，不能直接覆盖。

Cetus 的 `tools/uts_arm_job.sh` 已支持 `RAW_DATA` 参数。迁移后，prepare 作业应将该参数设为上述外置目录；预处理数据和训练输出仍可在仓库中的被忽略目录 `data/`、`results/` 重新生成。现有实验文件是否清理应另外决定，本次仅调整版本管理范围。

GitHub 官方说明：通过仓库根目录的 `.gitignore` 共享忽略规则；普通 Git 单文件超过 50 MiB 会有警告，超过 100 MiB 会被拒绝；需要在 GitHub 管理的大文件可使用 Git LFS，发布用的二进制文件也可附在 Releases。GitHub 没有规定本项目必须采用上述目录层级，这是根据原始数据体积及现有脚本作出的安排。

- [忽略文件](https://docs.github.com/en/get-started/git-basics/ignoring-files)
- [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)

上传脚本优先读取本机 data/upload_configs/<参数代码>/parameters.json；若不存在则使用 configs/upload/<参数代码>.example.json。示例配置纳入代码同步，当前运行中的上传继续使用原记录，保持现有目标。以后新上传默认放到外置 datasets/houdini 目录；使用示例前检查 local_source 是否对应本机数据路径。迁移完成后先同步代码，暂不重新生成实验数据。
