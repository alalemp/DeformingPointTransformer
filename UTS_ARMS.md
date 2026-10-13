# 在 UTS Cetus 上安装并运行手臂实验

本流程对应 GUIDE.md 第 0 节，使用自己的 Houdini 导出。Miniconda 安装到
`~/miniconda3`，原始数据和最终结果存放在仓库；PBS 作业在节点本地 `/scratch`
运行后复制结果回来。无需创建 `/data`，也不执行针对老师 iHPC 的安装脚本。

本账户已完成 CUDA 检查：PyTorch 2.5.1+cu121、Quadro RTX 6000，pointops 与 Chamfer 的正向及反向运算通过。记录见 results/uts_check/cuda_check.txt。

## 1. 安装一次：登录节点执行

```bash
cd /shared/homes/u24615736/git/DeformingPointTransformer
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -o "/tmp/miniconda-${USER}.sh"
bash "/tmp/miniconda-${USER}.sh" -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
CONDARC="$PWD/conda-forge.condarc.yml" conda env create -f environment.yml
conda install -n bodyseg --override-channels -c conda-forge \
    gcc_linux-64=12 gxx_linux-64=12 "setuptools<81" wheel -y
source cluster_env_uts.sh
python -c 'import torch, numpy, pandas, nibabel; print(torch.__version__, torch.version.cuda)'
```

已安装 Miniconda 时跳过下载和安装，只运行 source；已有 bodyseg 时使用
`CONDARC="$PWD/conda-forge.condarc.yml" conda env update -f environment.yml`。
不要同时运行两个环境安装进程。

使用老师原始的 `environment.yml`，保留其中完整依赖及版本；另补充 UTS 所需的
Conda GCC 12 和编译工具。创建环境时使用独立的 `conda-forge.condarc.yml`，避免新版
Miniconda 的 defaults 条款插件在读取环境文件之前触发检查；该配置只用于这一条
命令，不修改全局频道，也不自动接受条款。该配置还将新版 Conda 转发的
`nodefaults` 名称解析为空频道，保持旧版的排除默认源语义。
每次训练保存 pip_freeze.txt。
手臂实验无需下载 MRI 分割权重、设置许可证或克隆 VIBESegmentator。

登录节点没有 GPU，torch.cuda.is_available() 为 False 不表示安装失败。

## 2. 编译并测试 CUDA 扩展：登录节点提交 PBS 作业

```bash
qsub -N dpt_check -q small_gpuq \
    -l select=1:ncpus=4:mem=16gb:ngpus=1 -l walltime=01:00:00 \
    -v MODE=check tools/uts_arm_job.sh
qstat -u "$USER"
```

等待作业完成后：

```bash
cat results/uts_check/cuda_check.txt
```

成功标志为 `CUDA_CHECK_OK`。作业安装 pointops、JIT 编译 Chamfer3D，并测试
pointops 的采样/kNN 和 Chamfer 的前向及反向传播。PBS 输出文件也保留编译日志。
这里的队列选择依据 Cetus 官方文档，未在当前会话验证队列实时状态。

## 3. 放入自己的 Houdini 数据

```bash
mkdir -p $HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220
```

将数据上传至这个目录，例如 frame 160：

```text
$HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/
  final skin/<subject>/final_skin.00160.ply
  separated muscles/bicepsBrachialis/<group>/bicepsBrachialis.00160.ply
  separated muscles/tricepsLateralis/<group>/tricepsLateralis.00160.ply
  separated muscles/tricepsMedialis/<group>/tricepsMedialis.00160.ply
  separated muscles/tricepsLongus/<group>/tricepsLongus.00160.ply
  bones/bones.00160.ply
```

- subject 示例：`BB_0_TRL_0.6_TRM_-0.6_TRLN_0_T_1`。
- group 是去掉 subject 末尾 `_T_1` 的名称。
- 二进制 PLY、默认单位米、右臂 x > 0；每个肌肉文件含双臂，网格应闭合。
- 各结构使用同一坐标系，保持与脚本约定一致的姿态和朝向。
- 骨骼用于可视化；默认不做骨骼对齐时，训练预处理不依赖骨骼文件。
- 需要多个独立肌肉配置组。只有一个角色或一个配置无法划分训练/验证/测试；
  本流程要求每折至少 4 个训练臂样本才能使用 batch_size 4。
- 49 折是老师数据的规模。自己的有效组数决定实际折数，不能假定得到相同结果。

已有数据在其他目录时，prepare 作业可添加 `-v MODE=prepare,FRAME=160,RAW_DATA=/实际路径`。
不符合上述布局、坐标或命名规则的数据需要先适配，不能只改 --data 路径。

## 4. CPU 预处理：登录节点提交，等待完成

```bash
qsub -N dpt_prep160 -q smallq \
    -l select=1:ncpus=4:mem=32gb -l walltime=04:00:00 \
    -v MODE=prepare,FRAME=160 tools/uts_arm_job.sh
```

检查 `data/cactus160_loo_val/qc.png` 和 `subjects.csv`，核对皮肤裁剪与肌肉。
每折应含 train.txt、val.txt、test.txt、mean_shape.npz。
按肌肉配置组划分，模板仅由训练数据生成。

## 5. 先训练一个折：登录节点提交，等待完成

```bash
qsub -N dpt_fold160 -q small_gpuq \
    -l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=06:00:00 \
    -v MODE=train,FRAME=160,FOLD=fold_00,RUN_NAME=check160 tools/uts_arm_job.sh
```

训练参数固定为 batch_size 4、最多 120 epochs、residual、patience 15、seed 0。
使用 val.txt 选择 checkpoint，训练后默认重新校准 BatchNorm。

结果复制回来后检查：

```bash
tail -n 20 results/check160/fold_00/train.log
cat results/check160/fold_00/csv/best_epoch.txt
cat results/check160/summary_arm4.csv
```

比较模型与模板的 CD、HD95、体积误差，不仅观察训练损失。
作业运行期间的训练日志位于计算节点 `/scratch/$USER/dpt_<PBS_JOBID>/repo/results/`；
仓库内结果在作业结束后同步。PBS 标准输出显示当前折与日志位置。

## 6. 扩展到全部折及 frame 220

检查另外几个折后，再提交全部 frame 160 折：

```bash
qsub -N dpt_all160 -q small_gpuq \
    -l select=1:ncpus=4:mem=32gb:ngpus=1 -l walltime=48:00:00 \
    -v MODE=train,FRAME=160,FOLD=all,RUN_NAME=cactus160_loo_val tools/uts_arm_job.sh
```

frame 220 先用步骤 4 的命令改为 FRAME=220 准备数据，再把上述训练命令的
FRAME 改为 220、RUN_NAME 改为 cactus220_loo_val。
已完成折会跳过；未完成折会重新训练，不是从中断 epoch 恢复。
不要同时提交写入同一个 RUN_NAME 的作业。

训练结束后，summary_arm4.csv、每折 csv/volumes.csv、predictions.npz 和模型权重
位于对应 results 目录。作业同时记录源码 commit 和 pip freeze。

作业使用节点本地 scratch 并在退出时回传结果；硬性终止可能使回传来不及完成。
脚本保留 scratch，在同一节点上仍可恢复其已写入的文件，但不能将 scratch 当长期存储。

## 7. 单样本可视化与论文图

在已分配的计算会话中激活环境并运行，替换 SAMPLE 为自己的测试臂 ID：

```bash
cd /shared/homes/u24615736/git/DeformingPointTransformer
source cluster_env_uts.sh
SAMPLE="你的subject名称_R"
python tools/visualize_cactus.py --data $HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220 --frame 160 \
    --id "$SAMPLE" --prepared data/cactus160_loo_val \
    --results results/cactus160_loo_val --out vis/arm160
python tools/paper_figures.py --data $HOME/datasets/houdini/cactus_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220 --out figures/paper \
    --run "Frame 160=data/cactus160_loo_val:results/cactus160_loo_val:160" \
    --run "Frame 220=data/cactus220_loo_val:results/cactus220_loo_val:220" \
    --sample "$SAMPLE"
```

HTML 可下载到本地浏览器打开；当前查看器通过网络加载 Plotly。

参考：[Cetus 软件安装](https://cetus.hpc.uts.edu.au/software/)、
[Cetus 队列](https://cetus.hpc.uts.edu.au/pbs/queues/)、
[Cetus 存储](https://cetus.hpc.uts.edu.au/hardware/layout/)、
[PyTorch 版本安装](https://pytorch.org/get-started/previous-versions/)。

原始数据上传完成并迁移前，请等待数据整理状态确认；本次仅同步代码，暂不提交新的预处理或训练作业。
