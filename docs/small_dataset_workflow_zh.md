# 小数据集 Step 1–8 输出与运行记录

原始数据实际入口：
`/shared/homes/u24615736/datasets/houdini/cactus_small_BB_TRL_TRM_TRLN_n060p060s060_T_075100125_F_160_220/datasets_cactus pose frames_ply`

已检查每帧有108个完整subject、49个肌肉配置；双臂处理后预计216个手臂样本。配置组的T变体数量不完全相同，因此各折train/val/test样本数并不固定。查看预处理产生的small_split_counts.json。两帧分别训练，默认seed=0、batch4、最多120epochs、residual、patience15，仅使用原有Chamfer损失。

## 当前提交结构

- 160预处理：154410.hpc-head01。
- 220预处理：154420.hpc-head01。
- 160单折检查：154411.hpc-head01，训练fold_00，最终沿用其结果而非重复训练。
- 160批次：154412～154416，依次覆盖fold_01～fold_35。
- 后续调度：154450.hpc-head01，等待160的fold_00成功后提交220全部折训练及后处理，并安排160剩余fold_36～48。
- 最终各帧汇总、论文图与HTML依赖对应训练成功；两帧对照图依赖两帧后处理成功。
- 新任务编号会追加到results/small_workflow/jobs.csv，不要按本文件重复提交。

当前small_gpuq的max_queued是每用户6个，不能一次提交更多GPU等待任务。最初尝试的后处理因缺失前置训练而失败，不是训练结果失败；记录保存在submission_attempt1.csv。没有取消任何旧实验或新训练任务。已用延后提交方式修复工作流。

## 输出位置（相对于代码仓库）

| 目录 | 内容 |
|---|---|
| data/small_cactus160_loo_val/ | 160帧共享点云data.npz、坐标affines.npz、subjects.csv、labels、qc.png、各折名单与模板 |
| data/small_cactus220_loo_val/ | 220帧同类预处理结果 |
| results/small_cactus160_loo_val/fold_XX/ | 160帧每折日志、预测点云、最佳/最后权重、损失曲线、测试CD/HD95和体积误差 |
| results/small_cactus220_loo_val/fold_XX/ | 220帧同类训练结果 |
| results/small_cactus160_loo_val/summary_arm4.csv | 160帧49折最终汇总，只有后处理成功后才有完整表 |
| results/small_cactus220_loo_val/summary_arm4.csv | 220帧49折最终汇总 |
| figures/small_paper/frame160或frame220/完整样本ID/ | 每帧Step7五类图的PNG/PDF及CSV/LaTeX表 |
| figures/small_paper_comparison160_220/完整样本ID/ | 两帧并排比较的完整Step7图和表 |
| vis/small_interactive/frame160或frame220/完整样本ID.html | Step8交互3D视图 |
| results/small_workflow/ | 作业索引、源码commit、未提交代码补丁、作业脚本快照和实时作业日志 |
| results/small_pbs_logs/ | 此工作流的PBS标准输出与错误日志，不散落到仓库根目录 |

论文图使用负、零、正三个组中实际存在的手臂作示例；优先T=1右臂，不存在时选择该组实际可用样本。表格和体积散点覆盖全部已完成49折测试样本，不限于示例样本。此小数据集的全正组没有T=1皮肤，不能假设所有组都有三个T值。

每折文件：train.log；predictions.npz；models/model_weights_best.pth和model_weights_last.pth；csv/loss_history.csv、best_epoch.txt、trained_vs_target_arm4.csv、init_vs_target_arm4.csv、volumes.csv；figures/基础图。

五类论文图：figS1_full_body、fig1_data_overview、fig2_training_curves、fig3_volume_scatter、fig4_error_surface。指标表：table_metrics.csv和table_metrics.tex。前缀small用于数据集级输出目录；内部标准文件名保持工具默认以保证兼容。

## 如何监视

```bash
qstat -u "$USER"
cat results/small_workflow/jobs.csv
ls results/small_workflow/logs
```

`Q`排队、`R`运行、`H`等待成功依赖。查看实时主日志：`tail -n 30 results/small_workflow/logs/作业编号.log`。训练内部epoch日志仍写在计算节点scratch；新提交的后续训练脚本每折完成后即回传，最初160批次在批次退出后回传。若硬超时导致回传不全，需检查保留scratch并恢复，不要直接假定结果齐全。

Step1复用已安装并验证的bodyseg和CUDA扩展，不重新安装；Step3的fold_00结果被Step4沿用；Step5持续监视；Step6～8在训练成功后自动执行。完整训练可能持续数天，排队时间未知；48小时是单作业运行上限而非总完成时间承诺。任一前置失败，依赖任务不会继续，需诊断后恢复。
