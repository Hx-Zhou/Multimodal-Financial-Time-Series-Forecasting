# Transformer 逐步优化：共享执行工具

完整训练前计划：[PLAN.md](PLAN.md)。本轮最多四个Transformer初筛配置，再给一个候选做三种子复验；达到预定开发门槛才统一测试。

实验目录：

- `../e4_patchtst_capacity/`：P0选择规则基线、P1宽度64、P2较高dropout及Close线性对照。
- `../e5_itransformer/`：I0变量注意力模型及同输入七变量线性对照。
- 本目录保存共享数据、训练、调度、完整协议与汇总结果。

仓库根目录使用已有环境：

```powershell
./.venv/Scripts/python.exe -m unittest experiments.transformer_optimization_common.test_integrity
./.venv/Scripts/python.exe -u -m experiments.transformer_optimization_common.run --package-dir ../../hs300_train_val
```

中断后加 `--resume`：核对代码／计划与协议，保留完整记录，从最近完整epoch恢复模型、优化器及随机状态。现有结果不会静默覆盖。旧E1/E4/E6和公共读取器／划分保持不变，E2/E3不运行。

运行全部在原CPU环境中，模型与数值数组保存在内存，磁盘仅保留小型检查点、预测与元数据，不产生新闻／财报大缓存。所有`outputs/`和`.pt`均由gitignore排除。

检查点保存**原始权重＋alpha元数据**，验证／推理通过forward应用一次alpha=0.5；不要再次手动缩放权重。达到最大8轮只说明预算耗尽，不代表收敛。

已有开发／内部测试已多次观察。本轮属于在已知数据上的有限开发实验，不能称为全新盲测。完整结论写入本目录和两个实验目录的`RESULTS.md`。
