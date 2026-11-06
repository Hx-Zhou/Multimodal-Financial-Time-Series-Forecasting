# 多模态逐步优化：公共运行规则

三条实验路线分别位于：

- [E8 财报路线](../e8_dlinear_reports_refine/EXPERIMENT_PLAN.md)
- [E9 联合融合路线](../e9_dlinear_multimodal_refine/EXPERIMENT_PLAN.md)
- [E10 频域路线](../e10_dlinear_spectf_refine/EXPERIMENT_PLAN.md)

本目录只放共享的优化器、审计、训练调度和测试。各实验及每一步的记录分别放在对应目录。旧实验不覆盖；根模板、公共划分、E1/E6/E7 和其他组员的模型均不改。

## 本轮实际执行范围

只执行 F1/F2、M1/M2、S1/S2：

1. **第一步**：原结构和输入不变，将耦合 L2 的 Adam 改为 AdamW；仍然对全部可训练参数施加 wd=0.01，保持 lr=0.001。
2. **第二步**：在相同 AdamW 下，只对矩阵权重施加衰减，偏置和门控免衰减。这样能分别观察优化器类型和参数分组的作用。

都从同一个匹配 E6 检查点和相同的 seed42 初始化开始，最多 20 epoch，patience=6，min_delta=1e-5，batch=1024。保留初始纯行情状态，训练不会被迫选择辅助分支。20 epoch 是新一轮的固定预算；因此与旧 40 epoch 上限也有预算差异，不能声称严格只改变了优化器类型的完整复现。

每条路线运行前固定较强开发集对照。只有开发 MAE 至少改善 0.1%、MSE 不恶化超过 0.5%、内容置零及同股票打乱均使 MAE 恶化至少 1e-5，且开发早／晚两个时段 MAE 都不恶化超过 0.5%，才增加 seed43/44。多个合格配置取开发 MAE 较低者，差距不超过 1e-5 时优先第一步。

三种子复验要求平均 MAE 改善至少 0.1%、至少 2/3 个种子改善、至少 2/3 个种子有内容诊断支持、MSE 及开发两个时段仍通过门槛。诊断门槛不是显著性检验，早晚开发滑窗也不是独立盲测。

若未通过，就记录明确失败原因，按各自路线进入有针对性的下一项检查，不自动扩张搜索。本轮没有 holdout 参数，不构建或评估 holdout 样本；底层公共读取器会载入 val 文件，但只提供开发子集给优化流程。之前的内部测试已经揭晓，不能再称为新盲测。

## 运行

在仓库根目录执行：

```powershell
./.venv/Scripts/python.exe -m experiments.dlinear_refine_common.audit
./.venv/Scripts/python.exe -m unittest experiments.dlinear_refine_common.test_optimizer experiments.dlinear_modal_common.test_integrity
./.venv/Scripts/python.exe -u -m experiments.dlinear_refine_common.run --package-dir ../../hs300_train_val
```

单个实验可使用自己的 `run.py` 模块入口。已有运行不会静默覆盖；`--resume` 校验源文件／实验计划、协议、数据和旧权重后，保留已完成结果，从最近完整 epoch 恢复未完成任务及随机状态。修改下一阶段的实现必须开新阶段记录，不能带着变化覆盖当前实验。

## 权重退化审计

旧训练末尾的财报编码器范数约为初始化的 10^-5 量级；新闻编码器及频域 query/value 也有明显缩小。对负数 gate logit 做 L2 会把它向零推，使 sigmoid 门控从约 0.12 向 0.5 增大，与“限制辅助模态权重”的直觉相反。

审计是机制线索，不能单独证明 L2 是唯一原因。F1/F2 等对照用真实开发指标与内容响应检验这个假设。完整权重审计在每条路线的 `results/initial_audit.json`。
