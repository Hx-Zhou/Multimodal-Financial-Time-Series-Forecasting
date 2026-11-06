# E10 分阶段优化

[完整实验路线](EXPERIMENT_PLAN.md)按 S0–S8 分开处理频域通路退化、公平 TaTS 对照、幅度归一化、频率数和频域消融。

本轮只运行 S1/S2 两个正则化对照，保留 13 个频率、8 个复数通道及旧模型结构。每一步分别在 `results/S1/`、`results/S2/` 与对应本地 `outputs/` 中。S3–S8 是有条件的后续计划，尚未执行。

仓库根目录运行：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e10_dlinear_spectf_refine.run --package-dir ../../hs300_train_val
```

已有运行用 `--resume` 恢复。共同规则见 [公共说明](../dlinear_refine_common/README.md)。复杂频域融合不能稳定超过匹配的简单融合就停止扩张；本轮不重复评估内部 holdout。
