# E8 分阶段优化

[完整实验路线](EXPERIMENT_PLAN.md)按 F0–F7 明确每一步的假设、对照、改动、门槛和停止条件。

本轮只运行 F1/F2 两个正则化对照，模型结构复用旧 E8，所有结果在本目录独立保存。每一步分别位于 `results/F1/`、`results/F2/` 和对应的本地 `outputs/`。后续 F3–F7 是有条件的后续计划，尚未执行。

仓库根目录运行：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e8_dlinear_reports_refine.run --package-dir ../../hs300_train_val
```

已有运行用 `--resume` 恢复，不覆盖旧记录。共同规则和初筛／复验门槛见 [公共说明](../dlinear_refine_common/README.md)。这一系列只用开发集，不重复测试已经查看过的内部 holdout。
