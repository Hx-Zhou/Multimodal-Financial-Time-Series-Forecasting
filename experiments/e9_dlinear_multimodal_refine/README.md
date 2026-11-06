# E9 分阶段优化

[完整实验路线](EXPERIMENT_PLAN.md)按 M0–M7 分开处理训练退化、单模态对照、联合训练干扰和内容互补性。

本轮只运行 M1/M2 两个正则化对照，保持原 E9 结构不变。每一步分别保存在 `results/M1/`、`results/M2/` 和对应本地 `outputs/`。后续 M3–M7 仅在证据支持时推进，尚未执行。

仓库根目录运行：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e9_dlinear_multimodal_refine.run --package-dir ../../hs300_train_val
```

已有运行用 `--resume` 恢复。共同规则见 [公共说明](../dlinear_refine_common/README.md)。联合模型要超过同条件最强单模态，不能只与较弱的行情基线比较。本轮不重复评估内部 holdout。
