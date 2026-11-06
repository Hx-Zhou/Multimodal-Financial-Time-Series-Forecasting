# GRU 分时段评估

本目录集中保存固定检查点的分时段评估脚本和结果。它们把开发集按预测起始日期分为三个连续区间，对同一检查点逐段计分；每段都不重新训练，因此这是时间区间诊断，不是 walk-forward 重训练，也不读取 Holdout。

从仓库根目录运行：

```bash
python -m experiments.gru_validation.rolling_validation --device cuda
python -m experiments.gru_validation.rolling_validation_e10 --device cuda
python -m experiments.gru_validation.rolling_validation_e11 --device cuda
```

默认结果分别写入 `rolling_validation.json`、`rolling_validation_e10.json` 和 `rolling_validation_e11.json`。脚本默认读取 E3c、E9-2、E10-B、E10-C 与 E11 的检查点；运行前需确认对应检查点和 `data/` 可用。

公共时间切分实现及固定切分记录仍位于上一级 `experiments/validation_split.py` 和 `validation_split.json`，供训练脚本及 E1 DLinear 共用。
