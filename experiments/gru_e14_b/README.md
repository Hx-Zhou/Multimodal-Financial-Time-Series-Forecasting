# E14-B：财报可用滞后敏感性

E14-B 保持 E10-B 的行情模型、财报 16 维特征、标准化方法、损失函数和训练切分不变，只比较报告期末之后的可用滞后天数：

```text
30、60、90 个自然日
```

0 天不在本轮重新训练。历史 E9-2 提供了无滞后的参考，但 E10-A 除了增加 30 天滞后，还修正了报告年龄计算；E10-B 又增加了训练集标准化。因此目前没有同一 E10-B 配置下的严格 0 天对照，旧结果只作为历史背景，不和本轮候选一起选择。

每个候选滞后都重新计算训练集可用财报的均值/标准差，并独立训练财报分支。默认使用 seed 42、最多 20 轮、patience 6、batch size 256、学习率 `3e-4`。已有 E10-B 的 30 天结果默认复用，60 和 90 天需要训练。

运行：

```bash
python -m experiments.gru_e14_b.run --device cuda
```

输出目录：

```text
experiments/gru_e14_b/summary.md
experiments/gru_e14_b/summary.json
experiments/gru_e14_b/lag_30/
experiments/gru_e14_b/lag_60/
experiments/gru_e14_b/lag_90/
```

训练中断后重复同一命令会跳过通过校验的滞后。每个滞后单独保存 `result.json`、`checkpoints/best.pt`、`train.log` 和 `manifest.json`。中断中的训练不保存优化器状态，下一次会从该滞后重新开始。

开发集 MAE（MSE 平手时）选择最终滞后；Holdout 只在全部选择完成后作为辅助报告，不能用于选滞后。此实验只有一个随机种子，后续若某个滞后胜出，再对胜出滞后做多种子验证。

使用 `--no-reuse-lag30` 可重新训练 30 天版本，避免把历史 E10-B 结果和新实验混合：

```bash
python -m experiments.gru_e14_b.run --device cuda --no-reuse-lag30 --output-dir experiments/gru_e14_b_fresh
```

只整理已有结果：

```bash
python -m experiments.gru_e14_b.run --summarize-only
```
