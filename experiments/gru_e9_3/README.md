# E9-3：联合新闻与财报适配器

E9-3 冻结 E3c 的相对 OHLC GRU 和行情预测头，同时训练两个独立的时间分支：

- 新闻：每日均值和最大池化后的 1024 维向量，沿用 E9-1 的新闻投影、GRU 和逐步门控；
- 财报：沿用 E9-2 的 16 维最新财报快照、投影、GRU 和逐步门控。

最终预测为：

```text
E3c + news_gate * news_delta + financial_gate * financial_delta
```

默认从 E9-1/E9-2 的最佳开发集检查点初始化两个适配器。行情分支始终从
`experiments/gru_e3/checkpoints/best.pt` 加载并冻结。开发集选择最佳 epoch，holdout 只在训练结束后评估。

训练并自动评估 holdout：

```bash
python -m experiments.gru_e9_3.train --device cuda
```

结果和检查点：

```text
experiments/gru_e9_3/result.json
experiments/gru_e9_3/checkpoints/best.pt
```

如需确认训练完成后的检查点评估：

```bash
python -m experiments.gru_e9_3.train --evaluate-holdout --device cuda
```
