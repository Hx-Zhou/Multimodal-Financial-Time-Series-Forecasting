# E9-2：冻结 E3c 的财报适配器

E9-2 固定 E3c 的相对 OHLC GRU 和预测头，只训练财报分支。财报使用 E8 的 16 维最新报告快照，经过投影和时间 GRU 后，通过逐预测步门控残差接入 E3c。门控初值为 0，因此 epoch 0 严格等于 E3c。

训练：

```bash
python -m experiments.gru_e9_2.train --device cuda
```

结果和权重：

```text
experiments/gru_e9_2/result.json
experiments/gru_e9_2/checkpoints/best.pt
```

E9-2 使用开发集选择 epoch≥1 的最佳 checkpoint，训练结束后自动评估 holdout。
