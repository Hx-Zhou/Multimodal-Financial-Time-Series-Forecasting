# E3c-E8：E3c 相对 OHLC + 财报

E8 在 E3c 行情骨干上加入财报。对每个样本，只使用输入末日之前每类财报最新的一条记录，提取收入、利润、资产、负债、权益、现金流等核心字段，使用 signed `log1p` 压缩，并加入报告存在性和距输入末日的时间特征。财报状态在 96 日行情窗口中重复输入，避免把稀疏财报误当成逐日观测。

训练：

```bash
python experiments/gru_e8/train.py --device cuda
```

结果和权重：

```text
experiments/gru_e8/result.json
experiments/gru_e8/checkpoints/best.pt
```

训练结束后自动加载开发集最佳 epoch≥1 checkpoint 并评估 holdout。
