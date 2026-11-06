# E6-A：原始七字段行情对照

E6-A 在 E3c 的相同 GRU、数据切分和训练流程上，改用七个原始行情字段：

```text
Open, High, Low, Close, Volume, Dividends, Stock Splits
```

输入处理保持“必要标准化”原则：OHLC 使用该股票训练期的 `close_mean`、`close_std`；Volume 使用 `log1p` 后在当前 96 日窗口内标准化；Dividends 按收盘价标准差缩放；Stock Splits 保留事件值。模型仍预测标准化 Close 的残差，最后值作为 epoch 0 的 E0 参照。

这次是 E6-A 的正式重跑：最佳 checkpoint 只能来自 epoch 1 及以后，训练结束后自动在 holdout 上评估，不会使用旧 E3 的 epoch 0 结果。

训练命令：

```bash
python experiments/gru_e6_a/train.py --device cuda
```

默认输出：

```text
experiments/gru_e6_a/result.json
experiments/gru_e6_a/checkpoints/best.pt
```

只重新评估已保存的训练 checkpoint：

```bash
python experiments/gru_e6_a/train.py --evaluate-holdout --device cuda
```

E6-A 应与 [E3c](../gru_e3/README.md) 比较；两者只应在输入变量/表示上不同，训练切分和评价规则相同。
