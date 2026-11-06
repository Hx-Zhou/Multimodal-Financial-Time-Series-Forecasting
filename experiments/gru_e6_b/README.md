# E6-B：平稳化行情特征对照

E6-B 固定 E3c/E6-A 使用的单层 GRU、隐藏维度、训练参数、时间切分和 checkpoint 选择规则，只改变输入表示。每个 96 日窗口转换为 7 个特征：

```text
1. log return：log(Close[t] / Close[t-1])，窗口第一天设为 0
2. amplitude：(High - Low) / abs(Close)
3. candle body：(Close - Open) / abs(Open)
4. log volume：log1p(Volume)，按当前窗口 z-score
5. dividend event：Dividends != 0
6. split event：Stock Splits != 0
7. standardized close：(Close - close_mean) / close_std
```

前三个价格特征和 log volume 主要表达变化率与波动，最后保留一个必要的标准化价格水平，分红和拆股只作为事件指示。模型预测标准化 Close 的残差，最后标准化 Close 作为基线单独传入模型。

训练完成后会自动在开发集选择 epoch >= 1 的最佳权重，再在 holdout 上预测；holdout 不参与模型选择。

```bash
python experiments/gru_e6_b/train.py --device cuda
```

默认输出：

```text
experiments/gru_e6_b/result.json
experiments/gru_e6_b/checkpoints/best.pt
```

重新评估 checkpoint：

```bash
python experiments/gru_e6_b/train.py --evaluate-holdout --device cuda
```

应将 E6-B 与 [E3c](../gru_e3/README.md) 和 [E6-A](../gru_e6_a/README.md) 比较。
