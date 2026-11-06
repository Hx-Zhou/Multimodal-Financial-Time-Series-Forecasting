# E3c-E7：E3c 相对 OHLC + 新闻

E7 在当前最佳纯行情骨干 E3c 上加入新闻分支。过去 96 个交易日的 OHLC 仍按 E3c 方式相对最后 Close 表示；每天的新闻 embedding 先做均值池化，再经过 `512 -> 16` 的可学习投影。模型同时接收：

- `news_vector`：当天新闻的均值向量，无新闻时全零；
- `has_news`：当天是否有新闻；
- `log1p(news_count)`：当天新闻数量。

新闻分支只使用输入窗口内的新闻，不读取未来日期。新闻投影经过存在掩码，避免无新闻日的线性层偏置产生伪新闻信号。epoch 0 的残差头为零，因此仍严格等于 E0 最后值基线。

训练完成后按开发集 MAE 选择 epoch >= 1 的 checkpoint，并自动在 holdout 上评估。

```bash
python experiments/gru_e7/train.py --device cuda
```

输出：

```text
experiments/gru_e7/result.json
experiments/gru_e7/checkpoints/best.pt
```

单独评估已保存权重：

```bash
python experiments/gru_e7/train.py --evaluate-holdout --device cuda
```
