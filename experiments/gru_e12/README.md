# E12：E10-B 新闻增量

E12 冻结 E10-B 的行情和财报分支，只训练新闻适配器与 `news_gate`。新闻分支沿用 E9-1 的每日均值/最大池化、投影和时间 GRU，并将新闻门控初始化为 0，因此 epoch 0 严格等于 E10-B。

训练：

```bash
python -m experiments.gru_e12.train --device cuda
```

结果和检查点：

```text
experiments/gru_e12/result.json
experiments/gru_e12/checkpoints/best.pt
```
