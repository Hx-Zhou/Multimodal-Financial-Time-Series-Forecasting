# E9-1：冻结 E3c 的新闻适配器

E9-1 固定 E3c 的相对 OHLC GRU 和预测头，只训练新闻分支。已有 512 维文章向量按天同时做均值池化和最大池化，形成 1024 维日级新闻特征，再经过投影和时间 GRU。新闻分支通过逐预测步门控残差接入，门控初值为 0，因此 epoch 0 严格等于 E3c。

训练：

```bash
python -m experiments.gru_e9_1.train --device cuda
```

结果和权重：

```text
experiments/gru_e9_1/result.json
experiments/gru_e9_1/checkpoints/best.pt
```

holdout 只在训练确定后运行：

```bash
python -m experiments.gru_e9_1.train --evaluate-holdout --device cuda
```
