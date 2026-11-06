# E10-C：单次财报快照 MLP

E10-C 保留 E10-B 的 30 天财报可用滞后和训练集统计量标准化，只改变财报编码器。E10-B 把同一份静态财报复制到 96 个时间步再经过 GRU；E10-C 直接将最新财报快照 `[B, 16]` 经过 MLP 编码，再与冻结 E3c 的行情隐藏状态融合。

```text
market_hidden [B, 64] + financial_repr [B, 32]
    -> fusion head
    -> 24 步财报残差
```

训练：

```bash
python -m experiments.gru_e10_c.train --device cuda --report-lag-days 30
```

结果和检查点：

```text
experiments/gru_e10_c/result.json
experiments/gru_e10_c/checkpoints/best.pt
```
