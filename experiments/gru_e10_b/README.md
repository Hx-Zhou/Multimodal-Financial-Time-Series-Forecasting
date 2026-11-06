# E10-B：滞后财报的训练集标准化

E10-B 保留 E10-A 的 30 天财报可用滞后，同时只改变财报特征处理：12 个金额类字段的 signed-log 值使用训练集报告快照计算均值和标准差，再应用到训练集和验证集。三个报告存在标记和报告年龄保持独立，避免把缺失状态混入金额标准化。

训练集统计量会写入 checkpoint，holdout 评估时复用同一组统计量，不使用验证集信息。

训练：

```bash
python -m experiments.gru_e10_b.train --device cuda --report-lag-days 30
```

结果和检查点：

```text
experiments/gru_e10_b/result.json
experiments/gru_e10_b/checkpoints/best.pt
```
