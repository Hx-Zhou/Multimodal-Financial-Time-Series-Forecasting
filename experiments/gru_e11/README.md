# E11：财报比例与变化特征

E11 在 E10-B 的 30 天报告滞后和训练集标准化基础上扩展财报输入，仍然冻结 E3c 行情分支并使用财报 GRU。财报特征包括：

- 12 个 signed-log 原始金额类字段；
- 7 个规模调整比例，例如净利润/总资产、总负债/总资产；
- 12 个相邻报告期的 signed-log 变化量；
- 3 个报告类型存在标记；
- 报告年龄 `log1p(age)`。

前 31 个数值、比例和变化字段只用训练集统计量标准化，报告存在标记和年龄不标准化。

训练：

```bash
python -m experiments.gru_e11.train --device cuda --report-lag-days 30
```

结果和检查点：

```text
experiments/gru_e11/result.json
experiments/gru_e11/checkpoints/best.pt
```
