# E10-A：财报时间对齐

E10-A 沿用 E9-2 的冻结 E3c 财报适配器结构，修改财报的可用时间规则，并修正报告年龄的日历计算。
默认要求报告期末至少早于行情输入结束日 30 个自然日：

```text
report_end_date <= input_end - 30 days
```

数据中的 JSONL 记录没有明确的公告日期，因此这只是可用性的近似，不能保证报告当时已经公开。报告年龄也使用真实日历日计算，避免跨月时直接相减造成错误。因此与 E9-2 的比较并非只改变滞后天数的严格消融。训练和早停协议与 E9-2 相同，holdout 只在模型确定后评估。

训练：

```bash
python -m experiments.gru_e10_a.train --device cuda --report-lag-days 30
```

结果和检查点：

```text
experiments/gru_e10_a/result.json
experiments/gru_e10_a/checkpoints/best.pt
```

可以使用 `--report-lag-days 45` 或 `--report-lag-days 60` 做后续敏感性实验，但应先完成 30 天版本。
