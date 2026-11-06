# E14-A：财报字段级缺失掩码

E14-A 在 E10-B 基础上增加 12 个财报字段级存在标记。E10-B 将缺失值和真实零值都编码为 0，E14-A 让模型能够区分这两种情况。

输入财报维度从 16 维变为 28 维：

- 12 个训练集统计量标准化的 signed-log 数值；
- 12 个字段级存在标记；
- 3 个报告类型存在标记；
- 1 个报告年龄 `log1p(age)`。

行情骨干、E3c 初始化、30 天滞后、损失函数、早停和时间切分均与 E10-B 相同。结果不会覆盖 E10-B。

训练并自动评估 holdout：

```bash
python -m experiments.gru_e14_a.train --device cuda
```

结果和检查点：

```text
experiments/gru_e14_a/result.json
experiments/gru_e14_a/checkpoints/best.pt
```

只评估已经保存的检查点：

```bash
python -m experiments.gru_e14_a.train --device cuda --evaluate-holdout
```

只有开发集用于早停和选择检查点；holdout 仅在训练结束后评估。
