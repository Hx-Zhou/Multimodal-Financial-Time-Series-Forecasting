# E10-D：E3c 与 E10-B 固定比例融合

E10-D 不重新训练模型，只在开发集上测试固定融合比例：

```text
prediction = (1 - w) * E3c + w * E10-B
```

权重由开发集合计 MAE 选择，三个滚动开发窗口用于检查稳定性。holdout 不参与权重选择，只在权重确定后评估一次。

运行：

```bash
python -m experiments.gru_e10_d.blend --device cuda
```

结果：

```text
experiments/gru_e10_d/result.json
```
