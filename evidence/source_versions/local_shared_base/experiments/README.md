# 实验数据切分

## 模型代码管理

根目录的 `model.py` 保留为原始提交模板，不在探索实验时直接覆盖。每个模型实验在自己的子目录保存实现、训练脚本和结果，例如 `experiments/e1_gru/`。确定最终方案后，再将选定实现整理到根目录 `model.py`，满足评分程序的加载要求。

`validation_split.json` 固定了后续实验使用的时间切分：官方 `val.npz` 最晚 20% 的预测起始日期作为留出测试集。开发验证样本还需满足其 24 日预测标签全部早于测试集首个预测日期；标签会与测试期重叠的边界样本排除，不参与开发验证或测试。

```text
data/train.npz       训练模型参数
data/val.npz 较早段  开发验证：调参、早停、选择权重
data/val.npz 边界段  排除：预测标签与留出测试期重叠
data/val.npz 后段    留出测试：模型确定后再评估
```

留出测试集仍然包含标签，因此它是本地 holdout，不是官方隐藏测试集。反复根据 holdout 分数修改模型，会让它重新变成验证集；正式比较时应只用开发验证集做决策，最后在 holdout 上报告结果。

后续训练代码可以这样取得两个索引集合：

```python
from hs300_dataset import HS300Dataset
from experiments.validation_split import batches_for_indices, make_validation_split

val = HS300Dataset(".", "val")
split = make_validation_split(val, holdout_fraction=0.2)

for samples in batches_for_indices(val, split.dev_indices, batch_size=64):
    # 只用于早停和调参
    pass

for samples in batches_for_indices(val, split.holdout_indices, batch_size=64):
    # 只在模型和超参数确定后评估
    pass
```
