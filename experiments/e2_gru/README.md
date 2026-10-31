# E2：单变量 GRU

## 设计

E2 只使用过去 96 个交易日的标准化 `Close`，不读取新闻、财报或其他行情字段。单层单向 GRU（隐藏维度 64）读取 96 天序列，最后一个隐藏状态经过线性层直接输出未来 24 天的变化量：

```text
标准化 Close [B, 96, 1]
        ↓
单层 GRU
        ↓
最后隐藏状态 [B, 64]
        ↓
线性层 [B, 24] 预测残差
        ↓
最后一天 Close + 残差 = 预测 [B, 24]
```

训练损失使用 `MAE + 0.2 × MSE`，并进行梯度裁剪。残差头零初始化，因此模型的初始输出严格等于 E0 的最后值基线，训练只需学习是否加入有效修正。训练只使用 `data/train.npz`；每个 epoch 只在开发验证集上早停和选择最佳权重。留出测试集在最佳权重确定后评估。

训练脚本会预先构建标准化 Close 数组，只为 E2 读取行情和标签，不在每个 batch 创建新闻、财报对象；这可以避免纯行情实验被多模态读取器的 Python 开销拖慢。

## 运行

当前环境需要先安装 PyTorch。安装后从仓库根目录运行：

```bash
python -m experiments.e2_gru.train
```

结果写入 `experiments/e2_gru/results.json`，权重写入 `experiments/e2_gru/checkpoints/best.pt`。根目录 `model.py` 不会被修改。
