# E4 PatchTST

数值骨干探索：用过去 96 个交易日的**标准化 Close** 预测未来 24 个交易日，模型采用 PatchTST，残差输出。

## 配置（来自 `docs/experiment_route.md` §4）

| 项 | 值 |
| --- | --- |
| 输入 | 标准化 Close `[96]` |
| patch_len / stride | 16 / 8 |
| 层数 / d_model / heads / d_ff | 2 / 128 / 4 / 256 |
| dropout | 0.1 |
| 输出 | 残差 `ŷ_h = z_96 + Δ_h`，直接多步 |
| 损失 | `MAE + 0.2 × MSE` |
| 优化器 | AdamW，lr=1e-3，梯度裁剪 1.0 |

## 运行

```bash
python -m experiments.e4.e4_patchtst --epochs 20 --batch-size 128
```

快速冒烟（仅 2000 条训练样本、1 epoch）：

```bash
python -m experiments.e4.e4_patchtst --limit-train 2000 --epochs 1 --batch-size 64
```

结果写到 `experiments/e4/e4_val_results.json`，最佳权重保存为 `experiments/e4/e4_best.pt`（被 `.gitignore` 忽略）。

## 参照

- 开发验证集 E0 最后值基线：MAE 0.2173 / MSE 0.1312（`experiments/e0/`）。
- E4 与 E2（仅 Close 的 GRU）对比，两者输入一致；须稳定超过 E0。

## 结果（seed=0）

| 指标 | dev | holdout |
| --- | ---: | ---: |
| MAE | 0.2377 | 0.3941 |
| MSE | 0.1631 | 0.4251 |

最优 epoch=8（第 13 个 epoch 早停），参数量 305,560。

**结论**：该初始配置下 E4 未超过 E0 最后值基线（dev MAE 0.2173 / MSE 0.1312）。dev MAE 在 0.238–0.261 之间震荡、train_loss 仅从 0.340 缓慢降至 0.322，呈现过拟合特征（拟合了训练集，却未在验证集上超过“预测=最后值”）。下一步应降低学习率、加权重衰减或更强正则后复测；若仍不达标，可作为“PatchTST 初始配置不适配本任务”的初步证据上报。
