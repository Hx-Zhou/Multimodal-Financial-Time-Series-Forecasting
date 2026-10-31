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

| 运行 | lr / weight_decay | dev MAE | dev MSE | holdout MAE | holdout MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 初始 | 1e-3 / 0 | 0.2377 | 0.1631 | 0.3941 | 0.4251 |
| 调参 | 1e-4 / 1e-3 | 0.2383 | 0.1645 | 0.3787 | 0.4023 |

E0 最后值基线：dev MAE 0.2173 / MSE 0.1312。

**结论**：两次运行 dev MAE 都稳定在约 0.238，明显差于 E0；降低学习率并加权重衰减后无改善。dev MAE 在 0.238–0.262 之间震荡、train_loss 持续下降而验证集不涨，说明模型在拟合训练集噪声、却无法从 96 天 Close 中提取可泛化的未来 24 日信号。初步判断：PatchTST（仅 Close、本配置）不适配本任务，不建议作为骨干 B* 候选；后续可考虑改用全部行情字段（归 E5）、或加入 SpecTF 式窗口实例归一化后再试。
