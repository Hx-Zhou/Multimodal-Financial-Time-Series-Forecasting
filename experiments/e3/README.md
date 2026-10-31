# E3：行情 GRU 实验

原来的 E3、E3a～E3d 已合并到本目录，模型和训练循环只保留一份。默认方案是 **E3c：相对 OHLC 输入，预测标准化 Close 残差**。其他方案作为对照保存在 `archive/`。

## 文件结构

```text
experiments/e3/
├── model.py                 # 所有变体共用的 GRU 和输入转换
├── train.py                 # 统一训练和评估入口，默认 E3c
├── result.json              # 当前 E3c 的完整结果及逐轮历史
├── checkpoints/best.pt      # 当前 E3c 的最佳训练权重
└── archive/
    ├── e3a/                # OHLC，按股票训练期统计量标准化
    ├── e3b/                # E3a 的输出残差乘以 0.1
    ├── e3d/                # 相对 OHLC，训练目标为相对收益
    ├── e3_all/             # 原七字段模型的 epoch 0 权重
    └── legacy/             # 原七字段结果、迁移前快照及单次评估记录
```

`e3a/`、`e3b/`、`e3d/` 各自保留 `result.json` 和 `checkpoints/best.pt`，结果和权重原样迁移。旧的独立入口目录已移除，后续统一使用以下命令。

## 默认方案 E3c

输入为 `[B, 96, 4]` 的 OHLC，每个窗口按最后一天的原始 Close 做相对化：

```text
relative_ohlc[t] = raw_ohlc[t] / last_raw_close - 1
prediction = last_standardized_close + GRU_residual
```

最后标准化 Close 独立传给模型；它使用该股票训练期的 `close_mean` 和 `close_std`。单层 GRU 隐藏维度为 64，线性头输出未来 24 个交易日的残差。损失为 `MAE + 0.2 × MSE`。

训练命令：

```bash
python -m experiments.e3.train --device cuda
```

默认写入 `result.json` 和 `checkpoints/best.pt`。epoch 0 只作为 E0 参照；仅在 epoch ≥ 1 中按开发集 MAE 选择权重。训练结束后重新加载最佳训练权重，自动在 holdout 上预测一次，将开发集、holdout 指标和训练历史合并写入 `result.json`。临时文件 `result.progress.json` 在成功结束后删除。

仅重新评估现有 E3c 权重：

```bash
python -m experiments.e3.train --evaluate-holdout --device cuda
```

单独评估写入 `holdout_check.json`，不会替换含训练历史的 `result.json`。

## 对照方案复现

```bash
python -m experiments.e3.train --variant e3a --device cuda
python -m experiments.e3.train --variant e3b --device cuda
python -m experiments.e3.train --variant e3d --device cuda
python -m experiments.e3.train --variant e3_all --device cuda
```

对照方案默认读写 `archive/<variant>/result.json` 和 `archive/<variant>/checkpoints/best.pt`。重新运行会更新该方案已有结果和权重；需要保留多次运行时，用 `--output` 和 `--checkpoint` 指定新路径。`e3_all` 自动使用七字段，其他方案自动使用 OHLC；评估时模型配置从 checkpoint 读取。

## 已有结果

| 方案 | 最佳训练轮次 | 开发 MAE | 开发 MSE | Holdout MAE | Holdout MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| E0 最后值 | — | 0.217307 | 0.131206 | 0.354286 | 0.330340 |
| E3a | 1 | 0.242969 | 0.168926 | 0.399845 | 0.440116 |
| E3b | 1 | 0.243162 | 0.158174 | 0.384907 | 0.387999 |
| **E3c（当前默认）** | **2** | **0.213222** | **0.128779** | **0.358988** | **0.336930** |
| E3d | 2 | 0.212324 | 0.129458 | 0.363267 | 0.341936 |

E3c 的 holdout 指标优于其他已训练 E3 变体，但仍略差于 E0。E3d 的开发 MAE 更低，因此不能称 E3c 在所有指标上最优。原七字段实验曾保存 epoch 0，记录实质为 E0，不能作为训练后的七变量模型表现。

数据沿用固定的时间切分和标签重叠排除规则，详见 [实验说明](../README.md)。本次合并只调整目录、默认入口和文档，没有重新训练或更改历史指标。
