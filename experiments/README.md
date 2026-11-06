# 实验目录与评价约定

已完成的 GRU 系列对照、指标、架构和结论集中在 **[GRU 实验记录](../docs/gru_experiments.md)**。本页用于查找代码入口和理解数据切分。

用于组会或阶段汇报的精简叙述见 **[GRU 实验路径与结果汇报](../docs/gru_report.md)**。

## 仓库分工

| 位置 | 用途 |
| --- | --- |
| [根 README](../README.md) | 原始任务、数据接口、最终提交要求 |
| [docs/intro.md](../docs/intro.md) | 数据规模、原始数据类型与形状、预测目标 |
| [docs/experiment_route.md](../docs/experiment_route.md) | 最初实验计划；实际编号和进展以实验记录为准 |
| [docs/gru_experiments.md](../docs/gru_experiments.md) | 已完成 GRU 系列的对照与结果汇总 |
| `experiments/<实验>/` | 各实验的实现、运行说明、结果与检查点 |
| `data/` | 原始数据和样本索引，已被 `.gitignore` 排除 |

根目录 `model.py` 与 `model.pt` 是当前可提交的 E10-B 候选。探索实验不直接覆盖根模型；根模型使用相对 OHLC 行情和滞后、标准化财报特征，新闻不参与预测。

## 实验索引

下表命令均从仓库根目录运行。GPU 训练指定 `--device cuda`；重新运行单模型通常会更新默认结果及权重，保留多次运行时请指定独立输出路径，详见各目录说明。

| 阶段 | 实验与说明 | 主要入口 |
| --- | --- | --- |
| 无训练参照 | [E0 最后值与移动平均](gru_e0/README.md) | 运行方法见目录说明 |
| 独立线性路线 | [E1 DLinear](e1_dlinear/README.md) | [结果与分析](e1_dlinear/RESULTS.md) |
| 行情 GRU | [E2 单变量 Close](gru_e2/README.md) | `python -m experiments.gru_e2.train --device cuda` |
| 行情 GRU | [E3a/b/c/d 输入与目标对照](gru_e3/README.md)，默认 E3c | `python -m experiments.gru_e3.train --device cuda`，用 `--variant` 切换 |
| 行情特征 | [E6-A 七字段](gru_e6_a/README.md) | `python -m experiments.gru_e6_a.train --device cuda` |
| 行情特征 | [E6-B 平稳化特征](gru_e6_b/README.md) | `python -m experiments.gru_e6_b.train --device cuda` |
| 联合训练融合 | [E7 行情＋新闻](gru_e7/README.md) | `python -m experiments.gru_e7.train --device cuda` |
| 联合训练融合 | [E8 行情＋财报](gru_e8/README.md) | `python -m experiments.gru_e8.train --device cuda` |
| 冻结行情 | [E9-1 新闻适配器](gru_e9_1/README.md) | `python -m experiments.gru_e9_1.train --device cuda` |
| 冻结行情 | [E9-2 财报适配器](gru_e9_2/README.md) | `python -m experiments.gru_e9_2.train --device cuda` |
| 冻结行情 | [E9-3 新闻＋财报适配器](gru_e9_3/README.md) | `python -m experiments.gru_e9_3.train --device cuda` |
| 财报处理 | [E10-A 30 天滞后与日历年龄](gru_e10_a/README.md) | `python -m experiments.gru_e10_a.train --device cuda --report-lag-days 30` |
| 财报处理 | [E10-B 训练集标准化](gru_e10_b/README.md)，当前财报主线候选 | `python -m experiments.gru_e10_b.train --device cuda --report-lag-days 30` |
| 财报编码 | [E10-C 单次快照 MLP](gru_e10_c/README.md) | `python -m experiments.gru_e10_c.train --device cuda --report-lag-days 30` |
| 固定融合 | [E10-D E3c 与 E10-B 加权](gru_e10_d/README.md) | `python -m experiments.gru_e10_d.blend --device cuda` |
| 财报特征 | [E11 比例与变化特征](gru_e11/README.md) | `python -m experiments.gru_e11.train --device cuda --report-lag-days 30` |
| 新闻增量 | [E12 冻结 E10-B 加新闻](gru_e12/README.md) | `python -m experiments.gru_e12.train --device cuda` |
| 新闻诊断 | [E13 时间归属、去重与线性残差](gru_e13/README.md) | `python -m experiments.gru_e13.train --device cuda` |
| 财报缺失 | [E14-A 字段级掩码](gru_e14_a/README.md) | `python -m experiments.gru_e14_a.train --device cuda` |
| 滞后敏感性 | [E14-B 30/60/90 天](gru_e14_b/README.md) | `python -m experiments.gru_e14_b.run --device cuda` |
| 多种子对照 | [E15 E10-B 与 E14-A](gru_e15/README.md) | `python -m experiments.gru_e15.run --device cuda` |

E3 历史变体在 `gru_e3/archive/`；E14-B、E15 的简明结果看各自 `summary.md`，单模型一般看 `result.json`。保留历史结果，不将旧 epoch 0 的基线分数算成训练后的 GRU 成绩。

## 固定检查点的分时段评估

`gru_validation/` 中的分段评估脚本实际是在三个开发时间段评价同一个检查点，没有逐段重新训练。它们用于观察时间段差异，不是独立的 walk-forward 验证，也不读取 Holdout。

| 对照 | 命令 | 结果 |
| --- | --- | --- |
| E0、E3c、E9-2 | `python -m experiments.gru_validation.rolling_validation --device cuda` | [结果](gru_validation/rolling_validation.json) |
| E0、E3c、E10-B/C | `python -m experiments.gru_validation.rolling_validation_e10 --device cuda` | [结果](gru_validation/rolling_validation_e10.json) |
| E0、E3c、E10-B、E11 | `python -m experiments.gru_validation.rolling_validation_e11 --device cuda` | [结果](gru_validation/rolling_validation_e11.json) |

E13、E15 另在各自目录保存 `development_windows.json`。各段按预测起始日划分样本，但样本的历史输入和未来标签可能重叠，解释范围见 [GRU 实验记录](../docs/gru_experiments.md)。

## 固定时间切分

[validation_split.json](validation_split.json) 记录当前切分：[validation_split.py](validation_split.py) 从官方 `val.npz` 的预测起始日期构造索引。最晚 20% 的不同预测起始日期作为 Holdout；开发样本的未来 24 日标签必须全部早于 Holdout 首个预测日，否则排除。

| 部分 | 样本数 | 预测起始日期 |
| --- | ---: | --- |
| 训练集 | 587,587 | 使用给定 `train.npz` |
| 开发验证 | 42,930 | 2023-10-09～2023-12-20 |
| 边界排除 | 18,630 | 2023-12-21～2024-01-23 |
| Holdout | 15,390 | 2024-01-24～2024-02-27 |

```text
data/train.npz       训练参数、拟合预处理统计量
data/val.npz 较早段  开发验证：调参、早停、选择权重
data/val.npz 边界段  排除：未来标签覆盖 Holdout 时期
data/val.npz 后段    本地 Holdout：训练结束后报告
```

开发标签最晚到 2024-01-23。这里隔离的是未来标签日期，两个部分的历史输入允许重叠；表中日期是预测起始日，不是各部分所有标签的完整范围。

Holdout 有标签，且已经在研究过程中反复查看，当前应作为辅助证据。它不参与训练循环的早停，也不是官方隐藏测试集；后续仍优先按开发指标与预设稳定性对照作决策。

读取索引示例：

```python
from hs300_dataset import HS300Dataset
from experiments.validation_split import batches_for_indices, make_validation_split

val = HS300Dataset(".", "val")
split = make_validation_split(val, holdout_fraction=0.2)

for samples in batches_for_indices(val, split.dev_indices, batch_size=64):
    # 用于早停和调参
    pass

for samples in batches_for_indices(val, split.holdout_indices, batch_size=64):
    # 检查点确定后报告，不用于选择轮次
    pass
```
