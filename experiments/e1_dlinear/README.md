# E1：仅 Close 的残差 DLinear

> **实验定位：线性基线对照与可复现实验记录；冻结增强版在组内留出测试集上有小幅收益，尚不构成显著或普遍优势。**
>
> 原版 DLinear 未超过开发集 E0；增强版在开发集上取得约 0.58% / 0.66% 的 MAE/MSE 改善。2026-10-31 固定既有最终权重，在此前未用于 E1 选择的 15,390 条 holdout 上一次性评估，MAE/MSE 相对 E0 降低 **0.84% / 1.90%**，详见 [HOLDOUT.md](HOLDOUT.md)。这是一个时间段上的小幅改善，不代表统计显著或跨时期稳定优势。本实验供 E2–E5 比较，不代表团队已选择 DLinear 作为最终骨干。

本次交付使用独立分支 `experiment/e1-dlinear`，新增内容全部位于 `experiments/e1_dlinear/`。根目录 `model.py`、`hs300_dataset.py` 和公共验证切分均未修改；合并本 PR 只增加 E1 实验目录，不会自动替换主模型。原版与增强版结果应分别列出，避免把额外预处理、残差收缩和权重平均的收益全部归因于 DLinear 结构。

使用 96 日标准化收盘价预测未来 24 日。代码、结果和本地权重都位于本目录；不修改根目录提交模板、固定读取器或公共切分。原版结果见 [RESULTS.md](RESULTS.md)，中心化与正则化改进见 [OPTIMIZATION.md](OPTIMIZATION.md)。原版没有超过 E0，保留实现和结果供后续对照。

**最终配置见 [REFINEMENT.md](REFINEMENT.md)，留出测试见 [HOLDOUT.md](HOLDOUT.md)**：对中心化 + L2 模型按预测期限收缩残差，再平均三个种子的权重。开发 MAE **0.216052**、MSE **0.130342**；holdout MAE **0.351294**、MSE **0.324072**（同集 E0 为 **0.354286 / 0.330340**，越低越好）。最终本地权重为 `outputs/calibration/model.pt`，推理仍只有一个 DLinear，参数量不变。测试后没有训练、调参或改选候选。

## 实现与协议

- 输入为 `[B, 96]` 的 Close，按每只股票已有的训练期 `close_mean`、`close_std` 标准化；不重新估计验证期统计量。
- 对 96 日序列做复制端点填充的移动平均，得到趋势和剩余项；两个 `Linear(96, 24)` 分别输出残差，最终预测为 `最后一日标准化 Close + 两个残差之和`。
- 共 4,656 个可训练参数。两个输出层零初始化，初始预测等于 E0；随机种子影响训练批次顺序。
- 所有 587,587 条训练样本参与训练，每轮打乱训练样本顺序；训练/验证所属时间段保持固定。
- 复用 `experiments.validation_split.make_validation_split`，只用 42,930 条开发验证样本早停与选配置。18,630 条边界样本排除；15,390 条 holdout 在训练和选择期间不评估，最终固定权重后评估一次，今后不得据其结果继续调参。
- 损失 `MAE + 0.2 * MSE`，Adam，学习率 `1e-3`，无权重衰减，梯度裁剪 1.0，batch size 1,024。
- 最多 40 轮，开发 MAE 至少下降 `1e-5` 才更新最佳权重，连续 6 轮未达到该改进量即停止。每次运行至少训练一轮；E0 单独记录，不把零初始化当成训练后的最佳模型。
- seed 42 初筛窗口 25、15、5；选中窗口 25 后追加 seed 43、44。三次均值包含初筛的 seed 42，不额外挑选最好种子作为最终成绩。
- 额外做了窗口 25、seed 42、学习率 `1e-4` 的训练稳定性检查，保留在 `results/lr_1e-4/`。

默认 `center_last=False`、`weight_decay=0` 保持原版计算方式。改进版显式设置 `center_last=True`：分解前将输入减去最后一日 Close，输出时加回该值；输入仍使用原训练期统计量缩放。正则化使用 Adam 的 `weight_decay`（耦合 L2，作用于权重和偏置），训练损失的报告仍是 `MAE + 0.2 * MSE`。

## 环境与运行

所有命令在**仓库根目录**执行。已验证 Python 3.12.10、PyTorch 2.8.0+cpu、NumPy 2.2.6；完整依赖记录见 `results/environment.txt`。

PowerShell 安装示例（独立环境，不修改系统 Python）：

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r experiments/e1_dlinear/requirements.txt --index-url https://download.pytorch.org/whl/cpu
./.venv/Scripts/python.exe -m unittest experiments.e1_dlinear.test_model -v
```

运行全部数值检查：

```powershell
./.venv/Scripts/python.exe -m unittest experiments.e1_dlinear.test_model experiments.e1_dlinear.test_solver experiments.e1_dlinear.test_calibrate -v
```

`--package-dir` 指向**包含 `data/` 的目录**。数据已经在仓库根目录时使用 `--package-dir .`；本机原始交付包位于整理版仓库根目录的 `../../hs300_train_val`，不需要复制大文件。

复现完整初筛及三种子检查：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e1_dlinear.run --package-dir ../../hs300_train_val --output-dir experiments/e1_dlinear/outputs/reproduction --results-dir experiments/e1_dlinear/results/reproduction
```

复现低学习率对照：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e1_dlinear.run --package-dir ../../hs300_train_val --kernels 25 --seeds 42 --lr 0.0001 --output-dir experiments/e1_dlinear/outputs/reproduction_lr_1e-4 --results-dir experiments/e1_dlinear/results/reproduction_lr_1e-4
```

输出目录中已有同名权重或汇总结果时，脚本会拒绝覆盖；重复运行请更换两个输出目录。

复现第二轮有界优化（窗口固定 25；中心化后比较 L2 系数 0、1e-3、1e-2；seed 42 选配置，再追加 seed 43、44）：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e1_dlinear.optimize --package-dir ../../hs300_train_val --output-dir experiments/e1_dlinear/outputs/reproduction_optimization --results-dir experiments/e1_dlinear/results/reproduction_optimization
```

也可单独运行指定变体，例如：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e1_dlinear.run --package-dir ../../hs300_train_val --kernels 25 --center-last --weight-decay 0.01 --output-dir experiments/e1_dlinear/outputs/custom_centered --results-dir experiments/e1_dlinear/results/custom_centered
```

`optimize.py` 的候选范围和选择规则会在任何候选开始训练前写入 `protocol.json`。原版 `results/` 文件保留；新结果单独位于 `results/optimization/`。代码新增可选参数后源码摘要会变化，但旧配置和旧权重保持兼容；本机也核对了旧权重恢复后的开发指标与原记录一致。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `model.py` | DLinear、`build_model`、支持无标签样本的 `prepare_batch` 和损失函数 |
| `data.py` | 使用固定读取器的行情数组构造窗口；检查切分、标准化、标签和采样接口一致性 |
| `run.py` | 初筛、早停、保存最佳权重、种子复验、分期限和分组指标 |
| `optimize.py` | 中心化与正则化的固定候选对照，选中配置的三个种子复验 |
| `linear_solver.py` / `refine.py` | 解析最小二乘、近期加权与全量稳健目标的对照；该分支没有带来提升 |
| `calibrate.py` | 固定期限收缩曲线的对照及兼容 DLinear 权重平均，生成当前候选 |
| `diagnose.py` | 对指定权重比较训练集和开发验证集的 E0/E1 误差 |
| `evaluate_holdout.py` / `HOLDOUT.md` | 核对冻结权重摘要，一次性比较最终 E1 与 E0 的组内留出测试结果 |
| `test_model.py` | 标准化、分解边界、无标签推理、可学习性、中心化平移性质和旧权重兼容性检查 |
| `results/summary.json` | 主实验的配置选择、各次结果和三种子统计 |
| `results/protocol.json` | 数据数量、日期边界、参数、运行环境和代码/数据摘要 |
| `results/k*_seed*.json` | 每轮训练损失、开发 MAE/MSE、最佳轮次和耗时 |
| `results/*_per_stock.csv` | E0 和选中窗口三个种子的逐股票指标 |
| `results/news_coverage_groups.json` | 按输入窗口新闻记录数划分的指标；新闻不作为 E1 输入 |
| `results/optimization/` | 第二轮中心化与正则化对照的协议、逐轮日志和汇总 |
| `results/refinement/` | 稳定求解与近期加权的完整正负结果 |
| `results/calibration/` | 当前候选的曲线选择、各个种子、平均模型及分时间段指标 |
| `results/holdout/` | 留出测试的预先固定协议、总体/分期限指标、逐股票与新闻覆盖分组结果 |
| `outputs/` | 本地最佳权重、开发预测和运行过程记录，已被现有 `.gitignore` 排除 |

数据处理直接复用固定读取器载入的行情数组，以减少逐样本解析财报的开销；仍通过真实样本抽查与 `prepare_batch` 完全一致，并重算 E0 验证尺度。新闻覆盖分组统计整个输入日期范围内的记录数，包含非交易日和重复记录，仅用于评估分组。

实验权重是包含 `config`、`state_dict` 的检查点。恢复时使用保存的配置，不依赖默认窗口：

```python
import torch
from experiments.e1_dlinear.model import build_model, prepare_batch

saved = torch.load("experiments/e1_dlinear/outputs/calibration/model.pt", map_location="cpu", weights_only=True)
model = build_model(saved["config"])
model.load_state_dict(saved["state_dict"])
model.eval()
with torch.inference_mode():
    prediction = model(prepare_batch(samples))  # samples 可以没有 y
```

这份权重格式供实验恢复使用；最终课程提交需另行按根目录 README 整理为提交接口与纯 `state_dict`，并处理最终模型的模态要求。

当前候选的收缩规则已折入权重，无需在推理时再乘期限系数。详细复现流程与未获提升的实验见 [REFINEMENT.md](REFINEMENT.md)。

## 给 E2–E5 的交接

E2 可使用相同的 Close 标准化、96→24 残差输出、开发验证索引和误差计算，替换为 GRU。E3 相对 E2 只扩展行情输入；E4 与 E2 比较；E5 与 E3 比较。本目录无需被这些实验导入，公共切分仍使用原文件。后续所有模型继续同时比较 E0，不能只因优于 E1 就认定有效。
