# E15：E10-B 与 E14-A 多随机种子对照

比较原始 16 维财报输入与增加 12 个字段存在标记的 28 维输入。预先固定种子 **42、43、44**，报告全部结果及相同种子下的差值，不按 Holdout 选最好种子。

```bash
python -m experiments.gru_e15.run --device cuda
```

默认复用两种模型已有的 seed 42 结果，并复制对应检查点到本目录；新增训练各跑 seed 43 和 44，共 **4 次训练**。训练依次进行，单卡运行。每次训练仍为最多 20 轮、patience 6、batch size 256、学习率 3e-4、weight decay 1e-5、财报滞后 30 天。每次训练自动加载开发集 MAE 最低的训练后检查点，评估 Holdout。

**变化范围**：仅重训财报分支，随机种子改变财报分支的初始化和训练样本顺序。所有实验沿用同一个冻结的 E3c 行情检查点，所以本实验不评估行情骨干重新训练的随机性。同一种子在不同输入维度模型中不意味着各层初始权重完全相同。

训练、开发、Holdout 的划分、金额特征统计量计算、损失函数 `MAE + 0.2 MSE` 和早停规则与原实验一致。E10-B、E14-A 的原始结果与检查点不被覆盖。

## 输出与续跑

```text
experiments/gru_e15/
  summary.md                       # 简明均值、样本标准差、完成状态
  summary.json                     # 各种子、两模型同种子差值和胜出次数
  development_windows.json         # 开发集三个时间段的逐种子结果和均值/标准差
  progress.json                    # 当前运行阶段，成功结束后删除
  gru_e10_b/seed_42/...                 # 复制的现有结果和检查点
  gru_e10_b/seed_43/...                 # 新训练
  gru_e10_b/seed_44/...
  gru_e14_a/seed_42/...
  gru_e14_a/seed_43/...
  gru_e14_a/seed_44/...
```

每次训练单独保存 `result.json`、`checkpoints/best.pt`、`train.log`、`manifest.json` 和 `development_windows.json`。原训练循环的逐轮信息仍在该种子的结果文件中，主汇总不重复列出。

中断后再次执行同一命令，会跳过已经完整保存且通过检查的训练；缺失的分时段评估会补齐。**不会恢复未完成训练的优化器状态**：中途打断的某一个种子会从头开始训练。

检查会核对 E3c 冻结权重、训练/验证切分文件的 SHA256、对照参数、最佳轮次以及结果和检查点的 SHA256。更改对照设置请使用其他输出目录，避免混合结果。这不包含所有原始行情/财报文件的完整校验，实验期间应保持 `data/` 内容不变。

历史 seed 42 结果未记录 weight decay、最大轮数和 patience，复用时以其代码和文档默认值为依据，并在 manifest 中明确标注。核心架构、实际种子、学习率、batch、数据切分比例、报告滞后和冻结行情权重会核对。若不接受此历史元信息限制，可单独重新运行全部六次训练：

```bash
python -m experiments.gru_e15.run --device cuda --no-reuse-seed42 --output-dir experiments/gru_e15_fresh
```

只整理现有结果（不训练、不做模型推理）：

```bash
python -m experiments.gru_e15.run --summarize-only
```

只补齐已经完成训练的检查点的开发分时段评估：

```bash
python -m experiments.gru_e15.run --evaluate-existing --device cuda
```

其他预先指定的种子组可以使用 `--seeds`，如 `--seeds 42 43 44 45 46`。不要根据已看到的 Holdout 表现挑选种子组。

## 如何判断

- 以开发集均值、样本标准差和同种子的配对差值为主，`E14-A - E10-B < 0` 表示 E14-A 更好。
- 同时查看三个开发时间段是否一致，不只依靠总均值。
- 标准差使用 `ddof=1`；只有一个种子时为 `null`，不以 0 伪装为稳定。
- 三个开发时间段是固定检查点的时间分段诊断，不是独立滚动重训练。
- Holdout 仅报告所有预设种子的指标，不用于模型轮次或种子选择；该 Holdout 已多次查看，应作为辅助证据。
- 三个种子只能初步观察训练随机性，不能证明统计显著性，也不消除跨市场时间段的不确定性。

当前机器若没有 CUDA，可先运行 `--summarize-only`，在具备 GPU 的机器上执行正式命令。训练日志实时显示模型、种子及每轮进度。
