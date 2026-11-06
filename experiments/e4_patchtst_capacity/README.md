# E4 后续：选择规则、容量与正则化

本目录独立保存PatchTST新阶段，源码复用原E4架构，旧实验不覆盖。

| 编号 | 本步改变 |
|---|---|
| P0_selector | 原128维模型，固定残差×0.5，用校准后的验证MAE早停 |
| P1_width64 | 相对P0只将d_model改64；2层、4头、FF256等不变 |
| P2_dropout | 取P0/P1较好配置，只将dropout改0.3 |
| L_close | 相同数据、处理、优化器、校准与选择规则的DLinear参照 |

原8/3预算基线的含义、父配置选择、三种子复验与停止规则详见 [共享计划](../transformer_optimization_common/PLAN.md)。P0采用本轮8轮／patience3预算，历史E4为12／4，不能只把历史差异归因于早停指标。

仅使用Close；96天输入、24天输出，patch16/stride8，位置编码、pre-LN、GELU。训练仍用未收缩输出；验证／推理固定×0.5，只应用一次。检查点存原始权重与alpha元数据。

从仓库根目录运行共享入口：

```powershell
./.venv/Scripts/python.exe -u -m experiments.transformer_optimization_common.run --package-dir ../../hs300_train_val
```

本目录`results/<编号>/seed*.json`和本地`outputs/<编号>/`各自保存曲线、权重与预测；跨阶段决策在共享目录`results/`。结论见`RESULTS.md`。不安装新环境，不复制数据。
