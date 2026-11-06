# E8：DLinear 行情＋财报

本目录独立保存 E8 代码、记录和结果。行情来自已由开发集选择的 E6 stationary 三个种子，固定数值权重，只加入财报，不使用新闻输入。

**已完成结果：**三种子等权预测的开发 MAE **0.218715817**，内部测试 MAE **0.350379127**。E6 测试 MAE 为 **0.350345190**，本次 E8 略差约 **0.0097%**，没有实质收益。财报数值置零、时间打乱和延迟检查均未改变指标，不能声称财报内容有效。完整记录见 [RESULTS.md](RESULTS.md)。

## 结构

已达到可用日期的最新报表 → 12 个关键字段＋12 个同季度同比变化＋缺失掩码＋报告年龄／存在掩码（共 54 维）→ 54→16→8 的 SiLU MLP → 24 维零初始化残差头 → 小门控 → E6 预测修正。无财报时残差严格为零。

关键字段：

- 利润表：total_revenue、n_income_attr_p、basic_eps、operate_profit。
- 资产负债表：total_assets、total_liab、total_hldr_eqy_exc_min_int、money_cap。
- 现金流量表：n_cashflow_act、n_cashflow_inv_act、n_cash_flows_fnc_act、c_cash_equ_end_period。

没有公告日期，主设置固定期末＋180 天，冻结权重后另做 270 天延迟检查。源数据缺少原始发布版本，结果带有历史版本不确定性，不能声称严格无财报信息泄漏。

## 运行与记录

仓库根目录：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e8_dlinear_reports.run --package-dir ../../hs300_train_val --evaluate-holdout
```

已有运行需用 `--resume` 继续，保留原记录。全部实验的统一协议及恢复规则见 [公共工具说明](../dlinear_modal_common/README.md)。

- `results/protocol.json`：训练前固定的设置及代码／骨干哈希。
- `results/data_protocol.json`：数据划分、训练统计量和财报时间审计。
- `results/seed*.json`：每个种子的完整训练曲线和开发集最佳轮次。
- `results/development.json`：开发集等权平均、置零／打乱／270 天诊断。
- `results/holdout_protocol.json`、`results/holdout.json`：权重冻结记录及内部测试结果。
- `results/*_per_stock.csv`：各股票与预测期限误差。
- `outputs/`：本地权重、恢复状态与预测数组，不纳入 Git。

结果见 [RESULTS.md](RESULTS.md)。主要对照为匹配的 E6。
