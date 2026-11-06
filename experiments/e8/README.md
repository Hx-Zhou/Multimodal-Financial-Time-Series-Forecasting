# E8 —— 只加财报，不加入新闻

## 目的

纯行情骨干 E1-E5、以及 E7 新闻分支都未跑赢 E0（dev MAE 0.2173）。E8 单独测试
财报这一个低频基本面模态是否有效。

## 关键点

- 财报**没有公告日字段**，只有报告期末 `end_date`，因此采用保守可用性延迟：
  报告只有在 `end_date + report_lag_days <= 输入末日` 时才可用，默认 120 天
  （年报最坏情况），用 `--report-lag-days` 做敏感性实验。
- 字段大量缺失（银行无 `money_cap`/`ebit`/`free_cashflow`），靠缺失掩码处理。
- 所有取值特征都是**尺度无关**的（增长率 QoQ + 财务比率），避免银行 vs 小公司
  的原始量纲主导 MLP。

## 架构

```text
z_96（最后一日标准化收盘价）              → Linear(1→24)          → delta_market
22 维财报特征（比率/增长/年龄/掩码）       → MLP(22→64→16) * mask  → fin_z
[fin_z; has_fin]                        → MLP(17→32→24) * mask  → delta_fin

y_hat = z_96 + delta_market + delta_fin
```

`use_fin=False` 在评估时关掉财报分支，作为 R3 财报置零消融。

## 运行

```bash
python -m experiments.e8.e8_financials --epochs 20 --batch-size 256
```

特征缓存到 `experiments/e8/cache/{split}_lag120.npz`（按 lag 分文件）。
`--report-lag-days 60|90|120|180` 做时间对齐敏感性实验。
