# E7 —— 只加新闻（TaTS 风格，无 GPT-2）

## 目的

纯行情骨干 E1-E5 都未跑赢 E0（dev MAE 0.2173），所以市场分支刻意做到最简
（last-value + 一个小残差头），把容量全给新闻分支。E7 只回答一个问题：
**加了新闻能否把 dev MAE 压到 0.2173 以下。**

## 架构

```text
z_96（最后一日标准化收盘价）            → Linear(1→24)          → delta_market
512 维新闻（去重 + 掩码均值池化）        → MLP(512→128→8) * mask → news_z
[news_z; has_news; log(1+count)]        → MLP(10→32→24) * mask  → delta_news

y_hat = z_96 + delta_market + delta_news
```

- `has_news` 把新闻残差整体置零：无新闻样本自动退化为纯市场头。
- `use_news=False` 在评估时关掉新闻分支，作为 R1 新闻置零消融。

## 运行

```bash
python -m experiments.e7.e7_news --epochs 20 --batch-size 256
```

首次运行会提取并缓存特征到 `experiments/e7/cache/`（train 约 58.8 万样本，
提取较慢，之后复用）。`--no-cache` 强制重提取，`--limit-train N` 用于冒烟测试。
