# E9：DLinear 行情＋新闻＋财报

E9 的代码、权重和指标独立保存在本目录。冻结匹配种子的 E6 stationary 行情分支，联合训练新闻和财报残差：

**已完成结果：**三种子等权预测的开发 MAE **0.217979966**，内部测试 MAE **0.352400501**，比匹配 E6 的 **0.350345190** 差约 **0.5867%**。财报内容／新闻语义置零几乎不影响指标，没有证明多模态内容互补；保留为负结果，不作为升级方案。完整结果见 [RESULTS.md](RESULTS.md)。

`prediction = E6 + h/24 × (gate_news × news_delta + gate_report × report_delta)`。

新闻使用原 E7 的每日去重均值池化、512→8→8 MLP、has_news／log1p(count) 和 DLinear 新闻头。财报分支复用 E8 的 54→16→8 MLP。两分支使用小门控和按样本模态 dropout，头部从零初始化，epoch 0 与 E6 相同。各模态缺失时对应残差严格为零。

**归因限制：**与旧 E7 比较时还存在门控、dropout 和联合训练的变化，不是单独添加财报的严格单因素实验。报告同时列出 E7、E8 和 E6，并记录 E9 同一权重关闭各模态的诊断。

财报主设置为期末＋180 天；270 天为冻结权重的压力检查。缺少公告时间及历史发布版本，不能把保守延迟等同于严格 point-in-time 数据。

## 运行与记录

仓库根目录：

```powershell
./.venv/Scripts/python.exe -u -m experiments.e9_dlinear_multimodal.run --package-dir ../../hs300_train_val --evaluate-holdout
```

已有运行用 `--resume` 保留记录并继续。统一训练设置、三个固定种子与测试规则见 [公共工具说明](../dlinear_modal_common/README.md)。

本目录 `results/` 保存协议、数据审计、种子曲线、开发／测试集指标、单分支关闭、内容置零、同股票时间打乱及逐股票误差。`outputs/` 保存本地权重和预测，不上传 Git。结果见 [RESULTS.md](RESULTS.md)。
