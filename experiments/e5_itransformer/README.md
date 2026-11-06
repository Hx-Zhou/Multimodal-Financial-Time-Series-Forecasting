# E5 后续：七变量 iTransformer

本目录独立保存小型iTransformer与同输入线性对照。每个变量的96天历史映射为一个token，七个变量间用两层注意力交互，读取标准化Close对应的第6索引token预测24天残差。

固定配置：d_model64、4头、2层、FF256、GELU、pre-LayerNorm、dropout0.1。输出头零初始化，从E0最后值开始；训练损失使用原始残差，验证／推理固定alpha0.5和h/24缩放。

输入采用原E6已选出的七维平稳化表示，训练统计量保持不变：收益率、振幅、开收差、log成交量、分红／拆股指示、标准化Close。L_market线性对照读取完全相同的输入与锚点，使用同样优化器、损失、校准、早停与训练预算。

这是iTransformer核心思想的小型适配，不是原论文全部配置的完整复现。与Close-only PatchTST相比输入也不同，所以架构解释必须结合L_market；历史E6的优化器也不同，只作为既有性能参照。

完整执行次序、筛选与停止规则见 [共享计划](../transformer_optimization_common/PLAN.md)。最多8轮、patience3、batch512、CPU4线程，无新环境或大缓存。所有标准化统计量只由训练输入拟合。

运行共享入口：

```powershell
./.venv/Scripts/python.exe -u -m experiments.transformer_optimization_common.run --package-dir ../../hs300_train_val
```

I0_variables、L_market记录分别存入本目录的`results/`和本地`outputs/`。仅当它成为唯一复验候选时才补seed43/44；只有正式开发门槛全部通过才统一测试。结论见`RESULTS.md`。
