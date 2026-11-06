# HS300 三模态预测任务

提供训练集和验证集的原始行情、财报、逐篇新闻向量，以及读取器 `hs300_dataset.py` 和 `model.py` 模板。任务是用过去 **96 个交易日**的三种信息预测随后 **24 个交易日**的标准化 `Close`。你需要编写 `model.py`、训练得到 `model.pt`，最终只提交这两个文件。提交后，将在未公开测试集上运行模型，以最终的 MAE 和 MSE 作为评分依据。

## 要写的代码

`model.py` 提供了示例框架。你需要自行处理三种输入、设计模型并完成训练。模型结构、新闻和财报处理、损失函数、优化器及训练方式均可自行决定；模板中的 `build_optimizer`、`compute_loss` 和训练循环可以修改或删除。训练集用于训练模型，验证集用于调参和挑选最优模型。训练过程中应保留验证集表现最好的权重，最终将这份 `state_dict` 保存为 `model.pt`。

## 提交文件要求

评分程序按以下方式调用提交文件：

1. `model.py` 提供 `build_model(config=None)`；评分程序会无参数调用 `build_model()`，得到 PyTorch 模型，再从 `model.pt` 加载其 `state_dict`。
2. `model.py` 提供 `prepare_batch(samples, device)`，将行情、逐篇新闻向量和财报整理成模型所需的张量字典。评分时调用 `prepare_batch(samples, device)`；测试样本没有 `y`。
3. 模型的 `forward(batch)` 使用三种模态，返回有限的 `[batch_size, 24]` 数值，尺度与样本的标准化 `y` 相同。

模型和推理所需的自写代码应全部放在 `model.py`；提交目录只包含 `model.py`、`model.pt`。推理不能依赖训练目录的绝对路径或测试标签。`hs300_dataset.py` 是固定读取器，无需修改。

## 数据字段

拿到交付 ZIP 后解压一次即可。`data/` 中已经是行情 CSV、财报 JSONL 和原始新闻 JSONL 文件；读取器直接读取行情、财报文件及 `data/text/embeddings.npz` 中的新闻向量。请保持目录结构不变。

用 `HS300Dataset(".", "train")` 和 `HS300Dataset(".", "val")` 读取样本。`sample` 中主要字段如下：

| 字段 | 内容 |
| --- | --- |
| `numerical`、`dates` | 过去 96 个交易日的原始行情 `[96, 7]` 及对应日期。列顺序为 `Open, High, Low, Close, Volume, Dividends, Stock Splits`。 |
| `news[day]`、`news_time[day]` | 第 `day` 个交易日的逐篇新闻向量 `[篇数, 512]` 及发布时间；无新闻时为 `[0, 512]`。`news.events()` 可取得输入日期范围内包括非交易日的新闻。 |
| `financial[kind]` | `balance_sheet`、`cashflow`、`income` 三类原始财报记录，仅返回报告期末 `end_date` 不晚于输入末日的记录。 |
| `close_mean`、`close_std` | 该股票只用训练期 `Close` 算出的标准化参数。原始行情未预先标准化。 |
| `y` | 随后 24 个交易日的标准化 `Close`，仅训练和验证样本提供。 |

## 新闻和财报如何与行情配合

**新闻：** `news[day]` 返回当天的全部新闻，不会预先挑选或合并。例如当天有 8 篇，得到的就是 `[8, 512]`；`news_time[day]` 给出对应的发布时间。不同日期的篇数可能不同，模型需要自行决定如何使用这些新闻：逐篇处理、汇总、按时间筛选或学习权重均可。数据中有重复新闻，`news[day]` 会保留这些记录；使用新闻时需要自行处理重复内容，避免重复计权。`news.events()` 还可取到输入日期范围内非交易日的新闻。

**财报：** `financial[kind]` 返回该股票、该类别中所有 `end_date <= dates[-1]` 的历史记录，不限于这 96 个交易日内，也没有逐日映射或向前填充。`financial.latest(kind)` 只按报告期末 `end_date` 选最近的一条。例如输入最后一天是 2024-04-08，若有 2023-12-31 和 2024-03-31 两期财报，两期都会返回，`latest` 会选 2024-03-31。

模型需要自行确定财报与日频行情的使用和对齐规则，并处理缺失记录。
