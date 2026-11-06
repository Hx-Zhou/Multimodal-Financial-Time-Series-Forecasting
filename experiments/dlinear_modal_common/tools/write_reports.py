"""Render each experiment's handoff from completed, immutable metric records.

Run from repository root: python -m experiments.dlinear_modal_common.tools.write_reports
This post-processing tool does not train, select models, or recompute predictions.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "experiments"
OLD = EXPERIMENTS / "e1_dlinear/e6_plus"
DIRECTORIES = {"e8": "e8_dlinear_reports", "e9": "e9_dlinear_multimodal", "e10": "e10_dlinear_spectf"}
TITLES = {"e8": "E8：行情＋财报", "e9": "E9：行情＋新闻＋财报", "e10": "E10：SpecTF-style 频域新闻融合"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def improvement(value, base):
    return 100*(1-value/base)


def main():
    # Read everything before writing so an incomplete run cannot create partial reports.
    records = {stage: {split: read(EXPERIMENTS / directory / f"results/{split}.json")
                       for split in ("development", "holdout")}
               for stage, directory in DIRECTORIES.items()}
    summaries = {stage: [read(EXPERIMENTS / directory / f"results/seed{s}.json") for s in (42, 43, 44)]
                 for stage, directory in DIRECTORIES.items()}
    old_dev = read(OLD / "results/summary.json")
    old_averages = read(OLD / "results/decisions.json")["supplementary_equal_prediction_average"]
    old_test = read(OLD / "results/holdout/summary.json")["scores"]
    references = {
        "E0 最后值": (old_dev["e0_development"], old_test["e0"]),
        "原增强 E1": (old_dev["previous_e1_development"], old_test["e1"]),
        "E6 行情（三种子等权）": (old_averages["e6_selected"], old_test["e6_average"]),
        "E7 新闻（三种子等权）": (old_averages["e7"], old_test["e7_average"]),
    }
    for stage, directory in DIRECTORIES.items():
        dev, test = records[stage]["development"], records[stage]["holdout"]
        baseline = old_test["e6_average"]
        gain = improvement(test["equal_average"]["mae"], baseline["mae"])
        lines = [f"# {TITLES[stage]}：完成结果", "",
                 "本实验在独立目录运行，全部模型先按开发集选择权重，再统一冻结并评估内部测试集。下面 MAE／MSE 均为标准化 Close 误差，越低越好。主结果是 42、43、44 三个固定种子预测的等权平均。", "",
                 "## 主结果", "", "| 模型 | 开发 MAE | 开发 MSE | 测试 MAE | 测试 MSE |",
                 "|---|---:|---:|---:|---:|"]
        compare = dict(references)
        if stage == "e9":
            compare["E8 财报（三种子等权）"] = (records["e8"]["development"]["equal_average"], records["e8"]["holdout"]["equal_average"])
        compare[f"本实验 {stage.upper()}（三种子等权）"] = (dev["equal_average"], test["equal_average"])
        for name, (d, t) in compare.items():
            lines.append(f"| {name} | {d['mae']:.9f} | {d['mse']:.9f} | {t['mae']:.9f} | {t['mse']:.9f} |")
        lines += ["", f"相对匹配 E6，本实验测试 MAE 的改善率为 **{gain:+.4f}%**（负值表示变差）；MSE 改善率为 **{improvement(test['equal_average']['mse'], baseline['mse']):+.4f}%**。", "",
                  "## 三个种子", "", "| 种子 | 最佳 epoch | 开发 MAE | 开发 MSE | 测试 MAE | 测试 MSE |",
                  "|---|---:|---:|---:|---:|---:|"]
        for run, score in zip(summaries[stage], test["seeds"]):
            d = run["development"]
            lines.append(f"| {run['seed']} | {run['best_epoch']} | {d['mae']:.9f} | {d['mse']:.9f} | {score['mae']:.9f} | {score['mse']:.9f} |")
        stat = test["seed_statistics"]
        lines += ["", f"单种子测试 MAE 均值±总体标准差：{stat['mae']['mean']:.9f} ± {stat['mae']['std_population']:.9f}。这与先平均预测再算误差不是同一个指标。",
                  f"每个模型总参数 {summaries[stage][0]['parameters']:,}，其中辅助分支可训练参数 {summaries[stage][0]['trainable_parameters']:,}；其余为冻结的数值权重。", "",
                  "epoch 0 表示训练过辅助分支后，开发集没有达到预先设定的改善阈值，因此保留了精确的 E6 初始状态。不会按测试结果改用其他 epoch 或种子。", "",
                  "## 固定权重诊断", "", "| 操作 | 开发 MAE | 测试 MAE |", "|---|---:|---:|"]
        names = {"no_reports": "关闭财报分支", "report_values_zero": "财报数值／同比置零，保留掩码和年龄",
                 "report_lag270": "财报可用日延至期末＋270天（不重训）",
                 "reports_shuffled_within_stock": "同股票财报窗口打乱（仅开发集）",
                 "no_news": "关闭新闻分支", "news_semantics_zero": "新闻投影语义置零，保留数量／掩码",
                 "news_shuffled_within_stock": "同股票新闻窗口打乱（仅开发集）"}
        for key, d in dev["diagnostics"].items():
            t = test["diagnostics"].get(key)
            test_cell = f"{t['equal_average']['mae']:.9f}" if t else "未执行"
            lines.append(f"| {names[key]} | {d['equal_average']['mae']:.9f} | {test_cell} |")
        lines += ["", "诊断均不改变模型权重、不用来重新选择模型。270 天检查复用主训练的统计量。关闭单个分支是同一模型的依赖性检查，不能等同于该单模态模型的独立重训结果。", "",
                  "## 分期限与覆盖", "", "| 预测期限 | 测试 MAE | 测试 MSE |", "|---|---:|---:|"]
        for h in (1, 6, 12, 24):
            s = test["equal_average"]
            lines.append(f"| 第 {h} 天 | {s[f'mae_h{h}']:.9f} | {s[f'mse_h{h}']:.9f} |")
        lines += ["", "| 样本组 | 样本数 | 测试 MAE | 测试 MSE |", "|---|---:|---:|---:|"]
        for name, score in test["groups"].items():
            lines.append(f"| {name} | {score['samples']} | {score['mae']:.9f} | {score['mse']:.9f} |")
        lines += ["", "新闻覆盖按输入 96 天内每天去重后的新闻数合计，正样本中位数分高／低覆盖。逐股票误差见 `results/holdout_per_stock.csv`，所有单种子期限指标见 `results/holdout.json`。", "",
                  "## 结论与限制", ""]
        if gain <= 0:
            lines.append("本设置没有超过匹配 E6 的测试 MAE，不建议作为改善预测误差的升级方案；保留为完整负结果／对照实验。")
        else:
            lines.append("本设置在这一内部测试时间段的 MAE 小幅低于 E6；是否有实际模态增量，还必须结合下面的置零诊断判断，不能仅靠一个汇总数值宣称稳定提升。")
        for diagnostic, label in (("report_values_zero", "财报内容"), ("news_semantics_zero", "新闻语义")):
            if diagnostic in test["diagnostics"]:
                delta = test["diagnostics"][diagnostic]["equal_average"]["mae"] - test["equal_average"]["mae"]
                if abs(delta) < 1e-6:
                    lines.append(f"{label}置零的测试 MAE 变化仅 {delta:+.12f}，未证明模型利用了该内容带来收益；微小差异可能来自残差偏置、数量或掩码等其他信息。")
                else:
                    lines.append(f"{label}置零后的测试 MAE 变化为 {delta:+.9f}；正值代表置零后更差，但单次诊断不构成因果或跨时期稳定性的证明。")
        if stage == "e9":
            lines.append("E9 相对原 E7 还改变了门控、dropout 和联合训练方式，结果不能全部归因于加入财报。")
            d = dev["diagnostics"]["no_news"]["equal_average"]["mae"]
            t = test["diagnostics"]["no_news"]["equal_average"]["mae"]
            lines.append(f"关闭新闻的诊断得到开发 MAE {d:.9f}、测试 MAE {t:.9f}。它在开发集不优于 E6，不能因为测试数字较低，就事后把这个诊断选成新的推荐模型。")
        if stage == "e10":
            lines.append(f"相对 E7，本实验测试 MAE 改善率为 {improvement(test['equal_average']['mae'], old_test['e7_average']['mae']):+.4f}%；是否超越较弱的新闻模型与是否超越 E6 是两个不同判据。此实现是轻量 SpecTF-style 适配，不代表完整论文模型的上限。")
            lines.append("相对旧 E7，本版本还带有新增门控和模态 dropout；这是整体版本对照，不是只改变频域操作的严格单因素消融。")
        if stage != "e10":
            lines.append("财报没有公告日期，主设置固定期末＋180 天，另做 270 天压力检查；重复字段冲突已掩码。缺少历史发布版本仍可能含事后修订，延迟不能保证严格 point-in-time 无泄漏。")
            lines.append("统一 180 天会舍弃部分当时可能已公开的较新财报。本结果只适用于当前特征、结构、训练方式与可用性假设，不能推断财报模态普遍无效。")
        lines += ["", "训练样本 587,587，开发样本 42,930，边界隔离样本 18,630，内部测试样本 15,390。测试目标起点为 2024-01-24 至 2024-02-27，标签最晚到 2024-03-29。", "",
                  "这是此前已经查看过的内部时间留出集，不是新盲测，也不是官方隐藏测试。滑窗和股票间市场日期相关，未声称统计显著性；本次没有依据测试成绩重训、选种子、改财报延迟或改结构。", "",
                  "开发集重载指标一致；所有行情参数保持冻结；关闭全部辅助模态与 E6 逐元素一致；自然缺失全部模态的样本也检查了精确退化。测试指标经独立 PyTorch float64 归约核对，E6 测试基线复现此前记录。", "",
                  "具体训练前协议、数据审计、完整曲线、冻结候选哈希均保存在本目录 `results/`。代码只在 `experiments/`；根模板和共享划分未修改。", ""]
        destination = EXPERIMENTS / directory / "RESULTS.md"
        destination.write_text("\n".join(lines), encoding="utf-8")
        print(destination)


if __name__ == "__main__":
    main()
