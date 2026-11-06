"""Create the E6/E7 handoff from completed development-only runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split
from ..data import materialize
from ..run import metrics, write_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("e7"):
        raise ValueError("E7 has not completed")
    selected = summary["selected_representation"]
    dataset = HS300Dataset(args.package_dir.resolve(), "val")
    try:
        split = make_validation_split(dataset)
        dev = materialize(dataset, split.dev_indices)
    finally:
        dataset.close()
    target = dev.y.numpy()
    baseline = np.repeat(dev.x[:, -1:].numpy(), 24, axis=1)
    np.testing.assert_allclose(metrics(baseline, target)["mae"], summary["e0_development"]["mae"], rtol=0, atol=1e-10)
    rows = []
    names = [run["name"] for group in summary["e6"].values() for run in group["runs"]]
    names += [run["name"] for run in summary["e7"]["runs"]]
    ensembles = {}
    for label, group in (("e6_selected", summary["e6"][selected]), ("e7", summary["e7"])):
        average = np.mean([np.load(args.output_dir / run["name"] / "dev_predictions.npy", allow_pickle=False)
                           for run in group["runs"]], axis=0)
        ensembles[label] = metrics(average, target)
    for name in ["e0", *names]:
        prediction = baseline if name == "e0" else np.load(args.output_dir / name / "dev_predictions.npy", allow_pickle=False)
        if name != "e0":
            saved = json.loads((args.results_dir / f"{name}.json").read_text(encoding="utf-8"))
            actual = metrics(prediction, target)
            for key in ("mae", "mse"):
                np.testing.assert_allclose(actual[key], saved["development"][key], rtol=0, atol=1e-12)
        for stock, code in enumerate(dev.codes):
            mask = dev.stocks == stock
            if mask.any():
                score = metrics(prediction[mask], target[mask])
                rows.append({"model": name, "stock_code": code, "samples": int(mask.sum()),
                             "mae": score["mae"], "mse": score["mse"]})
    with (args.results_dir / "per_stock.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "stock_code", "samples", "mae", "mse"])
        writer.writeheader()
        writer.writerows(rows)
    market = summary["e6"][selected]["aggregate"]
    news = summary["e7"]["aggregate"]
    pairs = list(zip(summary["e6"][selected]["runs"], summary["e7"]["runs"]))
    decisions = {
        "selected_representation": selected,
        "e7_improvement_over_matched_e6_percent": {key: 100 * (1 - news[key]["mean"] / market[key]["mean"]) for key in ("mae", "mse")},
        "e7_all_seeds_improve_both": all(n["development"][key] < m["development"][key] for m,n in pairs for key in ("mae", "mse")),
        "e7_epoch_zero_seeds": [n["seed"] for _,n in pairs if n["best_epoch"] == 0],
        "zero_semantics_mean": {key: float(np.mean([n["zero_news_semantics"][key] for _,n in pairs])) for key in ("mae", "mse")},
        "shuffled_news_mean": {key: float(np.mean([n["within_stock_shuffled_news"][key] for _,n in pairs])) for key in ("mae", "mse")},
        "shuffled_news_mae_worse_seeds": [n["seed"] for _,n in pairs
                                         if n["within_stock_shuffled_news"]["mae"] > n["development"]["mae"] + 1e-6],
        "holdout_evaluated": False,
        "supplementary_equal_prediction_average": ensembles,
        "ensemble_note": "All three fixed seeds equally averaged; descriptive comparison only, not used to select the representation. E6 can fold linear weights; E7 requires three nonlinear models.",
    }
    decisions["semantic_metric_tolerance"] = 1e-6
    decisions["semantic_metric_change_negligible"] = all(
        abs(decisions["zero_semantics_mean"][key] - news[key]["mean"]) <= 1e-6
        for key in ("mae", "mse"))
    verified_epochs = set()
    for interruption_path in sorted(args.results_dir.glob("interruption*.json")):
        interrupted = json.loads(interruption_path.read_text(encoding="utf-8-sig"))
        restarted = json.loads((args.results_dir / f"{interrupted['run']}.json").read_text(encoding="utf-8"))
        resumed_epochs = {row["epoch"]: row for row in restarted["history"]}
        for old_row in interrupted["partial_history"]:
            new_row = resumed_epochs[old_row["epoch"]]
            for key in ("mae", "mse"):
                np.testing.assert_allclose(old_row["development"][key], new_row["development"][key], rtol=0, atol=1e-10)
        verified_epochs.update(row["epoch"] for row in interrupted["partial_history"])
    if verified_epochs:
        decisions["resume_reproduced_partial_history_epochs"] = sorted(verified_epochs)
    write_json(args.results_dir / "decisions.json", decisions)
    def mean_std(agg, key):
        return f"{agg[key]['mean']:.9f} ± {agg[key]['std_population']:.9f}"
    lines = [
        "# DLinear E6＋E7 开发集结果",
        "",
        "本轮仅使用固定训练集与开发集，没有重算或据留出测试选配置。三种子的均值/波动描述优化稳定性，不代表三个独立测试集。",
        "",
        "| 对照 | 开发 MAE ↓ | 开发 MSE ↓ |",
        "| --- | ---: | ---: |",
        f"| E0 最后值 | {summary['e0_development']['mae']:.9f} | {summary['e0_development']['mse']:.9f} |",
        f"| 原已交付增强 E1（单个平均权重模型，参考） | {summary['previous_e1_development']['mae']:.9f} | {summary['previous_e1_development']['mse']:.9f} |",
        f"| E6 原始七字段，三种子均值 ± 标准差 | {mean_std(summary['e6']['raw']['aggregate'], 'mae')} | {mean_std(summary['e6']['raw']['aggregate'], 'mse')} |",
        f"| E6 平稳化七特征，三种子均值 ± 标准差 | {mean_std(summary['e6']['stationary']['aggregate'], 'mae')} | {mean_std(summary['e6']['stationary']['aggregate'], 'mse')} |",
        f"| E7 固定选中 E6＋新闻，三种子均值 ± 标准差 | {mean_std(news, 'mae')} | {mean_std(news, 'mse')} |",
        "",
        f"E6 按预先固定的规则选择 **{selected}**。E7 相对匹配 E6 的 MAE/MSE 改善百分比分别为 **{decisions['e7_improvement_over_matched_e6_percent']['mae']:.4f}% / {decisions['e7_improvement_over_matched_e6_percent']['mse']:.4f}%**；负值表示退化。标准差使用 ddof=0。",
        "",
        "旧 E1 与 E6 的通道数及收缩训练流程不同，且旧 E1 是平均权重后的单模型，这一行仅提供已交付方案的性能参照；E6 A/B 与匹配种子的 E6/E7 才是本轮主要受控对照。",
        "",
        "## 固定三种子的等权预测平均（补充）",
        "",
        "| 模型 | MAE | MSE |",
        "| --- | ---: | ---: |",
        f"| 选定 E6 的三种子预测平均 | {ensembles['e6_selected']['mae']:.9f} | {ensembles['e6_selected']['mse']:.9f} |",
        f"| E7 的三模型预测平均 | {ensembles['e7']['mae']:.9f} | {ensembles['e7']['mse']:.9f} |",
        "",
        "这是报告阶段对三个既定种子做等权平均的补充，未用于重新选择表示或成员。E6 是线性模型，可等价合并权重；E7 含非线性新闻 MLP，需要运行三个模型，不能称为单模型提分。主对照仍是匹配种子的 E6/E7。",
        "",
        "## 各次运行",
        "",
        "| 实验 | 种子 | 最佳轮次 | MAE | MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for group in [summary["e6"]["raw"], summary["e6"]["stationary"], summary["e7"]]:
        for run in group["runs"]:
            lines.append(f"| {run['name']} | {run['seed']} | {run['best_epoch']} | {run['development']['mae']:.9f} | {run['development']['mse']:.9f} |")
    lines += [
        "",
        "E7 的第 0 轮就是冻结的 E6，若选中第 0 轮则不能说新闻有增益。",
        "",
        "## 新闻消融",
        "",
        "| 条件 | 三种子平均 MAE | 三种子平均 MSE |",
        "| --- | ---: | ---: |",
        f"| 正常新闻 | {news['mae']['mean']:.9f} | {news['mse']['mean']:.9f} |",
        f"| 语义向量置零，保留掩码/数量 | {decisions['zero_semantics_mean']['mae']:.9f} | {decisions['zero_semantics_mean']['mse']:.9f} |",
        f"| 同股票内置换窗口新闻 | {decisions['shuffled_news_mean']['mae']:.9f} | {decisions['shuffled_news_mean']['mse']:.9f} |",
        "",
    ]
    if decisions["semantic_metric_change_negligible"]:
        lines.append("语义置零后 MAE/MSE 几乎不变（绝对差均不超过 1e-6），未观察到有意义的新闻语义收益；不能把浮点量级的差异解读为有效利用语义。")
    elif decisions["zero_semantics_mean"]["mae"] <= news["mae"]["mean"] + 1e-6:
        lines.append("语义置零后平均 MAE 没有变差，不能据本轮宣称新闻语义提供了可靠增量。")
    else:
        lines.append("语义置零后平均 MAE 变差，提示模型使用了语义分支，但仍只是同一开发集上的诊断。")
    if decisions["shuffled_news_mean"]["mae"] <= news["mae"]["mean"] + 1e-6:
        lines.append("新闻时间对应被打乱后平均 MAE 没有变差，新闻的正确时间对应价值未得到支持。")
    else:
        lines.append("新闻时间对应被打乱后平均 MAE 变差；不能把这项破坏性诊断当作独立泛化评估。")
    lines.append(f"打乱后的平均 MAE 仅变化 {decisions['shuffled_news_mean']['mae'] - news['mae']['mean']:+.9f}；三个种子中只有 {len(decisions['shuffled_news_mae_worse_seeds'])} 个的 MAE 明确恶化（差值大于 1e-6）。本轮不支持将正确新闻时间对应的收益描述为稳定优势。")
    lines += ["", "## 交接结论", ""]
    if all(news[key]["mean"] < market[key]["mean"] for key in ("mae", "mse")):
        lines.append("E7 在本开发集上的两项平均指标优于匹配 E6，可作为新闻增量候选继续比较。")
    else:
        lines.append("E7 没有同时改善匹配 E6 的两项平均指标，本轮未获得稳定的新闻增量证据。")
    if all(news[key]["mean"] < summary["previous_e1_development"][key] for key in ("mae", "mse")):
        lines.append("E7 的三种子平均成绩也超过此前交付 E1 的参考成绩，但仍需关注消融与不同训练流程，不能直接当作已验证的最终替代模型。")
    else:
        lines.append("E7 的单种子平均成绩未同时超过此前交付 E1 的两项参考成绩，不应仅因超过较弱的 E6 就宣称整体优化成功。")
    if all(ensembles["e7"][key] < summary["previous_e1_development"][key] for key in ("mae", "mse")):
        lines.append("补充的三模型预测平均超过旧 E1 的两项参考指标，但推理成本和模型数量增加，且本次仍无新独立测试，不能把这一结果写成单模型优势。")
    else:
        lines.append("补充的 E7 三模型预测平均也未同时超过旧 E1 的两项参考指标，本轮不支持替换原 E1。")
    lines += [
        "",
        f"三种子是否均改善匹配 E6 的 MAE/MSE：**{decisions['e7_all_seeds_improve_both']}**。回退第 0 轮的种子：**{decisions['e7_epoch_zero_seeds']}**。",
        "",
        "协议、训练期标准化参数及所有逐轮日志均保存在 results/；每次 E7 JSON 含分期限、新闻覆盖分组及消融，per_stock.csv 保留所有实验的逐股票指标。原数据、权重、预测和缓存位于被忽略的 outputs/。参数量、冻结边界及复现命令见 README.md。",
        "",
        "本轮没有改动原 E1、共享读取器、公共切分或根目录模板，也没有运行 E8/E9。任何后续方案选择继续只用开发数据；已经查看的原 holdout 不再是针对这轮扩展的新独立证据。",
    ]
    if "resume_reproduced_partial_history_epochs" in decisions:
        lines += ["", "缓存与 CPU 微小数处理修复后，重启种子所有已记录轮次（含第 0 轮）的 MAE/MSE 与中断前记录一致（绝对误差不超过 1e-10）。原中断日志、代码版本与运行协议均保留。"]
    if (HERE / "HOLDOUT.md").exists() and args.results_dir.resolve() == (HERE / "results").resolve():
        lines[2:2] = ["> 后续已完成固定模型的留出测分，见 [HOLDOUT.md](HOLDOUT.md)。本文保留开发阶段记录，不将后来的测试结果混入选模过程。", ""]
    report_path = (HERE / "RESULTS.md" if args.results_dir.resolve() == (HERE / "results").resolve()
                   else args.results_dir / "REPORT.md")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decisions, indent=2), flush=True)


if __name__ == "__main__":
    main()
