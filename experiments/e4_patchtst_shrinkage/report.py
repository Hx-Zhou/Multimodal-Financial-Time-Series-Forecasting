"""Render frozen replication decisions; no fitting or scoring here."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_report():
    dev = read(HERE / "results/development.json")
    test_path = HERE / "results/holdout.json"
    test = read(test_path) if test_path.exists() else None
    lines = ["# PatchTST 固定残差收缩：三种子复验", "",
             "训练、输入和原始 epoch 选择规则保持原 E4 不变。复用 seed42，补跑 seed43/44，系数固定为 0.5 与 0.25；不为各个种子分别选系数，不选种子子集。", "",
             "## 开发集等权平均", "", "| 模型／系数 | MAE | MSE |", "|---|---:|---:|"]
    for label, key in (("E0", "e0"), ("增强 E1", "enhanced_e1")):
        row = dev[key]
        lines.append(f"| {label} | {row['mae']:.9f} | {row['mse']:.9f} |")
    for family, rows in dev["families"].items():
        for alpha, row in rows.items():
            score = row["average"]
            lines.append(f"| {family} ×{alpha} | {score['mae']:.9f} | {score['mse']:.9f} |")
    lines += ["", "这些是先等权平均三个种子的预测，再算误差；不是三个单种子 MAE 的简单平均。", "",
              "## PatchTST 各种子", "", "| 系数 | seed | 开发 MAE | 开发 MSE |", "|---|---:|---:|---:|"]
    for alpha, row in dev["families"]["patchtst"].items():
        for score in row["seeds"]:
            lines.append(f"| {alpha} | {score['seed']} | {score['mae']:.9f} | {score['mse']:.9f} |")
    lines += ["", "## 预定门槛", "",
              f"较强开发参照：{dev['reference']}，MAE {dev['reference_development']['mae']:.9f}，MSE {dev['reference_development']['mse']:.9f}。", "",
              "MAE 至少改善 0.1%，MSE 不恶化超过 0.5%，至少 2/3 种子的收缩版本优于各自原始版本，开发前／后时段 MAE 各自不恶化超过 0.5%。这些规则不是独立统计显著性证明。", ""]
    for alpha, decision in dev["decision"]["checks"].items():
        failed = [k for k, v in decision["checks"].items() if not v]
        lines.append(f"- ×{alpha}：{'通过' if decision['passed'] else '未通过：'+', '.join(failed)}。")
    lines.append("")
    if test is None:
        lines += ["本轮没有候选通过全部开发门槛，因此按预定规则停止，**未重新评估内部测试集**。不把未通过配置作为已验证的新最佳。", ""]
    else:
        previous = read(ROOT / "experiments/e1_dlinear/e6_plus/results/holdout/summary.json")["scores"]
        lines += [f"开发集选定全局系数 **{test['alpha']}**，固定全部权重及等权平均规则后统一评估。", "",
                  "## 内部测试集", "", "| 模型 | MAE | MSE |", "|---|---:|---:|"]
        rows = {"E0": test["scores"]["e0"], "原增强 E1（历史参考）": previous["e1"],
                "E6（历史三种子参考）": previous["e6_average"],
                "原始 PatchTST 三种子": test["scores"]["patchtst_raw"],
                f"PatchTST ×{test['alpha']} 三种子": test["scores"]["patchtst"],
                "开发集选定的 DLinear 对照": test["scores"]["dlinear_control"]}
        for label, row in rows.items():
            lines.append(f"| {label} | {row['mae']:.9f} | {row['mse']:.9f} |")
        score, base = test["scores"]["patchtst"], previous["e6_average"]
        lines += ["", f"相对 E6，测试 MAE 改善率 {100*(1-score['mae']/base['mae']):+.4f}%，MSE 改善率 {100*(1-score['mse']/base['mse']):+.4f}%（负数表示变差）。", ""]
        if score["mae"] >= base["mae"] or score["mse"] >= base["mse"]:
            lines += ["未同时超过 E6 的两个测试指标，因此不能称为全面升级。测试后不切换系数、种子或检查点。", ""]
        else:
            lines += ["这一已知留出时间段的两个指标均低于 E6；仍需未曾查看的新时段验证，不能仅凭本次已知测试证明稳定泛化。", ""]
    lines += ["## 限制与复现", "",
              "seed42 曾用于发现收缩方向；新增种子仍共享相同开发时间段，不能等同于新的时间外验证。内部测试此前已被多次查看，本轮也不属于新的盲测或官方隐藏测试。", "",
              "收缩只作用于最终残差头，原有 h/24 与输入窗口方差缩放保留；导出的 frozen 权重已经包含 alpha，推理不要再次乘 alpha。", "",
              "没有新安装依赖、复制原始数据或生成大缓存。旧 E4 的代码、结果和权重不覆盖；新的训练输出与协议在本目录及 replicas/ 内。", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_report()
