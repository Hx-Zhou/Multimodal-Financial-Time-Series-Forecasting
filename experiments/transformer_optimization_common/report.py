"""Render completed bounded stages; no model or threshold selection here."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_reports():
    screen=read(HERE / "results/screen.json")
    replication=read(HERE / "results/replication.json")
    test_path=HERE / "results/holdout.json"
    test=read(test_path) if test_path.exists() else None
    lines=["# Transformer 有限优化计划：执行结果", "",
           "共筛查四个 seed42 Transformer 配置，按预定规则最多复验一个候选。模型、输入和训练预算的改变均在 PLAN.md 中事先声明。旧实验保持原样。", "",
           "## 四个阶段：seed42", "", "| 阶段 | 最佳轮次 | 参数量 | 验证 MAE | 验证 MSE | 可进入复验 |", "|---|---:|---:|---:|---:|---|"]
    for row in screen["rows"]:
        score=row["development"];name=row["spec"]["id"]
        lines.append(f"| {name} | {row['best_epoch']} | {row['parameters']:,} | {score['mae']:.9f} | {score['mse']:.9f} | {'是' if screen['decisions'][name]['eligible'] else '否'} |")
    lines += ["", "P0：原宽度128、固定×0.5的校准感知早停；P1：仅宽度改64；P2：仅将前两者较好配置的 dropout 改0.3；I0：七维平稳化行情的 iTransformer64。", "",
              "P0 同时采用本轮的8轮上限／patience3，历史版本为12／4，不能将与历史结果的全部差异归因于检查点选择。P1/P2 与新 P0 才是同预算的逐项对照。I0 与 PatchTST 的输入路线不同，架构增量还需看同输入线性对照。", "",
              "## 验证参照", "", "| 参照 | MAE | MSE |", "|---|---:|---:|"]
    for name,row in screen["references"].items():
        lines.append(f"| {name} | {row['mae']:.9f} | {row['mse']:.9f} |")
    if screen["selected_spec"] is None:
        lines += ["", "没有单种子配置通过预先固定的候选储备门槛，未追加种子，未重新评估测试集。", ""]
    else:
        lines += ["",f"选中复验：**{replication['spec']['id']}**，三个种子42／43／44始终等权平均。", "",
                  "## 三种子复验", "", "| 模型 | 验证 MAE | 验证 MSE |", "|---|---:|---:|"]
        row=replication["development"]
        lines.append(f"| Transformer 平均预测 | {row['mae']:.9f} | {row['mse']:.9f} |")
        for name,score in replication["references"].items():
            lines.append(f"| {name} | {score['mae']:.9f} | {score['mse']:.9f} |")
        lines += ["", "| 种子 | Transformer MAE | 匹配线性 MAE | 最佳 epoch |", "|---|---:|---:|---:|"]
        for a,b in zip(replication["records"]["transformer"],replication["records"]["matched_linear"]):
            lines.append(f"| {a['seed']} | {a['development']['mae']:.9f} | {b['development']['mae']:.9f} | {a['best_epoch']} |")
        lines += ["",f"同种子胜过匹配线性模型：{replication['matched_linear_seed_wins']}/3。", ""]
        for name,passed in replication["decision"]["checks"].items():
            lines.append(f"- {name}：{'通过' if passed else '未通过'}。")
        lines.append("")
        if not replication["passed"]:
            lines += ["候选未通过全部正式开发门槛，按计划停止；**本轮未重新评估内部测试集，不作为升级模型。**", ""]
    if test:
        old=read(ROOT / "experiments/e1_dlinear/e6_plus/results/holdout/summary.json")["scores"]
        lines += ["## 冻结后的内部测试", "", "| 模型 | 测试 MAE | 测试 MSE |", "|---|---:|---:|"]
        rows={"E0（历史）":old["e0"],"增强E1（历史）":old["e1"],"E6（历史）":old["e6_average"],
              "新Transformer三种子":test["scores"]["transformer"],"新匹配线性三种子":test["scores"]["matched_linear"]}
        for name,row in rows.items():lines.append(f"| {name} | {row['mae']:.9f} | {row['mse']:.9f} |")
        s,b=test["scores"]["transformer"],old["e6_average"]
        lines += ["",f"相对 E6，MAE 改善 {100*(1-s['mae']/b['mae']):+.4f}%，MSE 改善 {100*(1-s['mse']/b['mse']):+.4f}%（负数表示变差）。", "",
                  "".join(["这一个已知时间段的两个指标都低于E6，仍需未曾查看的新时段确认泛化。" if all(s[k]<b[k] for k in ('mae','mse')) else "没有同时超过E6的两个测试指标，不能称为全面升级。"]),""]
    lines += ["## 限制和记录", "",
              "模型选择、父配置、完整曲线、两个开发时段指标和权重哈希均已保存。全程只固定alpha=0.5，不为不同种子重新选系数，不挑测试表现最好的种子或检查点。", "",
              "训练仍使用原完整587,587样本；开发42,930、隔离18,630、内部留出15,390。验证集与内部留出此前已多次查看，本轮不是新的独立盲测，多个种子也不能替代新的时间外验证。", "",
              "没有安装新依赖、复制原始数据或写新闻／财报大缓存。只使用纯行情。检查点存原始权重和alpha元数据；推理由forward应用一次alpha，不要再手动缩放权重。", ""]
    text="\n".join(lines)
    (HERE / "RESULTS.md").write_text(text,encoding="utf-8")
    for folder in (ROOT / "experiments/e4_patchtst_capacity",ROOT / "experiments/e5_itransformer"):
        (folder / "RESULTS.md").write_text(text,encoding="utf-8")
