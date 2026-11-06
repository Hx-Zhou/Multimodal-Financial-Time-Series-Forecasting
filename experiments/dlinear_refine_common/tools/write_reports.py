"""Render each first-stage result without training or accessing holdout records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORIES = {"e8": "e8_dlinear_reports_refine", "e9": "e9_dlinear_multimodal_refine", "e10": "e10_dlinear_spectf_refine"}
NEXT = {
    "e8": "按 F3 先比较常数残差、仅掩码／年龄与完整财报的重新训练对照；分清整体校准、覆盖信息和财报内容，之后才考虑比率或季度口径。",
    "e9": "按 M3 建立同条件仅新闻／仅财报的重训对照，检查联合失败来自哪个分支；在联合模型超过单模态前，不增加更复杂交互。",
    "e10": "按 S3 回到匹配门控／dropout／优化器的简单新闻融合对照，确认额外频域结构是否值得保留；不直接加大频率数或层数。",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    for stage, directory in DIRECTORIES.items():
        home = ROOT / "experiments" / directory
        summary = read(home / "results/regularization_summary.json")
        protocol = read(home / "results/regularization_protocol.json")
        audit = read(home / "results/initial_audit.json")["runs"][0]
        rows = [read(home / f"results/{step}/seed42.json") for step in protocol["steps"]]
        reference = summary["reference_development"]
        lines = [f"# {stage.upper()} 逐步优化：首轮正则化结果", "",
                 "完整后续路线见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。本轮只执行前两个正则化步骤，后续特征／结构实验尚未执行。旧实验、数据划分和数值骨干均未改变。", "",
                 "## 对照与结果", "",
                 f"预先登记的较强 seed42 对照为 **{summary['reference_family_seed42'].upper()}**，开发 MAE **{reference['mae']:.9f}**、MSE **{reference['mse']:.9f}**。下面都是同一个 seed42，不能与旧三种子等权平均直接混比。", "",
                 "| 步骤 | 正则化策略 | 最佳 epoch | 最佳开发 MAE | 最佳开发 MSE | 实际训练轮中的最低 MAE（不含 epoch0） | 是否进入复验 |",
                 "|---|---|---:|---:|---:|---:|---|"]
        for row in rows:
            score = row["development"]
            trained = min(r["development"]["mae"] for r in row["history"] if r["epoch"] > 0)
            passed = summary["screen_decisions"][row["step"]]["passed"]
            label = "AdamW，全部参数衰减" if row["policy"] == "all" else "AdamW，仅矩阵衰减"
            lines.append(f"| {row['step']} | {label} | {row['best_epoch']} | {score['mae']:.9f} | {score['mse']:.9f} | {trained:.9f} | {'通过' if passed else '未通过'} |")
        lines += ["", "epoch 0 是精确的纯行情 E6。若最佳轮次为 0，表示新增分支训练后没有达到开发集改善阈值，保留基线；这不表示没有运行训练。", "",
                  "与旧实验的对照也存在预算边界：本轮最多 20 epoch，旧版上限 40；本轮没有为追求结果临时延长训练。", "",
                  "## 编码器是否仍然退化", "", "| 步骤／模块 | 旧训练末尾范数／初始范数 | 新训练末尾范数／初始范数 | 选中权重范数／初始范数 |",
                  "|---|---:|---:|---:|"]
        for row in rows:
            final = row["history"][-1]
            for module, selected_ratio in row["module_norm_ratios"].items():
                old = audit["modules"][module]
                final_ratio = final["module_norms"][module]["l2_norm"]/old["initial"]["l2_norm"]
                lines.append(f"| {row['step']} / {module} | {old['final_to_initial_norm_ratio']:.8g} | {final_ratio:.8g} | {selected_ratio:.8g} |")
        lines += ["", "权重范数只能说明参数是否缩小，不能证明模态有效。若开发最优回到 epoch0，即使末轮编码器不再塌缩，也没有获得可采用的预测改善。门控轨迹和每轮误差见各步骤 JSON。", "",
                  "## 开发集内容检查", "", "| 步骤 | 诊断 | 开发 MAE | 相对正常输入 MAE 的变化 | 预测最大变化 |", "|---|---|---:|---:|---:|"]
        for row in rows:
            for name, score in row["diagnostics"].items():
                lines.append(f"| {row['step']} | {name} | {score['mae']:.9f} | {score['mae']-row['development']['mae']:+.9f} | {score['max_abs_prediction_change']:.9g} |")
        lines += ["", "诊断针对开发集选中的权重。若选中的是 epoch0，内容置零不影响预测是预期行为，不能用它单独评价未选中训练轮的内容响应。所有诊断都是开发集诊断，本轮没有新的测试集成绩。", "",
                  "## 去留决定", ""]
        for step, value in summary["screen_decisions"].items():
            failed = [name for name, passed in value["checks"].items() if not passed]
            lines += [f"- {step}：{'通过全部初筛门槛' if not failed else '未通过 '+', '.join(failed)}。"]
        lines.append("")
        if summary["selected_for_replication"] is None:
            lines += ["两个固定配置均未通过，按预先制定规则**不追加 seed43/44、不查看内部测试集、不作为升级模型**。本轮支持进一步研究正则化与内容学习，但没有证明预测性能改进。", ""]
        else:
            replica = summary["replication"]
            lines += [f"进入复验的是 {summary['selected_for_replication']}。三种子等权平均开发 MAE **{replica['equal_average']['mae']:.9f}**；匹配较强参照为 {replica['reference_family'].upper()}，MAE **{replica['reference_equal_average']['mae']:.9f}**。",
                      "", f"复验是否通过：**{'通过' if replica['passed'] else '未通过'}**；改善种子数 {replica['seeds_improved']}/3，内容证据支持种子数 {replica['content_supported_seeds']}/3。", ""]
        lines += ["## 下一步", "", NEXT[stage], "",
                  "下一步是有条件的后续计划，尚未在本轮执行。每个阶段仍需提前登记配置、对照和停止条件，保留失败记录，不能因为已知测试数字更好而跳过开发门槛。", "",
                  "本轮使用原训练集与开发集；早／晚开发时段只是稳定性诊断，不是独立盲测。内部测试之前已经揭晓，本轮未重评，也没有将其用于筛选。财报公告日期和历史发布版本缺失的限制仍然存在。", ""]
        (home / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
        print(home / "RESULTS.md")


if __name__ == "__main__":
    main()
