"""Run or summarize paired seeds for E10-B and E14-A, preserving original runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gru_e10_b import train as e10b
from experiments.gru_e10_b.model import build_model as build_e10b
from experiments.gru_e14_a import train as e14a
from experiments.gru_e14_a.model import build_model as build_e14a
from experiments.gru_e3.train import numeric_batch
from experiments.validation_split import make_validation_split
from hs300_dataset import HS300Dataset

HERE = Path(__file__).resolve().parent
FAMILIES = ("e10_b", "e14_a")
FAMILY_DIRS = {"e10_b": "gru_e10_b", "e14_a": "gru_e14_a"}
EXPERIMENTS = {"e10_b": "E10_B_frozen_E3c_normalized_financial",
               "e14_a": "E14_A_financial_field_masks"}
CONFIG = {"hidden_size": 64, "horizon": 24, "financial_dim": 16,
          "financial_hidden": 32, "dropout": 0.0}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_paths(args, family, seed):
    directory = args.output_dir / FAMILY_DIRS[family] / f"seed_{seed}"
    return directory, directory / "result.json", directory / "checkpoints/best.pt"


def protocol(args):
    return {
        "market_checkpoint_sha256": sha256(args.market_checkpoint),
        "train_split_sha256": sha256(args.package_dir / "data/train.npz"),
        "val_split_sha256": sha256(args.package_dir / "data/val.npz"),
        "config": CONFIG, "learning_rate": 3e-4, "weight_decay": 1e-5,
        "batch_size": args.batch_size, "report_lag_days": 30,
        "holdout_fraction": 0.2, "max_epochs": args.epochs,
        "patience": args.patience,
        "selection": "minimum development MAE among trained epochs; holdout never selects epochs or seeds",
    }


def validate_run(result, checkpoint, family, seed, args, market_state):
    expected = {
        "experiment": EXPERIMENTS[family], "seed": seed, "report_lag_days": 30,
        "holdout_fraction": 0.2, "batch_size": args.batch_size,
        "learning_rate": 3e-4, "config": CONFIG,
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise ValueError(f"{family} seed {seed}: {name} differs from this comparison")
    epoch = int(result.get("best_epoch", 0))
    if epoch < 1 or checkpoint.get("epoch") != epoch:
        raise ValueError(f"{family} seed {seed}: result/checkpoint epoch mismatch")
    if result.get("best_development", {}).get("epoch") != epoch:
        raise ValueError("best-development record does not match the checkpoint")
    if checkpoint.get("config") != CONFIG:
        raise ValueError("checkpoint architecture differs from this comparison")
    for key in ("report_lag_days", "holdout_fraction"):
        if checkpoint.get(key) != expected[key]:
            raise ValueError(f"checkpoint {key} does not match the result")
    for name in ("best_development", "holdout"):
        values = result.get(name, {})
        if not all(np.isfinite(values.get(metric, np.nan)) for metric in ("mae", "mse")):
            raise ValueError(f"missing or nonfinite {name} metrics")
    if result["holdout"].get("checkpoint_epoch") != epoch:
        raise ValueError("holdout did not use the selected development checkpoint")
    for prefix, original_prefix in (("market_gru.", "gru."), ("market_head.", "head.")):
        originals = {name[len(original_prefix):]: tensor for name, tensor in market_state.items()
                     if name.startswith(original_prefix)}
        if not originals:
            raise ValueError("the market checkpoint is not an E3c checkpoint")
        for name, tensor in originals.items():
            actual = checkpoint["model_state"].get(prefix + name)
            if actual is None or not torch.equal(actual, tensor):
                raise ValueError("frozen market weights do not match the reference E3c")
    stats = checkpoint.get("feature_stats")
    if not stats or not all(len(stats.get(key, [])) == 12 for key in ("mean", "std")):
        raise ValueError("missing training financial statistics")
    if not all(np.isfinite(stats[key]).all() for key in ("mean", "std")) or np.any(np.asarray(stats["std"]) <= 0):
        raise ValueError("invalid financial normalization statistics")


def import_seed42(args, family, current_protocol, market_state):
    directory, output, checkpoint_path = run_paths(args, family, 42)
    if not args.reuse_seed42 or output.exists():
        return
    source = ROOT / "experiments" / FAMILY_DIRS[family]
    source_result, source_checkpoint = source / "result.json", source / "checkpoints/best.pt"
    if not source_result.exists() or not source_checkpoint.exists():
        return
    # Historical scripts did not record max_epochs, patience or weight_decay.
    # Reuse is supported only for their documented default training protocol.
    if (args.epochs, args.patience, args.batch_size) != (20, 6, 256) or args.package_dir != ROOT:
        print(f"{family}: non-default protocol; seed 42 will be retrained", flush=True)
        return
    result = read_json(source_result)
    checkpoint = e10b.load_checkpoint(source_checkpoint, "cpu")
    validate_run(result, checkpoint, family, 42, args, market_state)
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, checkpoint_path)
    write_json(output, result)
    write_json(directory / "manifest.json", {
        "protocol": current_protocol,
        "origin": "existing_seed42",
        "source_result": str(source_result), "source_checkpoint": str(source_checkpoint),
        "unrecorded_historical_settings": {"max_epochs": 20, "patience": 6, "weight_decay": 1e-5},
        "note": "Historical settings above are the documented defaults, not independently recorded metadata.",
        "checkpoint_sha256": sha256(checkpoint_path), "result_sha256": sha256(output),
    })
    print(f"{family} seed 42: reused original result and copied checkpoint", flush=True)


def validate_existing(args, family, seed, current_protocol, market_state):
    directory, output, checkpoint_path = run_paths(args, family, seed)
    manifest_path = directory / "manifest.json"
    if not output.exists():
        return False
    if not checkpoint_path.exists() or not manifest_path.exists():
        raise ValueError(f"incomplete artifacts: {directory}; use another output directory or restore artifacts")
    manifest = read_json(manifest_path)
    if manifest.get("protocol") != current_protocol:
        raise ValueError(f"different comparison settings in {directory}; use another --output-dir")
    if manifest.get("checkpoint_sha256") != sha256(checkpoint_path) or manifest.get("result_sha256") != sha256(output):
        raise ValueError(f"result/checkpoint changed after completion: {directory}")
    validate_run(read_json(output), e10b.load_checkpoint(checkpoint_path, "cpu"), family, seed, args, market_state)
    return True


def train_one(args, family, seed, current_protocol, market_state):
    directory, output, checkpoint_path = run_paths(args, family, seed)
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", f"experiments.{FAMILY_DIRS[family]}.train",
               "--package-dir", str(args.package_dir), "--market-checkpoint", str(args.market_checkpoint),
               "--checkpoint", str(checkpoint_path), "--output", str(output),
               "--seed", str(seed), "--device", args.device, "--epochs", str(args.epochs),
               "--patience", str(args.patience), "--batch-size", str(args.batch_size)]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    write_json(args.output_dir / "progress.json", {"stage": "training", "model": family, "seed": seed,
                                                   "log": str(directory / "train.log")})
    print(f"\n{family} seed {seed}: starting training", flush=True)
    # Separate interpreters isolate E14-A's E10-B function substitutions.
    with (directory / "train.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise RuntimeError(f"{family} seed {seed} exited {returncode}; see {directory / 'train.log'}")
    result = read_json(output)
    validate_run(result, e10b.load_checkpoint(checkpoint_path, "cpu"), family, seed, args, market_state)
    write_json(directory / "manifest.json", {
        "protocol": current_protocol, "origin": "new_training", "command": command,
        "checkpoint_sha256": sha256(checkpoint_path), "result_sha256": sha256(output),
    })


def evaluate_windows(args, family, seed, dataset, split):
    directory, output, checkpoint_path = run_paths(args, family, seed)
    destination = directory / "development_windows.json"
    identity = {"checkpoint_sha256": sha256(checkpoint_path), "val_split_sha256": sha256(args.package_dir / "data/val.npz"),
                "folds": 3, "holdout_fraction": 0.2}
    if destination.exists() and read_json(destination).get("identity") == identity:
        return
    checkpoint = e10b.load_checkpoint(checkpoint_path, "cpu")
    builder, index_class, add_batch = (
        (build_e10b, e10b.FinancialIndex, e10b.add_financial_batch) if family == "e10_b" else
        (build_e14a, e14a.FinancialIndex, e14a.add_financial_batch))
    device = torch.device(args.device)
    model = builder(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    stats = checkpoint["feature_stats"]
    reports = index_class(dataset, 30, (np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)))
    dates = dataset.target_start_date
    groups = np.array_split(np.unique(dates[split.dev_indices]), 3)
    fold_ids = np.full(len(dataset), -1, dtype=np.int8)
    for i, group in enumerate(groups):
        fold_ids[split.dev_indices[np.isin(dates[split.dev_indices], group)]] = i
    totals = np.zeros((3, 3), dtype=np.float64)
    total_batches = (len(split.dev_indices) + args.batch_size - 1) // args.batch_size
    with torch.inference_mode():
        for step, start in enumerate(range(0, len(split.dev_indices), args.batch_size), 1):
            selected = split.dev_indices[start:start + args.batch_size]
            batch = numeric_batch(dataset, selected, device, 4, "relative", "standardized_close")
            batch = add_batch(dataset, selected, batch, reports)
            error = (model(batch) - batch["y"]).double()
            local_folds = fold_ids[selected]
            for i in np.unique(local_folds):
                part = error[torch.as_tensor(local_folds == i, device=device)]
                totals[i] += (part.abs().sum().item(), part.square().sum().item(), part.numel())
            if step == 1 or step % 50 == 0 or step == total_batches:
                print(f"{family} seed {seed} development windows: {step}/{total_batches}", flush=True)
                write_json(args.output_dir / "progress.json", {"stage": "development_windows", "model": family,
                                                               "seed": seed, "batch": step, "batches": total_batches})
    records = [{"start": int(group[0]), "end": int(group[-1]),
                "samples": int(np.count_nonzero(fold_ids == i)),
                "mae": float(totals[i, 0] / totals[i, 2]), "mse": float(totals[i, 1] / totals[i, 2])}
               for i, group in enumerate(groups)]
    aggregate = {"mae": float(totals[:, 0].sum() / totals[:, 2].sum()),
                 "mse": float(totals[:, 1].sum() / totals[:, 2].sum())}
    stored = read_json(output)["best_development"]
    if not all(np.isclose(aggregate[key], stored[key], rtol=1e-4, atol=1e-6) for key in aggregate):
        raise ValueError(f"reevaluated development metrics differ from result: {directory}")
    write_json(destination, {"identity": identity, "aggregate": aggregate, "windows": records})


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n": len(values)}


def summarize(args):
    by_family, missing, pending_windows = {}, [], []
    for family in FAMILIES:
        rows = []
        for seed in args.seeds:
            directory, output, _ = run_paths(args, family, seed)
            if not output.exists():
                missing.append({"model": family, "seed": seed})
                continue
            result = read_json(output)
            if not (directory / "development_windows.json").exists():
                pending_windows.append({"model": family, "seed": seed})
            rows.append({"seed": seed, "best_epoch": result["best_epoch"],
                         "development": {k: result["best_development"][k] for k in ("mae", "mse")},
                         "holdout": {k: result["holdout"][k] for k in ("mae", "mse")}})
        aggregate = {part: {metric: mean_std([row[part][metric] for row in rows])
                            for metric in ("mae", "mse")}
                     for part in ("development", "holdout")} if rows else {}
        by_family[family] = {"runs": rows, "aggregate": aggregate}
    paired = []
    first = {row["seed"]: row for row in by_family["e10_b"]["runs"]}
    second = {row["seed"]: row for row in by_family["e14_a"]["runs"]}
    for seed in args.seeds:
        if seed in first and seed in second:
            paired.append({"seed": seed, **{
                part: {metric: second[seed][part][metric] - first[seed][part][metric] for metric in ("mae", "mse")}
                for part in ("development", "holdout")}})
    summary = {
        "experiment": "E15_E10B_vs_E14A_multiseed", "requested_seeds": args.seeds,
        "status": "complete" if not missing and not pending_windows else "partial",
        "missing_runs": missing, "pending_window_evaluations": pending_windows,
        "scope": "financial branch initialization and shuffled training order; the same frozen E3c market checkpoint",
        "std": "sample standard deviation (ddof=1); null with fewer than two runs",
        "selection": "development MAE selects each run's epoch; all requested seeds are reported; holdout never selects seeds",
        "models": by_family, "paired_delta_e14a_minus_e10b": paired,
        "paired_delta_aggregate": {
            part: {metric: {**mean_std([row[part][metric] for row in paired]),
                            "e14a_wins": sum(row[part][metric] < 0 for row in paired)}
                   for metric in ("mae", "mse")} for part in ("development", "holdout")
        } if paired else {},
    }
    write_json(args.output_dir / "summary.json", summary)
    # Keep the detailed per-seed/time-window diagnostics in a separate file.
    window_rows = []
    for family in FAMILIES:
        for seed in args.seeds:
            directory, _, _ = run_paths(args, family, seed)
            path = directory / "development_windows.json"
            if path.exists():
                window_rows.append({"model": family, "seed": seed, "windows": read_json(path)["windows"]})
    window_aggregates = {}
    for family in FAMILIES:
        rows = [row for row in window_rows if row["model"] == family]
        if rows:
            window_aggregates[family] = [
                {"start": rows[0]["windows"][i]["start"], "end": rows[0]["windows"][i]["end"],
                 **{metric: mean_std([row["windows"][i][metric] for row in rows]) for metric in ("mae", "mse")}}
                for i in range(3)]
    write_json(args.output_dir / "development_windows.json", {"runs": window_rows, "aggregate": window_aggregates,
                                                               "note": "Fixed checkpoints on three development periods; not independent walk-forward retraining."})
    lines = ["# E10-B / E14-A 多种子对照", "", f"状态：{summary['status']}；预设种子：{args.seeds}", "",
             "标准差采用样本标准差；单个种子记为 —。以下结果不按 Holdout 选择种子。", "",
             "| 模型 | 已完成种子 | 开发 MAE 均值 ± 标准差 | 开发 MSE 均值 ± 标准差 | Holdout MAE 均值 ± 标准差 | Holdout MSE 均值 ± 标准差 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for family, record in by_family.items():
        entries = []
        for part in ("development", "holdout"):
            for metric in ("mae", "mse"):
                value = record["aggregate"].get(part, {}).get(metric)
                entries.append("—" if value is None else f"{value['mean']:.6f} ± " + (f"{value['std']:.6f}" if value["std"] is not None else "—"))
        lines.append(f"| {family} | {[row['seed'] for row in record['runs']]} | " + " | ".join(entries) + " |")
    if missing:
        lines += ["", "尚未完成全部种子，当前汇总不能用于多种子稳定性结论。"]
    elif pending_windows:
        lines += ["", "所有训练已完成，尚有分时段评估待完成。"]
    lines += ["", "同种子配对差值、胜出次数见 summary.json；时间窗口结果见 development_windows.json。", "",
              "三次训练只是初步稳定性检查，不是统计显著性的证明。"]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {args.output_dir / 'summary.md'} ({summary['status']})", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--market-checkpoint", type=Path, default=ROOT / "experiments/gru_e3/checkpoints/best.pt")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse-seed42", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize-only", action="store_true", help="reuse/import complete runs and summarize without training or inference")
    parser.add_argument("--evaluate-existing", action="store_true", help="evaluate development time windows of available runs; do not start training")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.patience, args.threads) < 1:
        parser.error("batch-size, epochs, patience and threads must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must be unique integers in [0, 2**32)")
    if args.evaluate_existing and args.summarize_only:
        parser.error("evaluate-existing and summarize-only are mutually exclusive")
    args.package_dir = args.package_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.market_checkpoint = args.market_checkpoint.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    current_protocol = protocol(args)
    market_state = e10b.load_checkpoint(args.market_checkpoint, "cpu")["model_state"]
    run_status = {}
    for family in FAMILIES:
        if 42 in args.seeds:
            import_seed42(args, family, current_protocol, market_state)
        for seed in args.seeds:
            run_status[family, seed] = validate_existing(args, family, seed, current_protocol, market_state)
    summarize(args)
    if args.summarize_only:
        return
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. The summary was prepared. Run on the GPU host, or use --evaluate-existing --device cpu.")
    dataset = None
    try:
        for seed in args.seeds:
            for family in FAMILIES:
                if not run_status[family, seed] and not args.evaluate_existing:
                    train_one(args, family, seed, current_protocol, market_state)
                    run_status[family, seed] = True
                    summarize(args)
                if run_status[family, seed]:
                    if dataset is None:
                        dataset = HS300Dataset(args.package_dir, "val")
                        split = make_validation_split(dataset, 0.2)
                    evaluate_windows(args, family, seed, dataset, split)
                    summarize(args)
        (args.output_dir / "progress.json").unlink(missing_ok=True)
    finally:
        if dataset is not None:
            dataset.close()


if __name__ == "__main__":
    main()
