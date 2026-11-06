"""Compare E10-B under several financial report availability lags."""

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
HERE = Path(__file__).resolve().parent
LAGS = (30, 60, 90)
EXPERIMENT = "E14_B_report_lag_sensitivity"


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


def paths(args, lag):
    directory = args.output_dir / f"lag_{lag}"
    return directory, directory / "result.json", directory / "checkpoints/best.pt"


def protocol(args):
    return {
        "market_checkpoint_sha256": sha256(args.market_checkpoint),
        "train_split_sha256": sha256(args.package_dir / "data/train.npz"),
        "val_split_sha256": sha256(args.package_dir / "data/val.npz"),
        "lags": list(args.lags), "seed": args.seed,
        "learning_rate": 3e-4, "weight_decay": 1e-5,
        "batch_size": args.batch_size, "max_epochs": args.epochs,
        "patience": args.patience, "holdout_fraction": 0.2,
        "selection": "minimum development MAE among trained epochs and lags; holdout never selects lag",
    }


def validate_result(result, checkpoint, lag, args):
    if result.get("report_lag_days") != lag:
        raise ValueError(f"lag {lag}: result has another report lag")
    if result.get("seed") != args.seed or result.get("batch_size") != args.batch_size:
        raise ValueError(f"lag {lag}: seed or batch size differs")
    if result.get("holdout_fraction") != 0.2 or checkpoint.get("holdout_fraction") != 0.2:
        raise ValueError(f"lag {lag}: holdout fraction differs")
    if checkpoint.get("report_lag_days") != lag:
        raise ValueError(f"lag {lag}: checkpoint has another report lag")
    epoch = int(result.get("best_epoch", 0))
    if epoch < 1 or checkpoint.get("epoch") != epoch:
        raise ValueError(f"lag {lag}: result/checkpoint epoch mismatch")
    if result.get("best_development", {}).get("epoch") != epoch:
        raise ValueError(f"lag {lag}: best development record mismatch")
    for section in ("best_development", "holdout"):
        if not all(np.isfinite(result.get(section, {}).get(metric, np.nan)) for metric in ("mae", "mse")):
            raise ValueError(f"lag {lag}: missing or nonfinite {section} metrics")
    if result["holdout"].get("checkpoint_epoch") != epoch:
        raise ValueError(f"lag {lag}: holdout did not use the selected checkpoint")
    stats = checkpoint.get("feature_stats")
    if not stats or len(stats.get("mean", [])) != 12 or len(stats.get("std", [])) != 12:
        raise ValueError(f"lag {lag}: missing train-only feature statistics")
    if not np.isfinite(stats["mean"]).all() or not np.isfinite(stats["std"]).all() or np.any(np.asarray(stats["std"]) <= 0):
        raise ValueError(f"lag {lag}: invalid feature statistics")


def import_lag30(args, current_protocol):
    if 30 not in args.lags or not args.reuse_lag30:
        return
    directory, output, checkpoint = paths(args, 30)
    if output.exists():
        return
    source_result = ROOT / "experiments/gru_e10_b/result.json"
    source_checkpoint = ROOT / "experiments/gru_e10_b/checkpoints/best.pt"
    if not source_result.exists() or not source_checkpoint.exists():
        return
    result = read_json(source_result)
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    validate_result(result, checkpoint, 30, args)
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, paths(args, 30)[2])
    write_json(output, result)
    write_json(directory / "manifest.json", {
        "protocol": current_protocol, "origin": "existing_e10_b_lag30",
        "source_result": str(source_result), "source_checkpoint": str(source_checkpoint),
        "unrecorded_historical_settings": {"max_epochs": 20, "patience": 6, "weight_decay": 1e-5},
        "checkpoint_sha256": sha256(paths(args, 30)[2]), "result_sha256": sha256(output),
    })
    print("lag 30: reused existing E10-B result and checkpoint", flush=True)


def validate_existing(args, lag, current_protocol):
    directory, output, checkpoint_path = paths(args, lag)
    manifest_path = directory / "manifest.json"
    if not output.exists():
        return False
    if not checkpoint_path.exists() or not manifest_path.exists():
        raise ValueError(f"incomplete artifacts: {directory}")
    manifest = read_json(manifest_path)
    if manifest.get("protocol") != current_protocol:
        # The initial scaffold included lag 0 as a candidate. It was never
        # trained here; permit the existing lag-30 artifact to migrate to the
        # corrected candidate set [30, 60, 90].
        old = dict(manifest.get("protocol", {}))
        new = dict(current_protocol)
        old.pop("lags", None)
        new.pop("lags", None)
        if old != new or manifest.get("origin") != "existing_e10_b_lag30" or lag != 30:
            raise ValueError(f"different protocol in {directory}; use another output directory")
        manifest["protocol"] = current_protocol
        write_json(manifest_path, manifest)
    if manifest.get("checkpoint_sha256") != sha256(checkpoint_path) or manifest.get("result_sha256") != sha256(output):
        raise ValueError(f"artifact changed after completion: {directory}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_result(read_json(output), checkpoint, lag, args)
    return True


def train_one(args, lag, current_protocol):
    directory, output, checkpoint = paths(args, lag)
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", "experiments.gru_e10_b.train",
               "--package-dir", str(args.package_dir), "--market-checkpoint", str(args.market_checkpoint),
               "--checkpoint", str(checkpoint), "--output", str(output),
               "--report-lag-days", str(lag), "--seed", str(args.seed),
               "--device", args.device, "--epochs", str(args.epochs),
               "--patience", str(args.patience), "--batch-size", str(args.batch_size)]
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    write_json(args.output_dir / "progress.json", {"stage": "training", "lag": lag,
                                                     "log": str(directory / "train.log")})
    print(f"\nlag {lag}: starting training", flush=True)
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
        raise RuntimeError(f"lag {lag} exited {returncode}; see {directory / 'train.log'}")
    validate_result(read_json(output), torch.load(checkpoint, map_location="cpu", weights_only=True), lag, args)
    write_json(directory / "manifest.json", {
        "protocol": current_protocol, "origin": "new_training", "command": command,
        "checkpoint_sha256": sha256(checkpoint), "result_sha256": sha256(output),
    })


def summarize(args, current_protocol):
    rows, missing = [], []
    for lag in args.lags:
        directory, output, _ = paths(args, lag)
        if not output.exists():
            missing.append(lag)
            continue
        result = read_json(output)
        rows.append({"lag_days": lag, "best_epoch": result["best_epoch"],
                     "development": {k: result["best_development"][k] for k in ("mae", "mse")},
                     "holdout": {k: result["holdout"][k] for k in ("mae", "mse")}})
    complete = not missing
    selected = None
    if rows:
        selected = min(rows, key=lambda row: (row["development"]["mae"], row["development"]["mse"]))["lag_days"]
    summary = {
        "experiment": EXPERIMENT, "requested_lags": list(args.lags),
        "status": "complete" if complete else "partial", "missing_lags": missing,
        "seed": args.seed, "selection": "development MAE, MSE tiebreak; holdout is report-only",
        "rows": rows, "selected_lag_by_development": selected,
        "protocol": current_protocol,
    }
    write_json(args.output_dir / "summary.json", summary)
    lines = ["# E14-B 财报可用滞后敏感性", "", f"状态：{summary['status']}；种子：{args.seed}", "",
             "只用开发集 MAE（MSE 平手时）选择滞后；Holdout 不参与选择。指标越低越好。", "",
             "| 财报滞后天数 | 最佳轮次 | 开发 MAE | 开发 MSE | Holdout MAE | Holdout MSE |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(rows, key=lambda row: row["lag_days"]):
        lines.append(f"| {row['lag_days']} | {row['best_epoch']} | {row['development']['mae']:.6f} | {row['development']['mse']:.6f} | {row['holdout']['mae']:.6f} | {row['holdout']['mse']:.6f} |")
    if selected is not None:
        lines += ["", f"开发集选择：**{selected} 天**。"]
    if missing:
        lines += ["", f"尚未完成：`{missing}`。"]
    lines += ["", "所有滞后都使用独立训练集财报统计量；Holdout 只在训练后的检查点上报告。", "",
              "这是一组固定种子的敏感性实验，不是多随机种子稳定性证明。"]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summary: {args.output_dir / 'summary.md'} ({summary['status']})", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--market-checkpoint", type=Path, default=ROOT / "experiments/gru_e3/checkpoints/best.pt")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--lags", type=int, nargs="+", default=list(LAGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reuse-lag30", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.lags or len(set(args.lags)) != len(args.lags) or any(lag < 0 for lag in args.lags):
        parser.error("lags must be unique nonnegative integers")
    if min(args.batch_size, args.epochs, args.patience, args.threads) < 1:
        parser.error("batch-size, epochs, patience and threads must be positive")
    args.package_dir = args.package_dir.resolve()
    args.market_checkpoint = args.market_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    current_protocol = protocol(args)
    import_lag30(args, current_protocol)
    status = {lag: validate_existing(args, lag, current_protocol) for lag in args.lags}
    summarize(args, current_protocol)
    if args.summarize_only:
        return
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use --device cuda on the GPU host, or --summarize-only here.")
    for lag in args.lags:
        if not status[lag]:
            train_one(args, lag, current_protocol)
            status[lag] = True
            summarize(args, current_protocol)
    (args.output_dir / "progress.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
