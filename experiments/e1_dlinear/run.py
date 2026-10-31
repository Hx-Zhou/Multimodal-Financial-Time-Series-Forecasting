"""Screen DLinear kernels on dev only, then repeat the selected kernel across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from .data import CloseData, load_data, sha256_file
from .model import build_model, compute_loss

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    result = {"samples": len(target), "mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean())}
    for h in (1, 6, 12, 24):
        result[f"mae_h{h}"] = float(np.abs(error[:, h - 1]).mean())
        result[f"mse_h{h}"] = float(np.square(error[:, h - 1]).mean())
    return result


@torch.inference_mode()
def predict(model, x: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    prediction = torch.cat([model({"close": b}) for b in x.split(batch_size)]).cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError("non-finite predictions")
    return prediction


def fit(train: CloseData, dev: CloseData, kernel: int, seed: int, args) -> dict:
    run_id = f"k{kernel}_seed{seed}"
    output = args.output_dir / run_id
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {checkpoint}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = {"moving_average": kernel, "center_last": args.center_last}
    model = build_model(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_x, train_y = train.x.to(args.device), train.y.to(args.device)
    dev_x = dev.x.to(args.device)
    target = dev.y.numpy()
    generator = torch.Generator().manual_seed(seed)
    history, best, best_epoch, stale = [], float("inf"), 0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        loss_sum = 0.0
        for indices in order.split(args.batch_size):
            indices = indices.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model({"close": train_x[indices]}), train_y[indices])
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        score = metrics(predict(model, dev_x, args.batch_size), target)
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_x),
                  "dev": score, "seconds": time.perf_counter() - epoch_start}
        history.append(record)
        if score["mae"] < best - args.min_delta:
            best, best_epoch, stale = score["mae"], epoch, 0
            torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)
        else:
            stale += 1
        print(f"{run_id} epoch={epoch:02d} train={record['train_loss']:.6f} "
              f"dev_mae={score['mae']:.6f} dev_mse={score['mse']:.6f} "
              f"best={best_epoch} time={record['seconds']:.1f}s", flush=True)
        write_json(output / "history.json", history)
        if stale >= args.patience:
            break
    saved = torch.load(checkpoint, map_location=args.device, weights_only=True)
    restored = build_model(saved["config"]).to(args.device)
    restored.load_state_dict(saved["state_dict"])
    prediction = predict(restored, dev_x, args.batch_size)
    result = {
        "run_id": run_id, "seed": seed, "config": config,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch, "epochs_run": len(history),
        "training_seconds": time.perf_counter() - start,
        "development": metrics(prediction, target), "history": history,
        "checkpoint_sha256": sha256_file(checkpoint),
        "holdout_evaluated": False,
    }
    np.testing.assert_allclose(result["development"]["mae"], best, rtol=0, atol=1e-8)
    write_json(args.results_dir / f"{run_id}.json", result)
    # Kept only locally to derive group reports after selecting a kernel.
    np.save(output / "dev_predictions.npy", prediction)
    return result


def group_report(dev: CloseData, prediction: np.ndarray, output: Path) -> dict:
    target = dev.y.numpy()
    rows = []
    for i, code in enumerate(dev.codes):
        mask = dev.stocks == i
        if np.any(mask):
            score = metrics(prediction[mask], target[mask])
            rows.append({"stock_code": code, "samples": score["samples"], "mae": score["mae"], "mse": score["mse"]})
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = dev.news_counts
    nonzero = counts[counts > 0]
    threshold = float(np.median(nonzero)) if len(nonzero) else 0.0
    groups = {"no_news": counts == 0, "low_coverage": (counts > 0) & (counts <= threshold),
              "high_coverage": counts > threshold}
    return {"news_count_definition": "all records in the input date range, including weekends and duplicates",
            "nonzero_news_count_median": threshold,
            "groups": {name: metrics(prediction[mask], target[mask]) for name, mask in groups.items() if np.any(mask)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--kernels", type=int, nargs="+", default=[25, 15, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--center-last", action="store_true", help="center the input on the last observed Close")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam L2 penalty on weights and biases")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) <= 0 or args.lr <= 0 or args.min_delta < 0 or args.weight_decay < 0:
        parser.error("epochs, patience, batch-size, threads, lr must be positive; min-delta and weight-decay must be nonnegative")
    if len(set(args.kernels)) != len(args.kernels) or len(set(args.seeds)) != len(args.seeds):
        parser.error("kernels and seeds must not contain duplicates")
    for kernel in args.kernels:
        build_model({"moving_average": kernel})
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    if (args.results_dir / "summary.json").exists():
        parser.error("results already exist; choose fresh --results-dir and --output-dir")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    train, dev, split = load_data(args.package_dir.resolve())
    target = dev.y.numpy()
    baseline_prediction = np.repeat(dev.x[:, -1:].numpy(), 24, axis=1)
    baseline = metrics(baseline_prediction, target)
    reference = json.loads((ROOT / "experiments/e0/e0_val_results.json").read_text())["development"]["last_value"]
    for key in ("mae", "mse"):
        np.testing.assert_allclose(baseline[key], reference[key], rtol=0, atol=2e-6)
    print(f"Data verified: train={len(train.x)}, dev={len(dev.x)}; E0 MAE={baseline['mae']:.6f}, MSE={baseline['mse']:.6f}", flush=True)
    write_json(args.results_dir / "protocol.json", {
        "split": split, "selection": "minimum development MAE; checkpoint improvement threshold min_delta",
        "loss": "MAE + 0.2 * MSE", "residual_heads_initialization": "zeros (initial predictions equal E0)",
        "optimizer": "Adam; gradient_clip_norm=1", "weight_decay": args.weight_decay,
        "center_last": args.center_last, "batch_size": args.batch_size,
        "learning_rate": args.lr, "max_epochs": args.epochs, "patience": args.patience,
        "min_delta": args.min_delta, "threads": args.threads, "device": args.device,
        "screening_kernels": args.kernels, "seeds": args.seeds,
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha256_file(p) for p in
                          sorted(Path(__file__).parent.glob("*.py")) + [ROOT / "hs300_dataset.py", ROOT / "model.py", ROOT / "experiments/validation_split.py"]},
    })
    screening = [fit(train, dev, kernel, args.seeds[0], args) for kernel in args.kernels]
    winner = min(screening, key=lambda r: (r["development"]["mae"], r["development"]["mse"]))
    kernel = winner["config"]["moving_average"]
    print(f"Selected kernel={kernel} on development MAE. Repeating remaining seeds...", flush=True)
    repeats = [winner] + [fit(train, dev, kernel, seed, args) for seed in args.seeds[1:]]
    aggregates = {}
    for key in baseline:
        if key == "samples":
            continue
        values = np.array([r["development"][key] for r in repeats])
        aggregates[key] = {"mean": float(values.mean()), "std_population": float(values.std()),
                           "e0": baseline[key], "improvement_percent": float(100 * (1 - values.mean() / baseline[key]))}
    grouped = {"e0": group_report(dev, baseline_prediction, args.results_dir / "e0_per_stock.csv")}
    for run in repeats:
        predictions = np.load(args.output_dir / run["run_id"] / "dev_predictions.npy", allow_pickle=False)
        grouped[run["run_id"]] = group_report(dev, predictions, args.results_dir / f"{run['run_id']}_per_stock.csv")
    write_json(args.results_dir / "news_coverage_groups.json", grouped)
    compact = lambda r: {k: v for k, v in r.items() if k != "history"}
    summary = {"experiment": "E1", "center_last": args.center_last, "weight_decay": args.weight_decay,
               "split": split, "e0_development": baseline,
               "screening": [compact(r) for r in screening], "selected_kernel": kernel,
               "confirmation": [compact(r) for r in repeats], "aggregate": aggregates,
               "holdout_evaluated": False,
               "interpretation_note": "Seed variability describes optimization only, not uncertainty across time periods."}
    write_json(args.results_dir / "summary.json", summary)
    print(json.dumps({"selected_kernel": kernel, "mae": aggregates["mae"], "mse": aggregates["mse"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
