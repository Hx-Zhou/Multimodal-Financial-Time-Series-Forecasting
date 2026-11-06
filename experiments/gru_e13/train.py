"""Audit news alignment, fit three small residual controls, then test real news."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gru_e10_b.model import build_model
from experiments.gru_e10_b.train import FinancialIndex, load_checkpoint, make_batch
from experiments.gru_e13.model import RidgeResidual
from experiments.gru_e13.news import (
    AlignedNewsIndex, fit_pca, project_vectors, shuffled_projection,
)
from experiments.validation_split import make_validation_split
from hs300_dataset import HS300Dataset

HERE = Path(__file__).resolve().parent
GROUPS = ("metadata_only", "real_content", "shuffled_content")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def metrics(prediction, target):
    error = np.asarray(prediction, dtype=np.float64) - target
    return {"mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean())}


def date_windows(dataset, indices, prediction):
    dates = dataset.target_start_date[indices]
    result = []
    for group in np.array_split(np.unique(dates), 3):
        if not len(group):
            continue
        rows = np.isin(dates, group)
        result.append({"start": int(group[0]), "end": int(group[-1]),
                       "samples": int(rows.sum()),
                       **metrics(prediction[rows], dataset.y[indices[rows]])})
    return result


def frozen_base(checkpoint, device):
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.requires_grad_(False)
    model.eval()
    return model


def financial_index(dataset, checkpoint):
    stats = checkpoint["feature_stats"]
    return FinancialIndex(dataset, int(checkpoint["report_lag_days"]), (
        np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32),
    ))


def base_predictions(dataset, indices, model, reports, args, label):
    output = np.empty((len(indices), dataset.y.shape[1]), np.float32)
    total = (len(indices) + args.batch_size - 1) // args.batch_size
    with torch.inference_mode():
        for step, start in enumerate(range(0, len(indices), args.batch_size), 1):
            selected = indices[start:start + args.batch_size]
            batch = make_batch(dataset, selected, reports, torch.device(args.device))
            output[start:start + len(selected)] = model(batch).cpu().numpy()
            if step == 1 or step % 100 == 0 or step == total:
                message = {"stage": label, "batch": step, "batches": total}
                write_json(args.output_dir / "progress.json", message)
                print(f"{label}: batch {step}/{total}", flush=True)
    if not np.isfinite(output).all():
        raise ValueError("non-finite E10-B predictions")
    return output


def subset(indices, size):
    return indices[np.linspace(0, len(indices) - 1, min(size, len(indices)), dtype=np.int64)]


def evaluate_saved(args):
    saved = load_checkpoint(args.output_dir / "checkpoints/best.pt", "cpu")
    if saved.get("smoke_test"):
        raise ValueError("a smoke-test checkpoint cannot be used for holdout evaluation")
    if saved["holdout_fraction"] != args.holdout_fraction:
        raise ValueError("holdout fraction differs from saved checkpoint")
    dataset = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(dataset, saved["holdout_fraction"])
        base = frozen_base(saved["base"], torch.device(args.device))
        prediction = base_predictions(dataset, split.holdout_indices, base,
                                      financial_index(dataset, saved["base"]), args, "holdout")
        news = AlignedNewsIndex(dataset, saved["cutoff_time"])
        projected = project_vectors(dataset.news_vectors, saved["pca_mean"].numpy(), saved["pca_basis"].numpy())
        content, metadata, active = news.features(dataset, split.holdout_indices, projected, saved["windows"])
        correction = RidgeResidual.from_state_dict(saved["residual"]).predict(
            np.concatenate((metadata, content), axis=1), active)
        result = {"experiment": "E13_news_residual", "selected_alpha": saved["alpha"],
                  "base": metrics(prediction, dataset.y[split.holdout_indices]),
                  "real_content": metrics(prediction + correction, dataset.y[split.holdout_indices])}
        write_json(args.output_dir / "holdout_evaluation.json", result)
        (args.output_dir / "progress.json").unlink(missing_ok=True)
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        dataset.close()


def run(args):
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch-size and threads must be positive")
    if not args.alphas or any(not np.isfinite(x) or x <= 0 for x in args.alphas):
        raise ValueError("all ridge alphas must be finite and positive")
    if not args.windows or any(x < 1 or x > 96 for x in args.windows):
        raise ValueError("windows must be between 1 and 96 trading intervals")
    if sum((args.evaluate_holdout, args.audit_only, args.smoke_test)) > 1:
        raise ValueError("evaluate-holdout, audit-only and smoke-test are exclusive")
    torch.set_num_threads(args.threads)
    if args.smoke_test:
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.evaluate_holdout:
        return evaluate_saved(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke test")
    train = HS300Dataset(args.package_dir, "train")
    val = None
    try:
        train_indices = np.arange(len(train), dtype=np.int64)
        if args.smoke_test:
            train_indices = subset(train_indices, 512)
        news = AlignedNewsIndex(train, args.cutoff_time)
        training_ids = news.training_vector_ids(train, train_indices, max(args.windows))
        audit = {"scope": "full news history of stocks in the dataset; not restricted to training dates",
                 "cutoff_time": args.cutoff_time, "deduplication": "earliest timestamp per (stock, vector_id)",
                 **news.audit, "pca_eligible_training_vectors": len(training_ids)}
        write_json(args.output_dir / "news_audit.json", audit)
        print(json.dumps(audit, indent=2), flush=True)
        if args.audit_only:
            return audit

        print("fit training-only PCA", flush=True)
        pca_mean, pca_basis, pca_count = fit_pca(
            train.news_vectors, training_ids, args.pca_components,
            min(args.pca_max_vectors, 2048) if args.smoke_test else args.pca_max_vectors, args.seed)
        projected = project_vectors(train.news_vectors, pca_mean, pca_basis)
        shuffled = shuffled_projection(projected, training_ids, args.seed)
        val = HS300Dataset(args.package_dir, "val")
        if train.stock_codes != val.stock_codes:
            val_news = AlignedNewsIndex(val, args.cutoff_time)
        else:
            val_news = news  # Both readers share the full price/news history.
        split = make_validation_split(val, args.holdout_fraction)
        dev_indices = subset(split.dev_indices, 256) if args.smoke_test else split.dev_indices
        checkpoint = load_checkpoint(args.base_checkpoint, "cpu")
        if checkpoint.get("holdout_fraction") != args.holdout_fraction:
            raise ValueError("holdout fraction differs from E10-B checkpoint")
        if not checkpoint.get("feature_stats"):
            raise ValueError("E10-B checkpoint is missing financial statistics")
        base = frozen_base(checkpoint, device)
        train_base = base_predictions(train, train_indices, base, financial_index(train, checkpoint), args, "train base")
        val_reports = financial_index(val, checkpoint)
        dev_base = base_predictions(val, dev_indices, base, val_reports, args, "development base")
        residual = train.y[train_indices] - train_base

        print("build aligned, deduplicated news features", flush=True)
        train_content, train_meta, train_active = news.features(train, train_indices, projected, args.windows)
        dev_content, dev_meta, dev_active = val_news.features(val, dev_indices, projected, args.windows)
        shuffled_train, _, _ = news.features(train, train_indices, shuffled, args.windows)
        shuffled_dev, _, _ = val_news.features(val, dev_indices, shuffled, args.windows)
        features = {
            "metadata_only": (train_meta, dev_meta),
            "real_content": (np.concatenate((train_meta, train_content), axis=1),
                             np.concatenate((dev_meta, dev_content), axis=1)),
            "shuffled_content": (np.concatenate((train_meta, shuffled_train), axis=1),
                                 np.concatenate((dev_meta, shuffled_dev), axis=1)),
        }
        search, selected = {}, {}
        window_diagnostics = {"e10_b": date_windows(val, dev_indices, dev_base)}
        models = {}
        for name in GROUPS:
            x_train, x_dev = features[name]
            prepared = RidgeResidual.prepare(x_train, train_active, residual)
            search[name] = []
            for alpha in args.alphas:
                model = RidgeResidual.fit_prepared(prepared, alpha)
                prediction = dev_base + model.predict(x_dev, dev_active)
                record = {"alpha": alpha, **metrics(prediction, val.y[dev_indices])}
                search[name].append(record)
                print(f"{name} alpha={alpha:g}: dev_MAE={record['mae']:.6f} dev_MSE={record['mse']:.6f}", flush=True)
                if name not in selected or (record["mae"], record["mse"]) < (selected[name]["mae"], selected[name]["mse"]):
                    selected[name] = record
                    models[name] = model
            prediction = dev_base + models[name].predict(x_dev, dev_active)
            window_diagnostics[name] = date_windows(val, dev_indices, prediction)
        write_json(args.output_dir / "development_search.json", search)
        write_json(args.output_dir / "development_windows.json", window_diagnostics)

        # Always retain the best trained real-content model, even when its
        # development score is worse than the unmodified base. Do not silently
        # replace it with E10-B or select a model using holdout labels.
        saved = {"experiment": "E13_news_residual", "smoke_test": args.smoke_test,
                 "base": checkpoint, "base_checkpoint": str(args.base_checkpoint),
                 "pca_mean": torch.from_numpy(pca_mean), "pca_basis": torch.from_numpy(pca_basis),
                 "cutoff_time": args.cutoff_time, "windows": args.windows,
                 "holdout_fraction": args.holdout_fraction,
                 "alpha": selected["real_content"]["alpha"],
                 "residual": models["real_content"].state_dict()}
        (args.output_dir / "checkpoints").mkdir(exist_ok=True)
        torch.save(saved, args.output_dir / "checkpoints/best.pt")
        base_dev = metrics(dev_base, val.y[dev_indices])
        result = {
            "experiment": "E13_news_residual", "smoke_test": args.smoke_test,
            "seed": args.seed, "base_checkpoint": str(args.base_checkpoint),
            "cutoff_time": args.cutoff_time, "windows": args.windows,
            "pca_components": args.pca_components, "pca_fit_articles": pca_count,
            "training_samples": len(train_indices), "development_samples": len(dev_indices),
            "training_news_coverage": float(train_active.mean()),
            "development_news_coverage": float(dev_active.mean()),
            "objective": "per horizon: sample-mean squared residual error + alpha * squared coefficients (including masked intercept)",
            "selection": "development MAE, then MSE; holdout only evaluates selected real-content model and base",
            "development": {"e10_b": base_dev, **selected},
            "real_content_beats_base_dev_mae": selected["real_content"]["mae"] < base_dev["mae"],
            "real_content_beats_both_controls_dev_mae": all(selected["real_content"]["mae"] < selected[name]["mae"] for name in ("metadata_only", "shuffled_content")),
            "holdout": None,
        }
        write_json(args.output_dir / "result.json", result)
        if not args.smoke_test and not args.skip_holdout:
            hold_base = base_predictions(val, split.holdout_indices, base, val_reports, args, "holdout base")
            content, meta, active = val_news.features(val, split.holdout_indices, projected, args.windows)
            correction = models["real_content"].predict(np.concatenate((meta, content), axis=1), active)
            result["holdout"] = {
                "e10_b": metrics(hold_base, val.y[split.holdout_indices]),
                "real_content": metrics(hold_base + correction, val.y[split.holdout_indices]),
            }
            write_json(args.output_dir / "result.json", result)
        (args.output_dir / "progress.json").unlink(missing_ok=True)
        print(f"completed; result: {args.output_dir / 'result.json'}", flush=True)
        print(json.dumps({"development": result["development"],
                          "holdout": result["holdout"]}, indent=2), flush=True)
        return result
    finally:
        train.close()
        if val is not None:
            val.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--base-checkpoint", type=Path, default=ROOT / "experiments/gru_e10_b/checkpoints/best.pt")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff-time", type=int, default=235959)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--pca-max-vectors", type=int, default=50000)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
