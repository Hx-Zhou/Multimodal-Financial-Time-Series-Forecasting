"""Train E3 variants (E3c by default), then evaluate the best trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset
from experiments.e2_gru.model import build_optimizer, compute_loss
from experiments.e2_gru.train import set_seed
from experiments.e3.model import build_model, transform_window
from experiments.validation_split import make_validation_split


def numeric_batch(
    dataset: HS300Dataset, indices: np.ndarray, device: torch.device, input_size: int,
    representation: str = "global", prediction_mode: str = "standardized_close",
) -> dict:
    """Gather windows by stock without allocating all overlapping windows."""
    indices = np.asarray(indices, dtype=np.int64)
    x = np.empty((len(indices), 96, input_size), dtype=np.float32)
    last_close = np.empty(len(indices), dtype=np.float32)
    last_raw = np.empty(len(indices), dtype=np.float32)
    close_std = np.empty(len(indices), dtype=np.float32)
    relative_target = (
        np.empty((len(indices), int(dataset.y.shape[1])), dtype=np.float32)
        if prediction_mode == "relative_return" else None
    )
    stock_indices = dataset.stock_index[indices]
    offsets = np.arange(96, dtype=np.int64)
    for stock in np.unique(stock_indices):
        mask = stock_indices == stock
        starts = dataset.input_start[indices[mask]].astype(np.int64)
        raw = dataset.price_values[dataset.stock_codes[int(stock)]][starts[:, None] + offsets]
        last_close[mask] = (
            raw[:, -1, 3] - float(dataset.close_mean[stock])
        ) / max(float(dataset.close_std[stock]), 1e-6)
        last_raw[mask] = raw[:, -1, 3]
        close_std[mask] = max(float(dataset.close_std[stock]), 1e-6)
        if relative_target is not None:
            horizon = relative_target.shape[1]
            future = dataset.price_values[dataset.stock_codes[int(stock)]][
                starts[:, None] + 96 + np.arange(horizon, dtype=np.int64)
            ][:, :, 3]
            relative_target[mask] = (
                future - raw[:, -1, 3, None]
            ) / np.maximum(np.abs(raw[:, -1, 3, None]), 1e-6)
        x[mask] = transform_window(
            raw, dataset.close_mean[stock], dataset.close_std[stock], input_size,
            representation,
        )
    batch = {
        "numerical": torch.as_tensor(x, device=device),
        "last_close": torch.as_tensor(last_close, dtype=torch.float32, device=device),
        "last_raw": torch.as_tensor(last_raw, dtype=torch.float32, device=device),
        "close_std": torch.as_tensor(close_std, dtype=torch.float32, device=device),
        "y": torch.as_tensor(dataset.y[indices], dtype=torch.float32, device=device),
    }
    if relative_target is not None:
        batch["target"] = torch.as_tensor(relative_target, dtype=torch.float32, device=device)
    return batch


def evaluate(dataset, indices, model, batch_size, device):
    model.eval()
    absolute = squared = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = numeric_batch(
                dataset,
                indices[start : start + batch_size],
                device,
                model.input_size,
                model.representation,
                model.prediction_mode,
            )
            prediction = model(batch)
            if model.prediction_mode == "relative_return":
                prediction = batch["last_close"][:, None] + prediction * (
                    batch["last_raw"] / batch["close_std"]
                )[:, None]
            error = prediction - batch["y"]
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
    if not count:
        raise ValueError("evaluation set is empty")
    return {"mae": absolute / count, "mse": squared / count}


def write_progress(path: Path, history: list[dict], split, experiment: str) -> None:
    """Persist development history while a long run is still active."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "development_samples": int(len(split.dev_indices)),
                "holdout_samples": int(len(split.holdout_indices)),
                "history": history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    if args.variant in ("e3b", "e3c", "e3d") and args.features != "ohlc":
        raise ValueError(f"{args.variant.upper()} is defined for the OHLC input only")
    set_seed(args.seed)
    device = torch.device(args.device)
    train = HS300Dataset(args.package_dir, "train") if not args.evaluate_holdout else None
    val = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(val, args.holdout_fraction)
        if args.evaluate_holdout:
            try:
                checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(args.checkpoint, map_location=device)
            if checkpoint.get("holdout_fraction") != args.holdout_fraction:
                raise ValueError("holdout fraction does not match the checkpoint")
            if int(checkpoint.get("epoch", 0)) < 1:
                raise ValueError(
                    "holdout evaluation requires a trained checkpoint (epoch >= 1); "
                    "the epoch 0 E0 reference is not a model checkpoint"
                )
            model = build_model(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            result = {
                "experiment": checkpoint.get(
                    "experiment",
                    "E3a_ohlc_GRU" if model.input_size == 4 else "E3_multivariate_GRU",
                ),
                "checkpoint_epoch": checkpoint["epoch"],
                "config": checkpoint["config"],
                "holdout": evaluate(
                    val, split.holdout_indices, model, args.batch_size, device
                ),
            }
        else:
            input_size = 4 if args.features == "ohlc" else 7
            if args.variant == "e3b":
                experiment = "E3b_damped_residual_GRU"
            elif args.variant == "e3c":
                experiment = "E3c_relative_ohlc_GRU"
            elif args.variant == "e3d":
                experiment = "E3d_relative_return_GRU"
            else:
                experiment = "E3a_ohlc_GRU" if input_size == 4 else "E3_multivariate_GRU"
            residual_scale = 0.1 if args.variant == "e3b" else 1.0
            representation = "relative" if args.variant == "e3c" else "global"
            if args.variant == "e3d":
                representation = "relative"
            prediction_mode = "relative_return" if args.variant == "e3d" else "standardized_close"
            config = {
                "hidden_size": args.hidden_size,
                "dropout": args.dropout,
                "horizon": 24,
                "input_size": input_size,
                "residual_scale": residual_scale,
                "representation": representation,
                "prediction_mode": prediction_mode,
            }
            model = build_model(config).to(device)
            optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
            indices = np.arange(len(train), dtype=np.int64)
            initial = evaluate(val, split.dev_indices, model, args.batch_size, device)
            # Epoch 0 is the E0 last-close reference only. It is logged for
            # comparison, but a trained epoch must win its own checkpoint.
            best_mae, best_epoch, wait = float("inf"), None, 0
            history = [{"epoch": 0, **initial}]
            progress_path = args.output.with_suffix(".progress.json")
            write_progress(progress_path, history, split, experiment)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"epoch 0 (E0 reference): dev_MAE={initial['mae']:.6f} "
                f"dev_MSE={initial['mse']:.6f}",
                flush=True,
            )
            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(indices)
                model.train()
                weighted_loss = 0.0
                total_batches = (len(indices) + args.batch_size - 1) // args.batch_size
                for step, start in enumerate(range(0, len(indices), args.batch_size), 1):
                    chosen = indices[start:start + args.batch_size]
                    batch = numeric_batch(
                        train, chosen, device, model.input_size,
                        model.representation, model.prediction_mode,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    target = batch.get("target", batch["y"])
                    loss = compute_loss(model(batch), target)
                    if not torch.isfinite(loss):
                        raise ValueError(f"non-finite loss at epoch {epoch}, batch {step}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    weighted_loss += loss.item() * len(chosen)
                    if step == 1 or step % 100 == 0 or step == total_batches:
                        print(f"epoch {epoch} batch {step}/{total_batches} loss={loss.item():.6f}", flush=True)
                dev = evaluate(val, split.dev_indices, model, args.batch_size, device)
                if not all(np.isfinite(value) for value in dev.values()):
                    raise ValueError(f"non-finite development metrics at epoch {epoch}")
                history.append({"epoch": epoch, "train_loss": weighted_loss / len(indices), **dev})
                write_progress(progress_path, history, split, experiment)
                print(f"epoch {epoch}: dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f}", flush=True)
                if dev["mae"] < best_mae:
                    best_mae, best_epoch, wait = dev["mae"], epoch, 0
                    torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch, "experiment": experiment, "holdout_fraction": args.holdout_fraction}, args.checkpoint)
                    print(f"saved best trained checkpoint: epoch {epoch} -> {args.checkpoint}", flush=True)
                else:
                    wait += 1
                    if wait >= args.patience:
                        break
            if best_epoch is None:
                raise RuntimeError("no trained epoch completed; no checkpoint was written")
            # Reload the selected trained checkpoint before evaluating holdout.
            # This guarantees that holdout is measured on the same weights that
            # were chosen from development MAE, rather than on the last epoch.
            try:
                selected = torch.load(args.checkpoint, map_location=device, weights_only=True)
            except TypeError:
                selected = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(selected["model_state"])
            best_trained = next(item for item in history if item["epoch"] == best_epoch)
            holdout = evaluate(val, split.holdout_indices, model, args.batch_size, device)
            progress_path.unlink(missing_ok=True)
            result = {
                "experiment": experiment,
                "seed": args.seed,
                "config": config,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "holdout_fraction": args.holdout_fraction,
                "e0_development": initial,
                "best_epoch": best_epoch,
                "best_development": best_trained,
                "holdout": {"checkpoint_epoch": best_epoch, **holdout},
                "mae_delta_vs_e0": best_trained["mae"] - initial["mae"],
                "history": history[1:],
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        if train is not None:
            train.close()
        val.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument(
        "--features", choices=("ohlc", "all"), default=None,
        help="input fields for training; holdout evaluation uses the checkpoint config",
    )
    parser.add_argument(
        "--variant", choices=("e3_all", "e3a", "e3b", "e3c", "e3d"), default="e3c",
        help="E3c (default), E3a global OHLC, E3b damped residual, E3d returns, E3_all seven fields",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    expected_features = "all" if args.variant == "e3_all" else "ohlc"
    if args.features is None:
        args.features = expected_features
    elif args.features != expected_features:
        parser.error(f"--variant {args.variant} requires --features {expected_features}")
    experiment_dir = Path("experiments/e3")
    if args.variant != "e3c":
        experiment_dir = experiment_dir / "archive" / args.variant
    if args.output is None:
        name = "holdout_check.json" if args.evaluate_holdout else "result.json"
        args.output = experiment_dir / name
    if args.checkpoint is None:
        args.checkpoint = experiment_dir / "checkpoints" / "best.pt"
    run(args)


if __name__ == "__main__":
    main()
