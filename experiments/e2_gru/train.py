"""Train and evaluate the E2 single-variable GRU experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split
from experiments.e2_gru.model import build_model, build_optimizer, compute_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def materialize_numeric(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialize only standardized Close and y for the E2 experiment."""
    sample_count = len(dataset)
    horizon = int(dataset.y.shape[1])
    close = np.empty((sample_count, 96, 1), dtype=np.float32)
    offsets = np.arange(96, dtype=np.int64)[None, :]
    for stock_index, code in enumerate(dataset.stock_codes):
        mask = dataset.stock_index == stock_index
        if not mask.any():
            continue
        starts = np.asarray(dataset.input_start[mask], dtype=np.int64)[:, None]
        raw_close = dataset.price_values[code][:, 3]
        selected = raw_close[starts + offsets]
        mean = np.float32(dataset.close_mean[stock_index])
        std = max(np.float32(dataset.close_std[stock_index]), np.float32(1e-6))
        close[mask, :, 0] = (selected - mean) / std
    target = np.asarray(dataset.y, dtype=np.float32).reshape(sample_count, horizon)
    return close, target


def train_batches(close, target, indices, model, optimizer, batch_size, device):
    model.train()
    losses = []
    total_batches = (len(indices) + batch_size - 1) // batch_size
    for batch_no, start in enumerate(range(0, len(indices), batch_size), start=1):
        selected = indices[start : start + batch_size]
        close_batch = torch.as_tensor(close[selected], dtype=torch.float32, device=device)
        batch = {
            "close": close_batch,
            "last_close": close_batch[:, -1, 0],
            "y": torch.as_tensor(target[selected], dtype=torch.float32, device=device),
        }
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model(batch), batch["y"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if batch_no == 1 or batch_no % 100 == 0 or batch_no == total_batches:
            print(
                f"  train batch {batch_no}/{total_batches} loss={loss_value:.6f}",
                flush=True,
            )
    return float(np.mean(losses)) if losses else float("nan")


def evaluate(close, target, indices, model, batch_size, device):
    model.eval()
    absolute = 0.0
    squared = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            close_batch = torch.as_tensor(close[selected], dtype=torch.float32, device=device)
            batch = {
                "close": close_batch,
                "last_close": close_batch[:, -1, 0],
                "y": torch.as_tensor(target[selected], dtype=torch.float32, device=device),
            }
            error = model(batch) - batch["y"]
            absolute += float(error.abs().sum().cpu())
            squared += float(error.square().sum().cpu())
            count += error.numel()
    return {"mae": absolute / count, "mse": squared / count}


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    train = HS300Dataset(args.package_dir, "train")
    val = HS300Dataset(args.package_dir, "val")
    split = make_validation_split(val, args.holdout_fraction)
    print("materializing standardized Close arrays...", flush=True)
    train_close, train_target = materialize_numeric(train)
    val_close, val_target = materialize_numeric(val)
    print(
        f"ready: train={len(train_close)} val={len(val_close)} "
        f"close_memory={(train_close.nbytes + val_close.nbytes) / 1024**2:.1f} MiB",
        flush=True,
    )
    train_indices = np.arange(len(train), dtype=np.int64)
    model = build_model({"hidden_size": args.hidden_size, "dropout": args.dropout}).to(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    best_mae = float("inf")
    best_epoch = 0
    wait = 0
    history = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(1, args.epochs + 1):
            if args.shuffle:
                np.random.shuffle(train_indices)
            train_loss = train_batches(
                train_close, train_target, train_indices, model, optimizer, args.batch_size, device
            )
            development = evaluate(
                val_close, val_target, split.dev_indices, model, args.batch_size, device
            )
            history.append({"epoch": epoch, "train_loss": train_loss, **development})
            print(
                f"epoch {epoch}: train_loss={train_loss:.6f} "
                f"dev_MAE={development['mae']:.6f} dev_MSE={development['mse']:.6f}",
                flush=True,
            )
            if development["mae"] < best_mae:
                best_mae = development["mae"]
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), args.checkpoint)
            else:
                wait += 1
                if wait >= args.patience:
                    break

        try:
            state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        development = evaluate(
            val_close, val_target, split.dev_indices, model, args.batch_size, device
        )
        holdout = evaluate(
            val_close, val_target, split.holdout_indices, model, args.batch_size, device
        )
        result = {
            "experiment": "E2_single_variable_GRU",
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "best_epoch": best_epoch,
            "development_samples": int(len(split.dev_indices)),
            "holdout_samples": int(len(split.holdout_indices)),
            "purged_overlap_samples": int(len(split.purged_indices)),
            "development": development,
            "holdout": holdout,
            "history": history,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("experiment", "best_epoch", "development", "holdout")}, indent=2))
        return result
    finally:
        train.close()
        val.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/e2_gru/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/e2_gru/results.json"))
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
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    parser.set_defaults(shuffle=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
