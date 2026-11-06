"""Train the joint E9-3 news/financial adapters and evaluate holdout."""

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

from experiments.gru_e2.model import compute_loss
from experiments.gru_e2.train import set_seed
from experiments.gru_e3.train import numeric_batch as market_batch
from experiments.gru_e9_1.train import DailyNewsIndex, add_news_batch
from experiments.gru_e8.train import FinancialIndex, add_financial_batch
from experiments.gru_e9_3.model import build_model
from experiments.validation_split import make_validation_split
from hs300_dataset import HS300Dataset


def make_batch(dataset, indices, news_index, financial_index, device):
    market = market_batch(dataset, indices, device, 4, "relative", "standardized_close")
    with_news = add_news_batch(dataset, indices, market, news_index)
    return add_financial_batch(dataset, indices, with_news, financial_index)


def evaluate(dataset, indices, model, news_index, financial_index, batch_size, device):
    model.eval()
    absolute = squared = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = make_batch(
                dataset, indices[start : start + batch_size], news_index, financial_index, device
            )
            error = model(batch) - batch["y"]
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
    if count == 0:
        raise ValueError("evaluation set is empty")
    return {"mae": absolute / count, "mse": squared / count}


def load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def run(args):
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    set_seed(args.seed)
    device = torch.device(args.device)
    train = None if args.evaluate_holdout else HS300Dataset(args.package_dir, "train")
    val = HS300Dataset(args.package_dir, "val")
    try:
        split = make_validation_split(val, args.holdout_fraction)
        val_news = DailyNewsIndex(val)
        val_financial = FinancialIndex(val)

        if args.evaluate_holdout:
            checkpoint = load_checkpoint(args.checkpoint, device)
            if checkpoint.get("holdout_fraction") != args.holdout_fraction:
                raise ValueError("holdout fraction does not match checkpoint")
            if int(checkpoint.get("epoch", 0)) < 1:
                raise ValueError("holdout requires an E9-3 trained checkpoint")
            model = build_model(checkpoint["config"]).to(device)
            model.load_state_dict(checkpoint["model_state"])
            result = {
                "experiment": "E9_3_frozen_E3c_joint_news_financial",
                "checkpoint_epoch": checkpoint["epoch"],
                "config": checkpoint["config"],
                "holdout": evaluate(
                    val, split.holdout_indices, model, val_news, val_financial,
                    args.batch_size, device,
                ),
            }
        else:
            e3c = load_checkpoint(args.market_checkpoint, device)
            news_checkpoint = load_checkpoint(args.news_checkpoint, device)
            financial_checkpoint = load_checkpoint(args.financial_checkpoint, device)
            config = {
                "hidden_size": int(e3c["config"].get("hidden_size", 64)),
                "horizon": int(e3c["config"].get("horizon", 24)),
                "news_dim": args.news_dim,
                "news_hidden": args.news_hidden,
                "financial_dim": args.financial_dim,
                "financial_hidden": args.financial_hidden,
                "dropout": args.dropout,
            }
            model = build_model(config).to(device)
            model.load_e3c_state(e3c["model_state"])
            if args.initialize_adapters:
                model.load_news_state(news_checkpoint["model_state"])
                model.load_financial_state(financial_checkpoint["model_state"])

            train_news = DailyNewsIndex(train)
            train_financial = FinancialIndex(train)
            indices = np.arange(len(train), dtype=np.int64)

            # Keep a pure E3c reference separate from the adapter-initialized
            # starting point, since loading E9-1/E9-2 gates can change epoch 0.
            saved_news_gate = model.news_gate.detach().clone()
            saved_financial_gate = model.financial_gate.detach().clone()
            model.zero_gates()
            e3c_reference = evaluate(
                val, split.dev_indices, model, val_news, val_financial, args.batch_size, device
            )
            with torch.no_grad():
                model.news_gate.copy_(saved_news_gate)
                model.financial_gate.copy_(saved_financial_gate)
            initial = evaluate(
                val, split.dev_indices, model, val_news, val_financial, args.batch_size, device
            )

            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable, lr=args.learning_rate, weight_decay=args.weight_decay
            )
            history = [{"epoch": 0, **initial}]
            best_mae = float("inf")
            best_epoch = None
            wait = 0
            progress = args.output.with_suffix(".progress.json")
            progress.write_text(
                json.dumps({"experiment": "E9_3_frozen_E3c_joint_news_financial", "history": history}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"epoch 0 (E3c reference): dev_MAE={e3c_reference['mae']:.6f} "
                f"dev_MSE={e3c_reference['mse']:.6f}; "
                f"initialized fusion: dev_MAE={initial['mae']:.6f} dev_MSE={initial['mse']:.6f}",
                flush=True,
            )

            for epoch in range(1, args.epochs + 1):
                np.random.shuffle(indices)
                model.train()
                weighted_loss = 0.0
                total_batches = (len(indices) + args.batch_size - 1) // args.batch_size
                for step, start in enumerate(range(0, len(indices), args.batch_size), 1):
                    batch = make_batch(
                        train, indices[start : start + args.batch_size],
                        train_news, train_financial, device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss = compute_loss(model(batch), batch["y"])
                    if not torch.isfinite(loss):
                        raise ValueError(f"non-finite loss at epoch {epoch}, batch {step}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    weighted_loss += loss.item() * len(batch["y"])
                    if step == 1 or step % 100 == 0 or step == total_batches:
                        print(
                            f"epoch {epoch} batch {step}/{total_batches} loss={loss.item():.6f}",
                            flush=True,
                        )

                dev = evaluate(
                    val, split.dev_indices, model, val_news, val_financial,
                    args.batch_size, device,
                )
                record = {"epoch": epoch, "train_loss": weighted_loss / len(indices), **dev}
                history.append(record)
                progress.write_text(
                    json.dumps(
                        {"experiment": "E9_3_frozen_E3c_joint_news_financial", "history": history},
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    f"epoch {epoch}: dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f}",
                    flush=True,
                )
                if dev["mae"] < best_mae:
                    best_mae, best_epoch, wait = dev["mae"], epoch, 0
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "config": config,
                            "epoch": epoch,
                            "experiment": "E9_3_frozen_E3c_joint_news_financial",
                            "holdout_fraction": args.holdout_fraction,
                            "market_checkpoint": str(args.market_checkpoint),
                            "news_checkpoint": str(args.news_checkpoint),
                            "financial_checkpoint": str(args.financial_checkpoint),
                        },
                        args.checkpoint,
                    )
                else:
                    wait += 1
                    if wait >= args.patience:
                        break

            if best_epoch is None:
                raise RuntimeError("no trained epoch completed")
            selected = load_checkpoint(args.checkpoint, device)
            model.load_state_dict(selected["model_state"])
            best_record = next(item for item in history if item["epoch"] == best_epoch)
            holdout = evaluate(
                val, split.holdout_indices, model, val_news, val_financial,
                args.batch_size, device,
            )
            progress.unlink(missing_ok=True)
            result = {
                "experiment": "E9_3_frozen_E3c_joint_news_financial",
                "seed": args.seed,
                "config": config,
                "market_checkpoint": str(args.market_checkpoint),
                "news_checkpoint": str(args.news_checkpoint),
                "financial_checkpoint": str(args.financial_checkpoint),
                "initialized_adapters": bool(args.initialize_adapters),
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "holdout_fraction": args.holdout_fraction,
                "e3c_development": e3c_reference,
                "initialized_development": initial,
                "best_epoch": best_epoch,
                "best_development": best_record,
                "holdout": {"checkpoint_epoch": best_epoch, **holdout},
                "mae_delta_vs_e3c": best_record["mae"] - e3c_reference["mae"],
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
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--market-checkpoint", type=Path, default=Path("experiments/gru_e3/checkpoints/best.pt"))
    parser.add_argument("--news-checkpoint", type=Path, default=Path("experiments/gru_e9_1/checkpoints/best.pt"))
    parser.add_argument("--financial-checkpoint", type=Path, default=Path("experiments/gru_e9_2/checkpoints/best.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/gru_e9_3/checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/gru_e9_3/result.json"))
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--initialize-adapters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--news-dim", type=int, default=16)
    parser.add_argument("--news-hidden", type=int, default=32)
    parser.add_argument("--financial-dim", type=int, default=16)
    parser.add_argument("--financial-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
