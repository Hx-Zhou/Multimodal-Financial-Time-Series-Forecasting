"""Optional starting template for HS300 forecasting.

You may replace the training helpers and model architecture. Scoring only
requires build_model(), prepare_batch(), forward(), and matching model.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from hs300_dataset import HS300Dataset


def prepare_batch(samples: list[dict], device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Convert fixed-reader samples into model inputs.

    Each sample provides:
      numerical: float32 [96, 7] (Open, High, Low, Close, Volume, Dividends, Stock Splits)
      dates: int32 [96]
      news[day]: a separate float32 [article_count, 512] array for that day
      news_time[day]: matching original article timestamps
      news.events(): all articles in the input range, including non-trading days
      financial[kind]: original reports with end_date <= input end date
      close_mean / close_std: training-period Close statistics for this stock
      y: float32 [24] standardized Close targets, only in train and val

    Return a dictionary of tensors. Teacher test samples never contain y.
    """
    raise NotImplementedError("Implement batch preparation in model.py")


def build_model(config: dict | None = None) -> nn.Module:
    """Return a model whose forward(batch) predicts float32 [batch, 24]."""
    raise NotImplementedError("Implement the model in model.py")


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Choose and configure an optimizer in this file."""
    raise NotImplementedError("Choose an optimizer in model.py")


def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Define the training objective in this file."""
    raise NotImplementedError("Define the loss function in model.py")


def batches(dataset: HS300Dataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        yield [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]


def train_and_validate(package_dir: Path, checkpoint: Path, epochs: int, batch_size: int) -> None:
    train = HS300Dataset(package_dir, "train")
    val = HS300Dataset(package_dir, "val")
    model = build_model()
    optimizer = build_optimizer(model)
    for epoch in range(epochs):
        model.train()
        for samples in batches(train, batch_size):
            batch = prepare_batch(samples)
            optimizer.zero_grad()
            prediction = model(batch)
            loss = compute_loss(prediction, batch["y"])
            loss.backward()
            optimizer.step()
        model.eval()
        absolute = 0.0
        squared = 0.0
        count = 0
        with torch.inference_mode():
            for samples in batches(val, batch_size):
                batch = prepare_batch(samples)
                difference = model(batch) - batch["y"]
                absolute += float(difference.abs().sum())
                squared += float(difference.square().sum())
                count += difference.numel()
        print(f"epoch {epoch + 1}: val_MAE={absolute / count:.6f}, val_MSE={squared / count:.6f}")
    torch.save(model.state_dict(), checkpoint)
    train.close()
    val.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", type=Path, default=Path("model.pt"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    train_and_validate(args.package_dir, args.checkpoint, args.epochs, args.batch_size)
