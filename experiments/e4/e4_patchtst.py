"""E4: PatchTST on standardized Close only, with residual prediction.

Follows experiments/e0/ conventions and the unified settings in
docs/experiment_route.md sections 2 and 4:

  - input: standardized Close [96] (last 96 trading days, per-stock close_mean/close_std)
  - patch_len=16, stride=8, 2 encoder layers, d_model=128, 4 heads, d_ff=256, dropout=0.1
  - residual prediction: y_hat_h = z_96 + delta_h (direct multi-step, no autoregression)
  - loss: MAE + 0.2 * MSE
  - chronological dev/holdout split from experiments.validation_split

Run from the repository root:

    python -m experiments.e4.e4_patchtst --epochs 20 --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split

# Unified settings (docs/experiment_route.md sections 2 and 4).
SEQ_LEN = 96
PRED_LEN = 24
PATCH_LEN = 16
STRIDE = 8
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
HORIZONS = (1, 6, 12, 24)
CLOSE_COLUMN = 3  # PRICE_COLUMNS: Open, High, Low, Close, Volume, Dividends, Stock Splits


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PatchTST(nn.Module):
    """Channel-independent PatchTST for a single standardized Close series."""

    def __init__(self) -> None:
        super().__init__()
        # Replicate the last value by `stride` before patching (SpecTF convention).
        self.n_patches = (SEQ_LEN - PATCH_LEN) // STRIDE + 2  # 12 for (96, 16, 8)
        self.patch_embed = nn.Linear(PATCH_LEN, D_MODEL)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, D_MODEL))
        self.dropout = nn.Dropout(DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(self.n_patches * D_MODEL, PRED_LEN)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, SEQ_LEN] standardized Close -> [B, PRED_LEN] residual prediction."""
        last = x[:, -1:]  # [B, 1]
        padded = F.pad(x, (0, STRIDE), mode="replicate")  # [B, SEQ_LEN + STRIDE]
        patches = padded.unfold(1, PATCH_LEN, STRIDE)  # [B, n_patches, PATCH_LEN]
        z = self.patch_embed(patches) + self.pos_embed  # [B, n_patches, D_MODEL]
        z = self.dropout(z)
        z = self.encoder(z)  # [B, n_patches, D_MODEL]
        delta = self.head(z.reshape(z.shape[0], -1))  # [B, PRED_LEN]
        return last + delta


def extract_numerical(
    dataset: HS300Dataset, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardized Close [N, SEQ_LEN] and targets [N, PRED_LEN] in memory.

    Reads only the in-memory price/stat arrays, skipping the news and report
    readers so the numerical-only experiment does not pay repeated JSONL IO.
    """
    n = len(indices)
    x = np.empty((n, SEQ_LEN), dtype=np.float32)
    y = np.empty((n, PRED_LEN), dtype=np.float32)
    for k, idx in enumerate(indices):
        si = int(dataset.stock_index[idx])
        code = dataset.stock_codes[si]
        start = int(dataset.input_start[idx])
        close = dataset.price_values[code][start : start + SEQ_LEN, CLOSE_COLUMN]
        mean = float(dataset.close_mean[si])
        std = max(float(dataset.close_std[si]), 1e-6)
        x[k] = (close - mean) / std
        y[k] = dataset.y[idx]
    return torch.from_numpy(x), torch.from_numpy(y)


def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unified training objective: MAE + 0.2 * MSE."""
    difference = prediction - target
    return difference.abs().mean() + 0.2 * difference.square().mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict:
    model.eval()
    predictions = []
    for start in range(0, x.shape[0], batch_size):
        predictions.append(model(x[start : start + batch_size].to(device)).cpu())
    prediction = torch.cat(predictions, dim=0)
    difference = prediction - y
    result = {
        "samples": int(x.shape[0]),
        "mae": float(difference.abs().mean()),
        "mse": float(difference.square().mean()),
    }
    for horizon in HORIZONS:
        error = difference[:, horizon - 1]
        result[f"mae_h{horizon}"] = float(error.abs().mean())
        result[f"mse_h{horizon}"] = float(error.square().mean())
    return result


def train_and_evaluate(
    package_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    patience: int,
    device: str,
    limit_train: int | None,
    checkpoint: Path | None = None,
) -> dict:
    set_seed(seed)
    train = HS300Dataset(package_dir, "train")
    val = HS300Dataset(package_dir, "val")
    split = make_validation_split(val, holdout_fraction=0.2)

    train_indices = np.arange(len(train))
    if limit_train is not None:
        train_indices = train_indices[:limit_train]
    x_train, y_train = extract_numerical(train, train_indices)
    x_dev, y_dev = extract_numerical(val, split.dev_indices)
    x_holdout, y_holdout = extract_numerical(val, split.holdout_indices)

    model = PatchTST().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )

    best_dev_mae = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * xb.shape[0]

        dev = evaluate(model, x_dev, y_dev, batch_size, device)
        train_loss = running_loss / len(loader.dataset)
        print(
            f"epoch {epoch:3d}: train_loss={train_loss:.6f} "
            f"dev_MAE={dev['mae']:.6f} dev_MSE={dev['mse']:.6f} "
            f"({time.time() - started:.1f}s)"
        )
        if dev["mae"] < best_dev_mae:
            best_dev_mae = dev["mae"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"early stop after {epoch} epochs")
            break

    model.load_state_dict(best_state)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint)
    dev = evaluate(model, x_dev, y_dev, batch_size, device)
    holdout = evaluate(model, x_holdout, y_holdout, batch_size, device)

    result = {
        "experiment": "e4",
        "model": "PatchTST",
        "input": "standardized Close [96]",
        "device": device,
        "seed": seed,
        "config": {
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "patch_len": PATCH_LEN,
            "stride": STRIDE,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "loss": "MAE + 0.2 * MSE",
            "optimizer": "AdamW",
            "lr": lr,
            "batch_size": batch_size,
            "epochs_run": best_epoch,
            "patience": patience,
            "params": n_params,
        },
        "train_samples": int(x_train.shape[0]),
        "dev_samples": int(x_dev.shape[0]),
        "holdout_samples": int(x_holdout.shape[0]),
        "best_epoch": best_epoch,
        "dev": dev,
        "holdout": holdout,
    }
    train.close()
    val.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root containing data/ (defaults to the repository root)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int, default=None, help="cap train samples for quick tests")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    result = train_and_evaluate(
        package_dir=args.package_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        device=args.device,
        limit_train=args.limit_train,
        checkpoint=args.checkpoint,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
