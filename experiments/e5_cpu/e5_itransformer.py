"""E5: iTransformer over all market variables using stationary features.

The inverted Transformer treats each input variable as a token and applies
attention across variables (not across time steps).  The Close variable's
token is projected to the 24-step residual, which is added to the last
standardized close to produce the final prediction.

Run from the repository root (the directory containing ``data/``):

    python -m experiments.e5.e5_itransformer --epochs 3 --max-train 2000 --max-val 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits")
CODE = re.compile(r"\d{6}\.(?:SZ|SH|SS)")

# Stationary feature set.  Feature 0 is the standardized close and is used as
# both the residual anchor (its last value) and the iTransformer output token.
NONPRICE_FEATURES = ("log_ret", "log_vol", "dividends", "splits")
N_VARS = 10
CLOSE_INDEX = 0
WINDOW = 96
HORIZON = 24


def _code(member: str) -> str:
    match = CODE.search(member)
    if not match:
        raise ValueError(f"no stock code in {member}")
    code = match.group(0)
    return code[:-2] + "SH" if code.endswith("SS") else code


def _price_components(numerical: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return close, log-volume, dividends, splits for the raw [96, 7] array."""
    close = numerical[:, 3].astype(np.float64)
    volume = numerical[:, 4].astype(np.float64)
    dividends = numerical[:, 5].astype(np.float64)
    splits = numerical[:, 6].astype(np.float64)
    log_vol = np.log1p(volume)
    log_ret = np.empty_like(close)
    log_ret[0] = 0.0
    np.log(np.maximum(close[1:] / close[:-1], 1e-6), out=log_ret[1:])
    return close, log_ret, log_vol, dividends, splits


def make_features(numerical: np.ndarray, close_mean: float, close_std: float, stats: dict) -> np.ndarray:
    """Build [96, N_VARS] stationary features from a raw [96, 7] window."""
    o = numerical[:, 0].astype(np.float64)
    h = numerical[:, 1].astype(np.float64)
    l = numerical[:, 2].astype(np.float64)
    c = numerical[:, 3].astype(np.float64)
    std = max(float(close_std), 1e-6)
    mean = float(close_mean)

    z_close = (c - mean) / std
    z_open = (o - mean) / std
    z_high = (h - mean) / std
    z_low = (l - mean) / std
    amplitude = (h - l) / std
    open_close = (o - c) / std

    _, log_ret, log_vol, dividends, splits = _price_components(numerical)
    log_ret = (log_ret - stats["log_ret_mean"]) / stats["log_ret_std"]
    log_vol = (log_vol - stats["log_vol_mean"]) / stats["log_vol_std"]
    dividends = (dividends - stats["dividends_mean"]) / stats["dividends_std"]
    splits = (splits - stats["splits_mean"]) / stats["splits_std"]

    feats = np.stack(
        [z_close, z_open, z_high, z_low, amplitude, open_close, log_ret, log_vol, dividends, splits],
        axis=1,
    )
    return feats.astype(np.float32)


class NumericalDataset:
    """Fast numerical-only reader for the HS300 package (no news or financials)."""

    def __init__(self, package_dir: Path, split: str):
        self.data_dir = Path(package_dir) / "data"
        with np.load(self.data_dir / f"{split}.npz", allow_pickle=False) as values:
            self.stock_codes = values["stock_codes"].tolist()
            self.stock_index = values["stock_index"]
            self.input_start = values["input_start"]
            self.target_start_date = values["target_start_date"]
            self.close_mean = values["close_mean"]
            self.close_std = values["close_std"]
            self.y = values["y"] if "y" in values.files else None
        self.price_days: dict[str, np.ndarray] = {}
        self.price_values: dict[str, np.ndarray] = {}
        price_root = self.data_dir / "numerical/HS300_time_series"
        for source_file in price_root.rglob("*.csv"):
            code = _code(str(source_file))
            days = []
            numbers = []
            with source_file.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                for row in reader:
                    day = int(row["Date"][:10].replace("-", ""))
                    if day < 20200331 or day > 20250331:
                        continue
                    days.append(day)
                    numbers.append([float(row[col]) for col in PRICE_COLUMNS])
            self.price_days[code] = np.asarray(days, dtype=np.int32)
            self.price_values[code] = np.asarray(numbers, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.stock_index)

    def raw(self, index: int) -> np.ndarray:
        stock_index = int(self.stock_index[index])
        code = self.stock_codes[stock_index]
        start = int(self.input_start[index])
        return self.price_values[code][start : start + WINDOW]

    def sample(self, index: int, stats: dict) -> tuple[np.ndarray, float, np.ndarray]:
        numerical = self.raw(index)
        stock_index = int(self.stock_index[index])
        mean = float(self.close_mean[stock_index])
        std = float(self.close_std[stock_index])
        x = make_features(numerical, mean, std, stats)
        z_last = float(x[-1, CLOSE_INDEX])
        y = self.y[index] if self.y is not None else None
        return x, z_last, y


def compute_input_stats(dataset: NumericalDataset) -> dict:
    """Compute mean/std of non-price features over the whole training set."""
    sums = {name: 0.0 for name in NONPRICE_FEATURES}
    sq_sums = {name: 0.0 for name in NONPRICE_FEATURES}
    count = 0
    t0 = time.time()
    for index in range(len(dataset)):
        numerical = dataset.raw(index)
        _, log_ret, log_vol, dividends, splits = _price_components(numerical)
        values = {"log_ret": log_ret, "log_vol": log_vol, "dividends": dividends, "splits": splits}
        for name in NONPRICE_FEATURES:
            v = values[name]
            sums[name] += float(v.sum())
            sq_sums[name] += float((v * v).sum())
        count += WINDOW
        if (index + 1) % 100000 == 0:
            print(f"  stats {index + 1}/{len(dataset)} ({(time.time() - t0):.1f}s)", flush=True)
    stats = {}
    for name in NONPRICE_FEATURES:
        mean = sums[name] / count
        var = max(sq_sums[name] / count - mean * mean, 0.0)
        stats[f"{name}_mean"] = mean
        stats[f"{name}_std"] = max(np.sqrt(var), 1e-6)
    return stats


class iTransformer(nn.Module):
    def __init__(
        self,
        n_vars: int = N_VARS,
        window: int = WINDOW,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = HORIZON,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.input_proj = nn.Linear(window, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, n_vars, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.output_proj = nn.Linear(d_model, horizon)
        # Zero-init the residual head so the model starts exactly at the
        # last-value baseline and learns the change from there.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, z_last: torch.Tensor) -> torch.Tensor:
        # x: [B, window, n_vars]
        tokens = x.permute(0, 2, 1)  # [B, n_vars, window]
        tokens = self.input_proj(tokens) + self.var_embed
        tokens = self.encoder(tokens)  # [B, n_vars, d_model]
        delta = self.output_proj(tokens)[:, CLOSE_INDEX, :]  # [B, horizon]
        return z_last.unsqueeze(1) + delta


def collate(dataset: NumericalDataset, indices, stats: dict, device: torch.device):
    batch_size = len(indices)
    xs = np.empty((batch_size, WINDOW, N_VARS), dtype=np.float32)
    zs = np.empty(batch_size, dtype=np.float32)
    ys = np.empty((batch_size, HORIZON), dtype=np.float32)
    for k, index in enumerate(indices):
        x, z, y = dataset.sample(int(index), stats)
        xs[k] = x
        zs[k] = z
        ys[k] = y
    return (
        torch.from_numpy(xs).to(device),
        torch.from_numpy(zs).to(device),
        torch.from_numpy(ys).to(device),
    )


def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = prediction - target
    return difference.abs().mean() + 0.2 * difference.square().mean()


def evaluate(
    model: nn.Module,
    dataset: NumericalDataset,
    indices: np.ndarray,
    stats: dict,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    count = 0
    horizon_abs = np.zeros(HORIZON, dtype=np.float64)
    horizon_sq = np.zeros(HORIZON, dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            xs, zs, ys = collate(dataset, selected, stats, device)
            prediction = model(xs, zs)
            difference = (prediction - ys).to(torch.float64)
            total_abs += float(difference.abs().sum())
            total_sq += float(difference.square().sum())
            count += difference.numel()
            horizon_abs += difference.abs().sum(dim=0).cpu().numpy()
            horizon_sq += difference.square().sum(dim=0).cpu().numpy()
    result = {
        "samples": int(len(indices)),
        "mae": total_abs / count,
        "mse": total_sq / count,
    }
    for horizon in (1, 6, 12, 24):
        result[f"mae_h{horizon}"] = float(horizon_abs[horizon - 1] / len(indices))
        result[f"mse_h{horizon}"] = float(horizon_sq[horizon - 1] / len(indices))
    return result


def run(args) -> dict:
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = NumericalDataset(args.package_dir, "train")
    val = NumericalDataset(args.package_dir, "val")

    stats_path = Path(args.stats_path)
    if stats_path.exists() and not args.recompute_stats:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"loaded input stats from {stats_path}")
    else:
        print("computing input stats over training set ...")
        stats = compute_input_stats(train)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"saved input stats to {stats_path}")

    from experiments.validation_split import make_validation_split

    split = make_validation_split(val, args.holdout_fraction)
    dev_indices = split.dev_indices
    holdout_indices = split.holdout_indices
    print(
        f"dev={len(dev_indices)} holdout={len(holdout_indices)} "
        f"purged={len(split.purged_indices)}"
    )

    # Optional subsetting for a quick smoke test.
    train_indices = np.arange(len(train))
    if args.max_train and args.max_train < len(train):
        train_indices = np.random.choice(len(train), args.max_train, replace=False)
    if args.max_val and args.max_val < len(dev_indices):
        dev_indices = dev_indices[: args.max_val]

    model = iTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"iTransformer parameters: {parameter_count}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    init_metrics = evaluate(model, val, dev_indices, stats, args.batch_size, device)
    print(
        f"epoch 0 (init): dev_MAE={init_metrics['mae']:.6f} dev_MSE={init_metrics['mse']:.6f}",
        flush=True,
    )
    best_dev_mae = init_metrics["mae"]
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(train_indices)
        running_loss = 0.0
        batches_seen = 0
        t0 = time.time()
        for start in range(0, len(order), args.batch_size):
            selected = order[start : start + args.batch_size]
            xs, zs, ys = collate(train, selected, stats, device)
            optimizer.zero_grad()
            prediction = model(xs, zs)
            loss = compute_loss(prediction, ys)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            running_loss += float(loss)
            batches_seen += 1
        train_loss = running_loss / max(batches_seen, 1)
        dev_metrics = evaluate(model, val, dev_indices, stats, args.batch_size, device)
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"loss={train_loss:.6f} dev_MAE={dev_metrics['mae']:.6f} "
            f"dev_MSE={dev_metrics['mse']:.6f} ({(time.time() - t0):.1f}s)",
            flush=True,
        )
        if dev_metrics["mae"] < best_dev_mae:
            best_dev_mae = dev_metrics["mae"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        elif args.patience and epoch + 1 - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch + 1} (no improvement for {args.patience} epochs)")
            break

    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = args.epochs

    model.load_state_dict(best_state)
    final_dev = evaluate(model, val, dev_indices, stats, args.batch_size, device)
    holdout = (
        evaluate(model, val, holdout_indices, stats, args.batch_size, device)
        if len(holdout_indices)
        else {}
    )

    result = {
        "experiment": "E5",
        "model": "iTransformer",
        "best_epoch": best_epoch,
        "init_dev_mae": init_metrics["mae"],
        "init_dev_mse": init_metrics["mse"],
        "parameters": parameter_count,
        "hyperparams": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "max_train": args.max_train,
            "max_val": args.max_val,
        },
        "dev": final_dev,
        "holdout": holdout,
    }
    return result, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stats-path", type=Path, default=Path("experiments/e5_cpu/input_stats.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/e5_cpu/e5_val_results.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/e5_cpu/e5_model.pt"))
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--recompute-stats", action="store_true")
    args = parser.parse_args()

    result, model = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
