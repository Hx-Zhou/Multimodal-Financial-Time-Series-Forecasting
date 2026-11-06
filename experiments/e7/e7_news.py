"""E7: minimal market backbone + pooled news branch (TaTS-style, no GPT-2).

The pure-market backbones E1-E5 all failed to beat E0 (dev MAE 0.2173), so the
market side is deliberately kept minimal -- a last-value residual head -- and
all capacity is spent on the news branch.  The only question E7 answers is
whether the news modality can pull dev MAE below the last-value baseline.

Architecture (residual prediction, per docs/experiment_route.md section 2):

    z_96 = last standardized Close                         [B, 1]
    pooled = dedup + masked mean-pool of 512-dim articles   [B, 512]
    has_news = 1.0 if the window has any article else 0.0   [B, 1]
    news_count = log(1 + unique_article_count)              [B, 1]

    delta_market = Linear(z_96)                             [B, 24]   (tiny)
    news_z       = MLP(pooled) * has_news                   [B, 8]
    delta_news   = Fusion([news_z; has_news; news_count]) * has_news   [B, 24]
    y_hat        = z_96 + delta_market + delta_news

    - When has_news == 0 the news residual is exactly zero, so the model
      degenerates to the market-only head on no-news samples.
    - `use_news=False` in forward() ablates the whole news branch at eval time
      (R1 news-zeroing check in docs/experiment_route.md section 8).

Run from the repository root:

    python -m experiments.e7.e7_news --epochs 20 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hs300_dataset import HS300Dataset, NewsWindow
from experiments.validation_split import make_validation_split

# Unified settings (docs/experiment_route.md sections 2 and 4).
SEQ_LEN = 96
PRED_LEN = 24
NEWS_DIM = 512
NEWS_HIDDEN = 128
NEWS_OUT = 8
FUSION_HIDDEN = 32
DROPOUT = 0.1
HORIZONS = (1, 6, 12, 24)
CLOSE_COLUMN = 3  # PRICE_COLUMNS: Open, High, Low, Close, Volume, Dividends, Stock Splits


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class E7News(nn.Module):
    """Last-value residual head plus a pooled news branch with masked gating."""

    def __init__(self) -> None:
        super().__init__()
        self.market_head = nn.Linear(1, PRED_LEN)
        self.news_mlp = nn.Sequential(
            nn.Linear(NEWS_DIM, NEWS_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(NEWS_HIDDEN, NEWS_OUT),
        )
        # Fusion sees news_z (NEWS_OUT) + has_news (1) + news_count (1).
        self.fusion = nn.Sequential(
            nn.Linear(NEWS_OUT + 2, FUSION_HIDDEN),
            nn.GELU(),
            nn.Linear(FUSION_HIDDEN, PRED_LEN),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.market_head.weight)
        nn.init.zeros_(self.market_head.bias)
        for linear in (self.news_mlp[0], self.news_mlp[3], self.fusion[0], self.fusion[2]):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        z96: torch.Tensor,
        pooled: torch.Tensor,
        has_news: torch.Tensor,
        news_count: torch.Tensor,
        use_news: bool = True,
    ) -> torch.Tensor:
        """z96 [B,1], pooled [B,512], has_news [B,1], news_count [B,1] -> [B,24]."""
        delta_market = self.market_head(z96)  # [B, 24]
        if not use_news:
            return z96 + delta_market
        news_z = self.news_mlp(pooled) * has_news  # [B, 8], zeroed when no news
        fused = torch.cat([news_z, has_news, news_count], dim=-1)  # [B, 10]
        delta_news = self.fusion(fused) * has_news  # [B, 24], zeroed when no news
        return z96 + delta_market + delta_news


def _empty_news_arrays() -> tuple[np.ndarray, np.ndarray]:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)


def extract_features(dataset: HS300Dataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Pool news per sample and build the compact market/news feature matrices.

    Uses only public reader surface: `price_days` / `price_values` / `news_by_stock`
    and the public `NewsWindow` class.  The window is the full 96 trading days;
    `NewsWindow.events()` covers the input date range including non-trading days.
    Duplicate article records (same vector) are collapsed before pooling.
    """
    n = len(indices)
    z96 = np.empty((n, 1), dtype=np.float32)
    pooled = np.empty((n, NEWS_DIM), dtype=np.float32)
    has_news = np.empty((n, 1), dtype=np.float32)
    news_count = np.empty((n, 1), dtype=np.float32)
    y = np.empty((n, PRED_LEN), dtype=np.float32)
    empty_times, empty_ids = _empty_news_arrays()

    for k, idx in enumerate(indices):
        si = int(dataset.stock_index[idx])
        code = dataset.stock_codes[si]
        start = int(dataset.input_start[idx])
        days = dataset.price_days[code][start : start + SEQ_LEN]
        close = dataset.price_values[code][start : start + SEQ_LEN, CLOSE_COLUMN]
        mean = float(dataset.close_mean[si])
        std = max(float(dataset.close_std[si]), 1e-6)
        z96[k, 0] = (close[-1] - mean) / std

        times, ids = dataset.news_by_stock.get(code, (empty_times, empty_ids))
        window = NewsWindow(days, times, ids, dataset.news_vectors)
        vectors, _ = window.events()  # [articles, 512], duplicates preserved
        if vectors.shape[0] == 0:
            pooled[k] = 0.0
            has_news[k, 0] = 0.0
            news_count[k, 0] = 0.0
        else:
            unique = np.unique(vectors, axis=0)  # dedup identical articles
            pooled[k] = unique.mean(axis=0, dtype=np.float32)
            has_news[k, 0] = 1.0
            news_count[k, 0] = float(np.log1p(unique.shape[0]))

        y[k] = dataset.y[idx]

        if (k + 1) % 50000 == 0:
            print(f"    extracted {k + 1}/{n}", flush=True)

    return {"z96": z96, "pooled": pooled, "has_news": has_news, "news_count": news_count, "y": y}


def _cache_path(package_dir: Path, split: str) -> Path:
    return package_dir / "experiments" / "e7" / "cache" / f"{split}_features.npz"


def load_or_extract(
    package_dir: Path, split: str, no_cache: bool, max_samples: int | None = None
) -> dict[str, np.ndarray]:
    dataset = HS300Dataset(package_dir, split)
    if max_samples is not None:
        # Smoke-test path: extract a subset only, never write a partial cache.
        try:
            print(f"[e7] extracting first {max_samples} {split} samples ...", flush=True)
            return extract_features(dataset, np.arange(min(max_samples, len(dataset))))
        finally:
            dataset.close()

    path = _cache_path(package_dir, split)
    if path.exists() and not no_cache:
        dataset.close()
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    print(f"[e7] extracting {split} features ...", flush=True)
    try:
        features = extract_features(dataset, np.arange(len(dataset)))
    finally:
        dataset.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **features)
    print(f"[e7] cached {split} features -> {path}", flush=True)
    return features


def _stack(features: dict[str, np.ndarray], indices: np.ndarray) -> tuple[torch.Tensor, ...]:
    z96 = torch.from_numpy(features["z96"][indices])
    pooled = torch.from_numpy(features["pooled"][indices])
    has_news = torch.from_numpy(features["has_news"][indices])
    news_count = torch.from_numpy(features["news_count"][indices])
    y = torch.from_numpy(features["y"][indices])
    return z96, pooled, has_news, news_count, y


def loss_fn(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unified training objective: MAE + 0.2 * MSE."""
    difference = prediction - target
    return difference.abs().mean() + 0.2 * difference.square().mean()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    z96: torch.Tensor,
    pooled: torch.Tensor,
    has_news: torch.Tensor,
    news_count: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: str,
    use_news: bool,
) -> dict:
    model.eval()
    predictions = []
    for start in range(0, z96.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        predictions.append(
            model(
                z96[sl].to(device),
                pooled[sl].to(device),
                has_news[sl].to(device),
                news_count[sl].to(device),
                use_news=use_news,
            ).cpu()
        )
    prediction = torch.cat(predictions, dim=0)
    difference = prediction - y
    result = {
        "samples": int(z96.shape[0]),
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
    weight_decay: float = 0.0,
    no_cache: bool = False,
    limit_val: int | None = None,
    checkpoint: Path | None = None,
) -> dict:
    set_seed(seed)

    train_feats = load_or_extract(package_dir, "train", no_cache, limit_train)
    val_feats = load_or_extract(package_dir, "val", no_cache, limit_val)

    if limit_val is None:
        # Chronological dev/holdout split needs the val reader's target dates.
        val_dataset = HS300Dataset(package_dir, "val")
        split = make_validation_split(val_dataset, holdout_fraction=0.2)
        val_dataset.close()
        dev_indices = split.dev_indices
        holdout_indices = split.holdout_indices
    else:
        dev_indices = np.arange(val_feats["z96"].shape[0])
        holdout_indices = np.empty(0, dtype=np.int64)

    train_indices = np.arange(train_feats["z96"].shape[0])
    z_train, p_train, h_train, c_train, y_train = _stack(train_feats, train_indices)
    z_dev, p_dev, h_dev, c_dev, y_dev = _stack(val_feats, dev_indices)
    z_holdout, p_holdout, h_holdout, c_holdout, y_holdout = _stack(val_feats, holdout_indices)

    model = E7News().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(z_train, p_train, h_train, c_train, y_train),
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
        for zb, pb, hb, cb, yb in loader:
            zb, pb, hb, cb, yb = zb.to(device), pb.to(device), hb.to(device), cb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(zb, pb, hb, cb, use_news=True), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * zb.shape[0]

        dev = evaluate(model, z_dev, p_dev, h_dev, c_dev, y_dev, batch_size, device, use_news=True)
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

    dev = evaluate(model, z_dev, p_dev, h_dev, c_dev, y_dev, batch_size, device, use_news=True)
    dev_news_off = evaluate(model, z_dev, p_dev, h_dev, c_dev, y_dev, batch_size, device, use_news=False)
    holdout = (
        evaluate(model, z_holdout, p_holdout, h_holdout, c_holdout, y_holdout, batch_size, device, use_news=True)
        if z_holdout.shape[0] > 0
        else {}
    )

    result = {
        "experiment": "e7",
        "model": "E7News",
        "input": "last-value Close + pooled 512-dim news (dedup + masked mean)",
        "device": device,
        "seed": seed,
        "config": {
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "news_dim": NEWS_DIM,
            "news_hidden": NEWS_HIDDEN,
            "news_out": NEWS_OUT,
            "fusion_hidden": FUSION_HIDDEN,
            "dropout": DROPOUT,
            "news_pooling": "dedup + mean over the full 96-day window",
            "loss": "MAE + 0.2 * MSE",
            "optimizer": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs_run": best_epoch,
            "patience": patience,
            "params": n_params,
        },
        "train_samples": int(z_train.shape[0]),
        "dev_samples": int(z_dev.shape[0]),
        "holdout_samples": int(z_holdout.shape[0]),
        "best_epoch": best_epoch,
        "dev": dev,
        "dev_news_off": dev_news_off,
        "holdout": holdout,
    }
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int, default=None, help="cap train samples for quick tests")
    parser.add_argument("--limit-val", type=int, default=None, help="cap val samples for quick tests (disables holdout)")
    parser.add_argument("--no-cache", action="store_true", help="re-extract features, ignore cache")
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
        weight_decay=args.weight_decay,
        no_cache=args.no_cache,
        limit_val=args.limit_val,
        checkpoint=args.checkpoint,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
