"""Bounded E6 raw/stationary comparison, followed by frozen-market E7 news."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..data import sha256_file
from ..run import ROOT, metrics, write_json
from .data import load_store
from .model import Model

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


@torch.inference_mode()
def predict(model, store, samples, representation, batch_size=256, zero_semantics=False, shuffle=None):
    model.eval()
    predictions = []
    for start in range(0, len(samples.y), batch_size):
        rows = np.arange(start, min(start + batch_size, len(samples.y)))
        batch = store.batch(samples, rows, representation, news=model.with_news, zero_semantics=zero_semantics,
                            news_rows=None if shuffle is None else shuffle[rows])
        predictions.append(model(batch).numpy())
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite predictions")
    return result


def train_one(store, representation, seed, args, source=None):
    name = f"e7_{representation}_seed{seed}" if source else f"e6_{representation}_seed{seed}"
    out = args.output_dir / name
    record = args.results_dir / f"{name}.json"
    if args.resume and record.exists():
        saved_result = json.loads(record.read_text(encoding="utf-8"))
        if (saved_result["seed"] != seed or saved_result["representation"] != representation
                or saved_result["news"] != (source is not None)
                or sha256_file(out / "model.pt") != saved_result["checkpoint_sha256"]
                or not (out / "dev_predictions.npy").exists()):
            raise ValueError(f"completed run cannot be safely resumed: {name}")
        if source and sha256_file(source) != saved_result["market_checkpoint_sha256"]:
            raise ValueError("frozen market checkpoint changed")
        print(f"Preserving completed run: {name}", flush=True)
        return saved_result
    out.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Model(news=source is not None)
    if source:
        saved = torch.load(source, map_location="cpu", weights_only=True)
        model.market.load_state_dict({k.removeprefix("market."): v for k, v in saved["state_dict"].items() if k.startswith("market.")})
        model.freeze_market()
        frozen = {key: value.clone() for key, value in model.market.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.01)
    generator = np.random.default_rng(seed)
    best, best_epoch, stale = float("inf"), 0, 0
    history = []
    start = time.perf_counter()
    # E7 epoch zero is the exact frozen E6; include it so adding news cannot silently discard the baseline.
    if source:
        prediction = predict(model, store, store.dev, representation, args.batch_size)
        initial = metrics(prediction, store.dev.y)
        best = initial["mae"]
        torch.save({"representation": representation, "with_news": True, "state_dict": model.state_dict()}, out / "model.pt")
        history.append({"epoch": 0, "development": initial})
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = generator.permutation(len(store.train.y))
        total_loss = 0.0
        for start_batch in range(0, len(order), args.batch_size):
            rows = order[start_batch:start_batch + args.batch_size]
            batch = store.batch(store.train, rows, representation, news=model.with_news)
            target = torch.from_numpy(store.train.y[rows])
            optimizer.zero_grad(set_to_none=True)
            error = model(batch) - target
            loss = error.abs().mean() + 0.2 * error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item() * len(rows)
        score = metrics(predict(model, store, store.dev, representation, args.batch_size), store.dev.y)
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "development": score,
                        "seconds": time.perf_counter() - epoch_start})
        if score["mae"] < best - 1e-5:
            best, best_epoch, stale = score["mae"], epoch, 0
            torch.save({"representation": representation, "with_news": model.with_news, "state_dict": model.state_dict()}, out / "model.pt")
        else:
            stale += 1
        write_json(out / "history.json", history)
        print(f"{name} epoch={epoch:02d} MAE={score['mae']:.6f} MSE={score['mse']:.6f} best={best_epoch} seconds={history[-1]['seconds']:.1f}", flush=True)
        if stale >= args.patience:
            break
    saved = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    if source:
        for key, value in model.market.state_dict().items():
            torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    prediction = predict(model, store, store.dev, representation, args.batch_size)
    score = metrics(prediction, store.dev.y)
    np.testing.assert_allclose(score["mae"], best, rtol=0, atol=1e-9)
    np.save(out / "dev_predictions.npy", prediction)
    result = {"name": name, "seed": seed, "representation": representation, "news": model.with_news,
              "best_epoch": best_epoch, "epochs_run": len(history) - int(source is not None),
              "development": score, "history": history, "seconds": time.perf_counter()-start,
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "checkpoint_sha256": sha256_file(out / "model.pt"), "holdout_evaluated": False}
    result["cpu_flush_denormal"] = True
    if source:
        result["market_checkpoint_sha256"] = sha256_file(source)
        result["zero_news_semantics"] = metrics(predict(model, store, store.dev, representation, args.batch_size, zero_semantics=True), store.dev.y)
        shuffle = np.arange(len(store.dev.y))
        rng = np.random.default_rng(20261031)
        for stock in np.unique(store.dev.stocks):
            rows = np.flatnonzero(store.dev.stocks == stock)
            shuffle[rows] = rng.permutation(rows)
        result["within_stock_shuffled_news"] = metrics(predict(model, store, store.dev, representation, args.batch_size, shuffle=shuffle), store.dev.y)
        # Group coverage refers to the same exact-day deduplicated buckets used as input.
        counts = np.asarray([store.news_counts[int(s):int(s)+96].sum() for s in store.dev.starts])
        positive = counts[counts > 0]
        threshold = float(np.median(positive)) if len(positive) else 0.0
        masks = {"no_news": counts == 0, "low_coverage": (counts > 0) & (counts <= threshold),
                 "high_coverage": counts > threshold}
        base_model = Model()
        base_model.load_state_dict(torch.load(source, map_location="cpu", weights_only=True)["state_dict"])
        base = predict(base_model, store, store.dev, representation, args.batch_size)
        result["coverage_groups"] = {key: {"market": metrics(base[mask], store.dev.y[mask]),
                                                "news": metrics(prediction[mask], store.dev.y[mask])}
                                     for key, mask in masks.items() if mask.any()}
        if masks["no_news"].any():
            np.testing.assert_array_equal(prediction[masks["no_news"]], base[masks["no_news"]])
    write_json(args.results_dir / f"{name}.json", result)
    return result


def aggregate(rows):
    return {key: {"mean": float(np.mean([r["development"][key] for r in rows])),
                  "std_population": float(np.std([r["development"][key] for r in rows]))}
            for key in ("mae", "mse")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    parser.add_argument("--stages", choices=("e6", "e6-e7"), default="e6-e7")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="preserve verified completed runs; unfinished output directories must be archived explicitly")
    args = parser.parse_args()
    if args.resume:
        if not (args.results_dir / "protocol.json").exists() or not args.output_dir.exists():
            parser.error("resume requires the original protocol and output directory")
        if (args.results_dir / "resume_protocol.json").exists():
            parser.error("a resume record already exists; preserve it and review before another restart")
    else:
        if args.results_dir.exists() or args.output_dir.exists():
            parser.error("choose fresh result/output directories; existing experiments are never overwritten")
        args.results_dir.mkdir(parents=True)
        args.output_dir.mkdir(parents=True)
    # L2 can shrink the auxiliary MLP into subnormal float32 ranges, which
    # are exceptionally slow on some CPUs. Values below ~1e-38 flush to zero.
    if not torch.set_flush_denormal(True):
        raise RuntimeError("this CPU does not support the configured flush-denormal mode")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protocol = {
        "experiment": "DLinear extension E6/E7; exploratory development-only branch, not a team backbone selection",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "representations": ["raw", "stationary"], "seeds": list(SEEDS),
        "selection": "E6 representation with minimum 3-seed mean dev MAE; raw wins differences <=1e-5",
        "e6": "Seven-channel DLinear; kernel=25, zero initialization, last-Close anchor, fixed h/24 output scaling during training/inference",
        "e7": "Freeze each matched E6 market checkpoint; fit masked daily 512->8->8 MLP plus 10-channel DLinear news heads; exact no-news fallback",
        "epoch_zero": "E7 includes its exact frozen E6 baseline among checkpoints; report epoch zero if news does not help",
        "optimizer": "Adam lr=0.001 weight_decay=0.01, clip_norm=1.0",
        "loss": "MAE + 0.2*MSE", "max_epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "min_delta": 1e-5,
        "news_validation": "Report semantic-zero and within-stock window-shuffle on dev only; no ablation-based refitting",
        "holdout_evaluated": False,
        "holdout_policy": "Previously observed E1 holdout is not used for this extension's model/feature decisions or rescoring",
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in HERE.glob("*.py")},
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu",
                        "cpu_flush_denormal": True},
    }
    if args.resume:
        original = json.loads((args.results_dir / "protocol.json").read_text(encoding="utf-8"))
        for key in ("representations", "seeds", "selection", "e6", "e7", "epoch_zero",
                    "optimizer", "loss", "max_epochs", "patience", "batch_size", "min_delta"):
            if original[key] != protocol[key]:
                raise ValueError(f"resume changes the experiment protocol: {key}")
        protocol["resume_reason"] = "Release reader caches and flush CPU subnormal arithmetic; retain all completed runs and restart unfinished runs with unchanged model/features/optimizer/seeds"
        protocol["original_protocol_sha256"] = sha256_file(args.results_dir / "protocol.json")
        write_json(args.results_dir / "resume_protocol.json", protocol)
    else:
        write_json(args.results_dir / "protocol.json", protocol)
    print("Preparing fixed-reader numerical tables and compact daily news cache...", flush=True)
    cache = args.output_dir / ("cache_resume" if args.resume else "cache")
    store = load_store(args.package_dir.resolve(), cache, args.stages == "e6-e7")
    if args.resume:
        previous_data = json.loads((args.results_dir / "data_protocol.json").read_text(encoding="utf-8"))
        if previous_data != store.metadata:
            raise ValueError("data preprocessing changed during resume")
    write_json(args.results_dir / "data_protocol.json", store.metadata)
    anchor = store.close[store.dev.starts + 95, None]
    baseline = metrics(np.repeat(anchor, 24, axis=1), store.dev.y)
    old = json.loads((ROOT / "experiments/e1_dlinear/results/calibration/summary.json").read_text())
    np.testing.assert_allclose([baseline["mae"], baseline["mse"]],
                               [old["e0_development"]["mae"], old["e0_development"]["mse"]], rtol=0, atol=1e-10)
    summary = {"e0_development": baseline, "previous_e1_development": old["averaged_model"]["development"],
               "e6": {}, "e7": None, "holdout_evaluated": False}
    for representation in ("raw", "stationary"):
        rows = [train_one(store, representation, seed, args) for seed in SEEDS]
        summary["e6"][representation] = {"runs": [{k: r[k] for k in ("name", "seed", "best_epoch", "development", "parameters")} for r in rows],
                                           "aggregate": aggregate(rows)}
        write_json(args.results_dir / "summary.json", summary)
    raw_mae = summary["e6"]["raw"]["aggregate"]["mae"]["mean"]
    stat_mae = summary["e6"]["stationary"]["aggregate"]["mae"]["mean"]
    chosen = "stationary" if stat_mae < raw_mae - 1e-5 else "raw"
    summary["selected_representation"] = chosen
    write_json(args.results_dir / "summary.json", summary)
    print(f"E6 selected={chosen}; raw MAE={raw_mae:.8f}, stationary MAE={stat_mae:.8f}", flush=True)
    if args.stages == "e6-e7":
        rows = [train_one(store, chosen, seed, args, args.output_dir / f"e6_{chosen}_seed{seed}/model.pt") for seed in SEEDS]
        summary["e7"] = {"runs": [{k: r[k] for k in ("name", "seed", "best_epoch", "development", "zero_news_semantics", "within_stock_shuffled_news")} for r in rows],
                         "aggregate": aggregate(rows)}
        write_json(args.results_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
