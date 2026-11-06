"""Bounded separate E8/E9/E10 runs; freeze every candidate before holdout scoring."""
import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.e1_dlinear.data import sha256_file
from experiments.e1_dlinear.run import ROOT, metrics, write_json
from experiments.e1_dlinear.e6_plus.model import Model as MarketModel
from experiments.e1_dlinear.e6_plus.run import predict as old_predict
from .data import load_store, add_reports, batch, load_holdout, PRIMARY_LAG, SENSITIVITY_LAG

COMMON = Path(__file__).resolve().parent
OLD = ROOT / "experiments/e1_dlinear/e6_plus"
DIRECTORIES = {"e8": "e8_dlinear_reports", "e9": "e9_dlinear_multimodal", "e10": "e10_dlinear_spectf"}
SEEDS = (42, 43, 44)
SETTINGS = {"seeds": list(SEEDS), "max_epochs": 40, "patience": 6, "min_delta": 1e-5,
            "batch_size": 1024, "learning_rate": 0.001, "weight_decay": 0.01,
            "loss": "MAE + 0.2*MSE", "gradient_clip": 1.0, "modality_dropout": 0.1,
            "initial_gate_logit": -2.0, "market_representation": "stationary",
            "epoch_zero_eligible": True, "primary_report_lag": PRIMARY_LAG,
            "sensitivity_report_lag": SENSITIVITY_LAG}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def here(stage):
    return ROOT / "experiments" / DIRECTORIES[stage]


def model_type(stage):
    return importlib.import_module(f"experiments.{DIRECTORIES[stage]}.model").Model


def source_hashes():
    paths = list(COMMON.glob("*.py"))
    for directory in DIRECTORIES.values():
        paths.extend((ROOT / "experiments" / directory).glob("*.py"))
    paths.extend([ROOT / "hs300_dataset.py", ROOT / "experiments/validation_split.py",
                  OLD / "data.py", OLD / "model.py", OLD / "run.py",
                  OLD.parent / "data.py", OLD.parent / "run.py"])
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def base_path(seed):
    return OLD / f"outputs/e6_stationary_seed{seed}/model.pt"


def verify_bases():
    assert read_json(OLD / "results/summary.json")["selected_representation"] == "stationary"
    hashes = {}
    for seed in SEEDS:
        digest = sha256_file(base_path(seed))
        assert digest == read_json(OLD / f"results/e6_stationary_seed{seed}.json")["checkpoint_sha256"]
        hashes[str(seed)] = digest
    return hashes


@torch.inference_mode()
def predict(model, store, samples, lag=PRIMARY_LAG, news_shuffle=None, report_shuffle=None, **flags):
    model.eval()
    values = []
    for start in range(0, len(samples.y), SETTINGS["batch_size"]):
        rows = np.arange(start, min(start+SETTINGS["batch_size"], len(samples.y)))
        item = batch(store, samples, rows, model.with_news, model.with_reports, lag=lag,
                     news_rows=None if news_shuffle is None else news_shuffle[rows],
                     report_rows=None if report_shuffle is None else report_shuffle[rows], **flags)
        values.append(model(item).numpy())
    result = np.concatenate(values)
    assert np.isfinite(result).all()
    return result


def shuffled_rows(samples):
    rng = np.random.default_rng(20261031)
    result = np.arange(len(samples.y))
    for stock in np.unique(samples.stocks):
        rows = np.flatnonzero(samples.stocks == stock)
        result[rows] = rng.permutation(rows)
    return result


def save_best(path, stage, seed, model):
    torch.save({"stage": stage, "seed": seed, "settings": SETTINGS, "state_dict": model.state_dict()}, path)


def load_model(stage, seed):
    path = here(stage) / f"outputs/seed{seed}/model.pt"
    record = read_json(here(stage) / f"results/seed{seed}.json")
    assert sha256_file(path) == record["checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["stage"] == stage and saved["seed"] == seed and saved["settings"] == SETTINGS
    model = model_type(stage)()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.requires_grad_(False)
    base = torch.load(base_path(seed), map_location="cpu", weights_only=True)
    for key, value in model.market.state_dict().items():
        torch.testing.assert_close(value, base["state_dict"]["market."+key], rtol=0, atol=0)
    return model


def train_one(stage, seed, store, resume):
    out = here(stage) / f"outputs/seed{seed}"
    result_path = here(stage) / f"results/seed{seed}.json"
    if result_path.exists():
        if not resume:
            raise FileExistsError(result_path)
        load_model(stage, seed)
        print(f"Preserving completed {stage} seed={seed}", flush=True)
        return read_json(result_path)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_type(stage)()
    model.load_market(torch.load(base_path(seed), map_location="cpu", weights_only=True))
    frozen = {k: v.clone() for k, v in model.market.state_dict().items()}
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    rng = np.random.default_rng(seed)
    best_epoch, stale, history, elapsed = 0, 0, [], 0.0
    if out.exists():
        if not resume:
            raise FileExistsError(out)
        state = torch.load(out / "training_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        best, best_epoch, stale, history, elapsed = (state[k] for k in ("best", "best_epoch", "stale", "history", "elapsed"))
    else:
        out.mkdir(parents=True)
        initial = predict(model, store, store.dev)
        score = metrics(initial, store.dev.y)
        reference = read_json(OLD / f"results/e6_stationary_seed{seed}.json")["development"]
        np.testing.assert_allclose([score[k] for k in ("mae", "mse")], [reference[k] for k in ("mae", "mse")], atol=1e-9, rtol=0)
        history = [{"epoch": 0, "development": score}]
        best = score["mae"]
        save_best(out / "model.pt", stage, seed, model)
        write_json(out / "history.json", history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "best": best, "best_epoch": 0, "stale": 0, "history": history, "elapsed": 0.0},
                   out / "training_state.pt")
    started = time.perf_counter()
    for epoch in range(history[-1]["epoch"]+1, SETTINGS["max_epochs"]+1):
        if stale >= SETTINGS["patience"]:
            break
        epoch_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(store.train.y))
        loss_sum = 0.0
        for left in range(0, len(order), SETTINGS["batch_size"]):
            rows = order[left:left+SETTINGS["batch_size"]]
            item = batch(store, store.train, rows, model.with_news, model.with_reports)
            target = torch.from_numpy(store.train.y[rows])
            optimizer.zero_grad(set_to_none=True)
            error = model(item)-target
            loss = error.abs().mean() + 0.2*error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, SETTINGS["gradient_clip"])
            optimizer.step()
            loss_sum += loss.item()*len(rows)
        score = metrics(predict(model, store, store.dev), store.dev.y)
        if score["mae"] < best-SETTINGS["min_delta"]:
            best, best_epoch, stale = score["mae"], epoch, 0
            save_best(out / "model.pt", stage, seed, model)
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": loss_sum/len(order), "development": score,
                        "seconds": time.perf_counter()-epoch_start})
        write_json(out / "history.json", history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "best": best, "best_epoch": best_epoch, "stale": stale, "history": history,
                    "elapsed": elapsed+time.perf_counter()-started}, out / "training_state.pt")
        print(f"{stage} seed={seed} epoch={epoch:02d} dev_MAE={score['mae']:.9f} best={best_epoch} seconds={history[-1]['seconds']:.1f}", flush=True)
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True, map_location="cpu")["state_dict"])
    for key, value in model.market.state_dict().items():
        torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
    prediction = predict(model, store, store.dev)
    score = metrics(prediction, store.dev.y)
    np.testing.assert_allclose(score["mae"], best, rtol=0, atol=1e-9)
    np.save(out / "dev_predictions.npy", prediction)
    result = {"stage": stage, "seed": seed, "best_epoch": best_epoch, "development": score,
              "history": history, "seconds": elapsed+time.perf_counter()-started,
              "parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in trainable),
              "checkpoint_sha256": sha256_file(out / "model.pt"), "market_sha256": sha256_file(base_path(seed)),
              "holdout_used_for_selection": False}
    write_json(result_path, result)
    return result


def seed_statistics(scores):
    return {key: {"mean": float(np.mean([s[key] for s in scores])),
                  "std_population": float(np.std([s[key] for s in scores]))} for key in ("mae", "mse")}


def diagnostic_options(model, samples, include_shuffle):
    options = {}
    if model.with_reports:
        options.update(no_reports={"disable_reports": True}, report_values_zero={"zero_report_values": True},
                       report_lag270={"lag": SENSITIVITY_LAG})
        if include_shuffle:
            options["reports_shuffled_within_stock"] = {"report_shuffle": shuffled_rows(samples)}
    if model.with_news:
        options.update(no_news={"disable_news": True}, news_semantics_zero={"zero_semantics": True})
        if include_shuffle:
            options["news_shuffled_within_stock"] = {"news_shuffle": shuffled_rows(samples)}
    return options


def evaluate(stage, store, samples, split_name):
    out = here(stage) / f"outputs/{split_name}"
    out.mkdir(parents=True, exist_ok=False)
    predictions, diagnostics, scores, checks = [], {}, [], {}
    for seed in SEEDS:
        model = load_model(stage, seed)
        dev = metrics(predict(model, store, store.dev), store.dev.y)
        reference = read_json(here(stage) / f"results/seed{seed}.json")["development"]
        np.testing.assert_allclose([dev[k] for k in ("mae", "mse")], [reference[k] for k in ("mae", "mse")], atol=1e-9, rtol=0)
        prediction = predict(model, store, samples)
        predictions.append(prediction)
        scores.append({"seed": seed, **metrics(prediction, samples.y)})
        np.save(out / f"seed{seed}.npy", prediction)
        base_model = MarketModel()
        base_model.load_state_dict(torch.load(base_path(seed), weights_only=True, map_location="cpu")["state_dict"])
        base = old_predict(base_model, store, samples, "stationary", 1024)
        disabled = predict(model, store, samples, disable_news=True, disable_reports=True)
        np.testing.assert_array_equal(disabled, base)
        missing = np.ones(len(samples.y), bool)
        if model.with_reports:
            missing &= ~np.asarray(store.reports[PRIMARY_LAG][samples.starts+95, -3:]).any(axis=1)
        if model.with_news:
            missing &= np.array([store.news_counts[int(s):int(s)+96].sum() == 0 for s in samples.starts])
        if missing.any():
            np.testing.assert_array_equal(prediction[missing], base[missing])
        checks[str(seed)] = {"development_reloaded": dev, "all_modalities_off_exact_market": True,
                             "naturally_missing_all_modalities_samples": int(missing.sum())}
        for name, kwargs in diagnostic_options(model, samples, split_name == "development").items():
            value = predict(model, store, samples, **kwargs)
            diagnostics.setdefault(name, []).append(value)
        print(f"{stage} seed={seed} {split_name} MAE={scores[-1]['mae']:.9f} MSE={scores[-1]['mse']:.9f}", flush=True)
    average = np.mean(predictions, axis=0)
    np.save(out / "average.npy", average)
    score = metrics(average, samples.y)
    error = torch.from_numpy(average).double()-torch.from_numpy(samples.y).double()
    np.testing.assert_allclose([score["mae"], score["mse"]], [error.abs().mean(), error.square().mean()], rtol=1e-12, atol=1e-12)
    diagnostic_scores = {}
    for name, values in diagnostics.items():
        averaged = np.mean(values, axis=0)
        diagnostic_scores[name] = {"equal_average": metrics(averaged, samples.y),
                                   "seeds": [{"seed": seed, **metrics(p, samples.y)} for seed, p in zip(SEEDS, values)]}
        np.save(out / f"{name}_average.npy", averaged)
    counts = np.array([store.news_counts[int(s):int(s)+96].sum() for s in samples.starts])
    threshold = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0
    report_available = np.asarray(store.reports[PRIMARY_LAG][samples.starts+95, -3:]).any(axis=1)
    masks = {"no_news": counts == 0, "low_news": (counts > 0) & (counts <= threshold),
             "high_news": counts > threshold, "no_report": ~report_available, "has_report": report_available}
    groups = {name: metrics(average[mask], samples.y[mask]) for name, mask in masks.items() if mask.any()}
    result = {"stage": stage, "split": split_name, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "seeds": scores, "seed_statistics": seed_statistics(scores), "equal_average": score,
              "diagnostics": diagnostic_scores, "groups": groups, "nonzero_news_count_median": threshold,
              "checks": checks, "selection_after_holdout": False,
              "caution": "Known internal temporal holdout; overlapping windows are not independent trials. Financial vintages/announcement times are incomplete."}
    write_json(here(stage) / f"results/{split_name}.json", result)
    stock_rows = []
    for stock, code in enumerate(store.codes):
        mask = samples.stocks == stock
        if mask.any():
            stock_rows.append({"stock": code, **metrics(average[mask], samples.y[mask])})
    with (here(stage) / f"results/{split_name}_per_stock.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stock_rows[0]))
        writer.writeheader()
        writer.writerows(stock_rows)
    return result


def main(default_stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", choices=tuple(DIRECTORIES), default=[default_stage] if default_stage else list(DIRECTORIES))
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.set_flush_denormal(True):
        raise RuntimeError("CPU flush-denormal mode unavailable")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    bases, sources = verify_bases(), source_hashes()
    for stage in args.stages:
        path = here(stage) / "results/protocol.json"
        if path.exists():
            if not args.resume:
                raise FileExistsError(f"Use --resume to preserve completed work: {path}")
            saved = read_json(path)
            for key, value in (("settings", SETTINGS), ("source_sha256", sources), ("market_sha256", bases)):
                if saved[key] != value:
                    raise ValueError(f"Resume would change {stage} {key}; keep this run and create a new declared experiment")
            write_json(here(stage) / f"results/resume_{time.time_ns()}.json", {"time_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256_file(path)})
        else:
            write_json(path, {"stage": stage, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                             "settings": SETTINGS, "source_sha256": sources, "market_sha256": bases,
                             "baseline": "The previously dev-selected stationary E6; identical frozen numerical weights for all modalities",
                             "selection": "Per-seed dev MAE early stopping; epoch zero eligible; fixed equal average of all 3 seeds",
                             "test_policy": "Freeze all requested experiments before any new holdout scores; no test-based tuning, lag choice, or seed choice",
                             "holdout_status": "Previously examined for E1/E6/E7; this is a known internal holdout, not an untouched blind/official test",
                             "e9_caveat": "E9 adds gates/dropout and jointly learns branches from zero heads; versus old E7 this is not a strict single-factor financial ablation",
                             "e10_caveat": "Compact SpecTF-style adaptation with 13 frequencies, 8 complex channels; not a full paper reproduction",
                             "environment": {"torch": str(torch.__version__), "numpy": str(np.__version__), "device": "cpu", "threads": 4}})
    cache = COMMON / f"outputs/cache_{time.time_ns()}"
    print("Loading unchanged numerical/news data and availability-aligned financial features...", flush=True)
    store = load_store(args.package_dir.resolve(), cache, with_news=True)
    assert store.metadata == read_json(OLD / "results/data_protocol.json")
    add_reports(store, args.package_dir.resolve(), cache)
    for stage in args.stages:
        path = here(stage) / "results/data_protocol.json"
        meta = {"market_news": store.metadata, "reports": store.report_metadata}
        if path.exists():
            assert read_json(path) == meta
        else:
            write_json(path, meta)
    print("Data ready. Training 3 fixed seeds for each separate experiment.", flush=True)
    for stage in args.stages:
        for seed in SEEDS:
            train_one(stage, seed, store, args.resume)
        dev_path = here(stage) / "results/development.json"
        if not dev_path.exists():
            evaluate(stage, store, store.dev, "development")
    if source_hashes() != sources or verify_bases() != bases:
        raise ValueError("Sources or frozen numerical weights changed while running")
    if args.evaluate_holdout:
        # Freeze every checkpoint before constructing or accessing holdout labels.
        for stage in args.stages:
            hashes = {str(s): read_json(here(stage) / f"results/seed{s}.json")["checkpoint_sha256"] for s in SEEDS}
            for seed in SEEDS:
                load_model(stage, seed)
            record = {"stage": stage, "checkpoint_sha256": hashes, "source_sha256": sources,
                      "development_sha256": sha256_file(here(stage) / "results/development.json"),
                      "settings": SETTINGS, "equal_average_seeds": list(SEEDS),
                      "diagnostics": "Disable branches, zero content, report lag270 with unchanged weights; no refitting or choosing lag on holdout",
                      "test_status": "Known internal holdout, 15390 samples; no significance or blind-test claim"}
            path = here(stage) / "results/holdout_protocol.json"
            if path.exists():
                assert read_json(path) == record
            else:
                write_json(path, record)
        holdout = load_holdout(store, args.package_dir.resolve())
        baseline = []
        for seed in SEEDS:
            model = MarketModel()
            model.load_state_dict(torch.load(base_path(seed), map_location="cpu", weights_only=True)["state_dict"])
            baseline.append(old_predict(model, store, holdout, "stationary", 1024))
        actual = metrics(np.mean(baseline, axis=0), holdout.y)
        expected = read_json(OLD / "results/holdout/summary.json")["scores"]["e6_average"]
        np.testing.assert_allclose([actual[k] for k in ("mae", "mse")], [expected[k] for k in ("mae", "mse")], atol=1e-10, rtol=0)
        for stage in args.stages:
            if not (here(stage) / "results/holdout.json").exists():
                evaluate(stage, store, holdout, "holdout")
        if source_hashes() != sources or verify_bases() != bases:
            raise ValueError("Sources or weights changed during evaluation")
    print("All requested runs and evaluations completed; results remain in their individual experiment directories.", flush=True)


if __name__ == "__main__":
    main()
