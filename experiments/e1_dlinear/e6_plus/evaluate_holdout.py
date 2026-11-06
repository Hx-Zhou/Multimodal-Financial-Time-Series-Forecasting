"""Score the dev-selected frozen E6/E7 models once on the shared holdout."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset, _load_embeddings
from experiments.validation_split import make_validation_split
from ..data import sha256_file
from ..model import build_model as build_e1
from ..run import ROOT, metrics, predict as predict_e1, write_json
from .data import Samples, load_store
from .model import Model
from .run import predict

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean_scores(scores):
    return {key: {"mean": float(np.mean([s[key] for s in scores])),
                  "std_population": float(np.std([s[key] for s in scores]))}
            for key in ("mae", "mse")}


def improvement(score, base):
    return {key: 100 * (1 - score[key] / base[key]) for key in ("mae", "mse")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/holdout")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/holdout")
    args = parser.parse_args()
    if args.results_dir.exists() or args.output_dir.exists():
        parser.error("evaluation records exist; choose fresh directories, never overwrite scores")
    if not torch.set_flush_denormal(True):
        raise RuntimeError("CPU flush-denormal mode unsupported")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    development = read_json(HERE / "results/summary.json")
    data_reference = read_json(HERE / "results/data_protocol.json")
    representation = development["selected_representation"]
    records, models, paths, hashes = {}, {}, {}, {}
    for stage in ("e6", "e7"):
        for seed in SEEDS:
            name = f"{stage}_{representation}_seed{seed}"
            records[name] = read_json(HERE / f"results/{name}.json")
            path = HERE / f"outputs/{name}/model.pt"
            digest = sha256_file(path)
            if digest != records[name]["checkpoint_sha256"]:
                raise ValueError(f"checkpoint changed: {name}")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["representation"] != representation or saved["with_news"] != (stage == "e7"):
                raise ValueError("checkpoint configuration differs from dev selection")
            model = Model(news=saved["with_news"])
            model.load_state_dict(saved["state_dict"], strict=True)
            model.requires_grad_(False)
            models[name], paths[name], hashes[name] = model, path, digest
    for seed in SEEDS:
        market_name, news_name = f"e6_{representation}_seed{seed}", f"e7_{representation}_seed{seed}"
        if records[news_name]["market_checkpoint_sha256"] != hashes[market_name]:
            raise ValueError("E7 market source changed")
        for key, value in models[market_name].market.state_dict().items():
            torch.testing.assert_close(value, models[news_name].market.state_dict()[key], rtol=0, atol=0)
    e1_path = HERE.parent / "outputs/calibration/model.pt"
    e1_reference = read_json(HERE.parent / "results/holdout/summary.json")
    hashes["e1"] = sha256_file(e1_path)
    paths["e1"] = e1_path
    if hashes["e1"] != e1_reference["checkpoint_sha256"]:
        raise ValueError("original E1 checkpoint changed")
    saved = torch.load(e1_path, map_location="cpu", weights_only=True)
    e1_model = build_e1(saved["config"])
    e1_model.load_state_dict(saved["state_dict"], strict=True)
    e1_model.requires_grad_(False)
    args.output_dir.mkdir(parents=True)
    # Freeze the candidate list and protocol before loading/scoring holdout labels.
    write_json(args.results_dir / "protocol.json", {
        "experiment": "Frozen dev-selected E6/E7 holdout comparison",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_representation": representation, "seeds": list(SEEDS),
        "candidates": list(models), "checkpoint_sha256": hashes,
        "selection": "Representation/checkpoints were selected on dev before this evaluation; report all three seeds, no test-seed selection",
        "averaging": "Equal prediction average of the same three seeds for each stage; no member or weight search",
        "diagnostic": "E7 semantic-output zeroing, retaining counts/masks, on all three fixed models; no refitting",
        "baseline": "E0 last value and the previously frozen E1 checkpoint",
        "metric": "Uniform mean over sample-horizon positions, standardized Close, float64 error reductions",
        "normalization": "Reuse the original training-only loader; require exact equality to committed preprocessing metadata before any scoring",
        "holdout_status": "Previously revealed for E1; first E6/E7 evaluation. This is a known internal holdout, not a new blind test or official hidden test.",
        "holdout_policy": "No training, hyperparameter changes, seed selection, or further tuning on these scores",
        "development_summary_sha256": sha256_file(HERE / "results/summary.json"),
        "data_protocol_sha256": sha256_file(HERE / "results/data_protocol.json"),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p)
                          for p in (Path(__file__), HERE / "data.py", HERE / "model.py", HERE / "run.py",
                                    ROOT / "hs300_dataset.py", ROOT / "experiments/validation_split.py")},
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu", "cpu_flush_denormal": True},
    })
    print("Frozen six E6/E7 checkpoints plus E1; verifying original preprocessing...", flush=True)
    store = load_store(args.package_dir.resolve(), args.output_dir / "cache", with_news=True)
    if store.metadata != data_reference:
        raise ValueError("preprocessing no longer matches the frozen training run")
    dataset = HS300Dataset(args.package_dir.resolve(), "val")
    try:
        split = make_validation_split(dataset)
        if dataset.stock_codes != store.codes:
            raise ValueError("stock order differs")
        if len(split.holdout_indices) != data_reference["split"]["holdout_samples"]:
            raise ValueError("holdout sample count changed")
        if (split.holdout_start_date != data_reference["split"]["holdout_target_start_date_min"]
                or split.holdout_end_date != data_reference["split"]["holdout_target_start_date_max"]
                or split.dev_target_end_date >= split.holdout_start_date):
            raise ValueError("holdout boundaries changed")
        offsets = np.r_[0, np.cumsum([len(dataset.price_values[code]) for code in store.codes])].astype(np.int64)
        idx = split.holdout_indices
        holdout = Samples(offsets[dataset.stock_index[idx]] + dataset.input_start[idx],
                          dataset.y[idx].astype(np.float32), dataset.stock_index[idx].copy(),
                          dataset.target_start_date[idx].copy())
        split_record = {**data_reference["split"],
                        "holdout_target_end_date_max": int(split.target_end_dates[idx].max()),
                        "unique_target_start_dates": int(len(np.unique(holdout.dates))),
                        "stocks": int(len(np.unique(holdout.stocks)))}
        # Verify the additional holdout mapping against the unchanged public reader.
        for position in np.linspace(0, len(idx)-1, 20, dtype=int):
            sample = dataset[int(idx[position])]
            normalized = (sample["numerical"][:, 3] - np.float32(sample["close_mean"])) / max(np.float32(sample["close_std"]), np.float32(1e-6))
            np.testing.assert_array_equal(store.close[holdout.starts[position]:holdout.starts[position]+96], normalized)
            np.testing.assert_array_equal(holdout.y[position], sample["y"])
    finally:
        dataset.close()
        HS300Dataset._reports.cache_clear()
        _load_embeddings.cache_clear()
    del dataset, sample
    gc.collect()
    target = holdout.y
    x = torch.from_numpy(store.close[holdout.starts[:, None] + np.arange(96)])
    predictions = {"e0": np.repeat(x[:, -1:].numpy(), 24, axis=1),
                   "e1": predict_e1(e1_model, x, 1024)}
    scores = {name: metrics(values, target) for name, values in predictions.items()}
    for name in ("e0", "e1"):
        for key in ("mae", "mse"):
            np.testing.assert_allclose(scores[name][key], e1_reference[f"{name}_holdout"][key], rtol=0, atol=1e-10)
    dev_verified = {}
    for name, model in models.items():
        check = metrics(predict(model, store, store.dev, representation, 1024), store.dev.y)
        for key in ("mae", "mse"):
            np.testing.assert_allclose(check[key], records[name]["development"][key], rtol=0, atol=1e-9)
        dev_verified[name] = check
        predictions[name] = predict(model, store, holdout, representation, 1024)
        scores[name] = metrics(predictions[name], target)
        print(f"{name}: holdout MAE={scores[name]['mae']:.9f} MSE={scores[name]['mse']:.9f}", flush=True)
        if model.with_news:
            zero = name + "_zero_semantics"
            predictions[zero] = predict(model, store, holdout, representation, 1024, zero_semantics=True)
            scores[zero] = metrics(predictions[zero], target)
    for stage in ("e6", "e7"):
        predictions[stage + "_average"] = np.mean([predictions[f"{stage}_{representation}_seed{seed}"] for seed in SEEDS], axis=0)
        scores[stage + "_average"] = metrics(predictions[stage + "_average"], target)
    predictions["e7_average_zero_semantics"] = np.mean([predictions[f"e7_{representation}_seed{seed}_zero_semantics"] for seed in SEEDS], axis=0)
    scores["e7_average_zero_semantics"] = metrics(predictions["e7_average_zero_semantics"], target)
    for name, values in predictions.items():
        error = torch.from_numpy(values).double() - torch.from_numpy(target).double()
        np.testing.assert_allclose([scores[name]["mae"], scores[name]["mse"]],
                                   [error.abs().mean().item(), error.square().mean().item()], rtol=1e-12, atol=1e-12)
        np.save(args.output_dir / f"{name}.npy", values)
    for name, path in paths.items():
        if sha256_file(path) != hashes[name]:
            raise ValueError("weights changed during evaluation")
    counts = np.asarray([store.news_counts[int(start):int(start)+96].sum() for start in holdout.starts])
    threshold = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0
    masks = {"no_news": counts == 0, "low_coverage": (counts > 0) & (counts <= threshold), "high_coverage": counts > threshold}
    if masks["no_news"].any():
        np.testing.assert_array_equal(predictions["e6_average"][masks["no_news"]], predictions["e7_average"][masks["no_news"]])
    groups = {"news_count_definition": "unique vector IDs per trading calendar day within the input window",
              "nonzero_count_median": threshold,
              "models": {name: {group: metrics(values[mask], target[mask]) for group, mask in masks.items() if mask.any()}
                         for name, values in predictions.items() if name in ("e0", "e1", "e6_average", "e7_average")}}
    write_json(args.results_dir / "news_coverage_groups.json", groups)
    rows = []
    for name in ("e0", "e1", "e6_average", "e7_average"):
        for stock, code in enumerate(store.codes):
            mask = holdout.stocks == stock
            if mask.any():
                score = metrics(predictions[name][mask], target[mask])
                rows.append({"model": name, "stock_code": code, "samples": int(mask.sum()), "mae": score["mae"], "mse": score["mse"]})
    with (args.results_dir / "per_stock.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": "Frozen E6/E7 holdout evaluation", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_representation": representation, "split": split_record, "holdout_evaluated": True,
        "selection_after_holdout": False, "checkpoint_sha256": hashes, "scores": scores,
        "seed_statistics": {stage: mean_scores([scores[f"{stage}_{representation}_seed{s}"] for s in SEEDS]) for stage in ("e6", "e7")},
        "average_improvements_percent": {
            "e6_vs_e0": improvement(scores["e6_average"], scores["e0"]),
            "e6_vs_e1": improvement(scores["e6_average"], scores["e1"]),
            "e7_vs_e0": improvement(scores["e7_average"], scores["e0"]),
            "e7_vs_e1": improvement(scores["e7_average"], scores["e1"]),
            "e7_vs_e6": improvement(scores["e7_average"], scores["e6_average"])},
        "e7_all_seeds_improve_matched_e6_both": all(scores[f"e7_{representation}_seed{s}"][key] < scores[f"e6_{representation}_seed{s}"][key] for s in SEEDS for key in ("mae", "mse")),
        "e7_average_semantic_zero_delta": {key: scores["e7_average_zero_semantics"][key]-scores["e7_average"][key] for key in ("mae", "mse")},
        "dev_reload_verified": dev_verified,
        "interpretation": "Previously known internal temporal holdout, overlapping windows and shared market dates; no significance claim or new model selection.",
    }
    write_json(args.results_dir / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("seed_statistics", "average_improvements_percent", "e7_average_semantic_zero_delta")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
