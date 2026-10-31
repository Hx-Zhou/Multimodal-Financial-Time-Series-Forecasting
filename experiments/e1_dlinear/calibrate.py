"""Constrained horizon shrinkage and exact averaging of compatible DLinear weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_data, sha256_file
from .model import build_model
from .run import ROOT, group_report, metrics, predict, write_json

HERE = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


def shrink_model(model, scale):
    config = {"moving_average": model.moving_average, "center_last": model.center_last}
    result = build_model(config)
    result.load_state_dict(model.state_dict())
    scale = torch.as_tensor(scale, dtype=torch.float32)
    if scale.shape != (24,) or not torch.isfinite(scale).all() or (scale < 0).any() or (scale > 1).any():
        raise ValueError("expected 24 finite residual scales in [0, 1]")
    with torch.no_grad():
        for layer in (result.trend_head, result.remainder_head):
            layer.weight.mul_(scale[:, None])
            layer.bias.mul_(scale)
    return result


def average_models(models):
    if not models:
        raise ValueError("models must not be empty")
    config = {"moving_average": models[0].moving_average, "center_last": models[0].center_last}
    if any(m.moving_average != config["moving_average"] or m.center_last != config["center_last"] for m in models):
        raise ValueError("averaging requires identical decomposition and centering")
    result = build_model(config)
    states = [m.state_dict() for m in models]
    result.load_state_dict({key: torch.stack([state[key].double() for state in states]).mean(dim=0).float()
                           for key in states[0]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--source-dir", type=Path, default=HERE / "outputs/optimization/centered_l2_1e-2")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/calibration")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/calibration")
    args = parser.parse_args()
    if (args.results_dir / "protocol.json").exists():
        parser.error("results exist; choose fresh output and results directories")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    steps = torch.arange(1, 25, dtype=torch.float32) / 24
    curves = {"unchanged": torch.ones(24), "constant_half": torch.full((24,), 0.5),
              "linear_horizon": steps, "quadratic_horizon": steps.square()}
    write_json(args.results_dir / "protocol.json", {
        "experiment": "E1 fixed residual shrinkage and weight averaging",
        "curves_fixed_before_scoring": {name: scale.tolist() for name, scale in curves.items()},
        "selection": "seed 42 only: minimum dev MAE among curves beating E0 on both MAE/MSE, otherwise minimum dev MAE",
        "confirmation": "freeze the selected curve, report all three seeds; then average all three checkpoints equally with no member selection",
        "source_model": "centered DLinear, kernel 25, Adam weight_decay 0.01", "source_seeds": list(SEEDS),
        "interpretation": "This is validation-selected residual shrinkage; seed runs do not constitute independent temporal test data.",
        "holdout_evaluated": False,
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha256_file(p) for p in sorted(HERE.glob("*.py"))},
    })
    training, dev, split = load_data(args.package_dir.resolve())
    del training
    baseline_prediction = np.repeat(dev.x[:, -1:].numpy(), 24, axis=1)
    baseline = metrics(baseline_prediction, dev.y.numpy())
    models, originals = [], []
    source_hashes = {}
    for seed in SEEDS:
        path = args.source_dir / f"k25_seed{seed}/model.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(saved["config"])
        model.load_state_dict(saved["state_dict"])
        models.append(model)
        originals.append(predict(model, dev.x, 1024))
        source_hashes[str(seed)] = sha256_file(path)
    screening = []
    for name, scale in curves.items():
        prediction = predict(shrink_model(models[0], scale), dev.x, 1024)
        expected = baseline_prediction + scale.numpy() * (originals[0] - baseline_prediction)
        np.testing.assert_allclose(prediction, expected, rtol=1e-5, atol=2e-6)
        score = metrics(prediction, dev.y.numpy())
        screening.append({"curve": name, "development": score})
        print(f"{name}: MAE={score['mae']:.8f}, MSE={score['mse']:.8f}", flush=True)
    eligible = [r for r in screening if r["development"]["mae"] < baseline["mae"] and r["development"]["mse"] < baseline["mse"]]
    winner = min(eligible or screening, key=lambda r: (r["development"]["mae"], r["development"]["mse"]))
    name, scale = winner["curve"], curves[winner["curve"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confirmation, calibrated, predictions = [], [], []
    config = {"moving_average": models[0].moving_average, "center_last": models[0].center_last}
    for seed, source in zip(SEEDS, models):
        model = shrink_model(source, scale)
        prediction = predict(model, dev.x, 1024)
        score = metrics(prediction, dev.y.numpy())
        confirmation.append({"seed": seed, "development": score})
        calibrated.append(model)
        predictions.append(prediction)
        torch.save({"config": config, "state_dict": model.state_dict()}, args.output_dir / f"{name}_seed{seed}.pt")
        print(f"selected={name}, seed={seed}: MAE={score['mae']:.8f}, MSE={score['mse']:.8f}", flush=True)
    ensemble = average_models(calibrated)
    ensemble_prediction = predict(ensemble, dev.x, 1024)
    np.testing.assert_allclose(ensemble_prediction, np.mean(predictions, axis=0), rtol=1e-5, atol=2e-6)
    checkpoint = args.output_dir / "model.pt"
    torch.save({"config": config, "state_dict": ensemble.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded = build_model(saved["config"])
    reloaded.load_state_dict(saved["state_dict"])
    np.testing.assert_array_equal(predict(reloaded, dev.x, 1024), ensemble_prediction)
    np.save(args.output_dir / "dev_predictions.npy", ensemble_prediction)
    ensemble_score = metrics(ensemble_prediction, dev.y.numpy())
    aggregate = {key: {"mean": float(np.mean([r["development"][key] for r in confirmation])),
                       "std_population": float(np.std([r["development"][key] for r in confirmation]))}
                 for key in baseline if key != "samples"}
    write_json(args.results_dir / "news_coverage_groups.json", {
        "e0": group_report(dev, baseline_prediction, args.results_dir / "e0_per_stock.csv"),
        "averaged": group_report(dev, ensemble_prediction, args.results_dir / "averaged_per_stock.csv")})
    time_blocks = []
    for dates in np.array_split(np.unique(dev.target_dates), 3):
        mask = np.isin(dev.target_dates, dates)
        time_blocks.append({"target_start_min": int(dates[0]), "target_start_max": int(dates[-1]),
                            "e0": metrics(baseline_prediction[mask], dev.y.numpy()[mask]),
                            "averaged": metrics(ensemble_prediction[mask], dev.y.numpy()[mask])})
    summary = {"experiment": "E1 residual shrinkage", "split": split,
               "e0_development": baseline, "screening_seed42": screening, "selected_curve": name,
               "confirmation": confirmation, "aggregate": aggregate,
               "all_seeds_beat_e0_on_both_metrics": all(r["development"]["mae"] < baseline["mae"] and r["development"]["mse"] < baseline["mse"] for r in confirmation),
               "averaged_model": {"development": ensemble_score, "parameters": sum(p.numel() for p in ensemble.parameters()),
                                  "improvement_over_e0_percent": {k: 100 * (1 - ensemble_score[k] / baseline[k]) for k in ("mae", "mse")},
                                  "checkpoint_sha256": sha256_file(checkpoint)},
               "source_checkpoint_sha256": source_hashes, "development_time_blocks": time_blocks,
               "holdout_evaluated": False, "time_block_note": "Descriptive slices of development; target dates overlap across slices."}
    write_json(args.results_dir / "summary.json", summary)
    print(json.dumps({"selected_curve": name, "aggregate": {k: aggregate[k] for k in ("mae", "mse")},
                      "all_seeds_beat_e0": summary["all_seeds_beat_e0_on_both_metrics"],
                      "averaged_model": summary["averaged_model"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
