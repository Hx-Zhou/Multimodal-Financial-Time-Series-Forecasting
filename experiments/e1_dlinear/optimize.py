"""Bounded E1 follow-up: endpoint centering, then L2 regularization."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from .data import load_data, sha256_file
from .model import build_model
from .run import ROOT, fit, group_report, metrics, predict, write_json

HERE = Path(__file__).resolve().parent
# Fixed before looking at follow-up scores. Only weight decay varies across these.
CANDIDATES = [
    ("centered", 0.0),
    ("centered_l2_1e-3", 1e-3),
    ("centered_l2_1e-2", 1e-2),
]
SEEDS = [42, 43, 44]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/optimization")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/optimization")
    args = parser.parse_args()
    if (args.results_dir / "protocol.json").exists():
        parser.error("follow-up results already exist; choose fresh output and results directories")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    train, dev, split = load_data(args.package_dir.resolve())
    original = json.loads((HERE / "results/summary.json").read_text(encoding="utf-8"))
    baseline_predictions = np.repeat(dev.x[:, -1:].numpy(), 24, axis=1)
    baseline = metrics(baseline_predictions, dev.y.numpy())
    for key in ("mae", "mse"):
        np.testing.assert_allclose(baseline[key], original["e0_development"][key], rtol=0, atol=1e-8)
    compatibility = {"checked": False, "reason": "original local checkpoint unavailable"}
    old_checkpoint = HERE / "outputs/k25_seed42/model.pt"
    if old_checkpoint.exists():
        saved = torch.load(old_checkpoint, map_location="cpu", weights_only=True)
        old_model = build_model(saved["config"])
        old_model.load_state_dict(saved["state_dict"])
        restored = metrics(predict(old_model, dev.x, 1024), dev.y.numpy())
        recorded = json.loads((HERE / "results/k25_seed42.json").read_text())["development"]
        for key in ("mae", "mse"):
            np.testing.assert_allclose(restored[key], recorded[key], rtol=0, atol=1e-8)
        compatibility = {"checked": True, "restored_development": restored}
    common = dict(epochs=40, patience=6, min_delta=1e-5, batch_size=1024, lr=1e-3,
                  device="cpu", center_last=True)
    protocol = {
        "experiment": "E1 endpoint-centering and regularization follow-up",
        "split": split, "common_training": common, "threads": 4, "kernel": 25,
        "candidates": [{"name": name, "weight_decay": wd} for name, wd in CANDIDATES],
        "seeds": SEEDS, "loss": "MAE + 0.2 * MSE",
        "optimizer": "Adam; coupled L2 on all weights and biases; gradient clipping at 1.0",
        "selection": "screen all three candidates on seed 42 dev MAE; break exact ties with dev MSE; repeat winner on seeds 43 and 44",
        "original_checkpoint_compatibility": compatibility,
        "holdout_evaluated": False,
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha256_file(p) for p in
                          sorted(HERE.glob("*.py")) + [ROOT / "hs300_dataset.py", ROOT / "model.py", ROOT / "experiments/validation_split.py"]},
    }
    # Save the scope and selection rule before running any candidate.
    write_json(args.results_dir / "protocol.json", protocol)

    def run_candidate(name, decay, seed):
        print(f"\nCandidate={name}, weight_decay={decay}, seed={seed}", flush=True)
        run_args = argparse.Namespace(**common, weight_decay=decay,
                                      output_dir=args.output_dir / name, results_dir=args.results_dir / name)
        result = fit(train, dev, 25, seed, run_args)
        result["variant"] = name
        return result

    screening = [run_candidate(name, decay, SEEDS[0]) for name, decay in CANDIDATES]
    winner = min(screening, key=lambda r: (r["development"]["mae"], r["development"]["mse"]))
    name, decay = winner["variant"], winner["weight_decay"]
    print(f"\nDev-only selection: {name}; repeating seeds 43 and 44", flush=True)
    confirmation = [winner] + [run_candidate(name, decay, seed) for seed in SEEDS[1:]]
    aggregate = {}
    for key in baseline:
        if key == "samples":
            continue
        values = np.array([r["development"][key] for r in confirmation])
        original_mean = original["aggregate"][key]["mean"]
        aggregate[key] = {"mean": float(values.mean()), "std_population": float(values.std()),
                          "e0": baseline[key], "original_e1_mean": original_mean,
                          "improvement_over_e0_percent": float(100 * (1 - values.mean() / baseline[key])),
                          "improvement_over_original_e1_percent": float(100 * (1 - values.mean() / original_mean))}
    groups = {"e0": group_report(dev, baseline_predictions, args.results_dir / "e0_per_stock.csv")}
    for run in confirmation:
        predictions = np.load(args.output_dir / name / run["run_id"] / "dev_predictions.npy", allow_pickle=False)
        groups[run["run_id"]] = group_report(dev, predictions, args.results_dir / f"{run['run_id']}_per_stock.csv")
    write_json(args.results_dir / "news_coverage_groups.json", groups)
    compact = lambda r: {k: v for k, v in r.items() if k != "history"}
    summary = {"experiment": "E1 follow-up", "split": split, "e0_development": baseline,
               "selected_variant": name, "selected_config": winner["config"], "selected_weight_decay": decay,
               "screening": [compact(r) for r in screening], "confirmation": [compact(r) for r in confirmation],
               "aggregate": aggregate, "holdout_evaluated": False,
               "all_three_seeds_beat_e0_on_both_metrics": all(
                   r["development"]["mae"] < baseline["mae"] and r["development"]["mse"] < baseline["mse"]
                   for r in confirmation),
               "interpretation_note": "Development selection only; seed variation does not measure uncertainty across future periods."}
    write_json(args.results_dir / "summary.json", summary)
    print(json.dumps({"selected_variant": name, "mae": aggregate["mae"], "mse": aggregate["mse"],
                      "all_three_seeds_beat_e0": summary["all_three_seeds_beat_e0_on_both_metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
