"""E1 deterministic fitting, recency weighting, and robust full-batch refinement."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_data, sha256_file
from .linear_solver import design_matrix, export_model, penalty_diagonal, sufficient_statistics, ridge_solution, robust_solution
from .model import build_model
from .run import ROOT, group_report, metrics, predict, write_json

HERE = Path(__file__).resolve().parent
REGULARIZATION = (0.001, 0.01, 0.1, 1.0)
REGIMES = (("uniform", None), ("recent_252", 252), ("recent_126", 126))


def weights_for_dates(dates, half_life):
    if half_life is None:
        return torch.ones(len(dates), dtype=torch.float64)
    _, ranks = np.unique(dates, return_inverse=True)
    age = torch.from_numpy(ranks.max() - ranks).double()
    return torch.exp2(-age / half_life)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/refinement")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/refinement")
    args = parser.parse_args()
    if (args.results_dir / "protocol.json").exists():
        parser.error("results exist; use fresh output and results directories")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    training, dev, split = load_data(args.package_dir.resolve())
    baseline_predictions = np.repeat(dev.x[:, -1:].numpy(), 24, axis=1)
    baseline = metrics(baseline_predictions, dev.y.numpy())
    original = json.loads((HERE / "results/summary.json").read_text(encoding="utf-8"))
    for key in ("mae", "mse"):
        np.testing.assert_allclose(baseline[key], original["e0_development"][key], rtol=0, atol=1e-8)
    protocol = {
        "experiment": "E1 deterministic linear fitting and recency weighting",
        "split": split, "model_config": {"moving_average": 25, "center_last": True},
        "ridge_regularization": list(REGULARIZATION),
        "regimes": [{"name": name, "half_life_in_unique_training_target_dates": half} for name, half in REGIMES],
        "ridge_objective": "weighted mean squared residual error + lambda * sum(all model parameters squared) / 24",
        "robust_objective": "weighted mean(sqrt(error^2 + 0.001^2) - 0.001 + 0.2 * error^2) + same normalized L2",
        "robust_candidates": "best dev MAE uniform ridge and best dev MAE recency ridge; each refined once on training loss only",
        "final_selection": "minimum dev MAE among candidates strictly better than E0 in both MAE and MSE; otherwise minimum dev MAE",
        "scope": "12 ridge candidates plus 2 full-batch refinements, fixed before fitting; no validation-label fitting or holdout scoring",
        "deterministic": True, "seed_repeats": "not applicable: closed-form and full-batch deterministic fitting",
        "device": "cpu", "dtype_for_solver": "float64", "threads": 4,
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha256_file(p) for p in
                          sorted(HERE.glob("*.py")) + [ROOT / "hs300_dataset.py", ROOT / "model.py", ROOT / "experiments/validation_split.py"]},
    }
    write_json(args.results_dir / "protocol.json", protocol)
    print("Building fixed DLinear features in float64...", flush=True)
    features = design_matrix(training.x)
    dev_features = design_matrix(dev.x)
    target = training.y.double() - training.x[:, -1:].double()
    stats, coefficients_by_id, all_results = {}, {}, []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def record(name, regime, regularization, coefficients, solver, details):
        model = export_model(coefficients)
        prediction = predict(model, dev.x, 1024)
        expected = dev.x[:, -1:].double() + dev_features @ coefficients
        maximum_difference = float(np.abs(prediction.astype(np.float64) - expected.numpy()).max())
        np.testing.assert_allclose(prediction, expected.numpy(), atol=5e-6, rtol=1e-5)
        checkpoint = args.output_dir / f"{name}.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        torch.save({"config": {"moving_average": 25, "center_last": True}, "state_dict": model.state_dict()}, checkpoint)
        np.save(args.output_dir / f"{name}_dev_predictions.npy", prediction)
        result = {"candidate": name, "regime": regime, "regularization": regularization,
                  "solver": solver, "development": metrics(prediction, dev.y.numpy()),
                  "export_max_absolute_difference": maximum_difference,
                  "checkpoint_sha256": sha256_file(checkpoint), "solver_details": details,
                  "holdout_evaluated": False}
        write_json(args.results_dir / f"{name}.json", result)
        coefficients_by_id[name] = coefficients
        all_results.append(result)
        print(f"{name}: dev MAE={result['development']['mae']:.8f} MSE={result['development']['mse']:.8f}", flush=True)
        return result

    for regime, half_life in REGIMES:
        started = time.perf_counter()
        weights = weights_for_dates(training.target_dates, half_life)
        print(f"Computing sufficient statistics: {regime}", flush=True)
        gram, cross = sufficient_statistics(features, target, weights)
        stats[regime] = (weights, gram, cross)
        effective_samples = float(weights.sum().square() / weights.square().sum())
        target_second_moment = (weights * target.square().mean(dim=1)).sum() / weights.sum()
        for regularization in REGULARIZATION:
            coefficients, _, equation_error = ridge_solution(gram, cross, regularization)
            weighted_mse = float(target_second_moment + ((coefficients * (gram @ coefficients)).sum()
                                                        - 2 * (coefficients * cross).sum()) / 24)
            penalty = float(regularization * (penalty_diagonal()[:, None] * coefficients.square()).sum() / 24)
            name = f"ridge_{regime}_l2_{regularization:g}".replace(".", "p")
            record(name, regime, regularization, coefficients, "closed_form_ridge",
                   {"normal_equation_max_residual": equation_error, "weighted_training_mse": weighted_mse,
                    "regularized_training_objective": weighted_mse + penalty,
                    "effective_samples": effective_samples, "half_life": half_life,
                    "regime_elapsed_seconds": time.perf_counter() - started})
    key = lambda r: (r["development"]["mae"], r["development"]["mse"])
    uniform = min((r for r in all_results if r["regime"] == "uniform"), key=key)
    recent = min((r for r in all_results if r["regime"] != "uniform"), key=key)
    for initial_result in (uniform, recent):
        regime, regularization = initial_result["regime"], initial_result["regularization"]
        weights, gram, cross = stats[regime]
        initial = coefficients_by_id[initial_result["candidate"]]
        _, cholesky, _ = ridge_solution(gram, cross, regularization)
        name = initial_result["candidate"].replace("ridge_", "robust_")
        print(f"Refining training objective: {name}", flush=True)
        coefficients, details = robust_solution(features, target, weights, initial, cholesky, regularization)
        details["starting_candidate"] = initial_result["candidate"]
        record(name, regime, regularization, coefficients, "full_batch_smoothed_mae_plus_mse", details)
    eligible = [r for r in all_results if r["development"]["mae"] < baseline["mae"]
                and r["development"]["mse"] < baseline["mse"]]
    winner = min(eligible or all_results, key=key)
    prediction = np.load(args.output_dir / f"{winner['candidate']}_dev_predictions.npy", allow_pickle=False)
    saved = torch.load(args.output_dir / f"{winner['candidate']}.pt", map_location="cpu", weights_only=True)
    restored = build_model(saved["config"])
    restored.load_state_dict(saved["state_dict"])
    np.testing.assert_array_equal(prediction, predict(restored, dev.x, 1024))
    groups = {"e0": group_report(dev, baseline_predictions, args.results_dir / "e0_per_stock.csv"),
              "selected": group_report(dev, prediction, args.results_dir / "selected_per_stock.csv")}
    write_json(args.results_dir / "news_coverage_groups.json", groups)
    time_groups = []
    for dates in np.array_split(np.unique(dev.target_dates), 3):
        mask = np.isin(dev.target_dates, dates)
        time_groups.append({"target_start_min": int(dates[0]), "target_start_max": int(dates[-1]),
                            "e0": metrics(baseline_predictions[mask], dev.y.numpy()[mask]),
                            "selected": metrics(prediction[mask], dev.y.numpy()[mask])})
    summary = {"experiment": "E1 deterministic refinement", "e0_development": baseline,
               "selected": winner, "all_candidates": all_results, "beats_e0_on_both_metrics": bool(eligible),
               "improvement_over_e0_percent": {k: 100 * (1 - winner["development"][k] / baseline[k]) for k in ("mae", "mse")},
               "development_time_blocks": time_groups,
               "time_block_note": "Descriptive only; blocks share overlapping target dates and were not independent holdouts.",
               "holdout_evaluated": False, "deterministic_solver": True}
    write_json(args.results_dir / "summary.json", summary)
    print(json.dumps({"selected": winner["candidate"], "development": winner["development"],
                      "improvement_over_e0_percent": summary["improvement_over_e0_percent"],
                      "beats_both": bool(eligible)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
