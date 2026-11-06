"""Evaluate only the frozen final E1 checkpoint and E0 on the shared holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hs300_dataset import HS300Dataset
from experiments.validation_split import make_validation_split
from .data import materialize, sha256_file
from .model import HORIZON, build_model
from .run import ROOT, group_report, metrics, predict, write_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=HERE / "outputs/calibration/model.pt")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results/holdout")
    args = parser.parse_args()
    if args.results_dir.exists():
        parser.error("results directory exists; preserve the original evaluation record")
    selected = json.loads((HERE / "results/calibration/summary.json").read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != selected["averaged_model"]["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the previously selected final model")
    package_dir = args.package_dir.resolve()
    val_hash = sha256_file(package_dir / "data/val.npz")
    if val_hash != selected["split"]["val_npz_sha256"]:
        raise ValueError("validation data differs from the development experiment")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.requires_grad_(False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    split_path = ROOT / "experiments/validation_split.json"
    reference = json.loads(split_path.read_text(encoding="utf-8"))
    source_paths = [Path(__file__), HERE / "data.py", HERE / "model.py", HERE / "run.py",
                    ROOT / "hs300_dataset.py", ROOT / "experiments/validation_split.py"]
    # This record is written before materializing or scoring any holdout samples.
    write_json(args.results_dir / "protocol.json", {
        "experiment": "E1 frozen final model versus E0 on shared holdout",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "config": saved["config"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "selection": "Previously selected calibration averaged_model; no holdout selection or training",
        "residual_scale": "Already folded into saved weights; no additional scaling",
        "baseline": "Repeat last standardized input Close for all 24 forecast positions",
        "metric": "Uniform mean across all sample-horizon positions, standardized Close, float64 errors",
        "holdout_policy": "Evaluate this single candidate once; do not tune or select on these scores",
        "val_npz_sha256": val_hash,
        "shared_split_sha256": sha256_file(split_path),
        "shared_split": reference,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in source_paths},
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu"},
    })
    print("Frozen checkpoint and data hashes verified; loading the shared holdout...", flush=True)
    dataset = HS300Dataset(package_dir, "val")
    try:
        split = make_validation_split(dataset, holdout_fraction=0.2)
        actual = {
            "development_samples": len(split.dev_indices),
            "purged_overlap_samples": len(split.purged_indices),
            "holdout_samples": len(split.holdout_indices),
            "development_target_start_date_min": split.dev_start_date,
            "development_target_start_date_max": split.dev_end_date,
            "development_target_end_date_max": split.dev_target_end_date,
            "holdout_target_start_date_min": split.holdout_start_date,
            "holdout_target_start_date_max": split.holdout_end_date,
        }
        for key, value in actual.items():
            if value != reference[key]:
                raise ValueError(f"shared split mismatch: {key}: {value} != {reference[key]}")
        if split.dev_target_end_date >= split.holdout_start_date:
            raise ValueError("development labels overlap holdout target dates")
        holdout = materialize(dataset, split.holdout_indices, news_counts=True)
        actual["holdout_target_end_date_max"] = int(split.target_end_dates[split.holdout_indices].max())
        actual["unique_target_start_dates"] = int(len(np.unique(holdout.target_dates)))
        actual["stocks"] = int(len(np.unique(holdout.stocks)))
    finally:
        dataset.close()
    target = holdout.y.numpy()
    baseline = np.repeat(holdout.x[:, -1:].numpy(), HORIZON, axis=1)
    prediction = predict(model, holdout.x, 1024)
    if prediction.shape != target.shape or baseline.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    e0, e1 = metrics(baseline, target), metrics(prediction, target)
    # Independent torch reductions check the shared NumPy metric implementation.
    for values, score in ((baseline, e0), (prediction, e1)):
        error = torch.from_numpy(values).double() - holdout.y.double()
        np.testing.assert_allclose(
            [score["mae"], score["mse"]],
            [error.abs().mean().item(), error.square().mean().item()], rtol=1e-12, atol=1e-12)
    if sha256_file(args.checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed during evaluation")
    groups = {
        "e0": group_report(holdout, baseline, args.results_dir / "e0_per_stock.csv"),
        "e1": group_report(holdout, prediction, args.results_dir / "e1_per_stock.csv"),
    }
    write_json(args.results_dir / "news_coverage_groups.json", groups)
    summary = {
        "experiment": "E1 frozen final checkpoint versus E0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_evaluated": True,
        "checkpoint_sha256": checkpoint_hash,
        "holdout_close_sha256": hashlib.sha256(holdout.x.numpy().tobytes()).hexdigest(),
        "split": actual,
        "e0_holdout": e0,
        "e1_holdout": e1,
        "e1_minus_e0": {key: e1[key] - e0[key] for key in ("mae", "mse")},
        "improvement_over_e0_percent": {key: 100 * (1 - e1[key] / e0[key]) for key in ("mae", "mse")},
        "e1_beats_e0_on_both_metrics": all(e1[key] < e0[key] for key in ("mae", "mse")),
        "selection_after_holdout": False,
        "interpretation": "One previously untouched temporal holdout; overlapping windows and shared market dates are not independent samples. No statistical significance claim.",
        "future_use": "Holdout is now observed for E1 and must not guide further tuning.",
    }
    write_json(args.results_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

