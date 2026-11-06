"""Compare a saved E1 checkpoint with E0 on train/dev; never evaluate holdout."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import load_data, sha256_file
from .model import build_model
from .run import metrics, predict, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    train, dev, _ = load_data(args.package_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    results = {"config": checkpoint["config"], "checkpoint_sha256": sha256_file(args.checkpoint),
               "holdout_evaluated": False}
    for name, data in (("train", train), ("development", dev)):
        baseline = np.repeat(data.x[:, -1:].numpy(), 24, axis=1)
        prediction = predict(model, data.x, 2048)
        target = data.y.numpy()
        e0, e1 = metrics(baseline, target), metrics(prediction, target)
        e0["loss"] = e0["mae"] + 0.2 * e0["mse"]
        e1["loss"] = e1["mae"] + 0.2 * e1["mse"]
        results[name] = {"e0": e0, "e1": e1,
                         "mean_target_change": float((target.astype(np.float64) - baseline).mean()),
                         "mean_predicted_change": float((prediction.astype(np.float64) - baseline).mean())}
        print(f"{name}: E0 loss={e0['loss']:.6f}; E1 loss={e1['loss']:.6f}; "
              f"target_change={results[name]['mean_target_change']:.6f}; "
              f"predicted_change={results[name]['mean_predicted_change']:.6f}", flush=True)
    write_json(args.output, results)


if __name__ == "__main__":
    main()
