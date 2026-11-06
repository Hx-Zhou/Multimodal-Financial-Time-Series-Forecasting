"""Tests for lag comparison validation and development-only selection."""

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.gru_e14_b.run import summarize, validate_result, write_json


class LagTests(unittest.TestCase):
    def test_validate_rejects_wrong_lag_or_epoch(self):
        args = argparse.Namespace(seed=42, batch_size=256)
        result = {"report_lag_days": 30, "seed": 42, "batch_size": 256,
                  "holdout_fraction": .2, "best_epoch": 2,
                  "best_development": {"epoch": 2, "mae": .2, "mse": .1},
                  "holdout": {"checkpoint_epoch": 2, "mae": .3, "mse": .2}}
        checkpoint = {"report_lag_days": 30, "holdout_fraction": .2, "epoch": 2,
                      "feature_stats": {"mean": np.zeros(12).tolist(), "std": np.ones(12).tolist()}}
        validate_result(result, checkpoint, 30, args)
        with self.assertRaisesRegex(ValueError, "another report lag"):
            validate_result(result, checkpoint, 60, args)
        checkpoint["epoch"] = 3
        with self.assertRaisesRegex(ValueError, "epoch mismatch"):
            validate_result(result, checkpoint, 30, args)

    def test_selection_uses_development_mae_not_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output_dir=Path(directory), lags=[0, 30], seed=42)
            for lag, dev_mae, holdout_mae in [(0, .20, .50), (30, .21, .10)]:
                _, output, _ = (args.output_dir / f"lag_{lag}", args.output_dir / f"lag_{lag}/result.json", None)
                write_json(output, {"report_lag_days": lag, "best_epoch": 1,
                                    "best_development": {"mae": dev_mae, "mse": dev_mae},
                                    "holdout": {"mae": holdout_mae, "mse": holdout_mae}})
            summary = summarize(args, {"lags": [0, 30]})
            self.assertEqual(summary["selected_lag_by_development"], 0)
            self.assertEqual(summary["status"], "complete")

    def test_default_candidates_exclude_zero_lag(self):
        from experiments.gru_e14_b.run import LAGS
        self.assertEqual(LAGS, (30, 60, 90))


if __name__ == "__main__":
    unittest.main()
