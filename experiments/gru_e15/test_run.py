"""Meaningful guards for paired aggregation and isolated E14-A execution."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.gru_e15.run import (
    CONFIG, EXPERIMENTS, mean_std, run_paths, summarize, validate_run, write_json,
)
from experiments.gru_e10_b import train as base
from experiments.gru_e14_a import train as masked


class ComparisonTests(unittest.TestCase):
    def test_sample_std_and_single_seed(self):
        self.assertIsNone(mean_std([0.2])["std"])
        self.assertAlmostEqual(mean_std([1, 2, 3])["std"], 1)

    def test_partial_summary_pairs_only_common_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output_dir=Path(directory), seeds=[42, 43, 44])
            for family, seed, value in [("e10_b", 42, .20), ("e10_b", 43, .22), ("e14_a", 42, .19)]:
                _, output, _ = run_paths(args, family, seed)
                write_json(output, {"best_epoch": 3, "best_development": {"mae": value, "mse": value * 2},
                                    "holdout": {"mae": value * 3, "mse": value * 4}})
            summary = summarize(args)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(len(summary["missing_runs"]), 3)
            self.assertEqual(summary["models"]["e10_b"]["aggregate"]["development"]["mae"]["n"], 2)
            pairs = summary["paired_delta_e14a_minus_e10b"]
            self.assertEqual([row["seed"] for row in pairs], [42])
            self.assertAlmostEqual(pairs[0]["development"]["mae"], -.01)
            self.assertEqual(summary["paired_delta_aggregate"]["development"]["mae"]["e14a_wins"], 1)

    def test_e14a_restores_e10b_functions_even_when_training_fails(self):
        names = ("FinancialIndex", "fit_feature_stats", "add_financial_batch", "build_model")
        originals = {key: getattr(base, key) for key in names}
        with patch.object(base, "run", side_effect=RuntimeError("interrupted training")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                masked.run(argparse.Namespace())
        for key in names:
            self.assertIs(getattr(base, key), originals[key])

    def test_reuse_rejects_different_market_weights_and_epochs(self):
        result = {"experiment": EXPERIMENTS["e10_b"], "seed": 42, "report_lag_days": 30,
                  "holdout_fraction": .2, "batch_size": 256, "learning_rate": 3e-4,
                  "config": CONFIG, "best_epoch": 2,
                  "best_development": {"epoch": 2, "mae": .2, "mse": .1},
                  "holdout": {"checkpoint_epoch": 2, "mae": .3, "mse": .2}}
        checkpoint = {"epoch": 2, "config": CONFIG, "report_lag_days": 30, "holdout_fraction": .2,
                      "model_state": {"market_gru.weight": torch.tensor([1.]), "market_head.weight": torch.tensor([2.])},
                      "feature_stats": {"mean": np.zeros(12).tolist(), "std": np.ones(12).tolist()}}
        reference = {"gru.weight": torch.tensor([1.]), "head.weight": torch.tensor([2.])}
        args = argparse.Namespace(batch_size=256)
        validate_run(result, checkpoint, "e10_b", 42, args, reference)
        checkpoint["model_state"]["market_gru.weight"] = torch.tensor([3.])
        with self.assertRaisesRegex(ValueError, "frozen market"):
            validate_run(result, checkpoint, "e10_b", 42, args, reference)
        checkpoint["epoch"] = 3
        with self.assertRaisesRegex(ValueError, "epoch mismatch"):
            validate_run(result, checkpoint, "e10_b", 42, args, reference)


if __name__ == "__main__":
    unittest.main()
