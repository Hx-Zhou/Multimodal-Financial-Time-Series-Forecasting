import unittest

import torch

from .calibrate import average_models, shrink_model
from .model import build_model


class CalibrationTests(unittest.TestCase):
    def test_weight_operations_match_prediction_operations(self):
        torch.manual_seed(31)
        torch.set_num_threads(2)
        x = torch.randn(12, 96)
        models = [build_model({"center_last": True}) for _ in range(3)]
        for model in models:
            for parameter in model.parameters():
                torch.nn.init.normal_(parameter, std=0.02)
        curve = (torch.arange(1, 25) / 24).square()
        scaled = [shrink_model(model, curve) for model in models]
        for original, result in zip(models, scaled):
            expected = x[:, -1:] + curve * (original({"close": x}) - x[:, -1:])
            torch.testing.assert_close(result({"close": x}), expected)
        expected_average = torch.stack([model({"close": x}) for model in scaled]).mean(dim=0)
        torch.testing.assert_close(average_models(scaled)({"close": x}), expected_average)

    def test_incompatible_models_cannot_be_averaged(self):
        with self.assertRaises(ValueError):
            average_models([build_model({"center_last": False}), build_model({"center_last": True})])


if __name__ == "__main__":
    unittest.main()
