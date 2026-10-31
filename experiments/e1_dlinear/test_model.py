"""Small numerical tests for scale, residual output, learning and checkpoint loading."""

import io
import unittest

import numpy as np
import torch

from .model import build_model, compute_loss, prepare_batch


class DLinearTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_normalization_and_unlabeled_inference(self):
        numerical = np.zeros((96, 7), dtype=np.float32)
        numerical[:, 3] = np.linspace(10, 20, 96)
        sample = {"numerical": numerical, "close_mean": 15, "close_std": 5}
        batch = prepare_batch([sample])
        self.assertNotIn("y", batch)
        torch.testing.assert_close(batch["close"][0], torch.linspace(-1, 1, 96))
        model = build_model()
        prediction = model(batch)
        self.assertEqual(prediction.shape, (1, 24))
        torch.testing.assert_close(prediction, torch.ones(1, 24))
        labeled = prepare_batch([{**sample, "y": np.arange(24, dtype=np.float32)}])
        torch.testing.assert_close(model(labeled), prediction)

    def test_decomposition_and_padding(self):
        for kernel in (25, 15, 5):
            model = build_model({"moving_average": kernel})
            x = torch.randn(4, 96)
            trend, remainder = model.decompose(x)
            torch.testing.assert_close(trend + remainder, x)
            trend, remainder = model.decompose(torch.full((2, 96), 3.0))
            torch.testing.assert_close(trend, torch.full((2, 96), 3.0))
            torch.testing.assert_close(remainder, torch.zeros(2, 96))
        with self.assertRaises(ValueError):
            build_model({"moving_average": 24})

    def test_learns_residual_and_roundtrips_checkpoint(self):
        model = build_model()
        x = torch.randn(64, 96)
        y = x[:, -1:] + 0.5 * x[:, -2:-1] + torch.linspace(0.1, 0.3, 24)
        initial = compute_loss(model({"close": x}), y).item()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            loss = compute_loss(model({"close": x}), y)
            loss.backward()
            optimizer.step()
        self.assertLess(compute_loss(model({"close": x}), y).item(), initial * 0.2)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = build_model()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(restored({"close": x}), model({"close": x}))

    def test_centered_variant_shift_equivariance_and_saved_config(self):
        config = {"moving_average": 25, "center_last": True}
        model = build_model(config)
        x = torch.randn(5, 96)
        # Nonzero heads make this check sensitive to incorrect centering.
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, std=0.03)
        shifts = torch.tensor([-3.0, 0.5, 1.0, 2.0, 7.0])[:, None]
        prediction = model({"close": x})
        torch.testing.assert_close(model({"close": x + shifts}), prediction + shifts)
        stream = io.BytesIO()
        torch.save({"config": config, "state_dict": model.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = build_model(saved["config"])
        restored.load_state_dict(saved["state_dict"])
        self.assertTrue(restored.center_last)
        torch.testing.assert_close(restored({"close": x}), prediction)

    def test_old_config_still_means_uncentered_model(self):
        original = build_model({"moving_average": 25})
        explicit = build_model({"moving_average": 25, "center_last": False})
        for parameter in original.parameters():
            torch.nn.init.normal_(parameter, std=0.02)
        explicit.load_state_dict(original.state_dict())
        x = torch.randn(3, 96)
        torch.testing.assert_close(original({"close": x}), explicit({"close": x}))


if __name__ == "__main__":
    unittest.main()
