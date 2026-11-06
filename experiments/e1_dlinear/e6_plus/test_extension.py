"""Leakage and missing-modality checks for the E6/E7 extension."""
import unittest
import numpy as np
import torch

from .data import stationary_fields, standardize
from .model import Model


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_training_statistics_ignore_future_rows(self):
        x = np.arange(70, dtype=np.float32).reshape(10, 7)
        mask = np.arange(10) < 6
        y, stats = standardize(x, mask, [4, 5, 6])
        modified = x.copy()
        modified[6:] *= 10000
        altered, alternate_stats = standardize(modified, mask, [4, 5, 6])
        self.assertEqual(stats, alternate_stats)
        np.testing.assert_array_equal(y[:6], altered[:6])

    def test_stationary_features_are_causal(self):
        p = np.ones((20, 7), dtype=np.float32)
        p[:, :4] = np.arange(10, 30)[:, None]
        original = stationary_fields(p, np.float32(10), np.float32(2))
        p[12:] *= 17
        changed = stationary_fields(p, np.float32(10), np.float32(2))
        np.testing.assert_array_equal(original[:12], changed[:12])

    def batch(self, with_news):
        return {"market": torch.randn(3, 96, 7), "anchor": torch.randn(3, 1),
                "news_unique": torch.randn(2, 512),
                "news_inverse": torch.ones((3, 96), dtype=torch.long),
                "has_news": torch.full((3, 96), float(with_news)),
                "news_count": torch.full((3, 96), float(with_news))}

    def test_no_news_exactly_restores_market_despite_biases(self):
        model = Model(news=True)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.1)
        batch = self.batch(False)
        expected = batch["anchor"] + model.scale * model.market(batch["market"])
        torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)

    def test_news_training_leaves_market_frozen_and_initially_equal(self):
        model = Model(news=True)
        for p in model.market.parameters():
            torch.nn.init.normal_(p, std=0.01)
        model.freeze_market()
        batch = self.batch(True)
        expected = batch["anchor"] + model.scale * model.market(batch["market"])
        torch.testing.assert_close(model(batch), expected, rtol=0, atol=0)
        frozen = {k: v.clone() for k, v in model.market.state_dict().items()}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
        for _ in range(2):
            optimizer.zero_grad()
            model(batch).square().mean().backward()
            optimizer.step()
        for k, v in model.market.state_dict().items():
            torch.testing.assert_close(v, frozen[k], rtol=0, atol=0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.news_mlp.parameters()))


if __name__ == "__main__":
    unittest.main()
