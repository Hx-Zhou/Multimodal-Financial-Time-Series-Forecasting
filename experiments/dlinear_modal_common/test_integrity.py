"""Behavioral tests for time leakage, missing modalities, and frozen backbones."""
import unittest
import numpy as np
import torch
from .data import consolidate, financial_block, normalize_reports, ordinal, REPORT_DIM
from experiments.e8_dlinear_reports.model import Model as E8
from experiments.e9_dlinear_multimodal.model import Model as E9
from experiments.e10_dlinear_spectf.model import Model as E10


class AlignmentTests(unittest.TestCase):
    def test_end_date_does_not_make_a_report_available(self):
        rows = [{"end_date": "20231231", "value": 2.0}]
        records, audit = consolidate(rows, ("value",), 180)
        dates = np.array([ordinal("20240131"), ordinal("20231231")+179, ordinal("20231231")+180])
        values, valid, ages, present = financial_block(dates, records, ("value",))
        np.testing.assert_array_equal(present, [0, 0, 1])
        np.testing.assert_array_equal(values[:2], 0)
        self.assertEqual(audit["rows_with_announcement_date"], 0)

    def test_publication_date_wins_and_future_reports_cannot_change_past(self):
        rows = [{"end_date": "20230331", "ann_date": "20230425", "value": 2.0}]
        days = np.array([ordinal("20230424"), ordinal("20230425"), ordinal("20230501")])
        original = financial_block(days, consolidate(rows, ("value",), 180)[0], ("value",))
        later = rows + [{"end_date": "20230630", "ann_date": "20230831", "value": 1e12}]
        changed = financial_block(days, consolidate(later, ("value",), 180)[0], ("value",))
        for a, b in zip(original, changed):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(original[-1], [0, 1, 1])

    def test_duplicate_revisions_are_masked_and_yoy_prior_must_be_available(self):
        rows = [{"end_date": "20230331", "ann_date": "20230420", "value": 4.0},
                {"end_date": "20230331", "ann_date": "20230520", "value": 7.0}]
        records, audit = consolidate(rows, ("value",), 180)
        self.assertIsNone(records[0]["values"]["value"])
        self.assertEqual(records[0]["available"], ordinal("20230520"))
        self.assertEqual(audit["conflicting_selected_fields_masked"], 1)
        rows = [{"end_date": "20220331", "ann_date": "20230530", "value": 2e8},
                {"end_date": "20230331", "ann_date": "20230430", "value": 4e8}]
        values, valid, _, _ = financial_block(np.array([ordinal("20230501"), ordinal("20230601")]), consolidate(rows, ("value",), 180)[0], ("value",))
        self.assertFalse(valid[0, 4])
        self.assertTrue(valid[1, 4])
        self.assertAlmostEqual(values[1, 4], 2/3, places=6)

    def test_scaling_ignores_validation_and_masks_missing_values(self):
        values = np.arange(24, dtype=np.float32).reshape(6, 4)
        valid = np.ones_like(values, bool)
        valid[0, 0] = False
        a, stats = normalize_reports(values, valid, np.array([0, 1, 2]))
        values[3:] *= 1e8
        b, changed = normalize_reports(values, valid, np.array([0, 1, 2]))
        self.assertEqual(stats, changed)
        np.testing.assert_array_equal(a[:3], b[:3])
        self.assertEqual(a[0, 0], 0)


class ModalityTests(unittest.TestCase):
    def setUp(self):
        torch.set_flush_denormal(True)
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def batch(self):
        return {"market": torch.randn(3, 96, 7), "anchor": torch.randn(3, 1),
                "news_unique": torch.randn(4, 512), "news_inverse": torch.randint(0, 4, (3, 96)),
                "has_news": torch.ones(3, 96), "news_count": torch.ones(3, 96),
                "reports": torch.randn(3, REPORT_DIM), "has_report": torch.ones(3, 1)}

    def test_all_start_at_market_and_training_keeps_market_frozen(self):
        for cls in (E8, E9, E10):
            with self.subTest(model=cls.__module__):
                model = cls()
                for p in model.market.parameters():
                    torch.nn.init.normal_(p, std=0.01)
                data = self.batch()
                model.eval()
                base = data["anchor"] + model.scale*model.market(data["market"])
                torch.testing.assert_close(model(data), base, rtol=0, atol=0)
                frozen = {k: v.clone() for k, v in model.market.state_dict().items()}
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(parameters, lr=0.001)
                model.train()
                for _ in range(3):
                    optimizer.zero_grad()
                    model(data).square().mean().backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters))
                    optimizer.step()
                for key, value in model.market.state_dict().items():
                    torch.testing.assert_close(value, frozen[key], rtol=0, atol=0)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))

    def test_missing_modalities_exact_fallback_after_nonzero_weights(self):
        for cls in (E8, E9, E10):
            with self.subTest(model=cls.__module__):
                model = cls().eval()
                for p in model.parameters():
                    torch.nn.init.normal_(p, std=0.1)
                data = self.batch()
                data["has_news"].zero_()
                data["news_count"].zero_()
                data["has_report"].zero_()
                expected = data["anchor"] + model.scale*model.market(data["market"])
                torch.testing.assert_close(model(data), expected, rtol=0, atol=0)

    def test_frequency_attention_masks_missing_day_content(self):
        model = E10().eval()
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.05)
        data = self.batch()
        data["news_inverse"][:, :48] = 0
        data["news_inverse"][:, 48:] = 1
        data["has_news"][:, :48] = 0
        a = model(data)
        data["news_unique"][0] = 1e4
        b = model(data)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
