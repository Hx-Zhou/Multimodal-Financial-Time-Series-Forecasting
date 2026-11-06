"""Check causal boundaries, duplicate handling, controls and residual gating."""

import unittest
from types import SimpleNamespace

import numpy as np

from experiments.gru_e13.model import RidgeResidual
from experiments.gru_e13.news import AlignedNewsIndex, fit_pca, shuffled_projection


def fixture():
    calendar = np.arange(np.datetime64("2023-07-01"), np.datetime64("2024-01-11"))
    days = np.asarray([int(str(day).replace("-", "")) for day in calendar[np.is_busday(calendar)][-100:]])
    end_dates = [20240105, 20240108, 20240109]
    starts = np.asarray([int(np.flatnonzero(days == day)[0]) - 95 for day in end_dates])
    return SimpleNamespace(
        stock_codes=["A", "B"], price_days={"A": days, "B": days},
        stock_index=np.zeros(3, dtype=int), input_start=starts,
        news_by_stock={
            "A": (np.asarray([20240105120000, 20240106100000, 20240106110000,
                              20240107160000, 20240108235959, 20240109000000, 20240112120000]),
                  np.asarray([0, 0, 1, 2, 3, 4, 5])),
            "B": (np.asarray([20240105120000]), np.asarray([0])),
        },
    )


class NewsTests(unittest.TestCase):
    def test_weekend_duplicates_and_future_boundary(self):
        dataset = fixture()
        index = AlignedNewsIndex(dataset)
        events = index.by_stock["A"]
        self.assertEqual(events.ids.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(events.times[0], 20240105120000)
        self.assertEqual(index.by_stock["B"].ids.tolist(), [0])
        self.assertEqual(index.audit["duplicate_records_removed"], 1)
        self.assertEqual(index.audit["assigned_to_later_calendar_date"], 2)
        projected = np.arange(6, dtype=np.float32)[:, None]
        content, meta, _ = index.features(dataset, [0, 1, 2], projected, windows=[1])
        np.testing.assert_allclose(np.expm1(meta[:, 0]), [1, 3, 1], rtol=1e-6)
        np.testing.assert_allclose(content[:, 0], [0, 2, 4])
        projected[4] = 99999  # Tuesday's news cannot affect Monday.
        changed, _, _ = index.features(dataset, [1], projected, windows=[1])
        np.testing.assert_array_equal(changed, content[1:2])

    def test_close_cutoff_includes_exact_boundary(self):
        dataset = fixture()
        dataset.news_by_stock["A"] = (
            np.asarray([20240105150000, 20240105150001, 20240108150000]), np.asarray([0, 1, 2]))
        index = AlignedNewsIndex(dataset, 150000)
        days = dataset.price_days["A"]
        np.testing.assert_array_equal(days[index.by_stock["A"].bins], [20240105, 20240108, 20240108])
        with self.assertRaises(ValueError):
            AlignedNewsIndex(dataset, 236000)

    def test_pca_uses_only_training_available_articles(self):
        dataset = fixture()
        index = AlignedNewsIndex(dataset)
        ids = index.training_vector_ids(dataset, [0, 1], lookback=1)
        np.testing.assert_array_equal(ids, [0, 1, 2, 3])
        vectors = np.arange(12, dtype=np.float32).reshape(6, 2)
        mean, basis, _ = fit_pca(vectors, ids, components=1)
        vectors[4:] = -1000000
        other_mean, other_basis, _ = fit_pca(vectors, ids, components=1)
        np.testing.assert_array_equal(mean, other_mean)
        np.testing.assert_array_equal(basis, other_basis)

    def test_shuffled_bank_contains_training_content_only(self):
        projected = np.arange(30, dtype=np.float32).reshape(10, 3)
        ids = np.asarray([0, 1, 2, 3])
        shuffled = shuffled_projection(projected, ids, seed=42)
        np.testing.assert_array_equal(np.sort(shuffled[ids, 0]), projected[ids, 0])
        self.assertTrue(np.isin(shuffled[:, 0], projected[ids, 0]).all())
        np.testing.assert_array_equal(shuffled, shuffled_projection(projected, ids, seed=42))

    def test_ridge_gating_and_reload(self):
        x = np.asarray([[1.], [2.], [3.], [0.]], dtype=np.float32)
        active = np.asarray([True, True, True, False])
        residual = np.asarray([[2., 3.], [4., 6.], [6., 9.], [99., 99.]], dtype=np.float32)
        model = RidgeResidual.fit_prepared(RidgeResidual.prepare(x, active, residual), 1e-8)
        prediction = model.predict(x, active)
        np.testing.assert_allclose(prediction[:3], residual[:3], atol=1e-5)
        np.testing.assert_array_equal(prediction[-1], [0, 0])
        other = RidgeResidual.from_state_dict(model.state_dict())
        np.testing.assert_array_equal(other.predict(x, active), prediction)


if __name__ == "__main__":
    unittest.main()
