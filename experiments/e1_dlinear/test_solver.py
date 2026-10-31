"""Numerical checks for deterministic fitting and exporting DLinear weights."""

import unittest

import torch

from .linear_solver import design_matrix, export_model, penalty_diagonal, ridge_solution, sufficient_statistics, smooth_objective, robust_solution


class LinearSolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_export_matches_feature_space(self):
        x = torch.randn(32, 96)
        features = design_matrix(x)
        coefficients = torch.randn(193, 24, dtype=torch.float64) * 0.02
        expected = x[:, -1:].double() + features @ coefficients
        model = export_model(coefficients)
        torch.testing.assert_close(model({"close": x}).double(), expected, atol=2e-6, rtol=2e-6)
        actual_norm = sum(p.double().square().sum() for p in model.parameters())
        expected_norm = (penalty_diagonal()[:, None] * coefficients.square()).sum()
        torch.testing.assert_close(actual_norm, expected_norm, atol=1e-6, rtol=1e-6)

    def test_ridge_is_stationary_for_weighted_objective(self):
        features = design_matrix(torch.randn(300, 96))
        target = torch.randn(300, 24, dtype=torch.float64)
        weights = torch.linspace(0.1, 1.0, 300, dtype=torch.float64)
        gram, cross = sufficient_statistics(features, target, weights)
        coefficients, _, residual = ridge_solution(gram, cross, 0.1)
        self.assertLess(residual, 1e-10)
        coefficients.requires_grad_()
        error = features @ coefficients - target
        objective = (weights * error.square().mean(dim=1)).sum() / weights.sum()
        objective += 0.1 * (penalty_diagonal()[:, None] * coefficients.square()).sum() / 24
        objective.backward()
        self.assertLess(float(coefficients.grad.abs().max()), 1e-10)

    def test_robust_refinement_reduces_training_objective(self):
        features = design_matrix(torch.randn(200, 96))
        target = torch.randn(200, 24, dtype=torch.float64) * 0.1
        target[::20] += 4.0
        weights = torch.ones(200, dtype=torch.float64)
        gram, cross = sufficient_statistics(features, target, weights)
        initial, cholesky, _ = ridge_solution(gram, cross, 0.1)
        before = float(smooth_objective(features, target, weights, initial, 0.1))
        result, details = robust_solution(features, target, weights, initial, cholesky, 0.1, max_iter=200)
        after = float(smooth_objective(features, target, weights, result, 0.1))
        self.assertLess(after, before)
        self.assertLess(details["preconditioned_gradient_max"], 1e-4)


if __name__ == "__main__":
    unittest.main()
