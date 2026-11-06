"""Deterministic solvers for the same endpoint-centered DLinear architecture."""

from __future__ import annotations

import time

import torch

from .model import HORIZON, LOOKBACK, build_model


@torch.no_grad()
def design_matrix(x: torch.Tensor) -> torch.Tensor:
    model = build_model({"moving_average": 25, "center_last": True}).double()
    x = x.double()
    trend, remainder = model.decompose(x - x[:, -1:])
    return torch.cat((trend, remainder, torch.ones((len(x), 1), dtype=x.dtype)), dim=1)


def penalty_diagonal() -> torch.Tensor:
    # The two model biases sum to one intercept. Splitting b equally minimizes
    # b_trend^2 + b_remainder^2, giving an effective penalty of 0.5 * b^2.
    diagonal = torch.ones(2 * LOOKBACK + 1, dtype=torch.float64)
    diagonal[-1] = 0.5
    return diagonal


@torch.no_grad()
def sufficient_statistics(features, target, sample_weights):
    weights = sample_weights / sample_weights.sum()
    gram = features.T @ (features * weights[:, None])
    cross = features.T @ (target * weights[:, None])
    return gram, cross


@torch.no_grad()
def ridge_solution(gram, cross, regularization):
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    system = gram + regularization * torch.diag(penalty_diagonal())
    cholesky = torch.linalg.cholesky(system)
    coefficients = torch.cholesky_solve(cross, cholesky)
    residual = float((system @ coefficients - cross).abs().max())
    return coefficients, cholesky, residual


def smooth_objective(features, target, sample_weights, coefficients, regularization, epsilon=1e-3):
    error = features @ coefficients - target
    per_sample = (torch.sqrt(error.square() + epsilon ** 2) - epsilon + 0.2 * error.square()).mean(dim=1)
    fit = (per_sample * sample_weights).sum() / sample_weights.sum()
    penalty = regularization * (penalty_diagonal()[:, None] * coefficients.square()).sum() / HORIZON
    return fit + penalty


def robust_solution(features, target, weights, initial, cholesky, regularization, max_iter=200):
    """Full-batch convex smooth-MAE + .2 MSE, preconditioned by the ridge system."""
    parameter = torch.nn.Parameter(cholesky.T @ initial)
    optimizer = torch.optim.LBFGS([parameter], lr=1.0, max_iter=max_iter, max_eval=max_iter * 2,
                                 tolerance_grad=1e-8, tolerance_change=1e-11,
                                 history_size=15, line_search_fn="strong_wolfe")
    evaluations = []
    started = time.perf_counter()

    def closure():
        optimizer.zero_grad()
        coefficients = torch.linalg.solve_triangular(cholesky.T, parameter, upper=True)
        loss = smooth_objective(features, target, weights, coefficients, regularization)
        if not torch.isfinite(loss):
            raise ValueError("non-finite full-batch objective")
        loss.backward()
        evaluations.append(float(loss.detach()))
        if len(evaluations) == 1 or len(evaluations) % 10 == 0:
            print(f"  full_batch evaluation={len(evaluations)} objective={evaluations[-1]:.8f} "
                  f"seconds={time.perf_counter() - started:.1f}", flush=True)
        return loss

    optimizer.step(closure)
    closure()
    coefficients = torch.linalg.solve_triangular(cholesky.T, parameter.detach(), upper=True)
    details = {"objective_evaluations": evaluations,
               "iterations": int(optimizer.state[parameter]["n_iter"]),
               "preconditioned_gradient_max": float(parameter.grad.abs().max()),
               "gradient_below_1e-6": bool(parameter.grad.abs().max() < 1e-6),
               "seconds": time.perf_counter() - started,
               "epsilon": 1e-3, "maximum_iterations": max_iter}
    return coefficients, details


@torch.no_grad()
def export_model(coefficients):
    """Export into the existing model class, so inference needs no solver."""
    model = build_model({"moving_average": 25, "center_last": True})
    model.trend_head.weight.copy_(coefficients[:LOOKBACK].T.float())
    model.remainder_head.weight.copy_(coefficients[LOOKBACK:2 * LOOKBACK].T.float())
    bias = coefficients[-1].float() * 0.5
    model.trend_head.bias.copy_(bias)
    model.remainder_head.bias.copy_(bias)
    return model
