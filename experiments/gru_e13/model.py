"""A regularized linear residual model, zero on samples without recent news."""

from __future__ import annotations

import numpy as np


class RidgeResidual:
    """For each horizon, minimize sample-mean squared error + alpha * ||w||².

    Use a masked intercept so that samples with no news in any configured
    window always receive exactly zero correction. All statistics are fit on
    training samples only. This diagnostic intentionally uses ridge's closed
    form rather than the earlier GRUs' MAE + 0.2 MSE training objective.
    """

    def __init__(self, mean, std, weights):
        self.mean, self.std, self.weights = mean, std, weights

    @staticmethod
    def design(x, active, mean, std):
        z = (np.asarray(x, dtype=np.float64) - mean) / std
        z[~active] = 0
        return np.column_stack((z, active.astype(np.float64)))

    @classmethod
    def prepare(cls, x, active, residual):
        if not np.any(active):
            raise ValueError("no training samples have recent news")
        mean = x[active].mean(axis=0, dtype=np.float64)
        std = np.maximum(x[active].std(axis=0, dtype=np.float64), 1e-6)
        # Normalize by all samples: this is the full training MSE objective,
        # including samples whose correction is constrained to zero.
        gram = np.zeros((x.shape[1] + 1, x.shape[1] + 1), np.float64)
        rhs = np.zeros((x.shape[1] + 1, residual.shape[1]), np.float64)
        for start in range(0, len(x), 8192):
            z = cls.design(x[start:start + 8192], active[start:start + 8192], mean, std)
            gram += z.T @ z
            rhs += z.T @ np.asarray(residual[start:start + 8192], dtype=np.float64)
        return mean, std, gram / len(x), rhs / len(x)

    @classmethod
    def fit_prepared(cls, prepared, alpha):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("ridge alpha must be finite and positive")
        mean, std, gram, rhs = prepared
        weights = np.linalg.solve(gram + alpha * np.eye(len(gram)), rhs)
        return cls(mean, std, weights)

    def predict(self, x, active):
        return (self.design(x, active, self.mean, self.std) @ self.weights).astype(np.float32)

    def state_dict(self):
        return {key: getattr(self, key).tolist() for key in ("mean", "std", "weights")}

    @classmethod
    def from_state_dict(cls, state):
        return cls(*(np.asarray(state[key], np.float64) for key in ("mean", "std", "weights")))
