"""NumPy ridge solvers matching the current research scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fit_ridge(features: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Use the current primal/dual branch with no hidden normalization."""

    x = np.asarray(features)
    y = np.asarray(targets)
    n_samples, n_features = x.shape
    if n_features <= n_samples:
        return np.linalg.solve(
            x.T @ x + ridge_lambda * np.eye(n_features),
            x.T @ y,
        )
    return x.T @ np.linalg.solve(
        x @ x.T + ridge_lambda * np.eye(n_samples),
        y,
    )


@dataclass(frozen=True)
class NormalizedRidgeFit:
    weights: np.ndarray
    bias: float
    predictions: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    r2: float
    rmse: float
    theta_normalized: np.ndarray


def fit_normalized_ridge(
    features: np.ndarray, targets: np.ndarray, ridge_lambda: float = 1.0e-6
) -> NormalizedRidgeFit:
    """Replicate the homeostasis trainer, including bias regularization."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    feature_mean = np.mean(x, axis=0)
    feature_std = np.std(x, axis=0)
    feature_std[feature_std == 0.0] = 1.0
    x_norm = (x - feature_mean) / feature_std
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1))], axis=1)
    regularizer = ridge_lambda * np.eye(x_aug.shape[1])
    theta = np.linalg.solve(x_aug.T @ x_aug + regularizer, x_aug.T @ y)
    weights = theta[:-1] / feature_std
    bias = float(theta[-1] - (feature_mean / feature_std) @ theta[:-1])
    predictions = x @ weights + bias
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))
    return NormalizedRidgeFit(
        weights=weights,
        bias=bias,
        predictions=predictions,
        feature_mean=feature_mean,
        feature_std=feature_std,
        r2=r2,
        rmse=rmse,
        theta_normalized=theta,
    )
