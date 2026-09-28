"""Pure single-joint decoder calibration functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..contracts import DecoderSpec
from ..math.control import decoder_features


@dataclass(frozen=True)
class DecoderFit:
    tau: float
    delay_steps: int
    k_pos: float
    k_neg: float
    desired_mse: float
    actual_mse: float
    desired_mae: float
    actual_mae: float
    n_valid: int
    X: np.ndarray
    y_desired: np.ndarray
    y_actual: np.ndarray
    y_pred: np.ndarray

    def as_legacy_dict(self) -> Dict[str, Any]:
        return {
            "tau": self.tau,
            "delay_steps": self.delay_steps,
            "k_pos": self.k_pos,
            "k_neg": self.k_neg,
            "desired_mse": self.desired_mse,
            "actual_mse": self.actual_mse,
            "desired_mae": self.desired_mae,
            "actual_mae": self.actual_mae,
            "n_valid": self.n_valid,
            "X": self.X,
            "y_desired": self.y_desired,
            "y_actual": self.y_actual,
            "y_pred": self.y_pred,
        }


def fit_signed_direction_gains(
    features: np.ndarray, targets: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Replicate calibration's regularization, clipping, and failure result."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if len(y) < 2:
        return None
    try:
        x_norm = np.max(np.abs(x)) + 1.0e-12
        lambda_reg = 1.0e-4 * x_norm
        xtx = x.T @ x + lambda_reg * np.eye(x.shape[1])
        xty = x.T @ y
        coefficients = np.linalg.solve(xtx, xty)
        k_pos, k_neg = float(coefficients[0]), float(coefficients[1])
        k_pos = np.clip(k_pos, -1.0e-2, 1.0e-2)
        k_neg = np.clip(k_neg, -1.0e-2, 1.0e-2)
        return k_pos, k_neg
    except Exception:
        return None


def fit_decoder_from_trials(
    trials: Sequence[Mapping[str, Any]],
    tau_candidates: Sequence[float] = (0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20),
    delay_candidates: Sequence[int] = (0, 1, 2, 3, 4),
    feature_extractor: Callable[..., Mapping[str, Any]] = decoder_features,
    gain_fitter: Callable[[np.ndarray, np.ndarray], Optional[Tuple[float, float]]] = fit_signed_direction_gains,
) -> Optional[DecoderFit]:
    """Run the current ordered tau/delay grid search.

    Ties retain the first candidate because the legacy comparison is strict
    ``<`` rather than ``<=``.
    """

    best: Optional[DecoderFit] = None
    for tau in tau_candidates:
        for delay_steps in delay_candidates:
            x_rows = []
            y_desired = []
            y_actual = []
            for trial in trials:
                features = feature_extractor(
                    trial["spike_history"],
                    trial["dt"],
                    tau,
                    delay_steps,
                )
                s_pos = features["S_pos"]
                s_neg = features["S_neg"]
                if abs(s_pos) + abs(s_neg) < 1.0e-9:
                    continue
                x_rows.append([s_pos, s_neg])
                y_desired.append(trial["dq_desired"])
                y_actual.append(trial["dq_actual"])
            if len(y_desired) < 2:
                continue
            gains = gain_fitter(x_rows, y_desired)
            if gains is None:
                continue
            k_pos, k_neg = gains
            x_array = np.asarray(x_rows, dtype=float)
            desired_array = np.asarray(y_desired, dtype=float)
            actual_array = np.asarray(y_actual, dtype=float)
            predictions = x_array @ np.array([k_pos, k_neg])
            candidate = DecoderFit(
                tau=tau,
                delay_steps=delay_steps,
                k_pos=k_pos,
                k_neg=k_neg,
                desired_mse=float(np.mean((desired_array - predictions) ** 2)),
                actual_mse=float(np.mean((actual_array - predictions) ** 2)),
                desired_mae=float(np.mean(np.abs(desired_array - predictions))),
                actual_mae=float(np.mean(np.abs(actual_array - predictions))),
                n_valid=len(y_desired),
                X=x_array,
                y_desired=desired_array,
                y_actual=actual_array,
                y_pred=predictions,
            )
            if best is None or candidate.desired_mse < best.desired_mse:
                best = candidate
    return best


def update_decoder_parameters(
    current: DecoderSpec, fit: Optional[DecoderFit], beta: float = 0.2
) -> DecoderSpec:
    if fit is None:
        return current
    return DecoderSpec(
        gain_positive=(1.0 - beta) * current.gain_positive + beta * fit.k_pos,
        gain_negative=(1.0 - beta) * current.gain_negative + beta * fit.k_neg,
        tau=(1.0 - beta) * current.tau + beta * fit.tau,
        delay_steps=int(fit.delay_steps),
        source="calibration_fit",
    )
