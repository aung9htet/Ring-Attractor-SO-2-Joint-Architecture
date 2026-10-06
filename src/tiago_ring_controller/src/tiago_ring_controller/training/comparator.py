"""Comparator (Homeostasis) weights for a bounded ring (pure NumPy).

The comparator turns the ring's Fourier readout into a scalar: warm (state
ring) and cold (goal ring) are both ``sum_k w_k * (sin_pos_k - sin_neg_k)`` of
their ring, and the decision pair moves the state bump toward the goal while
warm and cold differ.  The legacy weights were ridge-fitted to the sawtooth
``phase / 2pi``, which jumps at ring index 0.  Twenty sine harmonics cannot
follow a jump: the fitted ramp rings, non-monotone across the whole ring and
badly so within about 30 indices of the seam, where the legacy mapping put the
joint limits (docs/archive/velocity/bounded-ring.md).

:func:`bounded_ramp` is the target for a bounded ring instead: it rises
linearly across an arc centred on index ``N / 2`` (the joint range plus a
margin) and falls back across the sector around the seam.  It is continuous
and odd about the seam, which a sine-only feature set requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from ..features import embed_homeostasis_weights
from .ridge import fit_normalized_ridge

#: the legacy trainer counted features over a 50 ms injection plus 300 ms settle
LEGACY_FEATURE_WINDOW_MS = 350.0


def sawtooth_ramp(theta: Any) -> np.ndarray:
    """The legacy target: ``phase / 2pi``."""

    return np.mod(np.asarray(theta, dtype=float), 2.0 * np.pi) / (2.0 * np.pi)


def bounded_ramp(theta: Any, rise_fraction: float) -> np.ndarray:
    """0 to 1 across the arc of ``rise_fraction`` of the ring centred on pi, back across the seam sector.

    Piecewise linear, continuous, and ``ramp(-theta) = 1 - ramp(theta)``
    (odd about the seam after removing the mean 0.5).
    """

    if not 0.0 < rise_fraction < 1.0:
        raise ValueError("rise_fraction must be in (0, 1)")
    half = rise_fraction * np.pi
    u = np.mod(np.asarray(theta, dtype=float), 2.0 * np.pi) - np.pi        # angle from the arc centre, in [-pi, pi)
    rising = u / half
    falling = np.sign(u) * (np.pi - np.abs(u)) / (np.pi - half)
    return 0.5 + 0.5 * np.where(np.abs(u) <= half, rising, falling)


def signed_features(counts: Any) -> np.ndarray:
    """``(n, 2K)`` push-pull feature counts -> ``(n, K)`` signed ``sin_pos - sin_neg``."""

    values = np.atleast_2d(np.asarray(counts, dtype=float))
    return values[:, 0::2] - values[:, 1::2]


@dataclass(frozen=True)
class ComparatorFit:
    signed_weights: np.ndarray
    bias: float
    train_r2: float
    #: held-out rms and max |error| of the fitted ramp against the target
    test_rms: float
    test_max_error: float
    target: str

    def artifact(self) -> np.ndarray:
        """The ``(4K, 2)`` matrix ``Homeostasis`` loads."""

        return embed_homeostasis_weights(self.signed_weights)


def fit_comparator(
    feature_counts: Any,
    indices: Sequence[float],
    ring_size: int,
    target: Callable[[np.ndarray], np.ndarray],
    target_name: str,
    window_ms: float,
    ridge_lambda: float = 1e-6,
    reference_window_ms: float = LEGACY_FEATURE_WINDOW_MS,
) -> ComparatorFit:
    """Ridge-fit the signed weights on every other position, test on the rest.

    Feature counts are rescaled from ``window_ms`` to ``reference_window_ms``
    so the weights have the legacy artifact's scale and ``weight_scale`` keeps
    its meaning.
    """

    x = signed_features(feature_counts) * (reference_window_ms / float(window_ms))
    positions = np.asarray(indices, dtype=float)
    y = target(2.0 * np.pi * positions / ring_size)
    order = np.argsort(positions)
    train = np.zeros(len(positions), dtype=bool)
    train[order[0::2]] = True
    fit = fit_normalized_ridge(x[train], y[train], ridge_lambda)
    error = (x @ fit.weights + fit.bias) - y
    return ComparatorFit(
        signed_weights=np.asarray(fit.weights, dtype=float), bias=float(fit.bias), train_r2=float(fit.r2),
        test_rms=float(np.sqrt(np.mean(error[~train] ** 2))), test_max_error=float(np.max(np.abs(error[~train]))),
        target=target_name,
    )


def ramp(fit: ComparatorFit, feature_counts: Any, window_ms: float,
         reference_window_ms: float = LEGACY_FEATURE_WINDOW_MS) -> np.ndarray:
    """The fitted comparator output (before the NEST weight scale) for each feature row."""

    x = signed_features(feature_counts) * (reference_window_ms / float(window_ms))
    return x @ fit.signed_weights + fit.bias


def monotonicity(values: Any, indices: Sequence[float], low: float, high: float) -> Dict[str, float]:
    """How steadily ``values`` rise with the ring index between ``low`` and ``high``."""

    positions = np.asarray(indices, dtype=float)
    inside = (positions >= low) & (positions <= high)
    order = np.argsort(positions[inside])
    steps = np.diff(np.asarray(values, dtype=float)[inside][order])
    if len(steps) == 0:
        return {"steps": 0, "wrong_way": 0, "smallest_step": float("nan"), "mean_step": float("nan")}
    return {"steps": int(len(steps)), "wrong_way": int(np.sum(steps <= 0.0)),
            "smallest_step": float(steps.min()), "mean_step": float(steps.mean())}


def decision_summary(rows: Sequence[Mapping[str, float]], low: float, high: float, min_distance: float = 10.0) -> Dict[str, Any]:
    """Wrong-sign decisions and dead-zone widths of a frozen-transport decision map.

    ``rows`` hold ``state``, ``goal`` (ring indices) and ``drive`` (right - left
    gain spikes per tick).  Only pairs with both bumps on ``[low, high]`` count.
    The dead zone of a state is the largest distance to a goal that still gets
    no drive.
    """

    pairs = [r for r in rows if low <= r["state"] <= high and low <= r["goal"] <= high]
    wrong = [r for r in pairs if abs(r["goal"] - r["state"]) >= min_distance and np.sign(r["drive"]) == -np.sign(r["goal"] - r["state"]) and r["drive"] != 0]
    silent = [r for r in pairs if abs(r["goal"] - r["state"]) >= min_distance and abs(r["drive"]) < 1.0]
    widths = {}
    for state in sorted({r["state"] for r in pairs}):
        dead = [abs(r["goal"] - r["state"]) for r in pairs if r["state"] == state and abs(r["drive"]) < 1.0]
        widths[state] = max(dead) if dead else 0.0
    return {
        "pairs": len(pairs), "wrong_sign": len(wrong), "silent_beyond_min_distance": len(silent),
        "dead_zone_median": float(np.median(list(widths.values()))) if widths else float("nan"),
        "dead_zone_max": float(max(widths.values())) if widths else float("nan"),
        "dead_zone_by_state": widths,
    }


__all__ = [
    "ComparatorFit", "LEGACY_FEATURE_WINDOW_MS", "bounded_ramp", "decision_summary", "fit_comparator",
    "monotonicity", "ramp", "sawtooth_ramp", "signed_features",
]
