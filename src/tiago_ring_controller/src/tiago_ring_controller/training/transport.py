"""Transport calibration: how far the state bump moves per gain spike (pure NumPy).

The decoder turns the signed gain count of one tick, ``s = right - left``, into
a joint velocity.  The joint follows the state bump when it moves ``rho`` times
as far as the bump, ``rho`` being the joint angle per ring index of the encoder
mapping.  The bump's own displacement per tick is modelled as

    a * dz(s),    dz(s) = sign(s) * max(|s| - theta, 0)

``a`` (ring indices per spike above the dead zone) and ``theta`` (spikes per
tick) are properties of the network, not of a joint or of the robot: in
``once`` mode the gain spike train does not depend on the robot at all.  They
are therefore fitted on NEST trials alone, and the decoder gain of a joint is
``a * rho / dt`` (:func:`decoder_gains`).  ``theta = 0`` is the linear decoder.

Method and measurements: docs/archive/velocity/calibration.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..math.circular import decode_circular_centroid
from ..math.control import dead_zone

#: trial verdicts, in the order they are tested
STATUSES = ("silent", "no_motion", "edge", "stuck", "ok")


@dataclass(frozen=True)
class TransportTrial:
    """One trial as the calibration sees it: gain counts and state-bump position per NEST step."""

    left: np.ndarray
    right: np.ndarray
    #: unwrapped circular centroid of the state ring in ring indices (NaN where the ring was silent)
    centroid: np.ndarray
    ring_size: int
    start_index: Optional[int] = None
    goal_index: Optional[int] = None
    #: cross-validation group (the RNG seed by default)
    group: Any = None
    label: str = ""

    @property
    def signed(self) -> np.ndarray:
        return np.asarray(self.right, dtype=float) - np.asarray(self.left, dtype=float)


@dataclass(frozen=True)
class TransportOptions:
    #: first NEST steps left out: the bump injection makes both gain populations fire
    skip_steps: int = 3
    #: steps averaged for the bump position at the start and at the end of a trial
    window: int = 4
    #: a bump that moved less than this many indices did not move ("no_motion")
    min_displacement: float = 5.0
    #: a bump that came this close to either end of the ring left the interior ("edge")
    edge_margin: float = 15.0
    #: "stuck": a window of this many steps in which the fitted model expects at least
    #: ``stuck_expected`` indices of motion and the bump did less than ``stuck_fraction`` of it
    stuck_window: int = 20
    stuck_expected: float = 6.0
    stuck_fraction: float = 0.25
    #: dead zones tried, in spikes per tick
    thetas: Tuple[float, ...] = tuple(float(v) for v in range(0, 41, 2))
    #: the dead zone is kept only if its held-out p90 error is below this fraction of the linear model's
    selection_margin: float = 0.9
    #: folds when the trials have fewer than two groups
    folds: int = 4


@dataclass(frozen=True)
class ErrorSummary:
    """Error of the predicted bump displacement over a set of trials, relative to the true one."""

    n: int
    median_abs_rel: float
    p90_abs_rel: float
    max_abs_rel: float
    #: mean over-travel (+) or under-travel (-) of the prediction, by direction of the bump
    bias_left: float
    bias_right: float
    median_abs_index: float
    p90_abs_index: float

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TransportFit:
    #: "dead_zone" or "linear": the model the decoder should use
    model: str
    #: ring indices per spike (above the dead zone) of the chosen model
    index_per_spike: float
    #: spikes per tick; 0.0 for the linear model
    dead_zone: float
    #: both candidates fitted on all usable trials: {"linear": (a, 0.0), "dead_zone": (a, theta)}
    candidates: Dict[str, Tuple[float, float]]
    #: held-out error of each candidate
    cv: Dict[str, ErrorSummary]
    #: parameters chosen in each fold of the dead-zone model: (held-out group, theta, a)
    folds: List[Tuple[Any, float, float]]
    #: trials per verdict; only "ok" trials are fitted
    counts: Dict[str, int]
    statuses: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model, "index_per_spike": self.index_per_spike, "dead_zone": self.dead_zone,
            "candidates": {name: {"index_per_spike": a, "dead_zone": theta} for name, (a, theta) in self.candidates.items()},
            "cv": {name: summary.as_dict() for name, summary in self.cv.items()},
            "folds": [{"held_out": str(group), "dead_zone": theta, "index_per_spike": a} for group, theta, a in self.folds],
            "counts": dict(self.counts),
        }


# -- bump position ------------------------------------------------------------------
def unwrapped_centroid(counts: Any) -> np.ndarray:
    """Circular centroid of a ring per tick, in ring indices, continuous across the seam.

    ``counts`` has one row per tick and one column per ring neuron.  Ticks
    without spikes are NaN.  A bump that crosses the seam between the last and
    the first neuron keeps counting (index 200, 201, ... or -1, -2, ...), so
    differences of this trace are true displacements.
    """

    rates = np.asarray(counts, dtype=float)
    if rates.ndim != 2:
        raise ValueError("counts must be (ticks, neurons), got shape %r" % (rates.shape,))
    size = rates.shape[1]
    centroid, _ = decode_circular_centroid(rates.T)
    trace = np.full(len(centroid), np.nan)
    good = np.isfinite(centroid)
    if good.any():
        trace[good] = np.unwrap(centroid[good] * (2.0 * np.pi / size)) * (size / (2.0 * np.pi))
    return trace


def trial_displacement(trial: TransportTrial, options: TransportOptions = TransportOptions()) -> float:
    """Bump displacement over the trial in ring indices (end window minus start window); NaN if silent."""

    centroid = np.asarray(trial.centroid, dtype=float)
    start = centroid[options.skip_steps:options.skip_steps + options.window]
    end = centroid[-options.window:]
    if len(centroid) < options.skip_steps + 2 * options.window:
        return float("nan")
    if not np.isfinite(start).any() or not np.isfinite(end).any():
        return float("nan")
    return float(np.nanmean(end) - np.nanmean(start))


def trial_drive(trial: TransportTrial, theta: float, options: TransportOptions = TransportOptions()) -> float:
    """Sum of ``dz(right - left)`` between the middles of the start and end windows."""

    half = options.window // 2
    signed = trial.signed[options.skip_steps + half:len(trial.signed) - half]
    return float(np.sum(dead_zone(signed, theta)))


def _windows(trial: TransportTrial, options: TransportOptions) -> List[Tuple[float, np.ndarray]]:
    """``(bump displacement, signed counts)`` of consecutive ``stuck_window``-step stretches."""

    centroid = np.asarray(trial.centroid, dtype=float)
    signed = trial.signed
    half = options.window // 2
    rows = []
    start = options.skip_steps
    while start + options.stuck_window + options.window <= len(centroid):
        first = centroid[start:start + options.window]
        last = centroid[start + options.stuck_window:start + options.stuck_window + options.window]
        if np.isfinite(first).any() and np.isfinite(last).any():
            rows.append((
                float(np.nanmean(last) - np.nanmean(first)),
                signed[start + half:start + options.stuck_window + half],
            ))
        start += options.stuck_window
    return rows


def has_stuck_window(trial: TransportTrial, gain: float, theta: float, options: TransportOptions = TransportOptions()) -> bool:
    """True when the gain populations fired for a stretch and the bump did not follow."""

    for moved, signed in _windows(trial, options):
        expected = gain * float(np.sum(dead_zone(signed, theta)))
        if abs(expected) >= options.stuck_expected and moved * np.sign(expected) < options.stuck_fraction * abs(expected):
            return True
    return False


def classify_trial(
    trial: TransportTrial,
    options: TransportOptions = TransportOptions(),
    gain: Optional[float] = None,
    theta: float = 0.0,
) -> str:
    """One of :data:`STATUSES`; ``stuck`` is only tested when a fitted ``gain`` is given."""

    displacement = trial_displacement(trial, options)
    if not np.isfinite(displacement):
        return "silent"
    if abs(displacement) < options.min_displacement:
        return "no_motion"
    centroid = np.asarray(trial.centroid, dtype=float)[options.skip_steps:]
    centroid = centroid[np.isfinite(centroid)]
    if centroid.min() < options.edge_margin or centroid.max() > trial.ring_size - 1 - options.edge_margin:
        return "edge"
    if gain is not None and has_stuck_window(trial, gain, theta, options):
        return "stuck"
    return "ok"


# -- fit ----------------------------------------------------------------------------
def fit_gain(trials: Sequence[TransportTrial], theta: float, options: TransportOptions = TransportOptions()) -> float:
    """Least squares through the origin of displacement on drive: ring indices per spike."""

    drive = np.array([trial_drive(t, theta, options) for t in trials])
    moved = np.array([trial_displacement(t, options) for t in trials])
    denominator = float(np.sum(drive * drive))
    if denominator <= 0.0:
        raise ValueError("no gain spikes above a dead zone of %g in %d trials" % (theta, len(trials)))
    return float(np.sum(drive * moved) / denominator)


def prediction_errors(
    trials: Sequence[TransportTrial], gain: float, theta: float, options: TransportOptions = TransportOptions()
) -> Tuple[np.ndarray, np.ndarray]:
    """``(predicted - true, true)`` bump displacement per trial, in ring indices."""

    moved = np.array([trial_displacement(t, options) for t in trials])
    predicted = gain * np.array([trial_drive(t, theta, options) for t in trials])
    return predicted - moved, moved


def summarize_errors(errors: Any, moved: Any) -> ErrorSummary:
    errors = np.asarray(errors, dtype=float)
    moved = np.asarray(moved, dtype=float)
    relative = errors / np.abs(moved)
    aligned = relative * np.sign(moved)       # + : the prediction travels further than the bump
    left, right = aligned[moved < 0], aligned[moved > 0]
    return ErrorSummary(
        n=int(len(errors)),
        median_abs_rel=float(np.median(np.abs(relative))),
        p90_abs_rel=float(np.percentile(np.abs(relative), 90)),
        max_abs_rel=float(np.max(np.abs(relative))),
        bias_left=float(np.mean(left)) if len(left) else float("nan"),
        bias_right=float(np.mean(right)) if len(right) else float("nan"),
        median_abs_index=float(np.median(np.abs(errors))),
        p90_abs_index=float(np.percentile(np.abs(errors), 90)),
    )


def select_theta(trials: Sequence[TransportTrial], options: TransportOptions = TransportOptions()) -> Tuple[float, float]:
    """``(theta, a)`` with the smallest RMS relative displacement error over ``options.thetas``."""

    best: Optional[Tuple[float, float, float]] = None
    for theta in options.thetas:
        try:
            gain = fit_gain(trials, theta, options)
        except ValueError:
            continue
        errors, moved = prediction_errors(trials, gain, theta, options)
        score = float(np.sqrt(np.mean((errors / np.abs(moved)) ** 2)))
        if best is None or score < best[0]:
            best = (score, float(theta), gain)
    if best is None:
        raise ValueError("no dead zone in %r leaves any drive" % (options.thetas,))
    return best[1], best[2]


def _folds(trials: Sequence[TransportTrial], options: TransportOptions) -> List[Tuple[Any, List[int]]]:
    groups: Dict[Any, List[int]] = {}
    for index, trial in enumerate(trials):
        groups.setdefault(trial.group, []).append(index)
    if len(groups) >= 2:
        return sorted(groups.items(), key=lambda item: str(item[0]))
    count = max(2, min(options.folds, len(trials)))
    return [("fold %d" % k, [i for i in range(len(trials)) if i % count == k]) for k in range(count)]


def cross_validate(
    trials: Sequence[TransportTrial],
    options: TransportOptions = TransportOptions(),
    theta: Optional[float] = None,
) -> Tuple[ErrorSummary, List[Tuple[Any, float, float]]]:
    """Leave one group (seed) out.  ``theta`` fixed (0.0 = linear) or chosen inside each fold."""

    errors: List[float] = []
    moved: List[float] = []
    chosen: List[Tuple[Any, float, float]] = []
    for group, held in _folds(trials, options):
        held_set = set(held)
        train = [t for i, t in enumerate(trials) if i not in held_set]
        test = [trials[i] for i in held]
        if not train or not test:
            continue
        if theta is None:
            fold_theta, gain = select_theta(train, options)
        else:
            fold_theta, gain = float(theta), fit_gain(train, theta, options)
        fold_errors, fold_moved = prediction_errors(test, gain, fold_theta, options)
        errors.extend(fold_errors)
        moved.extend(fold_moved)
        chosen.append((group, fold_theta, gain))
    return summarize_errors(errors, moved), chosen


def fit_transport(trials: Sequence[TransportTrial], options: TransportOptions = TransportOptions()) -> TransportFit:
    """Fit the linear and the dead-zone model, cross-validate both and choose.

    Trials that never moved, left the ring interior, or fired without moving
    the bump are counted and left out: the decoder cannot represent them, and
    they are defects of the network, not of the gain.
    """

    statuses = [classify_trial(t, options) for t in trials]
    usable = [t for t, status in zip(trials, statuses) if status == "ok"]
    if len(usable) < 4:
        raise ValueError("only %d usable trials (verdicts: %s); at least 4 are needed" % (len(usable), _count(statuses)))
    # First pass for the "stuck" test: the median displacement per drive, which one trial with
    # hundreds of ticks of drive and no motion cannot pull down the way least squares would be.
    theta, _ = select_theta(usable, options)
    ratios = [trial_displacement(t, options) / trial_drive(t, theta, options)
              for t in usable if trial_drive(t, theta, options) != 0.0]
    gain = float(np.median(ratios)) if ratios else 0.0
    statuses = [classify_trial(t, options, gain, theta) for t in trials]
    usable = [t for t, status in zip(trials, statuses) if status == "ok"]
    if len(usable) < 4:
        raise ValueError("only %d usable trials (verdicts: %s); at least 4 are needed" % (len(usable), _count(statuses)))

    linear_cv, _ = cross_validate(usable, options, theta=0.0)
    dead_cv, folds = cross_validate(usable, options, theta=None)
    theta, gain = select_theta(usable, options)
    candidates = {"linear": (fit_gain(usable, 0.0, options), 0.0), "dead_zone": (gain, theta)}
    model = "dead_zone" if (theta > 0.0 and dead_cv.p90_abs_rel <= options.selection_margin * linear_cv.p90_abs_rel) else "linear"
    return TransportFit(
        model=model, index_per_spike=candidates[model][0], dead_zone=candidates[model][1],
        candidates=candidates, cv={"linear": linear_cv, "dead_zone": dead_cv}, folds=folds,
        counts=_count(statuses), statuses=statuses,
    )


def _count(statuses: Sequence[str]) -> Dict[str, int]:
    return {status: sum(1 for s in statuses if s == status) for status in STATUSES}


# -- description --------------------------------------------------------------------
def speed_rate_curve(
    trials: Sequence[TransportTrial],
    options: TransportOptions = TransportOptions(),
    edges: Sequence[float] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130),
) -> List[Dict[str, float]]:
    """Bump speed (indices per tick) against gain rate (|right - left| per tick), over windows.

    Left and right windows are pooled, the speed counted positive in the
    direction of the drive.  This is the curve the dead-zone model describes.
    """

    rates: List[float] = []
    speeds: List[float] = []
    for trial in trials:
        for moved, signed in _windows(trial, options):
            total = float(np.sum(signed))
            if total == 0.0 or len(signed) == 0:
                continue
            rates.append(abs(total) / len(signed))
            speeds.append(moved * np.sign(total) / options.stuck_window)
    rates_array, speeds_array = np.asarray(rates), np.asarray(speeds)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (rates_array >= low) & (rates_array < high)
        if mask.sum() >= 2:
            rows.append({
                "rate_low": float(low), "rate_high": float(high), "n": int(mask.sum()),
                "rate_mean": float(rates_array[mask].mean()),
                "speed_mean": float(speeds_array[mask].mean()), "speed_sd": float(speeds_array[mask].std()),
            })
    return rows


def trials_to_arrays(trials: Sequence[TransportTrial]) -> Dict[str, np.ndarray]:
    """Arrays for ``numpy.savez``: per-step traces concatenated, one row of bookkeeping per trial."""

    return {
        "lengths": np.array([len(t.left) for t in trials], dtype=int),
        "left": np.concatenate([np.asarray(t.left, dtype=float) for t in trials]) if trials else np.zeros(0),
        "right": np.concatenate([np.asarray(t.right, dtype=float) for t in trials]) if trials else np.zeros(0),
        "centroid": np.concatenate([np.asarray(t.centroid, dtype=float) for t in trials]) if trials else np.zeros(0),
        "ring_size": np.array([t.ring_size for t in trials], dtype=int),
        "start_index": np.array([-1 if t.start_index is None else t.start_index for t in trials], dtype=int),
        "goal_index": np.array([-1 if t.goal_index is None else t.goal_index for t in trials], dtype=int),
        "group": np.array([str(t.group) for t in trials]),
        "label": np.array([t.label for t in trials]),
    }


def trials_from_arrays(arrays: Any) -> List[TransportTrial]:
    """Inverse of :func:`trials_to_arrays` (groups come back as strings)."""

    trials = []
    offset = 0
    for k, length in enumerate(np.asarray(arrays["lengths"], dtype=int)):
        stop = offset + int(length)
        start_index, goal_index = int(arrays["start_index"][k]), int(arrays["goal_index"][k])
        trials.append(TransportTrial(
            left=np.asarray(arrays["left"][offset:stop], dtype=float),
            right=np.asarray(arrays["right"][offset:stop], dtype=float),
            centroid=np.asarray(arrays["centroid"][offset:stop], dtype=float),
            ring_size=int(arrays["ring_size"][k]),
            start_index=None if start_index < 0 else start_index,
            goal_index=None if goal_index < 0 else goal_index,
            group=str(arrays["group"][k]), label=str(arrays["label"][k]),
        ))
        offset = stop
    return trials


def decoder_gains(index_per_spike: float, radians_per_index: float, dt_s: float, actuator_gain: float = 1.0) -> Tuple[float, float]:
    """``(gain_positive, gain_negative)`` in rad/s per spike/tick: ``a * rho / dt``.

    ``actuator_gain`` is the measured displacement the robot delivers per
    commanded displacement (1.0 for an ideal integrator); the gains are
    divided by it.
    """

    if dt_s <= 0.0 or actuator_gain <= 0.0:
        raise ValueError("dt_s and actuator_gain must be positive")
    gain = float(index_per_spike) * float(radians_per_index) / float(dt_s) / float(actuator_gain)
    return gain, -gain


__all__ = [
    "ErrorSummary", "STATUSES", "TransportFit", "TransportOptions", "TransportTrial",
    "classify_trial", "cross_validate", "decoder_gains", "fit_gain", "fit_transport", "has_stuck_window",
    "prediction_errors", "select_theta", "speed_rate_curve", "summarize_errors", "trial_displacement",
    "trial_drive", "trials_from_arrays", "trials_to_arrays", "unwrapped_centroid",
]
