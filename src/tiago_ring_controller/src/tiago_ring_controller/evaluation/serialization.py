"""Schema-preserving, side-effect-free serialization preparation helpers."""

import json
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


RING_TRIAL_SCALAR_FIELDS: Tuple[str, ...] = (
    "joint_index",
    "batch_idx",
    "iteration_idx",
    "q_start",
    "q_goal",
    "q_final",
    "dq_desired",
    "dq_actual",
    "decoder_predicted_dq_rad",
    "raw_position_error_rad",
    "abs_position_error_rad",
    "aligned_error_rad",
    "S_raw",
    "S_filtered",
    "S_delayed",
    "S_pos",
    "S_neg",
    "S_vel",
    "n_steps",
    "stop_reason",
    "goal_ring_index",
    "initial_ring_index",
    "decoder_gain_positive",
    "decoder_gain_negative",
    "decoder_tau",
    "decoder_delay_steps",
)

PID_TRIAL_SCALAR_FIELDS: Tuple[str, ...] = (
    "joint_index",
    "batch_idx",
    "iteration_idx",
    "q_start",
    "q_goal",
    "q_final",
    "dq_desired",
    "dq_actual",
    "predicted_dq_rad",
    "raw_position_error_rad",
    "abs_position_error_rad",
    "aligned_error_rad",
    "S_vel",
    "n_steps",
    "stop_reason",
    "kp",
    "ki",
    "kd",
    "output_limit",
)

COMMON_BATCH_STAT_FIELDS: Tuple[str, ...] = (
    "joint_index",
    "batch_idx",
    "n_trials",
    "mean_abs_error_rad",
    "std_abs_error_rad",
    "median_abs_error_rad",
    "max_abs_error_rad",
    "min_abs_error_rad",
    "mean_aligned_error_rad",
    "std_aligned_error_rad",
    "mean_dq_desired_rad",
    "mean_dq_actual_rad",
    "mean_dq_pred_rad",
    "mae_pred_vs_desired_rad",
    "mae_pred_vs_actual_rad",
    "success_rate_le_0p02_rad",
    "success_rate_le_0p05_rad",
    "success_rate_le_0p10_rad",
    "max_steps_count",
)

RING_BATCH_STAT_FIELDS = COMMON_BATCH_STAT_FIELDS + ("drive_settled_count",)
PID_BATCH_STAT_FIELDS = COMMON_BATCH_STAT_FIELDS + ("error_settled_count",)

RING_TIMESERIES_FIELDS: Tuple[str, ...] = (
    "time",
    "joint_position",
    "joint_velocity",
    "signed_spike",
    "left_gain_spikes",
    "right_gain_spikes",
    "r1_spike_count",
    "r1_bump_index",
    "r1_centroid",
    "filtered_drive",
    "delayed_drive",
    "decoded_velocity",
    "position_error",
)

PID_TIMESERIES_FIELDS: Tuple[str, ...] = (
    "time",
    "joint_position",
    "joint_velocity",
    "position_error",
    "p_term",
    "i_term",
    "d_term",
    "decoded_velocity",
)

RING_RASTER_FIELDS: Tuple[str, ...] = (
    "r1_times",
    "r1_senders",
    "r2_times",
    "r2_senders",
    "left_times",
    "left_senders",
    "right_times",
    "right_senders",
)


def json_safe(
    value: Any,
    recursive_converter: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Mirror the recursive NumPy conversion used by analysis JSON writers."""

    convert_child = json_safe if recursive_converter is None else recursive_converter
    if isinstance(value, dict):
        return {str(key): convert_child(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert_child(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def legacy_json_text(value: Any, indent: int = 2) -> str:
    """Prepare JSON text using Python's current permissive NaN behavior."""

    return json.dumps(value, indent=indent)


def slug(value: Any) -> str:
    """Mirror the analysis filename slug conversion."""

    return "".join(
        char if char.isalnum() or char in ("_", "-") else "_"
        for char in str(value)
    )


def ordered_fields(
    values: Mapping[str, Any],
    fields: Sequence[str],
) -> "OrderedDict[str, Any]":
    """Select values in the exact order expected by ``csv.DictWriter``."""

    return OrderedDict((field, values[field]) for field in fields)


def array_payload(
    values: Mapping[str, Any],
    fields: Sequence[str],
) -> "OrderedDict[str, np.ndarray]":
    """Prepare an ordered NPY/NPZ payload without writing it."""

    return OrderedDict((field, np.asarray(values[field])) for field in fields)


def array_payload_schema(
    values: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Describe key order, shape, and dtype for a prepared artifact payload."""

    schema = []
    for key, value in values.items():
        array = np.asarray(value)
        schema.append(
            {
                "key": key,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    return tuple(schema)


def assert_exact_fields(values: Mapping[str, Any], fields: Iterable[str]) -> None:
    """Raise on a field mismatch without coercing or rewriting the payload."""

    expected = tuple(fields)
    actual = tuple(values.keys())
    if actual != expected:
        raise ValueError(
            "Payload fields differ: expected %r, got %r" % (expected, actual)
        )


__all__ = [
    "COMMON_BATCH_STAT_FIELDS",
    "PID_BATCH_STAT_FIELDS",
    "PID_TIMESERIES_FIELDS",
    "PID_TRIAL_SCALAR_FIELDS",
    "RING_BATCH_STAT_FIELDS",
    "RING_RASTER_FIELDS",
    "RING_TIMESERIES_FIELDS",
    "RING_TRIAL_SCALAR_FIELDS",
    "array_payload",
    "array_payload_schema",
    "assert_exact_fields",
    "json_safe",
    "legacy_json_text",
    "ordered_fields",
    "slug",
]
