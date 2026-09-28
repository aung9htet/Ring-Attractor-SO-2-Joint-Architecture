"""Immutable internal contracts derived from the legacy configuration.

These types deliberately do not validate away historical values.  Validation
belongs in report-only inspection code until all compatibility facades use the
same characterization suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class RingSpec:
    population_size: int
    num_positions: int
    num_fourier_k: int
    config_dir: str
    samples_per_position: int
    sim_settle_ms: float
    stimulus_half_width: int
    ridge_lambda: float
    output_dir: str
    harmonic_index: int
    readout_weight_scale: float
    output_dc_baseline: float
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class NeuronSpec:
    population: str
    parameters: Mapping[str, Any]
    model: str = "iaf_psc_alpha"


@dataclass(frozen=True)
class HomeostasisSpec:
    config_dir: str
    tie_epsilon: float
    warm_cold_weight_scale: float
    warm_cold_bias_scale: float
    warm_exc_weight: float
    warm_inh_weight: float
    cold_exc_weight: float
    cold_inh_weight: float
    decision_lateral_weight: float
    left_label: int
    right_label: int
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class GainSpec:
    left_homeostasis_gain_weight: float
    right_homeostasis_gain_weight: float
    ring_to_gain_weight: float
    gain_to_ring_weight: float
    cross_inhibition_weight: float
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class MultiRingSpec:
    population_size: int
    num_positions: int
    num_fourier_k: int
    sim_settle_ms: float
    stimulus_half_width: int
    ridge_lambda: float
    output_ring_size: int
    output_rate_baseline: float
    output_rate_amplitude: float
    num_joints: int
    joint_axes: Tuple[str, ...]
    max_train_samples: int
    max_test_samples: int
    n_vis_points: int
    feature_grid_size: int
    signed_product_dc_baseline: float
    signed_product_input_weight: float
    signed_product_output_weight_scale: float
    output_dc_baseline: float
    scalar_ramp_weight_scale: float
    scalar_dc_baseline: float
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class DecoderSpec:
    gain_positive: float = 1.0e-4
    gain_negative: float = -1.0e-4
    tau: float = 0.3
    delay_steps: int = 0
    source: str = "default"


@dataclass(frozen=True)
class PIDSpec:
    kp: float = 2.0
    ki: float = 0.1
    kd: float = 0.05
    output_limit: float = 0.5
    windup_limit: float = 2.0


@dataclass(frozen=True)
class JointCalibration:
    joint_index: int
    joint_min: Optional[float]
    joint_max: Optional[float]
    decoder: DecoderSpec
    pid: PIDSpec = field(default_factory=PIDSpec)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class ArtifactArrayDescriptor:
    key: Optional[str]
    shape: Tuple[int, ...]
    dtype: str
    scalar_value: Optional[Any] = None


@dataclass(frozen=True)
class ArtifactDescriptor:
    path: str
    kind: str
    size_bytes: int
    sha256: str
    arrays: Tuple[ArtifactArrayDescriptor, ...] = ()
    key_order: Tuple[str, ...] = ()
    json_top_level_type: Optional[str] = None
    json_keys: Tuple[str, ...] = ()
    strict_json: Optional[bool] = None
    errors: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors
