"""Co-simulation configuration derived from the legacy control profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional

from ..contracts import JointCalibration
from ..control.controller import DecoderParameters
from ..control.profiles import LegacyControlProfile, get_legacy_control_profile


NEST_STEP_MODES = ("run", "simulate")
STEPPERS = ("clock_wait", "plugin")


@dataclass(frozen=True)
class CosimConfig:
    """Everything a trial needs, serialisable so records can cite it."""

    profile: str = "collector"
    joint_index: int = 5
    dt_ms: float = 50.0
    nest_lead_steps: int = 4
    max_steps: int = 400
    drive_threshold: float = 5.0
    n_settle: int = 10
    stimulus_half_width: int = 5
    goal_mode: str = "once"
    proprioception_mode: str = "once"
    proprioception_rate_gain: float = 1.0
    decoder_gain_positive: float = 1e-4
    decoder_gain_negative: float = -1e-4
    decoder_tau_s: float = 0.3
    decoder_delay_steps: int = 0
    joint_min: Optional[float] = None
    joint_max: Optional[float] = None
    rng_seed: Optional[int] = None
    local_num_threads: int = 1
    nest_step_mode: str = "run"
    reset_mode: str = "rebuild"
    stepper: str = "clock_wait"
    max_step_size_s: float = 0.001
    step_timeout_s: float = 5.0
    ring_params_file: Optional[str] = None
    weights_dir: Optional[str] = None

    def __post_init__(self) -> None:
        get_legacy_control_profile(self.profile)
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        if self.nest_lead_steps < 0:
            raise ValueError("nest_lead_steps must be >= 0")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.nest_step_mode not in NEST_STEP_MODES:
            raise ValueError("nest_step_mode must be one of %r" % (NEST_STEP_MODES,))
        if self.stepper not in STEPPERS:
            raise ValueError("stepper must be one of %r" % (STEPPERS,))
        if self.reset_mode not in ("rebuild", "continue"):
            raise ValueError("reset_mode must be 'rebuild' or 'continue'")
        if self.goal_mode not in ("once", "continuous"):
            raise ValueError("goal_mode must be 'once' or 'continuous'")
        if self.proprioception_mode not in ("once", "continuous", "off"):
            raise ValueError("proprioception_mode must be 'once', 'continuous' or 'off'")
        if (self.joint_min is None) != (self.joint_max is None):
            raise ValueError("joint_min and joint_max must be given together")

    # -- derived ----------------------------------------------------------
    @property
    def dt_s(self) -> float:
        return self.dt_ms / 1000.0

    @property
    def control_profile(self) -> LegacyControlProfile:
        return get_legacy_control_profile(self.profile)

    @property
    def decoder(self) -> DecoderParameters:
        return DecoderParameters(
            gain_positive=self.decoder_gain_positive,
            gain_negative=self.decoder_gain_negative,
            tau_s=self.decoder_tau_s,
            delay_steps=self.decoder_delay_steps,
        )

    @property
    def physics_iterations_per_tick(self) -> int:
        iterations = self.dt_s / self.max_step_size_s
        rounded = int(round(iterations))
        if abs(iterations - rounded) > 1e-9:
            raise ValueError(
                "dt_ms=%r is not a whole number of physics iterations of %r s"
                % (self.dt_ms, self.max_step_size_s)
            )
        return rounded

    def require_limits(self) -> "tuple[float, float]":
        if self.joint_min is None or self.joint_max is None:
            raise ValueError("joint limits are not configured for joint %d" % self.joint_index)
        return float(self.joint_min), float(self.joint_max)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_profile(
        cls,
        profile: LegacyControlProfile,
        joint_index: int,
        calibration: Optional[JointCalibration] = None,
        **overrides: Any,
    ) -> "CosimConfig":
        values: Dict[str, Any] = {
            "profile": profile.name,
            "joint_index": int(joint_index),
            "dt_ms": profile.time_step_ms,
            "nest_lead_steps": profile.lookahead,
            "max_steps": profile.max_steps,
            "drive_threshold": profile.drive_threshold,
            "n_settle": profile.n_settle,
            "decoder_gain_positive": profile.decoder_gain_positive,
            "decoder_gain_negative": profile.decoder_gain_negative,
            "decoder_tau_s": profile.decoder_tau_s,
            "decoder_delay_steps": profile.decoder_delay_steps,
        }
        config = cls(**values)
        if calibration is not None:
            config = config.with_calibration(calibration)
        if overrides:
            config = replace(config, **overrides)
        return config

    def with_calibration(self, calibration: JointCalibration) -> "CosimConfig":
        values: Dict[str, Any] = {
            "joint_index": int(calibration.joint_index),
            "decoder_gain_positive": calibration.decoder.gain_positive,
            "decoder_gain_negative": calibration.decoder.gain_negative,
            "decoder_tau_s": calibration.decoder.tau,
            "decoder_delay_steps": calibration.decoder.delay_steps,
        }
        if calibration.joint_min is not None and calibration.joint_max is not None:
            values["joint_min"] = float(calibration.joint_min)
            values["joint_max"] = float(calibration.joint_max)
        return replace(self, **values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CosimConfig":
        return cls(**dict(value))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "CosimConfig":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str) -> "CosimConfig":
        with open(path, "r", encoding="utf-8") as stream:
            return cls.from_json(stream.read())

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(self.to_json())
            stream.write("\n")


__all__ = ["CosimConfig", "NEST_STEP_MODES", "STEPPERS"]
