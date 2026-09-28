"""Pure preprocessing core for the legacy neural joint controllers."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..math.control import (
    apply_delay as _apply_delay,
    decode_velocity as _decode_velocity,
    decoder_features as _decoder_features,
    exponential_filter as _exponential_filter,
)


@dataclass(frozen=True)
class DecoderParameters:
    """Parameters used by the asymmetric velocity decoder."""

    gain_positive: float = 1e-4
    gain_negative: float = -1e-4
    tau_s: float = 0.3
    delay_steps: int = 0


@dataclass(frozen=True)
class DriveSample:
    """One state transition of :class:`DriveControlCore`."""

    raw_signed_spike: float
    signed_drive: float
    filtered_drive: float
    delayed_drive: float
    drive_positive: float
    drive_negative: float
    velocity_command: float
    consecutive_settled: int
    is_settled: bool


def exponential_filter(
    spike_history: Iterable[float],
    dt_s: float,
    tau_s: float,
) -> np.ndarray:
    """Apply the controller's zero-initialized exponential filter.

    The operation order mirrors the three existing controller workflows.
    """

    return _exponential_filter(spike_history, dt_s, tau_s)


def apply_discrete_delay(
    drive_history: Iterable[float],
    delay_steps: int,
) -> np.ndarray:
    """Delay a drive trace with the current zero-fill convention."""

    return _apply_delay(drive_history, delay_steps)


def decode_asymmetric_velocity(
    delayed_drive: float,
    gain_positive: float,
    gain_negative: float,
) -> float:
    """Decode a signed drive using the existing positive/negative branches."""

    return _decode_velocity(delayed_drive, gain_positive, gain_negative)


def decoder_features_from_spikes(
    spike_history: Iterable[float],
    dt_s: float,
    tau_s: float,
    delay_steps: int,
) -> Dict[str, object]:
    """Return the exact offline filter/delay and integrated fit features."""

    return _decoder_features(spike_history, dt_s, tau_s, delay_steps)


def predicted_displacement(
    spike_history: Iterable[float],
    dt_s: float,
    parameters: DecoderParameters,
) -> float:
    """Return ``k_pos*S_pos + k_neg*S_neg`` for an offline trace."""

    features = decoder_features_from_spikes(
        spike_history,
        dt_s,
        parameters.tau_s,
        parameters.delay_steps,
    )
    return (
        parameters.gain_positive * features["S_pos"]
        + parameters.gain_negative * features["S_neg"]
    )


class DriveControlCore:
    """Stateful, ROS/NEST-independent filter-delay-drive controller.

    ``step`` accepts the raw ``right_count - left_count`` value.  The optional
    ``spike_scale`` captures calibration's ``100 / population_size`` scaling;
    collector and analysis use the default scale of one.
    """

    def __init__(
        self,
        dt_s: float,
        parameters: Optional[DecoderParameters] = None,
        spike_scale: float = 1.0,
        drive_threshold: float = 5.0,
        n_settle: int = 10,
        alpha: Optional[float] = None,
    ) -> None:
        self.dt_s = dt_s
        self.parameters = parameters or DecoderParameters()
        self.spike_scale = spike_scale
        self.drive_threshold = drive_threshold
        self.n_settle = n_settle
        self.alpha = (
            np.exp(-self.dt_s / self.parameters.tau_s)
            if alpha is None
            else alpha
        )
        self.reset()

    def reset(self) -> None:
        """Restore the zero filter, delay queue, and settle counter."""

        self.filtered_drive = 0.0
        self._delay_queue = (
            [0.0] * self.parameters.delay_steps
            if self.parameters.delay_steps > 0
            else []
        )
        self.consecutive_settled = 0

    def advance(
        self,
        raw_signed_spike: float,
        delay_steps: Optional[int] = None,
    ):
        """Advance only the legacy filter and delay state.

        This deliberately returns the uncoerced numerical values.  The flat
        robot-control facades use this narrower operation and continue to call
        their own private velocity-decoder methods.  That preserves subclass
        and monkeypatch dispatch as well as the exact point at which decoding
        occurs in each legacy control loop.
        """

        signed_drive = (
            raw_signed_spike
            if self.spike_scale == 1.0
            else raw_signed_spike * self.spike_scale
        )
        self.filtered_drive = (
            self.alpha * self.filtered_drive
            + (1.0 - self.alpha) * signed_drive
        )

        effective_delay_steps = (
            self.parameters.delay_steps
            if delay_steps is None
            else delay_steps
        )
        if effective_delay_steps > 0:
            self._delay_queue.append(self.filtered_drive)
            delayed_drive = self._delay_queue.pop(0)
        else:
            delayed_drive = self.filtered_drive

        return signed_drive, self.filtered_drive, delayed_drive

    def step(self, raw_signed_spike: float) -> DriveSample:
        """Consume one spike-count sample and return the decoded command."""

        signed_drive, filtered_drive, delayed_drive = self.advance(
            raw_signed_spike
        )

        velocity = decode_asymmetric_velocity(
            delayed_drive,
            self.parameters.gain_positive,
            self.parameters.gain_negative,
        )

        if abs(delayed_drive) < self.drive_threshold:
            self.consecutive_settled += 1
        else:
            self.consecutive_settled = 0

        return DriveSample(
            raw_signed_spike=float(raw_signed_spike),
            signed_drive=float(signed_drive),
            filtered_drive=float(filtered_drive),
            delayed_drive=float(delayed_drive),
            drive_positive=float(max(delayed_drive, 0.0)),
            drive_negative=float(max(-delayed_drive, 0.0)),
            velocity_command=float(velocity),
            consecutive_settled=int(self.consecutive_settled),
            is_settled=bool(self.consecutive_settled >= self.n_settle),
        )

    def prefill(self, raw_signed_spikes: Iterable[float]) -> List[DriveSample]:
        """Consume a lookahead sequence using ordinary ``step`` semantics."""

        return [self.step(spike) for spike in raw_signed_spikes]


class PIDControllerCore:
    """State-only implementation of the existing PID baseline controller."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        windup_limit: float = 2.0,
        output_limit: float = 1.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup_limit = windup_limit
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def step(self, error: float, dt: float):
        self._integral = np.clip(
            self._integral + error * dt,
            -self.windup_limit,
            self.windup_limit,
        )
        derivative = (
            0.0
            if self._prev_error is None
            else (error - self._prev_error) / dt
        )
        self._prev_error = error
        output = (
            self.kp * error
            + self.ki * self._integral
            + self.kd * derivative
        )
        return float(np.clip(output, -self.output_limit, self.output_limit)), {
            "p": float(self.kp * error),
            "i": float(self.ki * self._integral),
            "d": float(self.kd * derivative),
        }


__all__ = [
    "DecoderParameters",
    "DriveControlCore",
    "DriveSample",
    "PIDControllerCore",
    "apply_discrete_delay",
    "decode_asymmetric_velocity",
    "decoder_features_from_spikes",
    "exponential_filter",
    "predicted_displacement",
]
