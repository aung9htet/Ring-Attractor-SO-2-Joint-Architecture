"""Pure signal blocks: Decoder (counts → velocity horizon) and ProfileDecoder (profile → angle)."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, ClassVar, Deque, Dict, Optional, Tuple

import numpy as np

from ..control.controller import DecoderParameters, DriveControlCore
from ..math.circular import decode_population_angle, decode_sawtooth_profile, preferred_angles
from ..math.control import decode_velocity
from .base import Block, Port, signal_in, signal_out
from .params import DECODER_SCHEMA, PROFILE_DECODER_SCHEMA


class Decoder(Block):
    """``right - left`` → exponential filter → delay → asymmetric gains → velocity horizon.

    Wraps ``DriveControlCore`` exactly as the cosim ``MotorTF`` does: one sample
    per NEST step, the horizon is the last ``horizon`` decoded velocities, the
    settle counter runs over consumed samples on main ticks only.
    """

    type_name: ClassVar[str] = "Decoder"
    schema = DECODER_SCHEMA
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_in("left_counts", "left gain (or decision) spikes per tick"),
        signal_in("right_counts", "right gain (or decision) spikes per tick"),
        signal_in("centroid", "ring centroid per tick (source=centroid_velocity)", required=False),
        signal_out("velocity", "receding-horizon joint velocities"),
        signal_out("settled", "True once |delayed drive| stayed under the threshold n_settle times"),
    )

    def __init__(self, id: str, **params: Any) -> None:
        super().__init__(id, **params)
        self.dt_s = 0.05
        self.reset()

    def configure(self, dt_ms: float) -> None:
        self.dt_s = float(dt_ms) / 1000.0
        self.reset()

    def parameters(self) -> DecoderParameters:
        p = self.params
        return DecoderParameters(
            gain_positive=p["gain_positive"], gain_negative=p["gain_negative"],
            tau_s=p["tau_s"], delay_steps=p["delay_steps"],
        )

    def reset(self) -> None:
        p = self.params
        tau = max(float(p["tau_s"]), 1e-9)
        self.core = DriveControlCore(
            self.dt_s, self.parameters(), spike_scale=1.0,
            drive_threshold=p["drive_threshold"], n_settle=p["n_settle"], alpha=math.exp(-self.dt_s / tau),
        )
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=int(p["horizon"]))
        self.consecutive_settled = 0
        self.samples_seen = 0
        self._last_centroid: Optional[float] = None

    def signed_input(self, left: float, right: float, centroid: Optional[float] = None) -> float:
        source = self.params["source"]
        if source == "centroid_velocity":
            if centroid is None or not np.isfinite(centroid) or self._last_centroid is None:
                value = 0.0
            else:
                value = float(centroid) - self._last_centroid
            if centroid is not None and np.isfinite(centroid):
                self._last_centroid = float(centroid)
            return value * float(self.params["spike_scale"])
        return (float(right) - float(left)) * float(self.params["spike_scale"])

    def consume(self, left: float, right: float, centroid: Optional[float] = None, nest_step: Optional[int] = None) -> Dict[str, Any]:
        """Decode one NEST sample into the horizon buffer and return it."""

        signed = self.signed_input(left, right, centroid)
        _, filtered, delayed = self.core.advance(signed, self.params["delay_steps"])
        velocity = decode_velocity(delayed, self.params["gain_positive"], self.params["gain_negative"])
        self.samples_seen += 1
        sample = {
            "nest_step": int(nest_step if nest_step is not None else self.samples_seen),
            "signed_spike": float(signed), "left": float(left), "right": float(right),
            "filtered_drive": float(filtered), "delayed_drive": float(delayed),
            "decoded_velocity": float(velocity),
        }
        self.buffer.append(sample)
        return sample

    def step(self, left: float, right: float, is_lead: bool = False, centroid: Optional[float] = None,
             nest_step: Optional[int] = None) -> Dict[str, Any]:
        """One tick: consume the sample, update the settle counter, emit the horizon."""

        self.consume(left, right, centroid, nest_step)
        consumed = self.buffer[0]
        if not is_lead:
            if abs(consumed["delayed_drive"]) < self.params["drive_threshold"]:
                self.consecutive_settled += 1
            else:
                self.consecutive_settled = 0
        settled = (not is_lead) and self.consecutive_settled >= self.params["n_settle"]
        return {
            "velocity": [s["decoded_velocity"] for s in self.buffer],
            "settled": bool(settled),
            "consumed": dict(consumed),
            "consecutive_settled": int(self.consecutive_settled),
        }


class ProfileDecoder(Block):
    """Output-ring count profile → angle (sawtooth moments, scalar ramp or circular centroid)."""

    type_name: ClassVar[str] = "ProfileDecoder"
    schema = PROFILE_DECODER_SCHEMA
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_in("profile", "per-neuron counts of an output ring"),
        signal_out("angle", "decoded angle in rad"),
    )

    def decode(self, profile) -> float:
        counts = np.asarray(profile, dtype=float)
        method = self.params["method"]
        if counts.size == 0 or not np.any(counts > 0):
            return float("nan")
        if method == "sawtooth":
            return float(decode_sawtooth_profile(counts))
        if method == "centroid":
            return float(decode_population_angle(counts))
        # scalar ramp: mean preferred angle weighted by the baseline-subtracted profile
        shifted = np.maximum(counts - counts.min(), 0.0)
        if shifted.sum() <= 0:
            return float("nan")
        theta = preferred_angles(len(counts))
        return float(np.dot(shifted, theta) / shifted.sum())

    def step(self, profile) -> Dict[str, Any]:
        return {"angle": self.decode(profile)}


__all__ = ["Decoder", "ProfileDecoder"]
