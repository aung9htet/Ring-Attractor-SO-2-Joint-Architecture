"""Transceiver functions: pure mappings between engine datapacks.

Every TF is independent of ROS and NEST and is unit-tested with dict inputs.
The three TFs here reproduce the legacy single-joint control path:

* :class:`GoalTF` maps a goal angle to a ``goal_bump`` on the target ring.
* :class:`ProprioceptionTF` maps the measured joint to a ``state_bump`` on
  the state ring.  ``mode="once"`` is the legacy once-per-trial injection;
  ``mode="continuous"`` is the hook for closing the loop and is accepted by
  the loop but rejected by the legacy NEST stimulus port until build-time
  stimulus generators exist (plan B.4).
* :class:`MotorTF` maps ``ring_counts`` to an ``arm_velocity_cmd`` horizon
  through the unchanged :class:`DriveControlCore` filter/delay, the
  asymmetric decoder, the receding-horizon buffer and the settle counter.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, FrozenSet, Mapping, Optional

import numpy as np

from ..control.controller import DecoderParameters, DriveControlCore
from ..control.profiles import LegacyControlProfile
from ..math.control import decode_velocity
from .datapack import DataPack


STIMULUS_RATE_HZ = 200.0
STIMULUS_WEIGHT = 4.5e3
STIMULUS_DURATION_MS = 50.0

PROPRIOCEPTION_MODES = ("once", "continuous", "off")
GOAL_MODES = ("once", "continuous")


@dataclass(frozen=True)
class TickContext:
    """What a TF may know about *when* it runs."""

    t_ms: float
    tick: int
    phase: str = "main"
    lead_index: int = -1

    @property
    def is_lead(self) -> bool:
        return self.phase == "lead"


class TransceiverFunction(ABC):
    name: str = "tf"
    inputs: FrozenSet[str] = frozenset()
    outputs: FrozenSet[str] = frozenset()

    def reset(self) -> None:
        return None

    @abstractmethod
    def __call__(
        self, inputs: Mapping[str, DataPack], ctx: TickContext
    ) -> Dict[str, DataPack]:
        raise NotImplementedError


def bump_datapack(
    name: str,
    t_ms: float,
    center_index: int,
    half_width: int,
    source: str,
    **extra: Any,
) -> DataPack:
    data = {
        "center_index": int(center_index),
        "half_width": int(half_width),
        "rate_hz": STIMULUS_RATE_HZ,
        "weight": STIMULUS_WEIGHT,
        "duration_ms": STIMULUS_DURATION_MS,
        "source": source,
    }
    data.update(extra)
    return DataPack(name, t_ms, data)


class GoalTF(TransceiverFunction):
    """Goal angle → ``goal_bump`` via the profile's joint-to-ring mapping."""

    name = "goal"
    outputs = frozenset({"goal_bump"})

    def __init__(
        self,
        profile: LegacyControlProfile,
        joint_min: float,
        joint_max: float,
        population_size: int,
        requested_half_width: int = 5,
        goal_rad: Optional[float] = None,
        mode: str = "once",
    ) -> None:
        if mode not in GOAL_MODES:
            raise ValueError("goal mode must be one of %r" % (GOAL_MODES,))
        self.profile = profile
        self.joint_min = float(joint_min)
        self.joint_max = float(joint_max)
        self.population_size = int(population_size)
        self.requested_half_width = int(requested_half_width)
        self.mode = mode
        self.goal_rad = goal_rad
        self.last_index: Optional[int] = None
        self._emitted = False

    def set_goal(self, goal_rad: float) -> None:
        self.goal_rad = float(goal_rad)

    def reset(self) -> None:
        self._emitted = False
        self.last_index = None

    def ring_index(self, angle_rad: float) -> int:
        return self.profile.joint_to_ring_index(
            angle_rad,
            self.joint_min,
            self.joint_max,
            self.population_size,
            requested_half_width=self.requested_half_width,
        )

    def __call__(
        self, inputs: Mapping[str, DataPack], ctx: TickContext
    ) -> Dict[str, DataPack]:
        if self.goal_rad is None:
            raise ValueError("GoalTF has no goal; call set_goal() before the trial")
        if self.mode == "once" and self._emitted:
            return {}
        self._emitted = True
        index = self.ring_index(self.goal_rad)
        self.last_index = index
        return {
            "goal_bump": bump_datapack(
                "goal_bump",
                ctx.t_ms,
                index,
                # inject_stimulus receives the *effective* half width, exactly
                # as the legacy ``set_ring_goal`` passes it on.
                self.profile.effective_half_width(self.requested_half_width),
                source="goal",
                goal_rad=self.goal_rad,
                requested_half_width=self.requested_half_width,
            )
        }


class ProprioceptionTF(TransceiverFunction):
    """Measured joint → ``state_bump`` on the state ring."""

    name = "proprioception"
    inputs = frozenset({"joint_state"})
    outputs = frozenset({"state_bump"})

    def __init__(
        self,
        profile: LegacyControlProfile,
        joint_index: int,
        joint_min: float,
        joint_max: float,
        population_size: int,
        requested_half_width: int = 5,
        mode: str = "once",
        rate_gain: float = 1.0,
    ) -> None:
        if mode not in PROPRIOCEPTION_MODES:
            raise ValueError("proprioception mode must be one of %r" % (PROPRIOCEPTION_MODES,))
        self.profile = profile
        self.joint_index = int(joint_index)
        self.joint_min = float(joint_min)
        self.joint_max = float(joint_max)
        self.population_size = int(population_size)
        self.requested_half_width = int(requested_half_width)
        self.mode = mode
        self.rate_gain = float(rate_gain)
        self.last_index: Optional[int] = None
        self._emitted = False

    def reset(self) -> None:
        self._emitted = False
        self.last_index = None

    def ring_index(self, angle_rad: float) -> int:
        return self.profile.joint_to_ring_index(
            angle_rad,
            self.joint_min,
            self.joint_max,
            self.population_size,
            requested_half_width=self.requested_half_width,
        )

    def __call__(
        self, inputs: Mapping[str, DataPack], ctx: TickContext
    ) -> Dict[str, DataPack]:
        if self.mode == "off":
            return {}
        if self.mode == "once" and self._emitted:
            return {}
        state = inputs.get("joint_state")
        if state is None:
            return {}
        position = float(state["positions"][self.joint_index])
        index = self.ring_index(position)
        self._emitted = True
        self.last_index = index
        return {
            "state_bump": bump_datapack(
                "state_bump",
                ctx.t_ms,
                index,
                self.profile.effective_half_width(self.requested_half_width),
                source="proprioception",
                mode=self.mode,
                joint_position=position,
                rate_hz=STIMULUS_RATE_HZ * self.rate_gain,
                requested_half_width=self.requested_half_width,
            )
        }


@dataclass(frozen=True)
class MotorSample:
    """One decoded NEST step, in the order the legacy buffer stores it."""

    nest_step: int
    signed_spike: float
    left: int
    right: int
    r1_spike_count: float
    r1_bump_index: Optional[int]
    r1_centroid: float
    filtered_drive: float
    delayed_drive: float
    decoded_velocity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nest_step": self.nest_step,
            "signed_spike": self.signed_spike,
            "left": self.left,
            "right": self.right,
            "r1_spike_count": self.r1_spike_count,
            "r1_bump_index": self.r1_bump_index,
            "r1_centroid": self.r1_centroid,
            "filtered_drive": self.filtered_drive,
            "delayed_drive": self.delayed_drive,
            "decoded_velocity": self.decoded_velocity,
        }


class MotorTF(TransceiverFunction):
    """``ring_counts`` → ``arm_velocity_cmd`` (receding-horizon velocities).

    Mirrors the legacy loop exactly: each NEST sample is pushed through
    ``DriveControlCore.advance`` then ``decode_velocity``; the horizon is the
    last ``max(nest_lead_steps, 1)`` decoded velocities; the settle counter
    runs over *consumed* samples (the first horizon point) on main ticks only.
    """

    name = "motor"
    inputs = frozenset({"ring_counts"})
    outputs = frozenset({"arm_velocity_cmd"})

    def __init__(
        self,
        profile: LegacyControlProfile,
        joint_index: int,
        population_size: int,
        dt_ms: float,
        decoder: Optional[DecoderParameters] = None,
        nest_lead_steps: int = 4,
        drive_threshold: float = 5.0,
        n_settle: int = 10,
    ) -> None:
        self.profile = profile
        self.joint_index = int(joint_index)
        self.population_size = int(population_size)
        self.dt_ms = float(dt_ms)
        self.dt_s = self.dt_ms / 1000.0
        self.decoder = decoder or DecoderParameters(
            gain_positive=profile.decoder_gain_positive,
            gain_negative=profile.decoder_gain_negative,
            tau_s=profile.decoder_tau_s,
            delay_steps=profile.decoder_delay_steps,
        )
        self.nest_lead_steps = int(nest_lead_steps)
        self.horizon_len = max(self.nest_lead_steps, 1)
        self.drive_threshold = float(drive_threshold)
        self.n_settle = int(n_settle)
        self.core: DriveControlCore
        self.buffer: Deque[MotorSample]
        self.reset()

    def reset(self) -> None:
        alpha = math.exp(-self.dt_s / self.decoder.tau_s)
        self.core = DriveControlCore(
            self.dt_s,
            self.decoder,
            spike_scale=1.0,
            drive_threshold=self.drive_threshold,
            n_settle=self.n_settle,
            alpha=alpha,
        )
        self.buffer = deque(maxlen=self.horizon_len)
        self.consecutive_settled = 0
        self.samples_seen = 0
        self._last_nest_step: Optional[int] = None

    def consume_counts(self, counts: DataPack) -> Optional[MotorSample]:
        """Decode one NEST sample; ignore a sample already consumed."""

        nest_step = int(counts.get("nest_step", self.samples_seen + 1))
        if self._last_nest_step is not None and nest_step <= self._last_nest_step:
            return None
        self._last_nest_step = nest_step

        left = int(counts["left"])
        right = int(counts["right"])
        signed = self.profile.signed_drive(right, left, self.population_size)
        _, filtered, delayed = self.core.advance(signed, self.decoder.delay_steps)
        velocity = decode_velocity(
            delayed, self.decoder.gain_positive, self.decoder.gain_negative
        )
        bump = counts.get("r1_bump_index")
        sample = MotorSample(
            nest_step=nest_step,
            signed_spike=float(signed),
            left=left,
            right=right,
            r1_spike_count=float(counts.get("r1_spike_count", 0.0)),
            r1_bump_index=None if bump is None else int(bump),
            r1_centroid=float(counts.get("r1_centroid", float("nan"))),
            filtered_drive=float(filtered),
            delayed_drive=float(delayed),
            decoded_velocity=float(velocity),
        )
        self.buffer.append(sample)
        self.samples_seen += 1
        return sample

    def __call__(
        self, inputs: Mapping[str, DataPack], ctx: TickContext
    ) -> Dict[str, DataPack]:
        counts = inputs.get("ring_counts")
        if counts is not None:
            self.consume_counts(counts)
        if not self.buffer:
            return {}

        consumed = self.buffer[0]
        if not ctx.is_lead:
            if abs(consumed.delayed_drive) < self.drive_threshold:
                self.consecutive_settled += 1
            else:
                self.consecutive_settled = 0
        settled = (not ctx.is_lead) and self.consecutive_settled >= self.n_settle

        return {
            "arm_velocity_cmd": DataPack(
                "arm_velocity_cmd",
                ctx.t_ms,
                {
                    "joint_index": self.joint_index,
                    "dt_s": self.dt_s,
                    "velocities": [sample.decoded_velocity for sample in self.buffer],
                    "horizon_len": len(self.buffer),
                    "nest_lead_steps": self.nest_lead_steps,
                    "consumed": consumed.to_dict(),
                    "consecutive_settled": int(self.consecutive_settled),
                    "settled": bool(settled),
                    "phase": ctx.phase,
                },
            )
        }


__all__ = [
    "GOAL_MODES",
    "GoalTF",
    "MotorSample",
    "MotorTF",
    "PROPRIOCEPTION_MODES",
    "ProprioceptionTF",
    "STIMULUS_DURATION_MS",
    "STIMULUS_RATE_HZ",
    "STIMULUS_WEIGHT",
    "TickContext",
    "TransceiverFunction",
    "bump_datapack",
]
