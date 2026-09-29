"""The three reference architectures as template functions (plan 5f).

A template only calls ``Graph.add`` / ``Graph.connect`` on primitives or
composites; the example files under ``graph/examples/`` are written by
``write_examples`` and pinned by tests to these functions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

from ..blocks import (
    Decoder,
    Encoder,
    FourierReadout,
    Gain,
    Goal,
    Homeostasis,
    Joint,
    JointTriple,
    OutputRing,
    ProfileDecoder,
    Ring,
    SignedProduct,
)
from ..config import load_joint_calibration, source_config_path
from .graph import Graph

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
DEFAULT_LIMITS: Dict[int, Tuple[float, float]] = {}


def joint_limits(joint: int, calibration_path: Optional[str] = None) -> Tuple[float, float]:
    """Limits from the checked-in velocity calibration (the legacy workflows' source)."""

    path = calibration_path or source_config_path("calibration", "velocity_calibration.json")
    calibration = load_joint_calibration(path, joint)
    if calibration.joint_min is None or calibration.joint_max is None:
        return (-1.0, 1.0)
    return (float(calibration.joint_min), float(calibration.joint_max))


def decoder_params(joint: int, calibration_path: Optional[str] = None, mapping: str = "collector") -> Dict[str, Any]:
    path = calibration_path or source_config_path("calibration", "velocity_calibration.json")
    calibration = load_joint_calibration(path, joint)
    from ..control.profiles import get_legacy_control_profile

    profile = get_legacy_control_profile(mapping)
    return dict(
        gain_positive=calibration.decoder.gain_positive, gain_negative=calibration.decoder.gain_negative,
        tau_s=calibration.decoder.tau, delay_steps=calibration.decoder.delay_steps,
        drive_threshold=profile.drive_threshold, n_settle=profile.n_settle, horizon=max(profile.lookahead, 1),
    )


def two_ring_single_joint(joint: int = 5, goal_rad: float = 0.6, mapping: str = "collector",
                          calibration_path: Optional[str] = None, seed: Optional[int] = 13579) -> Graph:
    """Today's model: two rings, two readouts, comparator + gain, decoder, one joint."""

    from ..control.profiles import get_legacy_control_profile

    profile = get_legacy_control_profile(mapping)
    lo, hi = joint_limits(joint, calibration_path)
    g = Graph("two ring single joint %d" % joint, dt_ms=profile.time_step_ms, nest_lead_steps=profile.lookahead,
              max_steps=profile.max_steps, rng_seed=seed)
    limits = dict(mapping=mapping, joint_min=lo, joint_max=hi, half_width=5)
    r1, r2 = g.add(Ring("r1")), g.add(Ring("r2"))
    f1, f2 = g.add(FourierReadout("f1")), g.add(FourierReadout("f2"))
    cmp, gain = g.add(Homeostasis("cmp")), g.add(Gain("gain"))
    enc_state = g.add(Encoder("enc_state", mode="once", **limits))
    enc_goal = g.add(Encoder("enc_goal", mode="once", **limits))
    dec = g.add(Decoder("dec", **decoder_params(joint, calibration_path, mapping)))
    j = g.add(Joint("j%d" % joint, index=joint, joint_min=lo, joint_max=hi))
    goal = g.add(Goal("goal", angle_rad=goal_rad))
    g.connect(r1.spikes, f1.ring)
    g.connect(r2.spikes, f2.ring)
    g.connect(f1.features, cmp.state_features)
    g.connect(f2.features, cmp.target_features)
    g.connect(cmp.left, gain.left_in)
    g.connect(cmp.right, gain.right_in)
    g.connect(r1.spikes, gain.ring)
    g.connect(gain.feedback, r1.stim)
    g.connect(enc_state.stim, r1.stim)
    g.connect(enc_goal.stim, r2.stim)
    g.connect(j.angle, enc_state.angle)
    g.connect(goal.angle, enc_goal.angle)
    g.connect(gain.left_counts, dec.left_counts)
    g.connect(gain.right_counts, dec.right_counts)
    g.connect(dec.velocity, j.velocity)
    return g


def three_ring_single_joint(joint: int = 5, goal_rad: float = 0.6, mapping: str = "collector",
                            calibration_path: Optional[str] = None, seed: Optional[int] = 13579,
                            state_mode: str = "continuous", state_rate_hz: float = 200.0,
                            sense_feedback_weight: float = -0.6) -> Graph:
    """Target / belief / actual rings with CT_goal and CT_sense (plan 5d); characterised in phase 5."""

    from ..control.profiles import get_legacy_control_profile

    profile = get_legacy_control_profile(mapping)
    lo, hi = joint_limits(joint, calibration_path)
    g = Graph("three ring single joint %d" % joint, dt_ms=profile.time_step_ms, nest_lead_steps=1,
              max_steps=profile.max_steps, rng_seed=seed)
    triple = g.add(JointTriple(
        "jt", mapping=mapping, joint_min=lo, joint_max=hi, state_mode=state_mode,
        state_rate_hz=state_rate_hz, sense_feedback_weight=sense_feedback_weight,
    ))
    dec = g.add(Decoder("dec", **decoder_params(joint, calibration_path, mapping)))
    j = g.add(Joint("j%d" % joint, index=joint, joint_min=lo, joint_max=hi))
    goal = g.add(Goal("goal", angle_rad=goal_rad))
    g.connect(goal.angle, triple.goal_angle)
    g.connect(j.angle, triple.measured_angle)
    g.connect(j.angle, triple.initial_angle)
    g.connect(triple.left_counts, dec.left_counts)
    g.connect(triple.right_counts, dec.right_counts)
    g.connect(dec.velocity, j.velocity)
    return g


def two_joint_forward_kinematics(angles: Sequence[float] = (0.0, 0.0), seed: Optional[int] = 13579) -> Graph:
    """The legacy two-joint stack: rings N=100, signed-product conjunctions, three fitted output rings."""

    g = Graph("two joint forward kinematics", dt_ms=50.0, nest_lead_steps=0, max_steps=6, rng_seed=seed)
    q1, q2 = g.add(Ring("q1", population_size=100)), g.add(Ring("q2", population_size=100))
    enc1 = g.add(Encoder("enc_q1", mapping="circular", mode="once", half_width=5))
    enc2 = g.add(Encoder("enc_q2", mapping="circular", mode="once", half_width=5))
    goal1, goal2 = g.add(Goal("angle_q1", angle_rad=float(angles[0]))), g.add(Goal("angle_q2", angle_rad=float(angles[1])))
    product = g.add(SignedProduct("product"))
    g.connect(goal1.angle, enc1.angle)
    g.connect(goal2.angle, enc2.angle)
    g.connect(enc1.stim, q1.stim)
    g.connect(enc2.stim, q2.stim)
    g.connect(q1.spikes, product.ring_a)
    g.connect(q2.spikes, product.ring_b)
    for name in ("lift", "pitch", "yaw"):
        ring = g.add(OutputRing(name, weights_key="W_signed_%s" % name))
        decoder = g.add(ProfileDecoder("decode_" + name))
        g.connect(product.features, ring.features)
        g.connect(ring.profile, decoder.profile)
    return g


def multi_joint_two_ring(joints: Sequence[int] = (5, 6), goals: Sequence[float] = (0.6, -0.4), mapping: str = "collector",
                         calibration_path: Optional[str] = None, seed: Optional[int] = 13579) -> Graph:
    """Plan 5e level 1: one two-ring circuit per joint, independent goals, one robot engine."""

    from ..control.profiles import get_legacy_control_profile

    profile = get_legacy_control_profile(mapping)
    g = Graph("multi joint %s" % "-".join(str(j) for j in joints), dt_ms=profile.time_step_ms,
              nest_lead_steps=profile.lookahead, max_steps=profile.max_steps, rng_seed=seed)
    for joint, goal_rad in zip(joints, goals):
        lo, hi = joint_limits(joint, calibration_path)
        suffix = "_j%d" % joint
        limits = dict(mapping=mapping, joint_min=lo, joint_max=hi, half_width=5)
        r1, r2 = g.add(Ring("r1" + suffix)), g.add(Ring("r2" + suffix))
        f1, f2 = g.add(FourierReadout("f1" + suffix)), g.add(FourierReadout("f2" + suffix))
        cmp, gain = g.add(Homeostasis("cmp" + suffix)), g.add(Gain("gain" + suffix))
        enc_state = g.add(Encoder("enc_state" + suffix, mode="once", **limits))
        enc_goal = g.add(Encoder("enc_goal" + suffix, mode="once", **limits))
        dec = g.add(Decoder("dec" + suffix, **decoder_params(joint, calibration_path, mapping)))
        j = g.add(Joint("j%d" % joint, index=joint, joint_min=lo, joint_max=hi))
        goal = g.add(Goal("goal" + suffix, angle_rad=float(goal_rad)))
        g.connect(r1.spikes, f1.ring)
        g.connect(r2.spikes, f2.ring)
        g.connect(f1.features, cmp.state_features)
        g.connect(f2.features, cmp.target_features)
        g.connect(cmp.left, gain.left_in)
        g.connect(cmp.right, gain.right_in)
        g.connect(r1.spikes, gain.ring)
        g.connect(gain.feedback, r1.stim)
        g.connect(enc_state.stim, r1.stim)
        g.connect(enc_goal.stim, r2.stim)
        g.connect(j.angle, enc_state.angle)
        g.connect(goal.angle, enc_goal.angle)
        g.connect(gain.left_counts, dec.left_counts)
        g.connect(gain.right_counts, dec.right_counts)
        g.connect(dec.velocity, j.velocity)
    return g


TEMPLATES = {
    "two_ring_single_joint": two_ring_single_joint,
    "multi_joint_two_ring": multi_joint_two_ring,
    "three_ring_single_joint": three_ring_single_joint,
    "two_joint_forward_kinematics": two_joint_forward_kinematics,
}


def write_examples(directory: str = EXAMPLES_DIR) -> Dict[str, str]:
    os.makedirs(directory, exist_ok=True)
    written = {}
    for name, template in TEMPLATES.items():
        path = os.path.join(directory, name + ".graph.json")
        template().save(path)
        written[name] = path
    return written


__all__ = ["EXAMPLES_DIR", "TEMPLATES", "decoder_params", "joint_limits", "multi_joint_two_ring",
           "three_ring_single_joint", "two_joint_forward_kinematics", "two_ring_single_joint", "write_examples"]
