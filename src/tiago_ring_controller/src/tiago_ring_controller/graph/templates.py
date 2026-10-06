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
from ..blocks.params import RING_SCHEMA
from ..blocks.state import RingState
from ..config import load_joint_calibration, source_config_path
from ..control.mapping import ArcMapping, RingMapper
from .calibration import load_transport_calibration, transport_decoder_params
from .graph import Graph

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
DEFAULT_LIMITS: Dict[int, Tuple[float, float]] = {}
#: where a template's decoder gains come from: the frozen per-joint JSON, or the transport fit
CALIBRATIONS = ("legacy", "transport")
DEFAULT_CALIBRATION = "transport"
#: ``legacy``: the joint range on indices 5-195 (seam between the limits); ``bounded``: on an arc
BOUNDS = ("legacy", "bounded")
#: The bounded ring (docs/archive/velocity/bounded-ring.md).  ``arc_fraction`` of the ring holds the joint
#: range, centred on index N/2; the wall keeps bumps ``wall_margin`` indices beyond the arc; the
#: gain drive fades over ``limit_taper`` indices at each limit; ``comparator`` names the fitted
#: weights in config/homeostasis/, ``ramp_current`` lifts its warm/cold ramps off the zero-rate
#: floor, and ``transport`` names the transport calibration.  ``gain_to_ring_weight`` and
#: ``homeostasis_gain_weight`` set how far the bump moves per decision (docs/archive/velocity/speed.md).
BOUNDED: Dict[str, Any] = dict(
    arc_fraction=0.70, wall_margin=15, outside_current_scale=0.9, wall_taper=0, limit_taper=0.0,
    weight_scale=5.0e6, ramp_current=8.0e5, comparator="N_200_bounded_a070", transport="transport_calibration_bounded.json",
    gain_to_ring_weight=-0.6, homeostasis_gain_weight=10000.0,
    time_scale=3,
)
#: ``time_scale``: NEST steps per robot tick of the bounded graph.  3 makes a move 2.8 times
#: shorter with the same end point (docs/archive/velocity/speed.md); 1 is network time = robot time.


#: Architecture v4 (three rings, docs/v4-tuning.md).  ``motor_gain``: ring indices the joint
#: is commanded per goal-pair spike (a loop gain, not a fit: the body pair corrects what it gets
#: wrong); ``*_reference``: the ring-to-gain projections of the sharp comparison;
#: ``*_feedback_weight``: how far a spike of each pair moves the predictive bump (the body pair
#: pulls five times harder); ``goal_mode`` / ``state_mode``: how the goal and state rings are held
#: (``corrective``: stimulated only when the bump is ``*_dead_band`` indices off, so a resting
#: bump has its free amplitude and the ramp comparators are not biased); ``n_settle``: ticks
#: without motor drive that end a trial; ``dead_zone``: goal-pair spikes per NEST step that command
#: nothing (without it the joint hunts about one index around the goal); ``time_scale``: NEST
#: steps per robot tick.
FEEDBACK: Dict[str, Any] = dict(
    motor_gain=0.005266, tau_s=0.05, dead_zone=3.0, drive_threshold=5.0, n_settle=40,
    homeostasis_gain_weight=10000.0, body_homeostasis_gain_weight=10000.0,
    goal_feedback_weight=-0.6, body_feedback_weight=-3.0,
    goal_reference=dict(exc_weight=2.0e4, inh_weight=6.0e4, window=10),
    body_reference=dict(exc_weight=2.0e4, inh_weight=1.5e5, window=10),
    goal_mode="corrective", goal_dead_band=2,
    state_mode="corrective", state_dead_band=1, state_rate_hz=200.0,
    time_scale=3,
)


#: velocity limits of arm_1 .. arm_7 in tiago_description (arm.urdf.xacro, wrist.urdf.xacro), rad/s
ARM_MAX_VELOCITY = (1.95, 1.95, 2.35, 2.35, 1.95, 1.76, 1.76)


def bounded_settings(ring_size: int, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The bounded-ring parameters of every block, from :data:`BOUNDED` and ``overrides``."""

    settings = dict(BOUNDED, **(overrides or {}))
    arc = ArcMapping.bounded(-1.0, 1.0, ring_size, settings["arc_fraction"])
    first, last = arc.arc
    settings["limit_arc"] = [int(round(first)), int(round(last))]
    settings["wall"] = [max(int(round(first)) - int(settings["wall_margin"]), 0),
                        min(int(round(last)) + int(settings["wall_margin"]), ring_size - 1)]
    return settings


def feedback_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """:data:`FEEDBACK` with ``overrides``; a dict value (``goal_reference``) is merged key by key."""

    settings = {key: dict(value) if isinstance(value, dict) else value for key, value in FEEDBACK.items()}
    for key, value in (overrides or {}).items():
        if key not in settings:
            raise ValueError("unknown v4 setting %r; known: %s" % (key, sorted(settings)))
        if isinstance(settings[key], dict) and isinstance(value, dict):
            settings[key].update(value)
        else:
            settings[key] = value
    return settings


def joint_limits(joint: int, calibration_path: Optional[str] = None) -> Tuple[float, float]:
    """Limits from the checked-in velocity calibration (the legacy workflows' source)."""

    path = calibration_path or source_config_path("calibration", "velocity_calibration.json")
    calibration = load_joint_calibration(path, joint)
    if calibration.joint_min is None or calibration.joint_max is None:
        return (-1.0, 1.0)
    return (float(calibration.joint_min), float(calibration.joint_max))


def decoder_params(joint: int, calibration_path: Optional[str] = None, mapping: str = "collector",
                   calibration: Optional[str] = None, transport_path: Optional[str] = None,
                   population_size: Optional[int] = None, half_width: int = 5,
                   arc_fraction: float = 0.70, time_scale: int = 1) -> Dict[str, Any]:
    """``Decoder`` parameters of one joint.

    ``calibration="transport"``: gains ``a * rho / dt``, dead zone and filter
    from a transport calibration artifact (the joint follows the state bump;
    see ``graph.calibration``), with ``rho`` from the joint limits and the
    mapping; the bounded mapping uses ``transport_calibration_bounded.json``.
    ``calibration="legacy"``: the per-joint values of ``velocity_calibration.json``.
    With ``time_scale`` k NEST steps per tick (docs/archive/velocity/speed.md) a tick carries
    k times the spikes, so the dead zone and the settle threshold scale by k; the
    gain per spike does not change.
    """

    from ..control.profiles import get_legacy_control_profile

    calibration = calibration or DEFAULT_CALIBRATION
    if calibration not in CALIBRATIONS:
        raise ValueError("calibration must be one of %r, got %r" % (CALIBRATIONS, calibration))
    profile = get_legacy_control_profile("collector" if mapping in ("bounded", "circular") else mapping)
    k = int(time_scale)
    settle = dict(drive_threshold=profile.drive_threshold * k, n_settle=profile.n_settle, horizon=max(profile.lookahead, 1))
    if calibration == "transport":
        lo, hi = joint_limits(joint, calibration_path)
        size = int(population_size or RING_SCHEMA["population_size"].default)
        rho = RingMapper(mapping, lo, hi, size, half_width, arc_fraction).radians_per_index
        if transport_path is None and mapping == "bounded":
            transport_path = source_config_path("calibration", BOUNDED["transport"])
        params = transport_decoder_params(load_transport_calibration(transport_path), rho, joint)
        params["dead_zone"] = params["dead_zone"] * k
        return dict(params, **settle)
    path = calibration_path or source_config_path("calibration", "velocity_calibration.json")
    legacy = load_joint_calibration(path, joint)
    return dict(
        gain_positive=legacy.decoder.gain_positive, gain_negative=legacy.decoder.gain_negative,
        tau_s=legacy.decoder.tau, delay_steps=legacy.decoder.delay_steps, **settle
    )


def two_ring_single_joint(joint: int = 5, goal_rad: float = 0.6, mapping: str = "collector",
                          calibration_path: Optional[str] = None, seed: Optional[int] = 13579,
                          calibration: Optional[str] = None, bounds: str = "legacy",
                          bounded: Optional[Dict[str, Any]] = None, state_readout: Optional[bool] = None,
                          time_scale: Optional[int] = None) -> Graph:
    """Today's model: two rings, two readouts, comparator + gain, decoder, one joint.

    ``bounds="bounded"`` builds the bounded ring (:data:`BOUNDED`, ``bounded``
    overrides): the joint range on an arc, a wall around the seam, the gain
    drive faded at the limits, the bounded comparator, and a ``RingState``
    readout of the joint angle (``state_readout`` adds one to the legacy graph).
    ``time_scale`` runs that many NEST steps per robot tick (faster joints,
    docs/archive/velocity/speed.md); by default 1 for the legacy graph and
    ``BOUNDED["time_scale"]`` for the bounded one.
    """

    from ..control.profiles import get_legacy_control_profile

    if bounds not in BOUNDS:
        raise ValueError("bounds must be one of %r, got %r" % (BOUNDS, bounds))
    if time_scale is None:
        time_scale = dict(BOUNDED, **(bounded or {}))["time_scale"] if bounds == "bounded" else 1
    profile = get_legacy_control_profile(mapping)
    lo, hi = joint_limits(joint, calibration_path)
    g = Graph("two ring single joint %d%s" % (joint, " bounded" if bounds == "bounded" else ""), dt_ms=profile.time_step_ms,
              nest_lead_steps=profile.lookahead, max_steps=profile.max_steps, rng_seed=seed, time_scale=int(time_scale))
    size = int(RING_SCHEMA["population_size"].default)
    if bounds == "bounded":
        b = bounded_settings(size, bounded)
        limits = dict(mapping="bounded", joint_min=lo, joint_max=hi, half_width=5, arc_fraction=b["arc_fraction"])
        wall = dict(wall=b["wall"], outside_current_scale=b["outside_current_scale"], wall_taper=b["wall_taper"])
        r1, r2 = g.add(Ring("r1", **wall)), g.add(Ring("r2", **wall))
        f1, f2 = g.add(FourierReadout("f1")), g.add(FourierReadout("f2"))
        # a bare name lives in config/homeostasis/; a path (candidate fits) is used as given
        artifact = (b["comparator"] if os.path.isabs(b["comparator"]) else os.path.join("homeostasis", b["comparator"])) + "_homeostasis_%s"
        cmp = g.add(Homeostasis("cmp", weight_scale=b["weight_scale"], ramp_current=b["ramp_current"], weights_artifact=artifact % "weights.npy",
                                metadata_artifact=artifact % "metadata.json"))
        gain = g.add(Gain("gain", limit_arc=b["limit_arc"], limit_taper=b["limit_taper"], gain_to_ring_weight=b["gain_to_ring_weight"],
                          left_homeostasis_gain_weight=b["homeostasis_gain_weight"],
                          right_homeostasis_gain_weight=b["homeostasis_gain_weight"]))
        dec_params = decoder_params(joint, calibration_path, "bounded", calibration, arc_fraction=b["arc_fraction"],
                                    transport_path=source_config_path("calibration", b["transport"]), time_scale=time_scale)
    else:
        limits = dict(mapping=mapping, joint_min=lo, joint_max=hi, half_width=5)
        r1, r2 = g.add(Ring("r1")), g.add(Ring("r2"))
        f1, f2 = g.add(FourierReadout("f1")), g.add(FourierReadout("f2"))
        cmp, gain = g.add(Homeostasis("cmp")), g.add(Gain("gain"))
        dec_params = decoder_params(joint, calibration_path, mapping, calibration, time_scale=time_scale)
    enc_state = g.add(Encoder("enc_state", mode="once", **limits))
    enc_goal = g.add(Encoder("enc_goal", mode="once", **limits))
    dec = g.add(Decoder("dec", **dec_params))
    # The legacy graph keeps the 1 rad/s clip of the schema; the bounded graph may use the arm's own limit.
    clip = dict(max_velocity=ARM_MAX_VELOCITY[joint]) if bounds == "bounded" and 0 <= joint < len(ARM_MAX_VELOCITY) else {}
    j = g.add(Joint("j%d" % joint, index=joint, joint_min=lo, joint_max=hi, **clip))
    goal = g.add(Goal("goal", angle_rad=goal_rad))
    if state_readout or (state_readout is None and bounds == "bounded"):
        state = g.add(RingState("state", **limits))
        g.connect(r1.counts, state.counts)
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


def three_ring_single_joint(joint: int = 5, goal_rad: float = 0.6, calibration_path: Optional[str] = None,
                            seed: Optional[int] = 13579, bounded: Optional[Dict[str, Any]] = None,
                            feedback: Optional[Dict[str, Any]] = None, time_scale: Optional[int] = None) -> Graph:
    """Architecture v4: goal, predictive and state rings on the bounded ring, body feedback.

    The goal gain pair moves the predictive ring and drives the joint; the body
    gain pair pulls the predictive ring back to the measured joint
    (``JointTriple``).  ``feedback`` overrides :data:`FEEDBACK`, ``bounded``
    :data:`BOUNDED`.  The decoder gain is the loop gain ``motor_gain * rho / dt``
    and its dead zone a small noise floor: no transport calibration is used.  NEST
    leads by one step and one trajectory point is sent per tick, so the loop
    sees the newest command.  docs/v4-tuning.md.
    """

    from ..control.profiles import get_legacy_control_profile

    profile = get_legacy_control_profile("collector")
    f = feedback_settings(feedback)
    if time_scale is None:
        time_scale = f["time_scale"]
    k = int(time_scale)
    lo, hi = joint_limits(joint, calibration_path)
    size = int(RING_SCHEMA["population_size"].default)
    b = bounded_settings(size, bounded)
    g = Graph("three ring single joint %d" % joint, dt_ms=profile.time_step_ms, nest_lead_steps=1,
              max_steps=profile.max_steps, rng_seed=seed, time_scale=k)
    artifact = (b["comparator"] if os.path.isabs(b["comparator"]) else os.path.join("homeostasis", b["comparator"])) + "_homeostasis_%s"
    triple = g.add(JointTriple(
        "jt", mapping="bounded", joint_min=lo, joint_max=hi, arc_fraction=b["arc_fraction"],
        ring=dict(wall=b["wall"], outside_current_scale=b["outside_current_scale"], wall_taper=b["wall_taper"]),
        homeostasis=dict(weight_scale=b["weight_scale"], ramp_current=b["ramp_current"],
                         weights_artifact=artifact % "weights.npy", metadata_artifact=artifact % "metadata.json"),
        gain=dict(limit_arc=b["limit_arc"], limit_taper=b["limit_taper"],
                  left_homeostasis_gain_weight=f["homeostasis_gain_weight"],
                  right_homeostasis_gain_weight=f["homeostasis_gain_weight"]),
        goal_feedback_weight=f["goal_feedback_weight"], body_feedback_weight=f["body_feedback_weight"],
        goal_reference=dict(f["goal_reference"]), body_reference=dict(f["body_reference"]),
        goal_mode=f["goal_mode"], goal_dead_band=f["goal_dead_band"],
        state_mode=f["state_mode"], state_dead_band=f["state_dead_band"], state_rate_hz=f["state_rate_hz"],
        body_gain=dict(left_homeostasis_gain_weight=f["body_homeostasis_gain_weight"],
                       right_homeostasis_gain_weight=f["body_homeostasis_gain_weight"]),
    ))
    rho = RingMapper("bounded", lo, hi, size, 5, b["arc_fraction"]).radians_per_index
    gain = float(f["motor_gain"]) * rho / (profile.time_step_ms / 1000.0)
    dec = g.add(Decoder("dec", gain_positive=gain, gain_negative=-gain, dead_zone=f["dead_zone"] * k, tau_s=f["tau_s"], delay_steps=0,
                        drive_threshold=f["drive_threshold"] * k, n_settle=f["n_settle"], horizon=1))
    clip = dict(max_velocity=ARM_MAX_VELOCITY[joint]) if 0 <= joint < len(ARM_MAX_VELOCITY) else {}
    j = g.add(Joint("j%d" % joint, index=joint, joint_min=lo, joint_max=hi, **clip))
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
                         calibration_path: Optional[str] = None, seed: Optional[int] = 13579,
                         calibration: Optional[str] = None) -> Graph:
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
        dec = g.add(Decoder("dec" + suffix, **decoder_params(joint, calibration_path, mapping, calibration)))
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


__all__ = ["ARM_MAX_VELOCITY", "BOUNDED", "EXAMPLES_DIR", "FEEDBACK", "TEMPLATES", "bounded_settings", "decoder_params", "feedback_settings", "joint_limits", "multi_joint_two_ring",
           "three_ring_single_joint", "two_joint_forward_kinematics", "two_ring_single_joint", "write_examples"]
