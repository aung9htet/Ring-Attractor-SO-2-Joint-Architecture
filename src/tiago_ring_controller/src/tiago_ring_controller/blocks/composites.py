"""Composites: Transport (Homeostasis + Gain) and JointTriple (architecture v4: three rings, two transports)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

from .base import Block, Composite, Port, PortRef, signal_in, signal_out, spikes_in, spikes_out
from .comparator import Homeostasis
from .encoder import Encoder
from .gain import Gain
from .params import JOINT_TRIPLE_SCHEMA, TRANSPORT_SCHEMA
from .readout import FourierReadout
from .ring import Ring
from .state import RingState


class Transport(Composite):
    """Compare-and-transport motif: comparator plus opponent gain; moves ``ring`` toward the target features."""

    type_name: ClassVar[str] = "Transport"
    schema = TRANSPORT_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("state_features", "features of the ring being moved"),
        spikes_in("target_features", "features of the reference"),
        spikes_in("ring", "the ring being moved (gates the gain populations)"),
        spikes_in("reference", "the reference ring itself, for the sharp comparison (Gain.reference)",
                  required=False, multi=True, late=True),
        spikes_out("feedback", "shifted feedback into the moved ring's stim port"),
        signal_out("left_counts", "left gain spikes per tick"),
        signal_out("right_counts", "right gain spikes per tick"),
        signal_out("counts", "comparator warm/cold/left/right counts"),
    )

    def expand(self):
        comparator = Homeostasis(self.sub_id("cmp"), **self.params["homeostasis"])
        gain = Gain(self.sub_id("gain"), **self.params["gain"])
        edges = [
            (comparator.ref("left"), gain.ref("left_in"), {}),
            (comparator.ref("right"), gain.ref("right_in"), {}),
        ]
        port_map = {
            "state_features": comparator.ref("state_features"),
            "target_features": comparator.ref("target_features"),
            "ring": gain.ref("ring"),
            "reference": gain.ref("reference"),
            "feedback": gain.ref("feedback"),
            "left_counts": gain.ref("left_counts"),
            "right_counts": gain.ref("right_counts"),
            "counts": comparator.ref("counts"),
        }
        return [comparator, gain], edges, port_map


class JointTriple(Composite):
    """Architecture v4: goal, predictive and state rings with two gain pairs on the predictive ring.

    * ``G`` holds the goal, ``S`` the measured joint angle (encoders, every
      tick), ``P`` the predicted joint angle.  ``P`` starts from the first
      measured angle (``enc_pred``, mode ``once``); afterwards only the two
      gain pairs move it.
    * ``ct_goal`` compares ``P`` with ``G``: its spikes move ``P`` toward the
      goal and are the motor signal (``left_counts`` / ``right_counts``).
    * ``ct_body`` compares ``P`` with ``S``: its spikes pull ``P`` back to the
      body and command nothing.  The goal comparison therefore keeps
      commanding until the body itself has arrived.
    * Each comparison is the ramp comparator (far) plus the ring's own
      projection onto the gain pair (``Gain.reference``, near).

    docs/architecture.md, docs/v4-tuning.md.
    """

    type_name: ClassVar[str] = "JointTriple"
    schema = JOINT_TRIPLE_SCHEMA
    neural: ClassVar[bool] = True
    version: ClassVar[int] = 2
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_in("goal_angle", "desired angle → goal ring"),
        signal_in("measured_angle", "measured angle → state ring (mode state_mode)"),
        signal_in("initial_angle", "measured angle → predictive ring once, at trial start"),
        signal_out("left_counts", "goal pair, left gain counts (motor signal)"),
        signal_out("right_counts", "goal pair, right gain counts (motor signal)"),
        signal_out("body_left_counts", "body pair, left gain counts (prediction ahead of the body)"),
        signal_out("body_right_counts", "body pair, right gain counts"),
        signal_out("goal_counts", "goal ring counts"),
        signal_out("predicted_counts", "predictive ring counts"),
        signal_out("state_counts", "state ring counts"),
        signal_out("predicted_angle", "joint angle the predictive ring holds, rad"),
        signal_out("state_angle", "joint angle the state ring holds, rad"),
    )

    def expand(self):
        p = self.params
        size, harmonics = p["population_size"], p["num_fourier_k"]
        mapping = dict(mapping=p["mapping"], joint_min=p["joint_min"], joint_max=p["joint_max"],
                       half_width=p["half_width"], arc_fraction=p["arc_fraction"])
        rings = {name: Ring(self.sub_id(name), **dict(p["ring"], population_size=size)) for name in ("G", "P", "S")}
        readouts = {name: FourierReadout(self.sub_id("f" + name), num_fourier_k=harmonics) for name in ("G", "P", "S")}
        enc_goal = Encoder(self.sub_id("enc_goal"), mode=p["goal_mode"], dead_band=p["goal_dead_band"], **mapping)
        enc_pred = Encoder(self.sub_id("enc_pred"), mode="once", **mapping)
        enc_state = Encoder(self.sub_id("enc_state"), mode=p["state_mode"], rate_hz=p["state_rate_hz"],
                            dead_band=p["state_dead_band"], **mapping)
        gain = dict({"cross_inhibition_weight": 0.0}, **p["gain"])
        ct_goal = Transport(self.sub_id("ct_goal"), homeostasis=dict(p["homeostasis"]),
                            gain=dict(gain, gain_to_ring_weight=p["goal_feedback_weight"]))
        ct_body = Transport(self.sub_id("ct_body"), homeostasis=dict(p["homeostasis"]),
                            gain=dict(gain, gain_to_ring_weight=p["body_feedback_weight"], **p["body_gain"]))
        states = {name: RingState(self.sub_id(name + "_state"), window_ticks=p["window_ticks"], **mapping) for name in ("P", "S")}
        blocks: List[Block] = (list(rings.values()) + list(readouts.values()) + [enc_goal, enc_pred, enc_state, ct_goal, ct_body]
                               + list(states.values()))
        edges = [
            (enc_goal.ref("stim"), rings["G"].ref("stim"), {}),
            (enc_pred.ref("stim"), rings["P"].ref("stim"), {}),
            (enc_state.ref("stim"), rings["S"].ref("stim"), {}),
        ]
        for name in ("G", "P", "S"):
            edges.append((rings[name].ref("spikes"), readouts[name].ref("ring"), {}))
        for name, state in states.items():
            edges.append((rings[name].ref("counts"), state.ref("counts"), {}))
        for ct, reference, sharp in ((ct_goal, "G", p["goal_reference"]), (ct_body, "S", p["body_reference"])):
            edges += [
                (readouts["P"].ref("features"), ct.ref("state_features"), {}),
                (readouts[reference].ref("features"), ct.ref("target_features"), {}),
                (rings["P"].ref("spikes"), ct.ref("ring"), {}),
                (ct.ref("feedback"), rings["P"].ref("stim"), {}),
            ]
            if sharp:
                edges.append((rings[reference].ref("spikes"), ct.ref("reference"), dict(sharp)))
        port_map = {
            "goal_angle": enc_goal.ref("angle"),
            "measured_angle": enc_state.ref("angle"),
            "initial_angle": enc_pred.ref("angle"),
            "left_counts": ct_goal.ref("left_counts"),
            "right_counts": ct_goal.ref("right_counts"),
            "body_left_counts": ct_body.ref("left_counts"),
            "body_right_counts": ct_body.ref("right_counts"),
            "goal_counts": rings["G"].ref("counts"),
            "predicted_counts": rings["P"].ref("counts"),
            "state_counts": rings["S"].ref("counts"),
            "predicted_angle": states["P"].ref("angle"),
            "state_angle": states["S"].ref("angle"),
        }
        return blocks, edges, port_map


__all__ = ["JointTriple", "Transport"]
