"""Composites: Transport (Homeostasis + Gain) and JointTriple (T/B/A rings, two transports)."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Tuple

from .base import Block, Composite, Port, PortRef, signal_in, signal_out, spikes_in, spikes_out
from .comparator import Homeostasis
from .encoder import Encoder
from .gain import Gain
from .params import JOINT_TRIPLE_SCHEMA, TRANSPORT_SCHEMA
from .readout import FourierReadout
from .ring import Ring


class Transport(Composite):
    """Compare-and-transport motif: comparator plus opponent gain; moves ``ring`` toward the target features."""

    type_name: ClassVar[str] = "Transport"
    schema = TRANSPORT_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        spikes_in("state_features", "features of the ring being moved"),
        spikes_in("target_features", "features of the reference"),
        spikes_in("ring", "the ring being moved (gates the gain populations)"),
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
            "feedback": gain.ref("feedback"),
            "left_counts": gain.ref("left_counts"),
            "right_counts": gain.ref("right_counts"),
            "counts": comparator.ref("counts"),
        }
        return [comparator, gain], edges, port_map


class JointTriple(Composite):
    """Target / belief / actual rings with CT_goal (T moves B, motor signal) and CT_sense (A corrects B).

    The belief ring starts from the first measured angle (``enc_belief``, mode
    ``once``), like the state ring of the two-ring model; afterwards only the
    two transports move it.
    """

    type_name: ClassVar[str] = "JointTriple"
    schema = JOINT_TRIPLE_SCHEMA
    neural: ClassVar[bool] = True
    ports: ClassVar[Tuple[Port, ...]] = (
        signal_in("goal_angle", "desired angle → target ring"),
        signal_in("measured_angle", "measured angle → actual ring (mode state_mode)"),
        signal_in("initial_angle", "measured angle → belief ring once, at trial start"),
        signal_out("left_counts", "CT_goal left gain counts (motor signal)"),
        signal_out("right_counts", "CT_goal right gain counts (motor signal)"),
        signal_out("sense_left_counts", "CT_sense left counts (prediction error)"),
        signal_out("sense_right_counts", "CT_sense right counts (prediction error)"),
        signal_out("belief_counts", "belief ring counts"),
        signal_out("actual_counts", "actual ring counts"),
        signal_out("target_counts", "target ring counts"),
    )

    def expand(self):
        p = self.params
        size, harmonics = p["population_size"], p["num_fourier_k"]
        mapping = dict(mapping=p["mapping"], joint_min=p["joint_min"], joint_max=p["joint_max"], half_width=p["half_width"])
        rings = {name: Ring(self.sub_id(name), population_size=size) for name in ("T", "B", "A")}
        readouts = {name: FourierReadout(self.sub_id("f" + name), num_fourier_k=harmonics) for name in ("T", "B", "A")}
        enc_goal = Encoder(self.sub_id("enc_goal"), mode="once", **mapping)
        enc_belief = Encoder(self.sub_id("enc_belief"), mode="once", **mapping)
        enc_state = Encoder(self.sub_id("enc_state"), mode=p["state_mode"], rate_hz=p["state_rate_hz"], **mapping)
        ct_goal = Transport(self.sub_id("ct_goal"), gain={"gain_to_ring_weight": p["goal_feedback_weight"]})
        ct_sense = Transport(self.sub_id("ct_sense"), gain={"gain_to_ring_weight": p["sense_feedback_weight"]})
        blocks: List[Block] = list(rings.values()) + list(readouts.values()) + [enc_goal, enc_belief, enc_state, ct_goal, ct_sense]
        edges = [
            (enc_goal.ref("stim"), rings["T"].ref("stim"), {}),
            (enc_belief.ref("stim"), rings["B"].ref("stim"), {}),
            (enc_state.ref("stim"), rings["A"].ref("stim"), {}),
        ]
        for name in ("T", "B", "A"):
            edges.append((rings[name].ref("spikes"), readouts[name].ref("ring"), {}))
        for ct, reference in ((ct_goal, "T"), (ct_sense, "A")):
            edges += [
                (readouts["B"].ref("features"), ct.ref("state_features"), {}),
                (readouts[reference].ref("features"), ct.ref("target_features"), {}),
                (rings["B"].ref("spikes"), ct.ref("ring"), {}),
                (ct.ref("feedback"), rings["B"].ref("stim"), {}),
            ]
        port_map = {
            "goal_angle": enc_goal.ref("angle"),
            "measured_angle": enc_state.ref("angle"),
            "initial_angle": enc_belief.ref("angle"),
            "left_counts": ct_goal.ref("left_counts"),
            "right_counts": ct_goal.ref("right_counts"),
            "sense_left_counts": ct_sense.ref("left_counts"),
            "sense_right_counts": ct_sense.ref("right_counts"),
            "belief_counts": rings["B"].ref("counts"),
            "actual_counts": rings["A"].ref("counts"),
            "target_counts": rings["T"].ref("counts"),
        }
        return blocks, edges, port_map


__all__ = ["JointTriple", "Transport"]
