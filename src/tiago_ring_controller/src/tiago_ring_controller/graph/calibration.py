"""Transport calibration of a graph: trials, network fingerprint, artifact, decoder gains.

``training.transport`` holds the numerics.  This module connects them to
graphs: which blocks stand behind a decoder's gain counts, how the calibrated
network is identified, how a joint's decoder gains follow from the artifact
(``k = a * rho / dt``), and how the calibration trials are run.

The artifact is ``config/calibration/transport_calibration.json``; it is
written by ``scripts/calibrate_transport.py``.  Method and measurements:
docs/archive/velocity/calibration.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..artifacts import load_json_legacy, sha256_file
from ..blocks.base import Block
from ..blocks.decoder import Decoder
from ..config import source_config_path
from ..cosim.loop import TrialRecord
from ..training.transport import (
    TransportOptions,
    TransportTrial,
    classify_trial,
    decoder_gains,
    trial_displacement,
    unwrapped_centroid,
)
from .graph import Graph, GraphError

TRANSPORT_SCHEMA = "transport-calibration/1"
TRANSPORT_FILE = ("calibration", "transport_calibration.json")
#: |displacement gain - 1| the robot may show before the decoder gains are corrected for it
ACTUATOR_TOLERANCE = 0.03

_NETWORK_ROLES = ("state_ring", "goal_ring", "state_readout", "goal_readout", "comparator", "gain")
_ENCODER_ROLES = ("state_encoder", "goal_encoder")
_ENCODER_KEYS = ("half_width", "rate_hz", "weight", "duration_ticks", "mode")
#: bounded-ring parameters, by block type: the switch (empty / 0 = off) and the keys it governs.
#: Left out of the fingerprint while off, so networks calibrated before they existed still match.
_BOUNDED_KEYS = {"Ring": ("wall", ("wall", "outside_current_scale", "wall_taper")),
                 "Gain": ("limit_arc", ("limit_arc", "limit_taper")),
                 "Homeostasis": ("ramp_current", ("ramp_current",))}


def _fingerprint_params(block: Block) -> Dict[str, Any]:
    params = dict(block.params)
    switch, keys = _BOUNDED_KEYS.get(block.type_name, (None, ()))
    if switch is not None and not params.get(switch):
        for key in keys:
            params.pop(key, None)
    return params


def _source(graph: Graph, block: Block, port: str) -> Optional[Block]:
    edges = graph.edges_into(block, port)
    return edges[0].source.block if edges else None


def transport_motif(graph: Graph, decoder: Optional[Block] = None) -> Dict[str, Block]:
    """The blocks behind a decoder's gain counts, by role; roles that are absent are left out.

    Roles: ``decoder``, ``gain``, ``state_ring``, ``comparator``, ``state_readout``,
    ``goal_readout``, ``goal_ring``, ``state_encoder``, ``goal_encoder``, ``joint``.
    Without ``decoder`` the first decoder that commands a joint is used.
    """

    if decoder is None:
        decoders = [b for b in graph.blocks.values() if isinstance(b, Decoder)]
        commanding = [d for d in decoders if any(e.target.block.type_name == "Joint" for e in graph.edges_from(d, "velocity"))]
        if not decoders:
            return {}
        decoder = (commanding or decoders)[0]
    motif: Dict[str, Block] = {"decoder": decoder}
    for edge in graph.edges_from(decoder, "velocity"):
        if edge.target.block.type_name == "Joint":
            motif["joint"] = edge.target.block
    gain = _source(graph, decoder, "left_counts")
    if gain is None or gain.type_name != "Gain":
        return motif
    motif["gain"] = gain
    ring = _source(graph, gain, "ring")
    if ring is not None and ring.type_name == "Ring":
        motif["state_ring"] = ring
    comparator = _source(graph, gain, "left_in")
    if comparator is not None and comparator.type_name == "Homeostasis":
        motif["comparator"] = comparator
        for role, port in (("state_readout", "state_features"), ("goal_readout", "target_features")):
            readout = _source(graph, comparator, port)
            if readout is not None:
                motif[role] = readout
        goal_readout = motif.get("goal_readout")
        if goal_readout is not None and goal_readout.type_name == "FourierReadout":
            goal_ring = _source(graph, goal_readout, "ring")
            if goal_ring is not None and goal_ring.type_name == "Ring":
                motif["goal_ring"] = goal_ring
    for ring_role, encoder_role in (("state_ring", "state_encoder"), ("goal_ring", "goal_encoder")):
        target = motif.get(ring_role)
        if target is None:
            continue
        for edge in graph.edges_into(target, "stim"):
            if edge.source.block.type_name == "Encoder":
                motif[encoder_role] = edge.source.block
    return motif


def network_fingerprint(graph: Graph, decoder: Optional[Block] = None, config_dir: Optional[str] = None) -> str:
    """sha256 over everything that sets the bump's displacement per gain spike.

    Tick length, the parameters of the rings, readouts, comparator and gain
    behind the decoder, the injection parameters of the two encoders, and the
    files the build reads for them (neuron parameters, fitted comparator
    weights).  Joint limits and the mapping are left out: they are geometry
    and enter the decoder gain through ``rho``, so every joint driven by the
    same network has the same fingerprint.
    """

    motif = transport_motif(graph, decoder)
    document: Dict[str, Any] = {"dt_ms": float(graph.simulation.dt_ms), "blocks": {}, "files": {}}
    for role in _NETWORK_ROLES:
        block = motif.get(role)
        if block is not None:
            document["blocks"][role] = {"type": block.type_name, "version": block.version, "params": _fingerprint_params(block)}
    for role in _ENCODER_ROLES:
        block = motif.get(role)
        if block is not None:
            document["blocks"][role] = {key: block.params[key] for key in _ENCODER_KEYS}
    root = config_dir or source_config_path()
    files = [os.path.join(root, "model_params", "neuron_params.json")]
    ring, comparator = motif.get("state_ring"), motif.get("comparator")
    if ring is not None and comparator is not None:
        size = int(ring.params["population_size"])
        files.append(comparator.params["weights_artifact"] or os.path.join(root, "homeostasis", "N_%d_homeostasis_weights.npy" % size))
        files.append(comparator.params["metadata_artifact"] or os.path.join(root, "homeostasis", "N_%d_homeostasis_metadata.json" % size))
    for role in ("state_readout", "goal_readout"):
        block = motif.get(role)
        if block is not None and block.params.get("weights_artifact"):
            files.append(block.params["weights_artifact"])
    for path in files:
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(root, path)          # config-relative artifact names, as the blocks resolve them
        name = os.path.relpath(path, root) if os.path.abspath(path).startswith(os.path.abspath(root)) else path
        document["files"][name.replace(os.sep, "/")] = sha256_file(path) if os.path.exists(path) else None
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- artifact -----------------------------------------------------------------------
def transport_calibration_path(config_dir: Optional[str] = None) -> str:
    return os.path.join(config_dir or source_config_path(), *TRANSPORT_FILE)


def find_transport_calibration(graph: Graph, config_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The transport artifact in ``config/calibration/`` whose network fingerprint matches the graph's.

    Falls back to the default artifact when none matches (the report then
    flags the mismatch), and ``None`` when there is none at all.
    """

    folder = os.path.dirname(transport_calibration_path(config_dir))
    fingerprint = network_fingerprint(graph, config_dir=config_dir)
    candidates = []
    if os.path.isdir(folder):
        candidates = sorted(name for name in os.listdir(folder) if name.startswith("transport_calibration") and name.endswith(".json"))
    fallback = None
    for name in candidates:
        try:
            document = load_transport_calibration(os.path.join(folder, name))
        except (OSError, ValueError):
            continue
        if document.get("network_fingerprint") == fingerprint:
            return document
        if name == TRANSPORT_FILE[-1]:
            fallback = document
    return fallback


def load_transport_calibration(path: Optional[str] = None) -> Dict[str, Any]:
    """The transport calibration artifact as a dict; ``ValueError`` when it is not one."""

    path = path or transport_calibration_path()
    document = load_json_legacy(path)
    if not isinstance(document, dict) or document.get("schema") != TRANSPORT_SCHEMA:
        raise ValueError("%s is not a %s artifact" % (path, TRANSPORT_SCHEMA))
    missing = [key for key in ("index_per_spike", "dead_zone", "decoder_tau_s", "decoder_delay_steps", "dt_ms") if key not in document]
    if missing:
        raise ValueError("%s lacks %s" % (path, missing))
    return document


def actuator_gain(calibration: Mapping[str, Any], joint: Optional[int]) -> float:
    """Displacement gain to divide the decoder gain by: the measured one when it is out of tolerance, else 1."""

    if joint is None:
        return 1.0
    entry = (calibration.get("actuator") or {}).get(str(int(joint)))
    if not entry or "displacement_gain" not in entry:
        return 1.0
    gain = float(entry["displacement_gain"])
    return gain if abs(gain - 1.0) > ACTUATOR_TOLERANCE else 1.0


def transport_decoder_params(calibration: Mapping[str, Any], radians_per_index: float, joint: Optional[int] = None) -> Dict[str, Any]:
    """``Decoder`` parameters of one joint: gains ``a * rho / dt``, dead zone, filter."""

    positive, negative = decoder_gains(
        calibration["index_per_spike"], radians_per_index, float(calibration["dt_ms"]) / 1000.0, actuator_gain(calibration, joint)
    )
    return dict(
        gain_positive=positive, gain_negative=negative, dead_zone=float(calibration["dead_zone"]),
        tau_s=float(calibration["decoder_tau_s"]), delay_steps=int(calibration["decoder_delay_steps"]),
    )


def calibration_report(graph: Graph, calibration: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Per decoder driven by gain counts: what the artifact prescribes against what the graph holds.

    ``uses_transport_calibration`` is True when gains and dead zone equal the
    prescribed ones; ``network_matches`` when the network behind the decoder is
    the calibrated one.  A decoder that uses the calibration on a different
    network, or with another tick length, is stale (``warning`` says so).
    With ``simulation.time_scale`` k the prescribed dead zone is k times the
    artifact's (k NEST steps per tick; docs/archive/velocity/speed.md).  Decoders of
    the three-ring graph (a gain pair with a ``reference`` input) are left
    out: their gain is a loop gain.
    """

    if calibration is None:
        calibration = find_transport_calibration(graph)
        if calibration is None:
            return []
    rows = []
    for decoder in [b for b in graph.blocks.values() if isinstance(b, Decoder)]:
        motif = transport_motif(graph, decoder)
        encoder, ring, joint = motif.get("state_encoder"), motif.get("state_ring"), motif.get("joint")
        if encoder is None or ring is None or "gain" not in motif:
            continue
        if graph.edges_into(motif["gain"], "reference"):
            continue                                 # v4: the body feedback closes the loop, no transport calibration applies
        rho = encoder.radians_per_index(int(ring.params["population_size"]))
        expected = transport_decoder_params(calibration, rho, None if joint is None else joint.index)
        expected["dead_zone"] = expected["dead_zone"] * int(graph.simulation.time_scale)
        actual = {key: decoder.params[key] for key in expected}
        uses = all(abs(actual[key] - expected[key]) <= 1e-12 + 1e-6 * abs(expected[key])
                   for key in ("gain_positive", "gain_negative", "dead_zone"))
        network = network_fingerprint(graph, decoder) == calibration.get("network_fingerprint")
        same_dt = float(graph.simulation.dt_ms) == float(calibration["dt_ms"])
        warning = None
        if uses and not (network and same_dt):
            warning = ("decoder %r uses the transport calibration, but %s; rerun scripts/calibrate_transport.py"
                       % (decoder.id, "the network parameters differ from the calibrated network" if not network
                          else "the tick length differs from the calibrated one"))
        rows.append({
            "decoder": decoder.id, "joint": None if joint is None else joint.index, "radians_per_index": rho,
            "expected": expected, "actual": actual, "uses_transport_calibration": uses,
            "network_matches": network, "dt_matches": same_dt, "warning": warning,
        })
    return rows


# -- trials -------------------------------------------------------------------------
def transport_trial_from_record(
    record: TrialRecord, state_ring_id: str, gain_id: str, group: Any = None, label: str = ""
) -> TransportTrial:
    """Gain counts and state-bump trace of a trial, one entry per NEST step (lead steps included)."""

    samples: Dict[int, Tuple[float, float, Sequence[float]]] = {}
    for inputs in [tick.inputs for tick in record.ticks] + [record.end_state]:
        ring, gain = inputs.get(state_ring_id), inputs.get(gain_id)
        if ring is None or gain is None:
            continue
        samples[int(ring["nest_step"])] = (float(gain["left_counts"]), float(gain["right_counts"]), ring["counts"])
    if not samples:
        raise ValueError("the record has no %r and %r datapacks" % (state_ring_id, gain_id))
    steps = sorted(samples)
    counts = np.array([samples[step][2] for step in steps], dtype=float)
    return TransportTrial(
        left=np.array([samples[step][0] for step in steps]), right=np.array([samples[step][1] for step in steps]),
        centroid=unwrapped_centroid(counts), ring_size=counts.shape[1],
        start_index=record.meta.get("initial_ring_index"), goal_index=record.meta.get("goal_ring_index"),
        group=group, label=label,
    )


def index_pairs(starts: Sequence[int], goals: Sequence[int], min_distance: int) -> List[Tuple[int, int]]:
    """Start/goal ring-index pairs at least ``min_distance`` apart."""

    return [(int(s), int(g)) for s in starts for g in goals if abs(int(g) - int(s)) >= int(min_distance)]


def run_transport_trials(
    graph_factory: Callable[[float, Optional[int]], Graph],
    pairs: Sequence[Tuple[int, int]],
    seeds: Sequence[Optional[int]],
    engines: str = "nest",
    options: TransportOptions = TransportOptions(),
    on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
    backend: Any = None,
) -> Tuple[List[TransportTrial], List[Dict[str, Any]]]:
    """Run one trial per (seed, start index, goal index) with the fake robot.

    ``graph_factory(goal_rad, seed)`` returns a fresh graph.  The start pose is
    the fake robot's home, so ``engines`` is ``"nest"`` (or ``"fake"`` for
    plumbing tests): the transport does not depend on the robot.  Returns the
    trials and one summary row per trial.
    """

    from .compile import compile_graph
    from .run import run_graph_trial

    if engines not in ("nest", "fake"):
        raise GraphError(["transport trials use the fake robot: engines must be 'nest' or 'fake'"])
    probe = transport_motif(graph_factory(0.0, None))
    missing = [role for role in ("gain", "state_ring", "state_encoder", "joint") if role not in probe]
    if missing:
        raise GraphError(["the graph has no %s behind its decoder; transport trials need the single-joint motif" % missing])
    size = int(probe["state_ring"].params["population_size"])
    trials: List[TransportTrial] = []
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for start_index, goal_index in pairs:
            start_rad = probe["state_encoder"].index_to_angle(start_index, size)
            goal_rad = probe["state_encoder"].index_to_angle(goal_index, size)
            graph = graph_factory(goal_rad, seed)
            motif = transport_motif(graph)
            compiled = compile_graph(graph, engines=engines, backend=backend)
            compiled.robot_engine.home[motif["joint"].index] = float(start_rad)
            compiled.loop.initialize()
            try:
                record = run_graph_trial(compiled, goal_rad)
            finally:
                compiled.loop.shutdown()
            trial = transport_trial_from_record(
                record, motif["state_ring"].id, motif["gain"].id, group=seed,
                label="seed %s, index %d -> %d" % (seed, start_index, goal_index),
            )
            moved = trial_displacement(trial, options)
            centroid = trial.centroid[np.isfinite(trial.centroid)]
            row = {
                "seed": seed, "start_index": start_index, "goal_index": goal_index,
                "start_rad": float(start_rad), "goal_rad": float(goal_rad),
                "n_steps": record.n_steps, "stop_reason": record.stop_reason,
                "status": classify_trial(trial, options), "displacement": moved,
                "sum_left": float(np.sum(trial.left)), "sum_right": float(np.sum(trial.right)),
                "endpoint_error": float(np.mean(centroid[-options.window:]) - goal_index) if len(centroid) else float("nan"),
                "joint_displacement": float(record.final_state["joint_state"]["positions"][motif["joint"].index] - start_rad),
            }
            trials.append(trial)
            rows.append(row)
            if on_trial is not None:
                on_trial(row)
    return trials, rows


__all__ = [
    "ACTUATOR_TOLERANCE", "TRANSPORT_FILE", "TRANSPORT_SCHEMA", "actuator_gain", "calibration_report",
    "find_transport_calibration",
    "index_pairs", "load_transport_calibration", "network_fingerprint", "run_transport_trials",
    "transport_calibration_path", "transport_decoder_params", "transport_motif", "transport_trial_from_record",
]
