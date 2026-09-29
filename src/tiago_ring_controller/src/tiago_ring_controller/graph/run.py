"""Run a graph (object or file) through the co-simulation loop.

``run_graph`` returns the same ``TrialRecord``s the dashboard produces and
writes the collector layout when asked (through ``cosim.runner``).  Every
record's ``meta["graph"]`` holds the resolved graph.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

from ..cosim.loop import LoopObserver, TrialRecord
from ..cosim.runner import TrialWriter, legacy_collector_record, run_trial, trial_raster, trial_row
from ..evaluation.serialization import json_safe
from .compile import CompiledGraph, JointSensorTF, compile_graph
from .graph import Graph

GraphLike = Union[Graph, str, "os.PathLike[str]"]


def as_graph(graph: GraphLike) -> Graph:
    if isinstance(graph, Graph):
        return graph
    return Graph.load(os.fspath(graph))


def run_graph_trial(compiled: CompiledGraph, goal_rad: float, meta: Optional[Dict[str, Any]] = None,
                    reset_mode: Optional[str] = None) -> TrialRecord:
    """One trial of a compiled graph: sets the (first) Goal block, runs, annotates the record."""

    trial_meta = {"graph": compiled.graph.describe(), "graph_name": compiled.graph.name}
    if meta:
        trial_meta.update(meta)
    record = run_trial(compiled.loop, goal_rad, compiled.config, meta=trial_meta, reset_mode=reset_mode)
    for tf in compiled.tfs:
        if isinstance(tf, JointSensorTF) and compiled.primary.joint is tf.block:
            record.meta["initial_ring_index"] = tf.last_index
    if compiled.nest_engine is not None:
        record.meta["nest_build"] = None if compiled.nest_engine.built is None else compiled.nest_engine.built.describe()
        record.meta["nest_recalibrations"] = compiled.nest_engine.recalibrations
    return record


def run_graph(
    graph: GraphLike,
    engines: str = "fake",
    goals: Sequence[float] = (0.5,),
    out_dir: Optional[str] = None,
    backend: Any = None,
    observers: Optional[Sequence[LoopObserver]] = None,
    on_trial: Any = None,
    keep_loop: bool = False,
) -> List[Dict[str, Any]]:
    """Compile and run ``goals`` trials; returns ``[{"record", "legacy", "row"}, ...]``.

    ``legacy`` (the collector's record) is present when the graph has the
    single-joint motif; otherwise only the ``record``.
    """

    graph = as_graph(graph)
    compiled = compile_graph(graph, engines=engines, backend=backend, observers=observers)
    loop = compiled.loop
    results: List[Dict[str, Any]] = []
    writer = TrialWriter(out_dir, compiled.config, len(goals)) if (out_dir and compiled.primary.complete) else None
    try:
        loop.initialize()
        for index, goal in enumerate(goals, start=1):
            record = run_graph_trial(compiled, float(goal), meta={"iteration_idx": index})
            result: Dict[str, Any] = {"record": record}
            if compiled.primary.complete:
                legacy = legacy_collector_record(record, compiled.config, trial_raster(loop))
                row = trial_row(compiled.config, index, legacy)
                result.update(legacy=legacy, row=json_safe(row))
                if writer is not None:
                    writer.write(index, record, legacy, row)
            elif out_dir:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "trial_%04d_graph.json" % index), "w", encoding="utf-8") as stream:
                    stream.write(record.to_json(indent=1))
            results.append(result)
            if on_trial is not None:
                on_trial(index, record, result.get("legacy"))
    finally:
        if writer is not None:
            writer.close()
        if not keep_loop:
            loop.shutdown()
    if keep_loop:
        results.append({"compiled": compiled})
    return results


__all__ = ["as_graph", "run_graph", "run_graph_trial"]
