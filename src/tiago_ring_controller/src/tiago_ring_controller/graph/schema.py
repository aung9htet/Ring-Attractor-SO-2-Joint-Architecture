"""Graph file (schema ``ring-blocks/1``): JSON (de)serialisation and validation.

One file per experiment, produced by the editor or by ``Graph.save`` and
readable by hand.  Blocks are listed as declared (a composite is one entry;
its sub-blocks are not written), edges reference ``"block.port"`` as
declared, parameters are written *resolved* (every parameter, defaults filled)
so that a graph built in Python and the same graph drawn in the editor
serialise identically.  ``ui`` is opaque to the runtime.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Dict, List, Mapping

from ..blocks import REGISTRY, BlockError, block_from_type
from .graph import SCHEMA_ID, Graph, GraphError, Simulation

TOP_LEVEL_KEYS = ("schema", "name", "simulation", "blocks", "edges", "robot")


def graph_to_dict(graph: Graph) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    for block in graph.declared.values():
        entry: Dict[str, Any] = OrderedDict(id=block.id, type=block.type_name, params=OrderedDict(block.params))
        if block.ui:
            entry["ui"] = dict(block.ui)
        blocks.append(entry)
    edges: List[Dict[str, Any]] = []
    for edge in graph.declared_edges:
        source = edge.declared_source or edge.source
        target = edge.declared_target or edge.target
        entry = OrderedDict([("from", source.key), ("to", target.key)])
        if edge.params:
            entry["params"] = OrderedDict(edge.params)
        edges.append(entry)
    return OrderedDict(
        schema=SCHEMA_ID, name=graph.name, simulation=graph.simulation.to_dict(),
        blocks=blocks, edges=edges, robot=OrderedDict(graph.robot),
    )


def graph_from_dict(document: Mapping[str, Any]) -> Graph:
    """Build a Graph from a file document; every problem is reported at once."""

    problems: List[str] = []
    if not isinstance(document, Mapping):
        raise GraphError(["graph file must be a JSON object"])
    if document.get("schema") != SCHEMA_ID:
        problems.append("schema must be %r, got %r" % (SCHEMA_ID, document.get("schema")))
    unknown = sorted(set(document) - set(TOP_LEVEL_KEYS))
    if unknown:
        problems.append("unknown top-level keys %s" % unknown)
    simulation_doc = document.get("simulation", {})
    if not isinstance(simulation_doc, Mapping):
        problems.append("simulation must be an object")
        simulation_doc = {}
    known_sim = set(Simulation().to_dict())
    unknown_sim = sorted(set(simulation_doc) - known_sim)
    if unknown_sim:
        problems.append("unknown simulation keys %s" % unknown_sim)
    try:
        simulation = Simulation(**{k: v for k, v in simulation_doc.items() if k in known_sim})
    except TypeError as exc:
        problems.append("simulation: %s" % exc)
        simulation = Simulation()
    graph = Graph(str(document.get("name", "")), simulation=simulation)
    robot = document.get("robot")
    if robot is not None:
        if not isinstance(robot, Mapping):
            problems.append("robot must be an object")
        else:
            graph.robot = dict(robot)

    blocks_doc = document.get("blocks", [])
    if not isinstance(blocks_doc, list):
        problems.append("blocks must be a list")
        blocks_doc = []
    for index, entry in enumerate(blocks_doc):
        if not isinstance(entry, Mapping) or "id" not in entry or "type" not in entry:
            problems.append("blocks[%d]: needs 'id' and 'type'" % index)
            continue
        params = entry.get("params", {})
        if not isinstance(params, Mapping):
            problems.append("block %r: params must be an object" % entry["id"])
            continue
        extra = sorted(set(entry) - {"id", "type", "params", "ui"})
        if extra:
            problems.append("block %r: unknown keys %s" % (entry["id"], extra))
        try:
            block = block_from_type(str(entry["type"]), str(entry["id"]), params)
        except BlockError as exc:
            problems.append(str(exc))
            continue
        ui = entry.get("ui")
        if ui is not None:
            if isinstance(ui, Mapping):
                block.ui = dict(ui)
            else:
                problems.append("block %r: ui must be an object" % entry["id"])
        try:
            graph.add(block)
        except GraphError as exc:
            problems.extend(exc.problems)

    edges_doc = document.get("edges", [])
    if not isinstance(edges_doc, list):
        problems.append("edges must be a list")
        edges_doc = []
    for index, entry in enumerate(edges_doc):
        if not isinstance(entry, Mapping) or "from" not in entry or "to" not in entry:
            problems.append("edges[%d]: needs 'from' and 'to'" % index)
            continue
        params = entry.get("params", {})
        if not isinstance(params, Mapping):
            problems.append("edge %s -> %s: params must be an object" % (entry["from"], entry["to"]))
            continue
        refs = []
        for key in ("from", "to"):
            try:
                refs.append(graph.declared_ref(str(entry[key])))
            except GraphError as exc:
                problems.extend(exc.problems)
        if len(refs) != 2:
            continue
        graph.connect(refs[0], refs[1], **dict(params))
    if problems:
        raise GraphError(problems)
    return graph


def dumps(graph: Graph) -> str:
    return json.dumps(graph_to_dict(graph), indent=2) + "\n"


def loads(text: str) -> Graph:
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise GraphError(["not valid JSON: %s" % exc])
    return graph_from_dict(document)


def save_graph(graph: Graph, path: str) -> str:
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(dumps(graph))
    return path


def load_graph(path: str) -> Graph:
    with open(path, "r", encoding="utf-8") as stream:
        return loads(stream.read())


def block_types() -> List[str]:
    return list(REGISTRY)


__all__ = ["block_types", "dumps", "graph_from_dict", "graph_to_dict", "load_graph", "loads", "save_graph"]
