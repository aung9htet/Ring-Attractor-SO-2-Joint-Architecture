"""Backend-injected signed-product and output-ring builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..features import (
    mapped_ring_indices,
    signed_product_feature_order,
    signed_product_term_values,
)


@dataclass
class SignedProductLayer:
    feature_grid_size: int
    n_cells: int
    mapped_ring_indices: np.ndarray
    feature_order: Tuple[str, ...]
    populations: Dict[str, Dict[str, Any]]
    flat_nodes: Any


@dataclass
class OutputRings:
    populations: Dict[str, Dict[str, Any]]


def _node_global_ids(nodes: Any) -> Sequence[int]:
    global_ids = nodes.get("global_id")
    if np.isscalar(global_ids):
        return [int(global_ids)]
    return [int(global_id) for global_id in global_ids]


def build_signed_product_layer(
    backend: Any,
    q1_ring_neurons: Sequence[Any],
    q2_ring_neurons: Sequence[Any],
    population_size: int,
    feature_grid_size: int,
    dc_baseline: float,
    input_weight: float,
    epsilon: float = 1.0e-9,
    mapped_indices: Optional[np.ndarray] = None,
    term_values_factory: Optional[
        Callable[[], Mapping[str, Tuple[np.ndarray, np.ndarray]]]
    ] = None,
    feature_order: Optional[Sequence[str]] = None,
    on_population_built: Optional[Callable[[str, Any, Any], None]] = None,
) -> SignedProductLayer:
    n_cells = feature_grid_size * feature_grid_size
    mappings = (
        mapped_ring_indices(population_size, feature_grid_size)
        if mapped_indices is None
        else mapped_indices
    )
    q1_sources = [q1_ring_neurons[idx] for idx in mappings]
    q2_sources = [q2_ring_neurons[idx] for idx in mappings]
    term_values = (
        signed_product_term_values(population_size, feature_grid_size)
        if term_values_factory is None
        else term_values_factory()
    )
    populations: Dict[str, Dict[str, Any]] = {}

    for term_name, (q1_values, q2_values) in term_values.items():
        for sign in ("pos", "neg"):
            nodes = backend.Create("iaf_psc_alpha", n_cells)
            recorders = backend.Create("spike_recorder", n_cells)
            if dc_baseline > 0.0:
                dc = backend.Create("dc_generator", params={"amplitude": dc_baseline})
                backend.Connect(dc, nodes)
            for cell_idx in range(n_cells):
                backend.Connect(nodes[cell_idx], recorders[cell_idx])
            feature_name = "{}_{}".format(term_name, sign)
            populations[feature_name] = {
                "nodes": nodes,
                "recs": recorders,
            }
            if on_population_built is not None:
                on_population_built(feature_name, nodes, recorders)

        q1_only = term_name in ("cos1", "sin1")
        q2_only = term_name in ("cos2", "sin2")
        for i in range(feature_grid_size):
            a_value = q1_values[i]
            q1_source = q1_sources[i]
            for j in range(feature_grid_size):
                b_value = q2_values[j]
                product = a_value * b_value
                cell_idx = i * feature_grid_size + j
                if product > epsilon:
                    target = populations[term_name + "_pos"]["nodes"][cell_idx]
                elif product < -epsilon:
                    target = populations[term_name + "_neg"]["nodes"][cell_idx]
                else:
                    continue
                if not q2_only:
                    backend.Connect(
                        q1_source,
                        target,
                        syn_spec={"weight": float(input_weight * abs(a_value))},
                    )
                if not q1_only:
                    backend.Connect(
                        q2_sources[j],
                        target,
                        syn_spec={"weight": float(input_weight * abs(b_value))},
                    )

    order = tuple(
        signed_product_feature_order() if feature_order is None else feature_order
    )
    flat_ids = []
    for feature_name in order:
        flat_ids.extend(_node_global_ids(populations[feature_name]["nodes"]))
    flat_nodes = backend.NodeCollection(flat_ids)
    return SignedProductLayer(
        feature_grid_size=feature_grid_size,
        n_cells=n_cells,
        mapped_ring_indices=mappings,
        feature_order=order,
        populations=populations,
        flat_nodes=flat_nodes,
    )


def build_output_rings(
    backend: Any,
    layer: Any,
    weights_by_name: Mapping[str, np.ndarray],
    output_ring_size: int,
    output_dc_baseline: float,
    output_weight_scale: float,
    matrix_is_target_by_source: bool = True,
    on_population_built: Optional[Callable[[str, Any, Any], None]] = None,
    source_nodes: Any = None,
    matrix_orientation_getter: Optional[Callable[[], bool]] = None,
) -> OutputRings:
    if source_nodes is None:
        source_nodes = (
            layer["flat_nodes"] if isinstance(layer, Mapping) else layer.flat_nodes
        )
    populations: Dict[str, Dict[str, Any]] = {}
    for name in ("lift", "pitch", "yaw"):
        weights = weights_by_name[name]
        nodes = backend.Create("iaf_psc_alpha", output_ring_size)
        recorders = backend.Create("spike_recorder", output_ring_size)
        dc = backend.Create(
            "dc_generator", params={"amplitude": output_dc_baseline}
        )
        backend.Connect(dc, nodes)
        for idx in range(output_ring_size):
            backend.Connect(nodes[idx], recorders[idx])
        target_by_source = (
            matrix_is_target_by_source
            if matrix_orientation_getter is None
            else matrix_orientation_getter()
        )
        matrix = (
            (output_weight_scale * weights.T).tolist()
            if target_by_source
            else (output_weight_scale * weights).tolist()
        )
        backend.Connect(
            source_nodes,
            nodes,
            conn_spec={"rule": "all_to_all"},
            syn_spec={"weight": matrix},
        )
        populations[name] = {"nodes": nodes, "recs": recorders}
        if on_population_built is not None:
            on_population_built(name, nodes, recorders)
    return OutputRings(populations=populations)
