"""Fake-NEST helpers for the topology tests (the backend itself lives in the package)."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tiago_ring_controller.cosim.fake_backend import FakeNodes, RecordingNest, node_ids  # noqa: E402,F401
def _expand_weight(spec, n_source, n_target):
    """Return a ``(n_target, n_source)`` array of weights for one Connect call."""

    import numpy as np

    if spec is None or "weight" not in spec:
        return np.ones((n_target, n_source), dtype=float)
    weight = spec["weight"]
    array = np.asarray(weight, dtype=float)
    if array.ndim == 0:
        return np.full((n_target, n_source), float(array))
    if array.ndim == 1:
        # NEST one_to_one accepts a per-pair vector.
        return np.diag(array) if n_source == n_target else array.reshape(n_target, n_source)
    return array


def connection_rows(backend, calls=None):
    """Expand recorded Connect calls into ``(source_gid, target_gid, weight)`` rows.

    Scalar, ``one_to_one`` and ``all_to_all`` (with a NEST-shaped
    ``(n_target, n_source)`` weight matrix) calls are all expanded the same way
    so per-synapse and vectorised builders can be compared as sets.
    """

    rows = []
    for call in (backend.connect_calls if calls is None else calls):
        sources, targets = call["source"], call["target"]
        rule = call["conn_spec"]
        if isinstance(rule, dict):
            rule = rule.get("rule", "all_to_all")
        rule = rule or "all_to_all"
        if rule == "one_to_one":
            if len(sources) != len(targets):
                raise AssertionError("one_to_one with %d sources and %d targets" % (len(sources), len(targets)))
            weights = _expand_weight(call["syn_spec"], len(sources), len(targets))
            for index, (source, target) in enumerate(zip(sources, targets)):
                rows.append((source, target, float(weights[index, index])))
        elif rule == "all_to_all":
            weights = _expand_weight(call["syn_spec"], len(sources), len(targets))
            if weights.shape != (len(targets), len(sources)):
                raise AssertionError(
                    "weight shape %r for %d targets x %d sources" % (weights.shape, len(targets), len(sources))
                )
            for t_index, target in enumerate(targets):
                for s_index, source in enumerate(sources):
                    rows.append((source, target, float(weights[t_index, s_index])))
        else:
            raise AssertionError("unsupported rule %r" % (rule,))
    return rows


def relabel_rows(rows, labels):
    """Map ``(gid, gid, w)`` rows through ``labels[gid] -> (population, index)``."""

    return sorted((labels[s], labels[t], round(w, 9)) for s, t, w in rows)


def label_nodes(labels, population, nodes):
    """Register every id of ``nodes`` (FakeNodes or list of them) under ``population``."""

    ids = node_ids(nodes)
    for index, global_id in enumerate(ids):
        labels[global_id] = (population, index)
    return labels
