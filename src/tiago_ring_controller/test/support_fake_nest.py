"""Small call-recording backend shared by dependency-light topology tests."""

from copy import deepcopy


class FakeNodes:
    """Minimal stand-in for a NEST ``NodeCollection``."""

    def __init__(self, ids, events=None, backend=None):
        self.ids = tuple(int(value) for value in ids)
        self.events = {} if events is None else events
        # Optional owning backend so multi-node collections built through
        # ``NodeCollection`` can resolve per-node recorder events.
        self.backend = backend

    def _events_for(self, global_id):
        if self.backend is not None:
            owner = self.backend._nodes.get(global_id)
            if owner is not None:
                return owner.events
        return self.events

    def __getitem__(self, index):
        if isinstance(index, slice):
            return FakeNodes(self.ids[index], self.events)
        return FakeNodes((self.ids[index],), self.events)

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def get(self, key):
        if key == "global_id":
            return self.ids[0] if len(self.ids) == 1 else list(self.ids)
        if key == "events":
            return self.events
        if key == "n_events":
            counts = [
                len(self._events_for(global_id).get("times", []))
                for global_id in self.ids
            ]
            return counts[0] if len(self.ids) == 1 else counts
        raise KeyError(key)

    def __repr__(self):  # pragma: no cover - used only in assertion diagnostics
        return "FakeNodes(%r)" % (self.ids,)


def node_ids(value):
    if isinstance(value, FakeNodes):
        return value.ids
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(node_ids(item))
        return tuple(result)
    if isinstance(value, int):
        return (value,)
    return value


class RecordingNest:
    """Records NEST-like calls while assigning deterministic node IDs."""

    def __init__(self):
        self.next_id = 1
        self.create_calls = []
        self.connect_calls = []
        self.kernel_calls = []
        self.simulate_calls = []
        self.run_calls = []
        self.lifecycle_calls = []
        self.status_calls = []
        self._nodes = {}

    def Create(self, model, n=None, params=None):
        count = 1 if n is None else int(n)
        ids = tuple(range(self.next_id, self.next_id + count))
        self.next_id += count
        nodes = FakeNodes(ids)
        for global_id in ids:
            self._nodes[global_id] = nodes
        self.create_calls.append(
            {
                "model": model,
                "n": n,
                "params": deepcopy(params),
                "ids": ids,
            }
        )
        return nodes

    def Connect(self, source, target, conn_spec=None, syn_spec=None):
        self.connect_calls.append(
            {
                "source": node_ids(source),
                "target": node_ids(target),
                "conn_spec": deepcopy(conn_spec),
                "syn_spec": deepcopy(syn_spec),
            }
        )

    def ResetKernel(self):
        self.kernel_calls.append(("ResetKernel", None))

    def set_verbosity(self, value):
        self.kernel_calls.append(("set_verbosity", value))

    def SetKernelStatus(self, value):
        self.kernel_calls.append(("SetKernelStatus", deepcopy(value)))

    def Simulate(self, duration):
        self.simulate_calls.append(duration)

    def Prepare(self):
        self.lifecycle_calls.append("Prepare")

    def Run(self, duration):
        self.run_calls.append(duration)
        self.lifecycle_calls.append(("Run", duration))

    def Cleanup(self):
        self.lifecycle_calls.append("Cleanup")

    def SetStatus(self, nodes, value):
        self.status_calls.append((node_ids(nodes), deepcopy(value)))

    def GetStatus(self, nodes, key):
        if key != "events":
            raise KeyError(key)
        if isinstance(nodes, FakeNodes):
            return [nodes.events]
        return [{}]

    def NodeCollection(self, ids):
        return FakeNodes(ids, backend=self)


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
