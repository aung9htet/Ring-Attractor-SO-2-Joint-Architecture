# Graph file reference (`ring-blocks/1`)

One JSON file per experiment, written by `Graph.save` or the editor, readable
by hand. The runtime never reads anything else to decide the topology.

```json
{
  "schema": "ring-blocks/1",
  "name": "two ring single joint 5",
  "simulation": {"dt_ms": 50.0, "nest_lead_steps": 4, "max_steps": 400, "rng_seed": 13579,
                 "local_num_threads": 1, "step_mode": "run", "reset_mode": "rebuild"},
  "blocks": [
    {"id": "r1", "type": "Ring", "params": {"population_size": 200, "variant": "legacy", "...": "..."}, "ui": {"x": 40, "y": 60}},
    {"id": "enc_state", "type": "Encoder", "params": {"mode": "once", "mapping": "collector", "joint_min": -1.417, "joint_max": 1.417, "...": "..."}}
  ],
  "edges": [
    {"from": "r1.spikes", "to": "f1.ring"},
    {"from": "gain.feedback", "to": "r1.stim", "params": {"weight": -0.6}}
  ],
  "robot": {"engine": "fake", "stepper": "clock_wait"}
}
```

Rules (checked by `Graph.validate` and by `POST /api/graph/validate`, all
problems reported together):

- `schema` is exactly `ring-blocks/1`; no unknown top-level or simulation keys.
- `blocks`: `id` unique (no dots or spaces), `type` from the registry
  (`blocks.md`), `params` an object of *known* parameters — unknown names are
  rejected, missing ones take the defaults; a composite (`Transport`,
  `JointTriple`) is one entry, its internals are never written. `ui` is opaque
  to the runtime (the editor stores positions there).
- `edges`: `from` an output port, `to` an input port of the same kind
  (`spikes` or `signal`); required inputs connected; at most one producer per
  input unless the port is *multi* (`Ring.stim`); edge `params` only `weight`
  and `margin` (spike edges into a ring); the signal graph is acyclic within a
  tick; build-bound spike edges are acyclic (feedback goes through late-bound
  inputs).
- `simulation`: `dt_ms > 0`, `nest_lead_steps ≥ 0`, `max_steps > 0`,
  `step_mode` run|simulate, `reset_mode` rebuild|continue, `local_num_threads ≥ 1`.
- `robot.engine`: fake|gazebo (used with `--engines full`), `robot.stepper`:
  clock_wait|plugin.

Canonical form: parameters are written resolved (every parameter, defaults
filled) in schema order, `json.dumps(indent=2)`, one trailing newline. A graph
built in Python and the same graph drawn in the editor therefore serialise
identically (tested in the browser). The runtime records the *expanded* graph
(`Graph.describe()`, composites replaced by their primitives) into every
`TrialRecord.meta["graph"]`.

Python: `Graph.load(path)`, `graph.save(path)`, `graph.to_dict()`,
`Graph.from_dict(document)`, `graph.schema.dumps/loads`. Command line:
`scripts/run_graph.py FILE --validate`, `--save OUT` (canonicalise).
