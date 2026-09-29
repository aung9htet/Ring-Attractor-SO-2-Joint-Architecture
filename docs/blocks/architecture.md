# Block architecture

Branch `blocks-refactor`, phases 1–6 (see `progress.md`). This page is the map;
the block reference is `blocks.md`, the file format `graph-schema.md`, the
measurements `equivalence.md` and `feedback.md`.

## Layers

```text
 graph file (*.graph.json)  <──>  Graph (Python API)  <──>  editor (browser, dashboard server)
                                      │ build / compile
                                      v
              blocks (typed ports, ParamSchemas)  ──wrap──>  nest/populations.py builders
                                      │
                                      v
   GraphNestEngine + robot engine + generated transceivers  ──>  FTILoop (cosim)  ──>  TrialRecord
```

- **Primitives** (`tiago_ring_controller/blocks/`): `Ring`, `FourierReadout`,
  `Homeostasis`, `Gain`, `Encoder`, `FeatureEncoder`, `Decoder`,
  `ProfileDecoder`, `Joint`, `Goal`, `Probe`, `SignedProduct`, `OutputRing`,
  `TaskGain` (stub). Composites `Transport` (Homeostasis + Gain) and
  `JointTriple` (target / belief / actual rings, two transports) expand into
  primitives when added to a graph. Every parameter is a `ParamSpec` (type,
  default, bounds, unit, doc); defaults equal the checked-in `config/` files.
- **Ports** are `SPIKES` (NEST node collections, wired at build time) or
  `SIGNAL` (a value per tick). A spike input is *build-bound* (its source must
  be built first, e.g. a readout on its ring) or *late-bound* (connected after
  both blocks exist; the target owns the pattern: `Ring.stim` accepts encoder
  generators and the gain's shifted ±1 feedback). Build order is topological
  over build-bound edges, so feedback cycles are legal.
- **Graph** (`graph/graph.py`): `add`, `connect`, `validate` (every problem at
  once: kinds, directions, required inputs, producer counts, unknown edge
  parameters, signal cycles within a tick, build-bound cycles), `build(ctx)`.
- **Compiler** (`graph/compile.py`): one `GraphNestEngine` for all neural
  blocks, one robot engine (`FakeRobotEngine` or `GazeboRosEngine`), and one
  transceiver per signal block. Datapack convention: one datapack per block
  named by the block id, fields = its output ports; an edge `a.x -> b.y` reads
  datapack `a` field `x`. `Joint` reads `joint_state`; the decoders' horizons
  are merged into one `arm_velocity_cmd` per tick (`commands`, all joints).
  When the single-joint motif is present a `ring_counts` view keeps the
  dashboard, the monitor and the collector record working unchanged.
- **Runtime** (`graph/run.py`, `scripts/run_graph.py`): `run_graph(graph_or_path,
  engines="fake"|"nest"|"full", goals, out_dir)`; every `TrialRecord.meta["graph"]`
  holds the resolved graph. `--dashboard` serves the existing page and the
  editor; *Run in session* compiles a drawn graph and swaps the loop between
  trials.
- **NEST engine** (`cosim/graph_engine.py`): builds the graph on reset, reads
  per-neuron `n_events` deltas per block, applies encoder angles as generator
  rate changes followed by `Cleanup`/`Prepare` (NEST reads generator rates at
  `Prepare`), raster per ring, continue mode.

## What changed under the science

Nothing in the weights, neuron parameters or feedback rules. The phase 2
optimisation replaced ~98 600 per-synapse `Connect` calls by 31 vectorised ones
(same synapse set, proven on the fake NEST) and made bumps rate changes on
build-time generators; exact spike times differ from the legacy scripts (RNG
streams, summation order), rate profiles agree within a few percent, and a new
golden is pinned (`equivalence.md`). The legacy scripts stay runnable in
`legacy/`.

## Reference architectures (templates)

| template | blocks | status |
|---|---|---|
| `two_ring_single_joint` | Ring ×2, FourierReadout ×2, Homeostasis, Gain, Encoder ×2, Decoder, Joint, Goal | tick-for-tick equal to the cosim runner on the vectorised model |
| `multi_joint_two_ring` | the above per joint, one robot engine | both joints move toward their goals on the fakes with NEST |
| `three_ring_single_joint` | `JointTriple` (T/B/A rings, three encoders, two transports), Decoder, Joint, Goal | builds and runs; produces no drive in the first ten ticks (phase 5 open item) |
| `two_joint_forward_kinematics` | Ring ×2 (N=100), Encoder ×2 (circular), SignedProduct, OutputRing ×3, ProfileDecoder ×3 | matches legacy `MultiRingDecode` within tolerance |
| `task_space_error` | the above + `TaskGain` | not built: `TaskGain` is a port-contract stub |

## Open items

- Gazebo: no phase was run against the simulation (`run_graph.py --engines
  full`, `cosim_gazebo_smoke.py --joints 5 6`, `sweep_feedback.py --engines
  full`); all fake-robot numbers must be repeated there.
- Comparator sensitivity: below about 18 ring indices of goal distance the
  decision pair produces no drive (`feedback.md` §2); the motor-signal
  experiment (`Decoder.source`) is prepared but not run.
- The signed-product layer still builds with per-cell `Connect` calls (≈5 s).
