# Plan: block architecture, graph files and a visual editor (branch `blocks-refactor`)

Parent branch: `cosim-loop` (lock-stepped co-simulation harness, dashboard).
Date: 2026-09-29. Status: **plan, nothing implemented yet**. Phases below are meant to
be executed and ticked off in order; each ends with green tests in the container.

## 1. Goal

Turn the research package into a small set of robust, parameterised **blocks**
(Ring, Fourier readout, Homeostasis comparator, Gain, Encoder, Decoder, robot Joint,
Goal) with fixed internal topology, a **graph file** that says which blocks exist,
how they are wired and which robot joints they drive, a **runtime** that builds and
runs exactly the graph in that file through the existing co-simulation loop, and a
**visual editor** (drag and drop, in the browser) that reads and writes that file.

Everything that is not part of that pipeline moves to a `legacy/` folder and stays
runnable but unmaintained.

Non-goals for this branch: changing the science of any block (weights, neuron
parameters, feedback rules stay as they are unless a phase says otherwise), the
two-joint multi-ring orientation stack (kept legacy until the single-joint graph is
solid), hardware use.

## 2. Where we start

What exists after `cosim-loop`:

- `tiago_ring_controller/` internal package: pure math, backend-injected NEST
  topology builders (`nest/ring.py`, `readout.py`, `comparator.py`, `gain.py`,
  `multi_ring.py`), control (`DriveControlCore`, trajectories, profiles),
  evaluation, ROS contracts, `cosim/` (engines, datapacks, TFs, `FTILoop`, runner,
  monitor, dashboard).
- Flat scripts in `src/*.py`: the model facades (`ring_attractor`, `ring_component`,
  `homeostasis`, `gain_modulation`, `single_ring`), three robot workflows with their
  own loops, trainers, analyses, figure scripts, the multi-ring stack, camera and
  torque recorders.
- Golden tests that freeze the flat scripts byte-for-byte (API manifest, import
  graph, file modes, bytecode caches, install lists). They exist to protect the
  results behind the paper on `main`; they are incompatible with moving files.
- Known costs measured on `cosim-loop`: building the N=200 model takes 15–31 s
  (about 100k individual `nest.Connect` calls); stimulus injection hides 50 ms of
  simulated time per bump; robot state enters the ring once per trial.

## 3. Decisions taken by this plan (change them here if you disagree)

D1. **The compatibility freeze ends on this branch.** `main` keeps the frozen
    baseline (tag `legacy-loop-baseline`); on `blocks-refactor` the freeze tests
    (`test_api_compatibility`, `test_environment_contracts`, `test_bytecode_provenance`,
    `test_packaging_contracts`, the import-graph part) are deleted, not rewritten.
    Artifact contracts for scientific inputs under `config/` are kept.
D2. **Blocks wrap the existing builders; they do not re-derive the model.** Fixed
    topology per block is exactly what `nest/ring.py`, `readout.py`, `comparator.py`
    and `gain.py` build today. Parameters become declared, validated `ParamSpec`s.
D3. **The graph file is JSON**, one file per experiment, versioned schema, produced
    by the editor and readable by hand. The runtime never reads anything else to
    decide the topology.
D4. **The runtime is the co-simulation loop.** A compiled graph is a set of engines
    (one NEST engine holding all neural blocks, one robot engine) plus transceiver
    functions generated from the signal edges. `FTILoop`, records, dashboard and
    reset modes stay.
D5. **The editor lives in the dashboard**: same standard-library server, plain
    JavaScript and SVG, no build step, no external dependency. It is reachable from
    the host browser through the container's host networking like today.
D6. **Optimisation goes first** (phase 2) because the editor and the graph runtime
    need fast rebuilds and build-time stimulus generators. The gate for every
    optimisation is a documented equivalence test, statistical where exact spike
    parity is impossible.
D7. **Encoder becomes a real block** (build-time Poisson generators, rate set per
    tick). This is plan B's B.4 and is what closes the sensorimotor loop.

## 4. Target layout

```text
src/tiago_ring_controller/
  src/tiago_ring_controller/          # the package (importable without NEST/ROS)
    blocks/                           # phase 3
      __init__.py     registry: name -> Block class
      base.py         Block, Port, PortKind (SPIKES|SIGNAL), ParamSpec, BuildContext
      ring.py         Ring
      readout.py      FourierReadout
      comparator.py   Homeostasis
      gain.py         Gain
      encoder.py      Encoder (angle -> bump generators)
      decoder.py      Decoder (counts -> velocity, filter/delay/gains/settle)
      joint.py        Joint (robot binding), Goal (source)
      probe.py        Probe (record a signal or population)
    graph/                            # phase 4
      schema.py       dataclasses + JSON (de)serialisation + validation
      compile.py      graph -> NestNetwork (blocks built in order) + TF pipeline
      run.py          run_graph(path, engines) -> TrialRecord(s)
      examples/single_joint.graph.json   the current model, as a graph
    cosim/            unchanged runtime (engines, loop, dashboard) + GraphNestEngine
    ui/               phase 6: static editor files served by the dashboard server
    nest/, control/, math/, evaluation/, ros/, training/, config.py, contracts.py
  scripts/
    run_graph.py      run a graph file (fake | nest | full), optional --dashboard
    fit_homeostasis.py, fit_fourier.py   fitting tools that write block artifacts
    (existing cosim scripts stay)
  legacy/             phase 1: every flat script, builders/, trainer/, helpers/,
                      builders_analysis/, __pycache__/, plus a README
  config/             scientific inputs, unchanged paths
  test/               block, graph, cosim, artifact and real-NEST tests
```

## 5. The block model

A block is a Python class with:

- `name` (type name), `version`;
- `params: ParamSpec` — name, type, default, bounds, unit, doc; values live in the
  graph file, validated on load and by the editor;
- `ports` — typed, named, with a direction: `SPIKES` ports carry NEST node
  collections (neural wiring done at build time), `SIGNAL` ports carry floats or
  small vectors once per tick (datapacks);
- `build(ctx)` — creates its fixed internal topology in NEST through the existing
  backend-injected builders and returns its port node collections. Called once per
  rebuild; parameters only, never structure, may differ between two builds of the
  same block type;
- `connect(port, other_block, other_port, edge_params)` — the *pattern* for a given
  port pair is fixed by the block (e.g. `Gain.feedback -> Ring.recurrent` is the
  shifted ±1 rule excluding the edge margin); the edge carries only scalars such as
  the weight;
- `on_tick(inputs) -> outputs` for signal ports (pure Python, no NEST calls except
  device parameter updates), `read()` for spike-count outputs (NodeCollection
  `n_events` deltas, as in `cosim/nest_engine.py`).

Blocks and their fixed topology (what today's code builds):

| Block | Internal topology (fixed) | Editable params | Ports |
|---|---|---|---|
| `Ring` | N `iaf_psc_alpha`, N recorders, N² recurrent Gaussian excitation/inhibition, self-connections included; variants `legacy` / `builder` | N, neuron params, max_distance, exc/inh widths, variant | `stim` (SPIKES in, per-neuron targets), `spikes` (SPIKES out), `counts` (SIGNAL out) |
| `FourierReadout` | 2K `iaf_psc_alpha` + recorders + DC generator, analytic sine masks | K, weight scale, DC baseline, mask source (analytic or artifact) | `ring` (SPIKES in), `features` (SPIKES out) |
| `Homeostasis` | warm/cold/left/right `iaf_psc_alpha`, six fixed lateral weights, ridge-fitted feature weights | five weights, neuron params, weight scale, fitted weights artifact per N | `state_features`, `target_features` (SPIKES in), `left`, `right` (SPIKES out) |
| `Gain` | left/right populations of N, same-index ring input, decision input, cross-inhibition, shifted feedback with margin 5 | five weights, neuron params, margin | `ring` (SPIKES in), `left_in`, `right_in` (SPIKES in), `feedback` (SPIKES out), `left_counts`, `right_counts` (SIGNAL out) |
| `Encoder` | N Poisson generators (rate 0) connected one-to-one to the target ring | half width, rate, weight, duration in ticks, angle→index mapping (limits, margin mode), mode once/continuous | `angle` (SIGNAL in), `stim` (SPIKES out) |
| `Decoder` | none (pure) | spike scale, tau, delay, gains ±, drive threshold, settle count, horizon length | `left_counts`, `right_counts` (SIGNAL in), `velocity` (SIGNAL out), `settled` (SIGNAL out) |
| `Joint` | none (robot engine) | joint index, limits (URDF or calibration) | `angle` (SIGNAL out), `velocity` (SIGNAL in) |
| `Goal` | none | constant angle, or a schedule | `angle` (SIGNAL out) |
| `Probe` | optional recorders | what to log | any in |

Signal edges have an implicit one-tick delay (the loop's semantics); spike edges are
NEST synapses with the block's pattern. Cycles among spike edges are normal
(feedback); the signal graph must be acyclic within a tick.

## 5b. Programmatic use (the primary interface)

The Python API comes first; the graph file serialises it and the editor edits it.
Nothing is editor-only.

```python
from tiago_ring_controller.blocks import Ring, FourierReadout, Homeostasis, Gain, Encoder, Decoder, Joint, Goal
from tiago_ring_controller.graph import Graph, run_graph

g = Graph(dt_ms=50, nest_lead_steps=4, rng_seed=13579)
r1 = g.add(Ring("r1", population_size=200))
enc = g.add(Encoder("enc", half_width=5, mode="continuous"))
j6 = g.add(Joint("j6", index=5))
g.connect(j6.angle, enc.angle)
g.connect(enc.stim, r1.stim)
# ... remaining blocks and edges as in section 6
g.validate()
g.save("my_experiment.graph.json")      # the file the editor reads and writes
record = run_graph(g, engines="fake")   # "nest" or "full" for the simulators
```

Requirements that follow, checked by tests in phases 3 and 4:

- `Graph` round-trips through the file without loss; a graph built in Python and
  the same graph drawn in the editor serialise identically.
- Every block can be built alone in a bare NEST session (`block.build(ctx)` returns
  its port node collections) or against the fake NEST, without loop or robot.
- Sweeps, batch sessions and the fitting tools are plain Python over `Graph`
  objects; `run_graph` returns the same `TrialRecord`s the dashboard produces.

## 5c. Robot feedback path

Today the output side is defined (`Decoder.velocity -> Joint.velocity`, published as
the receding-horizon trajectory) but the feedback side is not: the measured joint
enters the ring once, at trial start, so the trial runs open loop.

Mechanism (already provided by the loop): the robot engine publishes measured
angles and velocities for all joints every tick; a `Joint` block exposes its angle
as a signal port; `Encoder` maps the angle to a ring index with the profile mapping
(limits, margin) and drives the build-time Poisson generators of the target ring.
Wiring is `Joint.angle -> Encoder.angle -> Ring.stim`, once per tick with the one-tick
delay; `Goal.angle -> Encoder.angle -> Ring.stim` encodes the target the same way, so
moving goals need no extra block. With several joints, the engine merges every
commanded joint into the single trajectory message it already sends.

Semantics (to be characterised, not assumed). The bump in r1 is not a sensor reading;
it is the state estimate the gain feedback transports, and its motion is the motor
command. A continuous stimulus competes with that transport. `Encoder.mode`:

| mode | behaviour | role |
|---|---|---|
| `once` | inject at trial start only | legacy parity |
| `continuous` | every tick, rate `r` on a window of half-width `h` around the measured index | observer correction; `r` is the sensor-vs-prediction weight, small `r` trusts the ring, large `r` slaves the bump to the joint |
| `corrective` | stimulate only when decoded centroid and measured index differ by more than a dead-band | easier to stabilise; the comparison is computed in the TF, not by neurons |

Consequences carried by the plan:

- Closed-loop graphs run with `nest_lead_steps` 0 or 1 and a one-point horizon; the
  four-step lead feeds the ring a 200 ms old state and stays a legacy option.
- Optional `Joint` output ports for later blocks: measured velocity, effort (the
  wrist FT stream is already subscribed), so other feedback designs can be tried
  without touching the engine.
- Phase 5 gate: with the fake robot and then in Gazebo, sweep `r` and `h` for
  `continuous` and the dead-band for `corrective`; report tracking error, bump-to-joint
  lag and settle time against `once`; the example graph's default mode and gain are
  chosen from that report (`docs/blocks/feedback.md`).

## 6. Graph file (schema v1)

```json
{
  "schema": "ring-blocks/1",
  "name": "single joint 6, collector profile",
  "simulation": {"dt_ms": 50, "nest_lead_steps": 4, "max_steps": 400,
                 "rng_seed": 13579, "step_mode": "run", "reset_mode": "rebuild"},
  "blocks": [
    {"id": "enc_state", "type": "Encoder", "params": {"half_width": 5, "mode": "once", ...}, "ui": {"x": 40, "y": 120}},
    {"id": "r1", "type": "Ring", "params": {"population_size": 200, ...}, "ui": {...}},
    {"id": "r2", "type": "Ring", "params": {...}},
    {"id": "f1", "type": "FourierReadout", "params": {"num_fourier_k": 20, ...}},
    {"id": "f2", "type": "FourierReadout", "params": {...}},
    {"id": "cmp", "type": "Homeostasis", "params": {...}},
    {"id": "gain", "type": "Gain", "params": {...}},
    {"id": "dec", "type": "Decoder", "params": {"tau_s": 0.51, "gain_positive": 0.0011, ...}},
    {"id": "j6", "type": "Joint", "params": {"index": 5}},
    {"id": "goal", "type": "Goal", "params": {"angle_rad": 0.6}}
  ],
  "edges": [
    {"from": "j6.angle", "to": "enc_state.angle"},
    {"from": "enc_state.stim", "to": "r1.stim"},
    {"from": "goal.angle", "to": "enc_goal.angle"}, {"from": "enc_goal.stim", "to": "r2.stim"},
    {"from": "r1.spikes", "to": "f1.ring"}, {"from": "r2.spikes", "to": "f2.ring"},
    {"from": "f1.features", "to": "cmp.state_features"}, {"from": "f2.features", "to": "cmp.target_features"},
    {"from": "cmp.left", "to": "gain.left_in"}, {"from": "cmp.right", "to": "gain.right_in"},
    {"from": "r1.spikes", "to": "gain.ring"}, {"from": "gain.feedback", "to": "r1.stim", "params": {"weight": -0.6}},
    {"from": "gain.left_counts", "to": "dec.left_counts"}, {"from": "gain.right_counts", "to": "dec.right_counts"},
    {"from": "dec.velocity", "to": "j6.velocity"}
  ],
  "robot": {"engine": "gazebo", "stepper": "clock_wait"}
}
```

Rules: block ids unique; every edge names existing ports with matching kinds and
directions; required input ports connected; at most one producer per signal port;
unknown params rejected, missing ones defaulted; `ui` is opaque to the runtime. A
`validate` command reports all problems at once. The runtime records the resolved
graph (defaults filled) into every `TrialRecord.meta`.

## 7. Phases

### Phase 0 — plan and branch (this document)

Deliverable: this file, reviewed. Decide the open questions in section 9.

### Phase 1 — legacy folder and test policy (½–1 day)

- `git mv` every flat script, `builders/`, `builders_analysis/`, `trainer/`,
  `helpers/` and their `__pycache__/` into `src/tiago_ring_controller/legacy/` with
  a README stating they are frozen, unmaintained, and run only with
  `PYTHONPATH=legacy:src` from `legacy/`.
- Delete the freeze tests listed in D1 and their golden manifests; keep
  `test_artifact_contracts` (paths under `config/` do not move),
  `test_output_schemas`, the fake-NEST topology tests, the real-NEST smoke test,
  all cosim tests. Update `setup.py` (packages only, no `py_modules`),
  `CMakeLists.txt` (install `scripts/`, not the legacy scripts), the READMEs.
- Gate: container suite green; `python3 legacy/single_ring.py` still runs from
  `legacy/`; `scripts/run_cosim_trial.py --engines nest` still runs (it imports
  `single_ring` from the legacy folder until phase 4 replaces it).

### Phase 2 — optimisation and robustness pass (2–3 days)

Each item: measure before, change, measure after, add the equivalence test.

1. **Vectorised ring build.** Replace the N² loop of `nest.Connect` calls with one
   `Connect(pre, post, "all_to_all", syn_spec={"weight": W})` using the weight
   matrix computed in NumPy (`math/ring.py` already has the formulas). Same for
   readout (`one_to_one` / matrix), gain (`one_to_one`) and comparator. Target:
   N=200 model in under 3 s (today 15–31 s).
2. **Build-time stimulus generators** (plan B.4): one `poisson_generator` per ring
   neuron at rate 0; a bump is `SetStatus` on a window. Removes the hidden
   `Simulate(50)`, allows one `Prepare` per trial and continuous encoding.
3. **Recorders**: one `spike_recorder` per population with `n_events` reset per
   tick, versus today's one-per-neuron. Keep whichever measures faster at 20 Hz;
   per-neuron deltas must remain available for the raster and the bump index.
4. **Parameter specs and validation** for every block parameter (types, bounds,
   units), with a `params.json` round-trip; neuron parameter sets validated against
   NEST's model defaults at build time.
5. **Threads and seeds**: `local_num_threads` exposed, seed recorded, determinism test
   (same seed → identical record) kept.
6. **Equivalence gates.** The optimised builders change the order in which NEST
   creates connections, so exact spike times may differ from the pinned goldens.
   Gate: fake-NEST manifests prove the *set* of connections and weights is
   identical; real-NEST tests compare rate profiles (bump position, width, gain
   left/right rates, comparator counts) over 300 ms with tolerances, then a **new
   pinned golden** is recorded and documented in `docs/blocks/equivalence.md`. If
   an item does reproduce exact spikes, say so there.
7. Robustness: build errors carry the block id; every block has a unit test with
   the fake NEST, a real-NEST smoke and a parameter-range test.

### Phase 3 — block API (2 days)

- `blocks/base.py` and the eight blocks of section 5, wrapping the existing
  builders (`nest/*.py`) and control code (`control/`, `math/`). `Encoder` uses the
  phase-2 generators; `Decoder` wraps `DriveControlCore`; `Joint`/`Goal` are thin
  signal blocks bound to the robot engine and to the TFs.
- A `GraphNestEngine` in `cosim/` that holds any number of neural blocks (today's
  `NestEngine` holds the fixed `SingleRingModel`) and reads their `counts` ports.
- Gate: building the single-joint architecture from blocks by hand (Python, no
  graph file) reproduces the phase-2 golden.

### Phase 4 — graph schema, compiler, run from file (2 days)

- `graph/schema.py`, `graph/compile.py`, `graph/run.py`, `scripts/run_graph.py`,
  `examples/single_joint.graph.json`.
- `run_graph.py --engines fake|nest|full --dashboard` runs the file; the dashboard
  shows the graph name and lets you start/stop/reset as today.
- The three legacy workflows are *not* ported; their outputs (collector layout) are
  produced by `cosim/runner.py` from graph runs, as now.
- Gate: `run_graph.py examples/single_joint.graph.json --engines nest` equals the
  phase-3 hand-built result tick for tick; validation errors are tested; a graph
  with two rings driving two joints builds and runs on the fakes.

### Phase 5 — robot binding and multi-joint (1–2 days)

- `Joint` blocks map to joints of the one Gazebo engine; the engine publishes one
  trajectory with all commanded joints per tick (the message already carries all
  seven). Limits from the URDF (`cosim/limits.py`) by default.
- Gate: Gazebo smoke with two joints commanded from two decoders on the fakes and
  in Gazebo (phase-3 plan-B smoke extended).

### Phase 6 — visual editor (3–4 days)

- Dashboard gets a *Graph* view: palette of block types (from the registry, with
  their `ParamSpec`s), SVG canvas with drag-and-drop blocks, ports as handles,
  edges drawn by dragging between compatible ports (kind and direction checked in
  the browser and again on the server), a parameter panel per block with typed
  fields, bounds and units, validation messages, *Load* / *Save* (writes the graph
  file through `POST /api/graph`), *Run* (compiles and hot-swaps the loop between
  trials). Block positions are stored in `ui`.
- Live overlay: while a trial runs, the ring blocks show their bump and the signal
  edges their last value, using the existing event stream.
- Gate: a graph drawn from an empty canvas and saved is byte-identical to a graph
  written by hand with the same content; opening the example file shows the
  single-joint architecture; headless-Chrome screenshot test in the suite.

### Phase 7 — documentation, cleanup, PR (1 day)

- `docs/blocks/`: architecture, graph schema reference, block reference generated
  from the `ParamSpec`s, equivalence report, how to add a block.
- Update `architecture.md`, `robot_safety.md`, both READMEs; retire plan-B phase 7
  items that are now done; open the PR to `main` with the legacy tag noted.

Rough total: 12–15 working days.

## 8. Testing strategy

- **Unit (no simulators)**: every block against `support_fake_nest.RecordingNest`
  (topology manifests: node counts, connection lists, weights); `ParamSpec`
  validation; graph schema round-trips and validation errors; compiler output
  (which TFs, which engines); dashboard/editor API.
- **Real NEST (container)**: per-block smoke with pinned counts (re-baselined once
  in phase 2); single-joint graph parity with the hand-built model; determinism.
- **Gazebo (container, manual gate)**: `cosim_gazebo_smoke.py` extended to graph
  runs; one recorded trial per phase that touches the robot path.
- **Browser**: headless-Chrome screenshot and console check of dashboard and
  editor, as already done for the dashboard.
- Container command stays `python3 -B -m unittest discover -s test -p "test_*.py"`.

## 9. Open questions (answer before phase 1)

Q1. D1 deletes the freeze tests on this branch. Alternative: keep them pointing at
    `legacy/` paths. Recommendation: delete; `main` and the tag keep the baseline.
Q2. Exact-spike goldens after phase 2: accept new pinned values (recommended) or
    require bit-exact reproduction of the legacy builders (rules out vectorised
    `Connect` if NEST's summation order changes results)?
Q3. Multi-ring two-joint orientation stack: legacy only for now (recommended), or
    blocks in this pass (adds SignedProduct and OutputRing blocks, +2 days)?
Q4. Editor stack: vanilla JS/SVG in the dashboard (recommended, no dependencies in
    the image) or a vendored graph library (single file, faster to build, one more
    licence to carry)?
Q5. Should the `Joint` block support hardware later? It changes nothing now, but
    the safety notes would need a per-block interlock design.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Vectorised build changes spike-level results | statistical gates + re-baselined goldens (Q2) |
| Editor scope creep | phase 6 gate is a fixed feature list; no undo/redo, no zoom, no groups in v1 |
| Legacy scripts silently break | they are frozen and documented as such; one smoke test runs `legacy/single_ring.py` in the container |
| Graph cycles in signal edges | rejected at validation; spike cycles allowed |
| Encoder as generators changes RNG streams vs. legacy injection | documented in the phase-2 equivalence report; seeds recorded per trial |
| Two engines out of step with more joints | unchanged: one robot engine, one NEST engine, the loop's clock check |
