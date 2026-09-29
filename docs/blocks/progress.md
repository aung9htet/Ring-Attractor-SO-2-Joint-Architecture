# Progress: block architecture (branch `blocks-refactor`)

Dated entries per phase of `plan.md`. Each entry records what changed, the gate
results with the actual command output, and what was not run.

## 2026-09-29 — Phase 1: legacy folder and test policy

Open questions of plan section 9 were taken as answered by the hand-over brief:
Q1 delete the freeze tests (D1 restated as not to re-open), Q2 accept new pinned
goldens after phase 2 (the brief says "keep that parity test passing until the
phase 2 re-baseline"), Q4 vanilla JS/SVG (D5 restated), Q5 no hardware joints in
this pass (plan non-goal "hardware use"; a per-block interlock is not designed).

### Changes

- `git mv` of every flat script, `builders/`, `builders_analysis/`, `helpers/`,
  `trainer/`, the recovered `__pycache__/` folders and the entrypoint guide
  (`src/README.md`) into `src/tiago_ring_controller/legacy/`. The scripts are
  byte-identical (rename-only diff). `legacy/config` is a symlink to `../src/config`
  so the historical `./config/...` defaults resolve when a script runs from
  `legacy/`; `legacy/README.md` carries the frozen notice and the run recipe
  (`PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH`, appended so NEST's own path survives).
  `src/Neurips_2026.pdf` stays where it is.
- Deleted the freeze tests and their manifests: `test_api_compatibility`,
  `test_environment_contracts`, `test_bytecode_provenance`,
  `test_packaging_contracts`; `api_manifest`, `import_graph_manifest`,
  `bytecode_manifest`, `orphan_bytecode_manifest`, `environment_manifest`,
  `checked_output_manifest`. `test_output_schemas` lost the PNG byte freeze, the
  checked-in `fourier_results.npz` check and the external-world hash; it keeps the
  configuration, calibration, local world and launch contracts. `test_ring_math`
  lost the two import-shim tests that ran `builders/ring_attractor.py` from `src/`
  without the package on the path (the shim now points at `legacy/`, where the
  package is not). `test_artifact_contracts` is scoped to `src/config/**` and
  `worlds/*.world`; the two golden manifests were pruned to those entries
  (34 → 23 hashed files, 32 → 22 schemas), nothing re-hashed.
- Kept tests repointed at `legacy/`: `test_control_ros_contracts`,
  `test_multi_ring_contracts`, `test_ring_math`, `test_evaluation_facade_plots`,
  `test_real_nest_smoke`, `test_cosim_real_nest`.
- Package: `config.legacy_root()`; `cosim/runner.make_nest_engine` imports
  `single_ring` from there; `scripts/measure_legacy_tick.py` likewise.
- Packaging: `setup.py` lists the package only (no `py_modules`); `CMakeLists.txt`
  installs `scripts/*.py` plus `legacy/record_experiments.py` (the wrist FT logger
  that `tiago_ring_controller.launch` starts); the devel-space `env-hooks/`
  PYTHONPATH hook for flat modules is gone. `package.xml` unchanged.
- Untracked generated results: `src/outputs/` (34 MB), `results/`, `outputs/`
  (incl. the nine `fourier_results.npz` copies) and the 23 recovered `.pyc` files
  inside the package. Files stay on disk; `.gitignore` rewritten (package
  bytecode ignored, legacy bytecode kept, legacy working-directory outputs ignored).
  `docker/check_recovery.py` will report these as differences, as documented.
- READMEs: package README describes the layout, the container test command and the
  legacy run recipe; cosim commands now run from the package root.

### Gates

Host (`/usr/bin/python3` 3.13, `-B`, `unittest discover`): before the change
178 tests, 66 failures + 1 error, all in the deleted freeze tests plus
`test_evaluation_facade_plots` (no `colorcet` on the host). After:

```
Ran 162 tests in 6.128s
FAILED (errors=1, skipped=2)      # test_evaluation_facade_plots: No module named 'colorcet' (host only, pre-existing)
```

Container (`aung9htet/ubuntu-20.04:tiago_ring_forward`, package bind-mounted,
`python3 -B -m unittest discover -s test -p "test_*.py"`), previously 178 tests
with 59 pre-existing inventory failures:

```
Ran 170 tests in 73.796s
OK
```

`scripts/run_cosim_trial.py --engines nest --seed 13579 --goal 0.6` from the
package root in the container (imports `SingleRingModel` from `legacy/`):

```
trial 1 timing: reset [nest 12.3 s, robot 0.0 s]; lead 0.06 s; 177 ticks in 1.3 s (7.5 ms/tick)
trial 1: goal=0.6000 start=0.0000 final=0.4628 |err|=0.1372 steps=177 stop=drive_settled tick_wall_ms(mean=7.54 max=17.28)
```

`cd legacy && PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH python3 -B single_ring.py`
in the container (100 s of simulated time, figures under `legacy/outputs/single_ring/`):

```
exit=0   real 16m29.729s  user 260m43.119s   (one 100 s run, 10 repeat runs, 10x10 goal sweep)
legacy/outputs/single_ring/: goal_distance_mean_std_across_runs.{csv,png}, raster.png,
  relative_goal_error_boxplot.png, relative_goal_error_per_run.csv, relative_goal_error_summary.csv,
  ring_activity_heatmap.png, ring_centroid_change_over_time.png
```

The first attempt failed with `No module named 'nest'` because `export
PYTHONPATH=$PWD:$PWD/../src` replaced the path that `nest_vars.sh` sets; the
READMEs append `:$PYTHONPATH` for that reason.

Not run: Gazebo (needs `./run_model_docker.bash` with the simulation launched; no
robot code changed in this phase) and the browser dashboard (no dashboard code
changed; its end-to-end test with the fakes is part of the suite above).

## 2026-09-29 — Phase 2: optimisation and robustness pass

Full report with the measured numbers: `equivalence.md`.

### Changes

- `math/ring.py`: `ring_weight_matrix` (the legacy per-synapse weights as a
  `(pre, post)` matrix, both variants).
- `nest/populations.py`: vectorised builders (`Population` = neuron collection +
  one recorder per neuron; ring, Fourier readout, decision circuit with feature
  projections, gain populations with cross-inhibition and shifted feedback,
  build-time Poisson generators with `set_bump` / `clear_stimulus`);
  `validate_neuron_parameters` against `GetDefaults`; `BuildError` names the
  population. The per-synapse builders the legacy scripts import are untouched.
- `nest/single_ring.py`: `load_single_ring_artifacts` (all `config/` inputs
  validated up front) and `build_single_ring_network` (seed and threads recorded,
  `describe()`, `set_bump` / `clear_bumps`, recorder lists in the facade's shape).
- `blocks/params.py`: `ParamSpec`, `ParamSchema`, schemas for the eight primitives,
  JSON round trip, all problems of a block reported at once.
- cosim: `GeneratorStimulusPort` (bumps at any tick, no hidden time, windows expire
  after `duration_ms` of loop time, `NestEngine.recalibrate()` = Cleanup/Prepare
  after a rate change because NEST reads generator rates at Prepare);
  `RingModelPorts.from_network`; `CosimConfig.nest_model` (`legacy` default,
  `vectorised`); `run_cosim_trial.py --model`; `scripts/benchmark_build.py`.
- Tests: `test_vectorised_topology.py` (fake-NEST connection-set equivalence per
  builder and for the whole N=200 model, 11 tests), `test_block_params.py` (5),
  generator-port tests in `test_cosim_engines.py` (3), `test_vectorised_real_nest.py`
  (rate-profile equivalence, determinism + build time, pinned golden, mid-trial
  bump; 4, container only). Golden: `test/golden/vectorised_single_ring_seed13579.json`.

### Gates

Host (`/usr/bin/python3` 3.13):

```
Ran 181 tests in 7.171s
FAILED (errors=1, skipped=3)      # test_evaluation_facade_plots: No module named 'colorcet' (host only, pre-existing)
```

Container (full suite; the pinned-golden, equivalence, determinism and mid-trial
tests ran, as did the legacy parity test `test_cosim_real_nest.py` unchanged):

```
Ran 193 tests in 87.411s
OK
EQUIVALENCE {'r1_l1': 0.049, 'r2_l1': 0.051, ...}
MIDTRIAL {'centroid_before': 60.0, 'centroid_after': 104.43, 'window_spikes': 209.0}
```

Build time N=200 (`scripts/benchmark_build.py`, container, 1 thread, min of 3):
legacy 12.01 s → vectorised 0.041 s (target was < 3 s). Per tick: `Run(50)` 5.8 ms,
readout of r1 + left + right 1.6 ms.

`run_cosim_trial.py --engines nest --model vectorised --seed 13579 --goal 0.6`:

```
trial 1 timing: reset [nest 0.1 s, robot 0.0 s]; lead 0.04 s; 203 ticks in 1.4 s (7.0 ms/tick)
trial 1: goal=0.6000 start=0.0000 final=0.5917 |err|=0.0083 steps=203 stop=drive_settled
```

Same with `--proprioception continuous` (a 200 Hz state bump every tick):

```
trial 1: goal=0.6000 start=0.0000 final=-0.0840 |err|=0.6840 steps=400 stop=max_steps
```

The continuous stimulus at the legacy rate pins the state bump to the measured
joint and the loop does not converge; this is exactly the phase 5 sweep
(`Encoder.mode`, rate, half width, dead band), not a defect of the port. Recorded
here so the phase 5 baseline is known.

Item 3 (recorders): measured both options, kept per-neuron recorders (flat cost,
per-neuron deltas); numbers in `equivalence.md`.

Not run: Gazebo (no robot code changed) and the browser dashboard (unchanged).

## 2026-09-29 — Phase 3: block API

### Changes

- `blocks/base.py`: `Block` (type name, `ParamSchema`, typed `Port`s, `build(ctx,
  inputs)`, `connect_late`, `counts_sources`, `describe_type`), `PortKind`
  SPIKES/SIGNAL, `PortRef` (`ring.stim`), `Connection`, `BuildContext` (backend,
  config dir, seed, threads, neuron-parameter loader, build log), `Composite`
  (`expand()` → sub-blocks, internal edges, port map). Spike inputs are
  *build-bound* (source built first: readout ← ring) or *late-bound* (wired
  after both exist, pattern owned by the target: `Ring.stim` ← encoder
  generators / gain feedback), which is how feedback cycles build.
- Primitives, each wrapping the phase-2 builders: `Ring`, `FourierReadout`
  (analytic masks by default, identical to the artifacts), `Homeostasis` (fitted
  artifact keyed by the readout's ring size), `Gain` (`connect_stim` implements
  the shifted ±1 feedback; the edge may override weight and margin), `Encoder`
  (generators created when connected to a ring; modes once / continuous /
  corrective with dead band; `drive`, `expire`, `drive_index`), `FeatureEncoder`
  (2K push-pull generators for a reference with no ring), `Decoder` (wraps
  `DriveControlCore`; `step` equals `MotorTF` sample for sample; sources
  gain_counts / centroid_velocity / decision_counts), `ProfileDecoder`
  (sawtooth / centroid / scalar ramp), `Joint`, `Goal` (constant or schedule),
  `Probe`, `SignedProduct` and `OutputRing` (wrap `nest/multi_ring.py`; one
  output ring per block, artifact key per block), `TaskGain` (port contract
  only; `build` raises). Composites `Transport` (Homeostasis + Gain) and
  `JointTriple` (T/B/A rings, two encoders, two transports). `blocks.REGISTRY`,
  `block_from_type`, `describe_types()` (the editor palette). Schema defaults now
  equal the config files (Homeostasis, Gain, readout, ring; tested).
- `graph/graph.py`: `Graph` (`add`, `connect`, composites expanded on `add`,
  `problems()`/`validate()` reporting everything at once: kinds, directions,
  required inputs, producer counts, unknown edge parameters, signal cycles,
  build-bound cycles; `build_order()`; `build(ctx)` → `BuiltGraph`).
- `cosim/graph_engine.py`: `GraphNestEngine` holds any graph: inputs
  `"<encoder>.angle"`, one output datapack per neural block (ring: per-neuron
  deltas, total, bump index, centroid; gain: left/right; comparator:
  warm/cold/left/right; conjunction/output rings: per-cell deltas), generator
  rate changes followed by Cleanup/Prepare, raster per ring, continue mode.
- Tests: `test_blocks_fake.py` (15: hand-built graph creates the vectorised
  network with the *same node creation sequence and the same Connect calls in
  the same order*; Transport composite equals the primitives; validation
  messages; block errors name the block; schema defaults vs config; registry
  descriptions; encoder modes and expiry; Decoder vs MotorTF; Goal/Joint/Probe/
  ProfileDecoder; SignedProduct + OutputRing on the fake with a temporary
  artifact; JointTriple expands to 12 blocks and builds; engine datapacks and
  recalibration), `test_graph_real_nest.py` (2, container).

### Gates

Phase 3 gate, container: the single-joint architecture built by hand from
blocks (no graph file) reproduces the phase-2 golden **exactly** (per-neuron
r1/r2/left/right counts and decision counts, seed 13579), because the block
build order yields the same NEST node ids as `build_single_ring_network`.
`GraphNestEngine` runs the graph with once-mode encoders and accepts a
continuous proprioceptive encoder mid-trial (window driven, recalibrations per
tick).

```
Ran 210 tests in 103.139s   (container, full suite; one failure fixed below)
Ran 2 tests in 0.286s  OK   (test_graph_real_nest.py after the fix)
```

The one failure was the test's own naive angle→index formula (62 ≠ 60); it now
inverts the encoder's collector mapping by search. Host (`/usr/bin/python3`
3.13): 196 tests, only the pre-existing `colorcet` import error.

Not run: Gazebo and the dashboard (unchanged).

## 2026-09-29 — Phase 4: graph schema, compiler, run from file, templates

### Changes

- `graph/schema.py`: JSON file (`ring-blocks/1`): blocks as declared (a composite
  is one entry, its internals are not written), edges as `"block.port"`,
  parameters written resolved (defaults filled) so Python and editor output are
  byte-identical, `ui` opaque; every file problem reported at once (schema id,
  unknown keys, unknown types/params, duplicate ids, bad port references, both
  ends of an edge). `Graph.save/load/to_dict/from_dict`; `Graph.declared` and
  `declared_edges` keep the file's view next to the expanded one.
- `graph/compile.py`: `compile_graph(graph, engines="fake"|"nest"|"full")` →
  `GraphNestEngine` + robot engine (fake, or Gazebo from `robot.engine`) +
  transceivers generated from the signal blocks (`GoalBlockTF`,
  `JointSensorTF`, `DecoderTF`, `JointCommandTF` → `arm_velocity_cmd` in the
  MotorTF schema, `ProfileDecoderTF`, `ProbeTF`) + `LegacyViewTF` (`ring_counts`
  for the dashboard, monitor and collector record when the single-joint motif
  is present) + `FTILoop` with the legacy stop order. Datapack convention: one
  datapack per block, named by the block id, fields = its output ports; an edge
  `a.x -> b.y` reads datapack `a` field `x`. `graph_cosim_config` derives the
  `CosimConfig` (joint, limits, profile, decoder, lead, seed) so `TrialWriter`,
  `legacy_collector_record` and `run_dashboard_session` work unchanged.
- Loop change: transceivers now see the datapacks produced earlier in the same
  tick (`FTILoop._transceive` passes a merged view), so signal chains
  sensor → decoder → command run within a tick (the signal graph is acyclic by
  validation). Existing transceivers only read engine datapacks; unaffected.
- `GraphNestEngine` inputs are the datapacks of the blocks feeding its
  encoders (`encoder_sources`), not synthetic names.
- `cosim/fake_backend.py`: the recording fake NEST moved into the package
  (`engines="fake"` runs need no simulator); `test/support_fake_nest.py`
  re-exports it and keeps the row-expansion helpers.
- `graph/run.py`: `run_graph(graph_or_path, engines, goals, out_dir)` →
  `TrialRecord`s with `meta["graph"]` (the resolved graph), the collector
  layout when the motif is present, otherwise one JSON record per trial;
  `run_graph_trial`. `cosim/runner.run_trial`/`trial_raster` duck-type the
  NEST engine.
- `graph/templates.py`: `two_ring_single_joint`, `three_ring_single_joint`
  (`JointTriple`; the belief ring now gets a once-mode encoder from the first
  measured angle, otherwise the goal transport has nothing to move),
  `two_joint_forward_kinematics` (circular encoder mapping, one `OutputRing`
  per output with its artifact key); `write_examples` → `graph/examples/*.graph.json`
  (pinned to the templates by test).
- `scripts/run_graph.py` (`--engines`, `--goal`, `--template`, `--validate`,
  `--save`, `--out`, `--dashboard`: the existing page with the graph name in
  its status; start/stop/reset as today). `Encoder.mapping="circular"`.
- Tests: `test_graph_files.py` (10: round trips, example files, file problems,
  compiler structure and config derivation, all templates run on the fakes,
  collector layout, JSON records), phase 4 gates in `test_graph_real_nest.py`.

### Gates (container)

```
Ran 222 tests in 112.989s   (full suite; the one failure was the test's row extraction, fixed)
test_graph_real_nest.py after the fix: Ran 4 tests in 23.101s  OK
PARITY ticks=203 stop=drive_settled q_final=0.5917
```

- `run_graph.py two_ring_single_joint.graph.json --engines nest` equals the
  Python template run **and** the vectorised cosim runner (legacy transceivers,
  `--model vectorised`) tick for tick: identical left/right/r1 deltas, velocity
  horizons and joint positions on every one of the 203 ticks, same stop reason
  and final angle (the same numbers as the phase-2 smoke).
- Forward-kinematics template vs legacy `MultiRingDecode` (same seed, burn-in
  300 ms, q1 then q2 injected 50 ms each, 300 ms; angles (0.9, −1.3)):

  | | legacy | graph |
  |---|---:|---:|
  | q1 / q2 ring centroid | 13.41 / 78.40 | 14.79 / 78.50 |
  | lift: total, decoded angle | 1086, 1.273 | 1084, 1.244 (L1 0.066, Δ 0.029 rad) |
  | pitch | 1050, −0.705 | 1124, −0.551 (L1 0.162, Δ 0.154 rad) |
  | yaw | 1251, −1.171 | 1283, −1.029 (L1 0.117, Δ 0.142 rad) |
  | build time | | 5.2 s (signed-product layer still per-cell Connects) |

  Exact counts are not expected (per-synapse legacy ring vs vectorised, RNG
  streams); tolerances in the test: totals ±25 %, L1 ≤ 0.5, angle ≤ 0.35 rad.
- Validation errors tested (`test_graph_files.py`); the three-ring template
  builds and runs on the fakes and with NEST.

Smokes with NEST (`scripts/run_graph.py`): two-ring 203 ticks, |err| 0.0083;
forward kinematics 6 ticks, build 5.9 s, decoded angles per tick; three-ring
`--max-steps 120`: settles after 10 ticks with |err| 0.5995, i.e. CT_goal
produces no drive above the threshold in the first ten ticks. That is the
phase 5 characterisation (trust weights, state mode and rate, lead), not a
phase 4 gate.

Not run: Gazebo (`--engines full --dashboard` needs the simulation), the
browser (dashboard code unchanged; its fake end-to-end test is in the suite).

## 2026-09-29 — Phase 5: robot binding and multi-joint

Report: `feedback.md` (encoder-mode sweep, goal-distance behaviour, multi-joint);
raw sweep data `feedback_sweep.json`, table `feedback_sweep_table.md`.

### Changes

- `control/trajectory.py`: `build_multi_joint_trajectory` and
  `CommandState.build_multi` (several joints per receding horizon; equals the
  single-joint builder point for point when one joint is commanded).
- `arm_velocity_cmd` gains an optional `commands` list; `FakeRobotEngine` and
  `GazeboRosEngine` build one trajectory from it (`command_horizons`), record
  `joint_indices`, and stop-and-settle every commanded joint at trial end.
- `graph/compile.py`: `JointCommandTF` now emits `"<joint>.command"`; a single
  `ArmCommandTF` merges all of them into `arm_velocity_cmd` (legacy fields from
  the primary joint, `settled` = all joints, `commands` for the engines).
  `resolve_joint_limits(graph, transport)` applies URDF limits to `Joint` blocks
  with `limits_source="urdf"` and to the encoders mapped with the old limits;
  `run_graph` calls it after the loop initialised when the robot engine has a
  transport. `Joint.limits_source` parameter (`params` default).
- `Homeostasis` exposes `left_counts` / `right_counts` signal ports (the
  decision pair as a motor-signal source; engine emits them).
- `graph/templates.py`: `multi_joint_two_ring(joints, goals)` (plan 5e level 1:
  one two-ring circuit per joint, one robot engine); example file added.
- `scripts/sweep_feedback.py` (plan 5c sweep: once / continuous r×h / corrective
  dead band; tracking error and bump-to-joint lag; JSON + markdown);
  `scripts/cosim_gazebo_smoke.py --joints 5 6` commands two joints in one
  trajectory; CMake installs the new scripts.
- Tests: `test_graph_files.MultiJointTests` (trajectory equality and padding,
  multi-joint template compile/run on the fakes, fake robot integrates and
  settles two joints, Gazebo engine with the fake transport publishes both
  joints in one trajectory, URDF limits follow into the encoders),
  `test_graph_real_nest.MultiJointRealNestTests` (container).

### Gates

Container: full suite `Ran 228 tests … FAILED (failures=1)` where the failure
was the multi-joint test's first goal set, then `test_graph_real_nest.py`
`Ran 5 tests in 30.727s OK` after retargeting it (below). Host: 211 tests,
only the pre-existing `colorcet` import error.

Multi-joint on the fakes with real NEST (`multi_joint_two_ring`, goals
(0.6, −1.2)): `steps=400 stop=max_steps j5=0.5245 j6=-0.8191`; both joints move
in the right direction (87 % and 68 % of the way), uncommanded joints hold. The
first goal set (0.6, −0.4) moved joint 6 the wrong way, which led to the
goal-distance finding in `feedback.md` §2: below about 18 ring indices the
comparator produces no drive, and joint 6's outcome is seed dependent. Recorded
as model behaviour, not a plumbing defect (the single-joint parity test is
unchanged and green).

Feedback sweep (fake robot, real NEST, joint 5, goals ±0.5/0.6): `once` error
0.035 rad in 225 steps; every `continuous` configuration (r 25–400 Hz, h 2–10)
pins the belief to the joint and never settles (error 0.17–1.69 rad);
`corrective` with dead band 5 reaches 0.113 rad. Decision: `once` stays the
default; `corrective`(5) is the candidate for the Gazebo repeat.

**Not run:** Gazebo (`scripts/cosim_gazebo_smoke.py --joints 5 6`,
`run_graph.py --engines full`, `sweep_feedback.py --engines full`) — needs the
simulation launched through `./run_model_docker.bash`; the plan's "in Gazebo"
half of the phase 5 gate is therefore open and documented as such in
`feedback.md`. Browser: unchanged.

## 2026-09-30 — Phase 6: visual editor

### Changes

- `tiago_ring_controller/ui/`: `editor.html`, `editor.css`, `editor.js` — plain
  JavaScript and SVG, no build step, no dependency. Palette of block types from
  `blocks.describe_types()` (parameters with type, default, bounds, unit, doc;
  ports with kind and direction), blocks on an SVG canvas (drag to move,
  positions in `ui`), ports as handles (inputs left, outputs right, colour by
  kind), edges by dragging from an output to an input port (kind checked in the
  browser, one producer per non-multi input, the server validates again),
  parameter panel per block (typed fields, bounds, choices, JSON for dict/list,
  id rename with edge rewrite), Delete for the selection, simulation and robot
  forms, Load example / Load file / Save / Validate / Run in session / Clear,
  live overlay of the primary state ring (bump bars, centroid), gain (L/R
  counts), decoder (drive) and joint (angle) from the existing event stream.
  `?static=1` disables the event stream, `?selftest=<example>` draws the example
  from an empty canvas through the editor's own code paths and reports whether
  the canonical text equals the example's (`data-selftest="ok"`).
- Dashboard server: `/editor`, `/ui/<file>`, `GET /api/blocks`, `GET /api/graph`
  (the session's document), `GET /api/graph/examples[/<name>]`,
  `GET /api/graph/file?path=`, `POST /api/graph/validate` (problems + canonical
  document and text), `POST /api/graph/save` (writes the canonical file),
  `POST /api/graph` (queues a `graph` command). `run_dashboard_session(...,
  on_graph=)` swaps the loop between trials; `run_graph.py --dashboard` provides
  the swap (same engines), publishes the primary block ids for the overlay and
  keeps writing the collector layout with the current config. The dashboard page
  links to the editor.
- Tests `test/test_editor.py`: static files and palette, validation problems and
  canonical text, a document assembled the editor's way saved byte-identically
  to `dumps(two_ring_single_joint())` with the same positions, the graph
  command swapping a session's loop, and two headless-Chrome tests on the host
  (`google-chrome --headless=new --dump-dom --screenshot`): the editor renders
  the single-joint architecture (all 11 blocks, ≥ 9 spike and ≥ 5 signal edges,
  no console error, screenshot written) and the in-browser self-test draws the
  example from an empty canvas with the same result (11 blocks, 15 edges).

### Gates

```
host:      Ran 217 tests in 12.377s  FAILED (errors=1, skipped=7)   # colorcet import only; both Chrome tests ran and passed
container: Ran 234 tests in 125.399s OK (skipped=2)                 # the two Chrome tests: no browser in the image
```

Screenshot checked by eye (host, 1400×900): palette, blocks with typed ports,
orange spike edges and green signal edges, inspector. Headless Chrome tiles the
page below the viewport in the PNG; in a browser window the page does not
scroll (`html, body {height: 100%; overflow: hidden}`, the canvas pane scrolls).

Not exercised by tests: mouse drag-and-drop itself (the connect logic behind it
is exercised by the self-test through the same function), the live overlay
during a trial (the message fields it reads are the dashboard's, unchanged),
and the editor against Gazebo.

## 2026-09-30 — Phase 7: documentation, cleanup, PR

### Changes

- `docs/blocks/`: `architecture.md` (the map: layers, what changed under the
  science, templates, open items), `graph-schema.md` (file reference and
  rules), `blocks.md` (generated from the `ParamSpec`s by
  `scripts/write_block_reference.py`; a test pins the file to the generator),
  `adding-a-block.md`. `equivalence.md` and `feedback.md` from phases 2 and 5.
- Package docs: `architecture.md` compatibility rule and robot control flow
  rewritten for the branch; `robot_safety.md` names the maintained drivers,
  the `commands`-per-tick rule for graph runs and the legacy location. Both
  READMEs point at the graph drivers, the editor and the legacy recipe;
  plan-B phase 7 marked as superseded with what remains open.
- Local tag `legacy-loop-baseline` on `main` (`e717286`), not pushed.

### Gates

```
host:      Ran 218 tests in 12.829s  FAILED (errors=1, skipped=7)   # colorcet import only
container: Ran 235 tests in 148.756s OK (skipped=3)                 # two Chrome tests, one docs test (docs/ not mounted)
```

### Still open after this branch

- Every Gazebo gate (phase 5 multi-joint smoke, feedback sweep repeat, phase 6
  editor with the simulation, plan-B A1/A3): needs `./run_model_docker.bash`
  with the simulation launched.
- Comparator goal-distance behaviour and the motor-signal experiment
  (`feedback.md` §2, plan 5d follow-up), the three-ring template's transports.
- `TaskGain` (plan 5e level 3) is a port-contract stub; the signed-product
  layer is not vectorised.
