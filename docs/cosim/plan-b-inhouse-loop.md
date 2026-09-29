# Plan B — in-house NRP-style co-simulation loop (recommended)

Goal: replace the three copies of the free-running tick with one deterministic,
lock-stepped loop that owns time for both NEST and Gazebo, feeds robot state into the
network every tick, and is testable without either simulator. Semantics follow
nrp-core's FTILoop so that a later migration to NRP (Plan A) would be a re-hosting of
the same engines/TFs rather than a rewrite.

Everything here lives under `src/tiago_ring_controller/src/tiago_ring_controller/`
(the internal package) in a new sub-package `cosim/`, plus one optional C++ plugin.

## B.1 Concepts (mirroring nrp-core)

| Term | Meaning here |
| --- | --- |
| **Engine** | Something that owns simulated time and advances in steps: `NestEngine`, `GazeboRosEngine`. Fakes: `FakeNestEngine`, `FakeRobotEngine`. |
| **DataPack** | A named, plain-dict payload an engine publishes after a step or accepts before one. Serialisable (JSON) so it can be logged. |
| **Transceiver function (TF)** | A pure callable `tf(inputs: dict[str, DataPack], t_ms) -> dict[str, DataPack]` mapping datapacks between engines. Holds its own state (e.g. `DriveControlCore`) if needed. |
| **FTILoop** | Fixed-time-increment scheduler. Owns `t_ms`, engines, TFs; performs one tick in a fixed order. |
| **Trial** | Reset + N ticks until a stop condition; produces a `TrialRecord`. |

### Tick order (one-step delay, as in NRP)

```text
tick(t):
  1. for each engine due at t: wait until its previous advance has completed
  2. cache = {name: engine.get_datapacks()}          # state at time t
  3. outputs = run all TFs on cache                  # pure, deterministic order
  4. for each engine: engine.set_datapacks(outputs)  # inputs applied for [t, t+dt)
  5. for each engine due at t: engine.advance(dt)    # may be async
  6. t += dt
```

Robot data measured at `t` therefore influences NEST during `[t, t+dt)` and the NEST
counts from `[t, t+dt)` influence the robot command applied at `t+dt`. That is the
documented, fixed latency; the legacy code had an undocumented, variable one.

### NEST lead (receding horizon)

Legacy publishes a 4-point horizon computed from 4 NEST steps run ahead. Support this
as `nest_lead_steps` (default 4 for legacy equivalence, 0 for pure lock-step): the
NEST engine's clock is allowed to run `nest_lead_steps * dt` ahead of the robot
clock; the motor TF keeps a ring buffer of the last `nest_lead_steps + 1` decoded
velocities and emits the whole horizon. With lead > 0, proprioceptive feedback
(B.5) is delayed by the lead; document that in the record.

## B.2 Module layout

```text
tiago_ring_controller/cosim/
  __init__.py
  datapack.py        DataPack dataclass (name, t_ms, data: dict), JSON helpers
  engine.py          Engine Protocol/ABC; EngineClock; errors
  loop.py            FTILoop, TickRecord, TrialRecord, StopCondition
  tf.py              TF protocol + registry; ProprioceptionTF, GoalTF, MotorTF
  nest_engine.py     NestEngine (real PyNEST); build via existing nest/ builders
  gazebo_ros_engine.py  GazeboRosEngine (rospy): pause/step/unpause, joint_states,
                     trajectory publish, play_motion reset
  stepping.py        GazeboStepper implementations: ClockWaitStepper (phase 1),
                     PluginStepper (phase 2)
  fakes.py           FakeNestEngine, FakeRobotEngine (kinematic joint model),
                     used by tests
  runner.py          run_trial(profile, loop_cfg, goal) -> TrialRecord;
                     run_session(...) for multi-trial workflows
  config.py          CosimConfig dataclass; loads from JSON + LegacyControlProfile
```

Optional C++ (phase 2): `src/tiago_ring_controller/plugins/cosim_step_plugin.cc`,
built by the package's `CMakeLists.txt`.

## B.3 Owning Gazebo time

Facts to rely on (verify with the README checklist):

- `gazebo_ros` exposes `/gazebo/pause_physics` and `/gazebo/unpause_physics`
  (`std_srvs/Empty`) and publishes `/clock`. There is **no** step service in ROS 1
  `gazebo_ros`.
- The world runs `max_step_size 0.001`, so `dt = 50 ms` is exactly 50 physics
  iterations.
- `gazebo_ros_control` runs `controller_manager.update()` inside Gazebo's world
  update event, so PAL's `arm_controller` advances exactly with physics when
  stepped. `JointTrajectory` interpolation uses ROS (sim) time, so `time_from_start`
  is honoured exactly under stepping.
- With `/use_sim_time true`, `rospy.sleep` and `rospy.Rate` block on `/clock`. **They
  must never be called while Gazebo is paused.** The new code uses wall-clock waits
  (`time.sleep`, `threading.Event.wait`) and compares `/clock` values explicitly.

Two stepper implementations behind one interface `GazeboStepper.step(n_iters)`:

**Phase 1 — `ClockWaitStepper` (pure Python, no build changes).**
```text
target = clock_now + n_iters * max_step_size
unpause_physics()
wait (wall clock, with timeout) until /clock >= target
pause_physics()
overshoot = clock_now - target        # record it; typically 0-3 ms at RTF≈1
```
Non-deterministic by a few physics iterations, but bounded and logged; enough to
develop and validate everything else, and already far better than today. The TF that
integrates commanded position must use the *measured* elapsed sim time from the
datapack, not nominal `dt`.

**Phase 2 — `PluginStepper` (exact).** A ~60-line Gazebo *world* plugin that
advertises a ROS service `/cosim/step` (`uint32 iterations` → `float64 sim_time`)
and calls `world->Step(iterations)` (blocking, physics stays paused). Loaded through
`tiago_ring_controller.launch` with an extra `<arg name="extra_gazebo_args"
value="-s libcosim_step_plugin.so">` or as a `<plugin>` inside the override world
file. This is the same mechanism nrp-core's world plugin uses. Message definition:
`srv/StepWorld.srv` in the package; add `message_generation` to `package.xml` /
`CMakeLists.txt`.

Resets and limit discovery (B.7) run *unpaused* with explicit conditions; trials run
paused + stepped.

## B.4 NEST engine

- Build the network once per trial through the existing builders
  (`nest/ring.py`, `nest/gain.py`, `nest/readout.py`, `nest/comparator.py`) via
  `SingleRingModel`, exactly as `ring_component.py` does today, so topology tests
  keep applying. `reset()` = `ResetKernel` + rebuild (NEST 3 has no `ResetNetwork`).
- **Stimulus devices created at build time.** Replace per-injection
  `inject_stimulus` (creates a generator, `Simulate(50)`, rate 0) with, per ring, one
  `poisson_generator` **per neuron** (`N` generators, all `rate = 0`, each connected
  to its neuron with weight 4500 — same weight/rate/half-width semantics). A bump at
  index `i` with half-width `h` is then `SetStatus(gens[window], {"rate": 200})` for
  the next tick(s) and back to 0 afterwards. Legacy equivalence mode: rates on for
  exactly one 50 ms tick before the loop starts (same 50 ms exposure as
  `inject_stimulus`). Continuous mode: rates follow the measured joint every tick.
  Keep `inject_stimulus` in place for the non-cosim scripts; add
  `attach_stimulus_generators(backend, network)` beside it.
- **Stepping.** `nest.Prepare()` once after build; `nest.Run(dt)` per tick;
  `nest.Cleanup()` at trial end. Between `Run` calls only parameter changes are
  allowed (rates, currents) — no node/connection creation — which the design above
  respects. Keep `Simulate` for the non-cosim analysis scripts.
- **Readout.** Collect all recorders of interest into NodeCollections at build time
  (`nest.NodeCollection(sorted(ids))` works for non-contiguous ids). Per tick, read
  `n_events` once per collection (`nc.get("n_events")` → list) and either subtract
  the previous value or reset with `nc.set(n_events=0)`. This turns ~600 calls into
  ~3. Rasters for plots still come from `events` at trial end (`record_to: memory`
  unchanged).
- Datapacks: `ring_counts` (out: `left`, `right`, `r1_delta[N]`, `t_nest_ms`);
  `state_bump`, `goal_bump` (in: `center_index`, `half_width`, `rate_hz`,
  `duration_ticks`).
- Threads/seed through `configure_kernel` as today; record `rng_seed` in the trial.

## B.5 Transceiver functions

- `GoalTF` (runs once at trial start, or every tick if the goal moves): goal angle →
  `goal_bump` via `profile.joint_to_ring_index`.
- `ProprioceptionTF`: `joint_state` → `state_bump`. Modes: `once` (legacy: only at
  tick 0), `continuous` (every tick; rate scaled by a config gain), `off`.
- `MotorTF`: `ring_counts` → `arm_velocity_cmd`. Wraps `DriveControlCore`
  (filter + delay), `decode_velocity` (asymmetric gains from the calibration JSON),
  the `nest_lead_steps` ring buffer, and the stop condition (`|delayed| <
  drive_threshold` for `n_settle` ticks, or `max_steps`). Emits the full horizon
  as velocities; the Gazebo engine turns it into the `JointTrajectory` with
  `build_receding_trajectory` (unchanged) and advances the commanded state by the
  first point (unchanged contract).

All TFs are pure w.r.t. ROS and NEST; they are unit-tested with dict inputs.

## B.6 Gazebo/ROS engine

`GazeboRosEngine` composes, not replaces, the existing `TiagoPublisher`:

- `initialize()`: `rospy.init_node` (if needed), wait for `/joint_states` with a
  wall-clock timeout, wait for pause/unpause services, create the stepper.
- `get_datapacks()`: `joint_state` = latest positions/velocities snapshot taken
  **under a lock** in the callback, plus `/clock` value and the wrist wrench if
  subscribed. Snapshots are copied, never shared.
- `set_datapacks()`: stores `arm_velocity_cmd`.
- `advance(dt)`: publish the pending trajectory (header stamp = current sim time),
  then `stepper.step(dt / max_step_size)`; record `sim_time_before/after`.
- `reset()`: unpause → `play_motion` `tiago_experiment_start_1` with result *checked*
  (`get_state() == SUCCEEDED`, else raise) → `wait_for_settled` (wall-clock polling
  of the snapshot, timeout) → `prime_command_state(force=True)` → pause.
- `shutdown()`: `stop_joint` horizon, unpause (leave Gazebo running for the next
  workflow), unregister.

## B.7 Limit discovery and safety

Legacy discovers joint limits by commanding ±1 rad/s and sleeping 5 s. Replace with:
read limits from the URDF on the parameter server (`/robot_description`, via
`urdf_parser_py`) and keep the empirical sweep as an opt-in fallback that runs
unpaused and stops on `|velocity| < eps` for 0.5 s, timeout 8 s. Document in
`docs/robot_safety.md`.

## B.8 Phased tasks

Each phase ends with tests green in the container
(`python3 -B -m unittest discover -s test -p "test_*.py"`) and a short note in
this file's "Progress" section.

**Phase 0 — baseline measurement (½ day).**
- Add `scripts/measure_legacy_tick.py` (or a flag in the collector) that timestamps
  each tick with `time.monotonic()` and `rospy.Time.now()` and prints mean/p95
  period and NEST-vs-sim-time drift over one trial. Save output under
  `docs/cosim/baseline/`. This is the before number.

**Phase 1 — loop core + fakes (1-2 days, no simulators needed).**
- `datapack.py`, `engine.py`, `loop.py`, `tf.py`, `fakes.py`, `config.py`.
- `FakeRobotEngine`: 7-joint kinematic model integrating commanded velocity with a
  first-order lag (tau from the calibration JSON) so `MotorTF` closes a loop.
- `FakeNestEngine`: deterministic counts from a stub (e.g. proportional to the
  distance between goal and state bump), enough to test ordering and delay.
- Tests `test/test_cosim_loop.py`: tick order, one-step delay, `nest_lead_steps`
  buffer, stop conditions, determinism (two runs → identical `TrialRecord`),
  JSON round-trip of records.

**Phase 2 — real NEST engine (1 day).**
- `nest_engine.py` with `attach_stimulus_generators`, `Prepare/Run/Cleanup`,
  NodeCollection `n_events` readout.
- Extend `test/support_fake_nest.py` with `Prepare/Run/Cleanup`, `NodeCollection`
  and `n_events` so `test_nest_topology.py`-style tests cover the engine.
- Real-NEST check in the container: `NestEngine` + `FakeRobotEngine` for one trial;
  compare `left/right/r1` count statistics with the legacy `run_ring_simulation`
  driven by the same seed and the same 50 ms exposure (expect equal up to RNG stream
  differences caused by the extra generator nodes — record the actual difference).

**Phase 3 — Gazebo engine with `ClockWaitStepper` (1-2 days, container).**
- `gazebo_ros_engine.py`, `stepping.py`.
- Smoke: pause; step 50 iterations ×20; assert `/clock` advanced 1.000 s ± overshoot;
  publish a constant-velocity horizon and assert the joint moves the expected angle
  (validates ros_control under stepping — this is Plan A's gate G3 too).
- First lock-stepped trial with `ProprioceptionTF(mode="once")`, collector profile.
  Save the record and the overshoot histogram.

**Phase 4 — workflow migration (2-3 days).**
- `runner.py` with `run_trial` / `run_session`.
- Make `single_joint_data_collector.py`, `analysis_single_joint.py`,
  `calibrate_single_joint.py` delegate `run_trial` to the runner behind a
  `--cosim` flag (legacy path untouched until parity is shown), mapping their
  profile and outputs (`TrialRecord` → the existing CSV/JSON/plot schemas in
  `evaluation/serialization.py`). `demo_graphs.py` hooks camera capture into a
  per-tick callback.
- Parity report: same seeds, legacy vs cosim, per profile — final error, settle
  steps, count traces. Expect differences (that is the point) but explain each.

**Phase 5 — exact stepping plugin (1 day, optional but recommended).**
- `plugins/cosim_step_plugin.cc`, `srv/StepWorld.srv`, CMake/package.xml,
  launch arg. `PluginStepper` selected by config when the service exists.
- Re-run phase 3 smoke: overshoot must be exactly 0.

**Phase 6 — continuous proprioception (research, open-ended).**
- `ProprioceptionTF(mode="continuous")`; sweep rate gain and half-width; this is
  the new science the loop enables. Not part of the engineering acceptance.

**Phase 7 — retire legacy loop.** Flip `--cosim` default, delete the three loop
bodies, update `docs/architecture.md` "Robot control flow", `docs/robot_safety.md`,
`README.md`. Open PR to `main`.

## B.9 Acceptance criteria

- A1. With `PluginStepper`, two runs of the same trial (same seed, same goal) produce
  byte-identical `TrialRecord`s. With `ClockWaitStepper`, identical NEST traces and
  robot traces equal within the logged overshoot.
- A2. Per-tick wall time is independent of trial length (readout cost flat).
- A3. A trial runs correctly at Gazebo RTF 0.2 and RTF 1.0 with identical results
  (A1), demonstrating independence from host speed.
- A4. No `rospy.sleep` / `rospy.Rate` inside `cosim/`; no `Simulate` inside stimulus
  code; no blind sleeps > 0.1 s outside explicit `wait_for(condition, timeout)`.
- A5. Unit suite runs without NEST or ROS installed (fakes), and the container
  suite passes including the real-NEST smoke test.
- A6. The three workflows produce their existing artifact schemas via `--cosim`.

## B.10 Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| `arm_controller` misbehaves when physics is paused/stepped in 50 ms chunks (e.g. trajectory rejected as "in the past") | Stamp header with current sim time or 0 ("now"); phase 3 smoke catches it early. |
| `play_motion` needs unpaused time | Reset runs unpaused by design. |
| `/clock` subscriber latency makes `ClockWaitStepper` overshoot large | Log it; if p95 > 5 ms go to phase 5 sooner. |
| Extra generator nodes change NEST RNG streams vs. legacy | Accept; document in parity report; seeds are recorded either way. |
| NEST `Prepare/Run` forbids structural changes | All structure is built before `Prepare`; bumps are rate changes only. |
| Fake-based tests give false confidence | Phases 2 and 3 each have a real-simulator smoke with numeric assertions. |

## B.11 Reference sketch

```python
# loop.py (sketch, not final)
class FTILoop:
    def __init__(self, engines, tfs, dt_ms, nest_lead_steps=0):
        ...
    def tick(self):
        cache = {e.name: e.get_datapacks() for e in self.engines if self.due(e)}
        outputs = {}
        for tf in self.tfs:
            outputs.update(tf(cache, self.t_ms))
        for e in self.engines:
            e.set_datapacks({k: v for k, v in outputs.items() if k in e.inputs})
        for e in self.engines:
            if self.due(e):
                e.advance(self.dt_ms)
        self.records.append(TickRecord(self.t_ms, cache, outputs))
        self.t_ms += self.dt_ms
```

```python
# nest_engine.py (sketch)
class NestEngine:
    inputs = {"state_bump", "goal_bump"}
    def advance(self, dt_ms):
        self._apply_pending_bumps()          # SetStatus rates
        nest.Run(dt_ms)
        counts = self._recs_all.get("n_events")   # one call
        self._delta = np.subtract(counts, self._prev); self._prev = counts
```

## Progress

*(Linux session: append dated entries here — phase, what was done, test status,
numbers.)*

### 2026-09-29 — phases 1–2 done, phase 3 code written, phase 4 partial; model unchanged

Scope decision for this step: change the harness, not the model. `inject_stimulus`
still creates one generator per injection and runs `Simulate(50)` inside it; B.4's
build-time generators and continuous proprioception stay behind a hook
(`StimulusPort`, see below). Everything lives in
`src/tiago_ring_controller/src/tiago_ring_controller/cosim/`; the flat scripts are
untouched.

Done:

- **Phase 1** — `datapack.py`, `engine.py`, `loop.py` (`FTILoop`, `TickRecord`,
  `TrialRecord`, stop conditions `MaxSteps` / `SettledFlag` / `AnyOf`), `tf.py`
  (`GoalTF`, `ProprioceptionTF` with modes `once` / `continuous` / `off`, `MotorTF`),
  `fakes.py` (`FakeRobotEngine` 7-joint first-order-lag model on the unchanged
  `CommandState`; `FakeNestEngine` scripted or goal-seeking stub), `config.py`
  (`CosimConfig.from_profile(profile, joint, calibration)`).
  Tests `test/test_cosim_loop.py`: tick order, one-step delay, lead phase and clock
  invariant, horizon/consumed-sample/settle/stop parity against an inline
  re-implementation of the collector loop (collector and calibration profiles,
  delayed decoder, step budget, lead 0), determinism, JSON round-trip.
- **Phase 2** — `nest_engine.py`: `NestEngine(backend, model_factory)` builds the
  unchanged `SingleRingModel` via `RingModelPorts.from_single_ring_model`;
  `nest_step_mode="run"` (Prepare once, Run per tick, Cleanup at trial end; default)
  or `"simulate"`; readout is NodeCollection `n_events` deltas (three calls per tick,
  per-recorder fallback); bumps go through a `StimulusPort`. The default
  `LegacyInjectStimulusPort` calls the model's `_inject_bump` and therefore only
  accepts bumps before the first step of a trial; a mid-trial `state_bump` raises
  `StimulusNotSupportedError`. **This is the hook for closing the loop**: a port
  that sets generator rates plugs in without touching loop or TFs.
  Real-NEST check `test/test_cosim_real_nest.py`: seed 13579, goal index 140, state
  index 60, six 50 ms ticks — the engine's `left` / `right` / `r1_delta` sequence is
  **identical** to the legacy `run_ring_simulation` sequence in both step modes.
  (No RNG-stream difference, because the model and its generator nodes are
  unchanged.)
- **Phase 3 (code only)** — `stepping.py` (`ClockWaitStepper`; `PluginStepper` hook
  that raises until the world plugin exists), `gazebo_ros_engine.py`
  (`RosTransport` ABC; `RospyTransport` imports rospy lazily and keeps a locked
  joint-state snapshot and a `/clock` condition variable; `GazeboRosEngine`: reset
  runs unpaused with the `play_motion` result checked, then pauses; each tick
  publishes the horizon stamped with sim time and steps `dt / max_step_size`
  iterations; `stop_and_settle` steps physics instead of sleeping), `limits.py`
  (joint limits from the URDF on the parameter server). Unit-tested against a fake
  transport only — **not yet run against Gazebo**. Smoke script:
  `scripts/cosim_gazebo_smoke.py`.
- **Phase 4 (partial)** — `runner.py`: `build_loop`, `run_trial`, `run_session`,
  `legacy_collector_record` (`TrialRecord` → the collector's
  `{"scalars", "timeseries", "raster"}` with the exact `RING_*` field sets from
  `evaluation/serialization.py`), `make_nest_engine` / `make_gazebo_engine` /
  `make_fake_engines`. Driver `scripts/run_cosim_trial.py --engines fake|nest|full`
  writes the collector layout plus one `trial_XXXX_cosim.json` per trial. The
  `--cosim` flag in the three flat scripts waits for the Gazebo parity report
  (their signatures and main blocks are frozen by the golden tests).
- **Phase 0** — `scripts/measure_legacy_tick.py` written, not yet run (needs the
  simulation).
- **Live view** — `FTILoop` accepts `LoopObserver`s (`on_reset` / `on_tick` /
  `on_trial_end`, read-only, called after the engines advanced). `visualization.py`
  provides `RingMonitor` (polar state ring with goal / start / centroid markers,
  rolling r1 raster, left/right gain counts and filtered drive, joint angle vs goal
  with decoded velocity); `run_cosim_trial.py --monitor [--monitor-every N]
  [--monitor-frames DIR] [--monitor-hold]`. Off-screen rendering is tested in
  `test/test_cosim_visualization.py`. This is also the hook `demo_graphs.py` needs
  for per-tick camera capture (phase 4).
- **Browser dashboard** — `dashboard.py` / `dashboard_page.py`: a standard-library
  HTTP server (JSON endpoints + server-sent events) and a single static page that
  draws the same panels on canvases and starts / stops trials (`/api/start`,
  `/api/stop`, `/api/goal`, `/api/quit`). `run_cosim_trial.py --dashboard` runs
  trials on demand in the main thread; the loop is unchanged (an observer publishes
  ticks, a stop condition honours the page's stop). Tested end to end with the fakes
  in `test/test_cosim_dashboard.py`.

Deviations from the design above (all deliberate):

- TF signature is `tf(inputs, ctx: TickContext)`; `ctx` carries `t_ms`, `tick`,
  `phase` (`lead` / `main`) so `MotorTF` counts settled samples only on main ticks.
- The horizon buffer holds `max(nest_lead_steps, 1)` samples, which is what the
  legacy scripts publish (four points for lookahead 4), not `lead + 1`.
- Lead phase: `FTILoop.run_trial` advances only the lead engine `nest_lead_steps`
  times before tick 0, running the TFs at each sub-step; afterwards
  `nest.t_ms == loop.t_ms + lead * dt` is asserted after every tick.
- Two harmless differences from the legacy loop, both covered by the parity tests:
  legacy publishes one more horizon after its last NEST step when the step budget
  ends a trial, and runs no NEST step after the settle break; the loop stops after
  the tick in which the settle flag was raised (one unconsumed NEST step).
- Horizons, consumed samples and stop reasons otherwise match sample-for-sample.

Test status (image `aung9htet/ubuntu-20.04:tiago_ring_forward` — the pinned `_2`
digest is not present on this host; Python 3.8.10, NEST `HEAD@41892a5`, numpy 1.24.4;
package bind-mounted at `/tiago_public_ws/src/tiago_ring_controller`;
`python3 -B -m unittest discover -s test -p "test_*.py"`):

- cosim tests: 37 passed in the container (including the two real-NEST parity
  tests); 35 passed + 1 skipped on a host without the pinned NEST.
- full suite: 178 tests, 59 failures — the same 59 (file-mode, checked-in PNG and
  bytecode inventory subtests) fail on a clean `main` checkout under the same mount,
  so none are new.

Numbers:

- Real NEST + fake robot, collector profile, seed 13579, 40 ticks: per-tick wall
  time 6.3–8.7 ms (mean 6.95 ms), flat from the first to the last tick (A2).
  Hidden stimulus time 100 ms per trial (two injections), recorded in the trial meta.
- Fake engines: ≈0.13 ms per tick.

Next: run `scripts/cosim_gazebo_smoke.py` and `scripts/measure_legacy_tick.py` in
the container with the simulation launched (README checklist), then the first
lock-stepped trial with `--engines full` and the phase-3 overshoot histogram; then
phase 5 (plugin), then B.4 generators + a rate-setting `StimulusPort` for
continuous proprioception (phase 6).
