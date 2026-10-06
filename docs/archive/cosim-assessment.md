# Assessment of the current NEST → TIAGo control loop

Date: 2026-09-28. Baseline commit: `dee98aa` on `main`.
Paths below are relative to `src/tiago_ring_controller/` unless stated otherwise.

## 1. What exists

The coupling between the spiking model and the robot is a hand-written,
single-threaded Python loop. There is no co-simulation layer: no MUSIC, no NRP, no
shared clock. The loop body is copy-pasted, with deliberate small differences, into
three scripts:

| Script | Loop | Purpose |
| --- | --- | --- |
| `src/single_joint_data_collector.py` | `run_trial`, lines 219-317 | dataset collection (joint 5, 50 trials by default) |
| `src/analysis_single_joint.py` | lines 615-798 | decoder analysis (joint 6, 25 trials) |
| `src/calibrate_single_joint.py` | lines 479-620 | fits decoder gain/tau/delay, rewrites `config/calibration/velocity_calibration.json` |

`src/demo_graphs.py` subclasses the analysis class and grabs a camera frame after
each NEST step. `src/single_joint_pid_data_collector.py` is the PID baseline and uses
the same publisher without NEST.

The robot side is `src/tiago_controller.py` (`TiagoPublisher`, `TiagoSubscriber`):
concrete rospy classes, not an interface. Contracts for topics/actions live in
`src/tiago_ring_controller/ros/transport.py`. Pure control math lives in
`src/tiago_ring_controller/control/` (`DriveControlCore`, `build_receding_trajectory`,
the three `LegacyControlProfile`s). NEST topology builders live in
`src/tiago_ring_controller/nest/` and take a `backend` argument, so they run against
the fake NEST in `test/support_fake_nest.py`.

### One tick, as implemented (collector version)

```text
rospy.sleep(0.05)                                   # collector 267
read cumulative events of ~300 spike recorders      # run_ring_simulation 165-167
nest.Simulate(50)                                   # 169
read them again, delta = after - before             # 171-178
signed = right - left                               # 180
DriveControlCore.advance -> filter (tau 0.3 s) + integer delay   # control/controller.py 143-178
decode_velocity (asymmetric gains, default ±1e-4)   # math/control.py
publish 4-point JointTrajectory on /arm_controller/command       # tiago_controller.py 132-207
```

Before the loop, NEST is pre-run `lookahead = 4` steps (200 ms) so that a 4-point
receding horizon can be published (collector 249-264). Each subsequent tick shifts the
buffer by one and re-publishes (307-315). The publisher advances its internal
commanded state by only the first point (tiago_controller.py 202-207).

Parameters (all from `control/profiles.py:32-40`): `time_step_ms = 50`,
`lookahead = 4`, `drive_threshold = 5.0`, `n_settle = 10`, `max_steps = 400`.
NEST resolution is never set (default 0.1 ms, min delay 1 ms).

### Sensory input to NEST

Robot state enters the network **once per trial**, at the start:

- `set_ring_goal` / `set_ring_state` (collector 134-143) map the joint angle to a
  ring index through the profile's `joint_to_ring_index` (`control/profiles.py:64-151`,
  three different mappings) and call `RingAttractorComponent._inject_bump`
  (`src/ring_component.py:105`).
- That calls `inject_stimulus` (`src/tiago_ring_controller/nest/ring.py:139-159`),
  which creates a **new** `poisson_generator` (200 Hz, weight 4500), connects it to
  `2*half_width+1` neurons, **calls `Simulate(50)` itself**, then sets its rate to 0.

During the motion nothing from the robot is fed back into NEST. `/joint_states` is
read only for logging and the error history (collector 269-284). `/wrist_ft` is
logged to CSV. The camera is used only for figures. There are no `dc_generator`,
`step_current_generator` or `spike_generator` inputs driven by robot data.

Between trials `nest.ResetKernel()` runs and the whole network is rebuilt
(collector 374-382).

## 2. Why it is defective

1. **No shared clock.** NEST time and Gazebo time are never compared. The real tick
   period is `rospy.sleep(0.05)` + wall time of `Simulate(50)` + ~600 `GetStatus`
   calls + publish, so it is always longer than 50 ms and it drifts. The
   `time_from_start` values in the published trajectory assume exactly 50 ms
   between publishes. Results therefore depend on host speed and on Gazebo's
   real-time factor (the world runs at `max_step_size 0.001`,
   `real_time_update_rate 1000`, never paused —
   `overrides/pal_gazebo_worlds/worlds/tiago_ring_controller.world:19-20`).
2. **`rospy.sleep` under `use_sim_time`.** gazebo_ros normally sets
   `/use_sim_time true`, so `rospy.sleep(0.05)` measures *simulated* time. If Gazebo
   runs slower than real time the loop slows with it; if Gazebo were paused the loop
   would hang. This must be verified on Linux (see README checklist) and is the main
   reason the loop cannot simply be "stepped" as is.
3. **Open loop during the movement.** The r1 state ring integrates its own gain
   feedback; the measured joint angle never corrects it. Any modelling of
   proprioception is impossible in this structure.
4. **Hidden simulated time.** Each `inject_stimulus` advances NEST by 50 ms outside
   the loop's accounting (two injections per trial → 100 ms), and leaves orphan
   generator nodes behind.
5. **Readout cost grows with trial length.** Recorders are never cleared; each tick
   reads the full cumulative `events` of ~300 recorders one by one
   (`nest/kernel.py:48-52`). `nest.Prepare/Run/Cleanup` are unused, so every
   `Simulate` call also pays kernel prepare/cleanup overhead.
6. **Blind waits.** 5 s sleeps in joint-limit discovery (collector 384-395), a 10 s
   `play_motion` wait whose result is not checked (tiago_controller.py 240-259),
   `wait_for_joint_state` polling on every publish (94-103, via `prime_command_state`),
   `wait_for_settled` polling (215-238), `rospy.sleep(3.0)` in the `tuck_arm.py`
   override.
7. **Unsynchronised shared state.** ROS callbacks write `current_positions` from
   rospy's threads with no lock; `(current_velocities or [0.0]*7)` masks missing data.
8. **Triplication.** The tick logic (~100 lines) exists three times with divergent
   preprocessing that `control/profiles.py:1-13` documents as intentional. Any fix
   must be made three times.

## 3. What is fine and should be kept

- The NEST model itself (`iaf_psc_alpha`, static synapses, `spike_recorder`,
  `poisson_generator`, `dc_generator`; no custom modules, no plasticity). It has no
  feature that constrains the choice of loop.
- The pure layers already extracted under `src/tiago_ring_controller/`: `control/`,
  `math/`, `nest/` builders, `ros/transport.py`. The new loop should be built on
  them, not beside them.
- The fake-NEST and mocked-ROS test infrastructure (`test/support_fake_nest.py`,
  `test/test_control_ros_contracts.py`). The new loop must be testable the same way.
- TIAGo's upstream stack (`tiago_gazebo`, `ros_control` joint trajectory controller
  on `/arm_controller/command`, `play_motion` for resets, `/wrist_ft`). Replacing it
  is a cost, not a benefit.

## 4. Requirements for any replacement

R1. One owner of time. Per tick, Gazebo advances exactly `dt` and NEST advances
    exactly `dt` (optionally with a fixed NEST lead for the receding horizon).
    Results must not depend on host speed or real-time factor.
R2. Deterministic order of operations with documented one-step data delay
    (NRP semantics: data produced at `t` is consumed at `t + dt`).
R3. Robot state can be fed into NEST every tick (continuous proprioception), with the
    legacy once-per-trial injection available as a mode for reproducing old results.
R4. Stimulus devices are created once at build time; no `Simulate` inside injection.
R5. Readout uses per-tick deltas (`n_events` or reset counters), cost independent of
    trial length.
R6. One loop implementation, parameterised by the three legacy profiles.
R7. Runs against fake NEST + fake robot in the unit suite; runs against real NEST and
    real Gazebo in the Docker/VM environment.
R8. No blind sleeps in the trial loop; resets and limit discovery use explicit
    conditions with timeouts and checked results.

## 5. Size of the job

| Part | Lines today |
| --- | --- |
| `src/tiago_controller.py` | 443 |
| `src/tiago_ring_controller/ros/` | ~457 |
| `src/tiago_ring_controller/control/` | ~718 |
| duplicated loop bodies (3 scripts) | ~800 |
| **plumbing total** | **~2.4k** |

Whole Python codebase: 74 files, ~25k lines (16k flat legacy scripts, 4.5k internal
package, 4.6k tests). Pins: Ubuntu 20.04, ROS Noetic, Gazebo 11, CPython 3.8.10,
NEST `HEAD@41892a5` (NEST 3.x API), numpy 1.24.4.

## 6. Verified on Linux

Recorded as checks are done; the Gazebo items are listed in `README.md`
("Gazebo checks still open").

- `/use_sim_time`: not yet verified (no simulation launched by the automated work).
- Gazebo pause/unpause services present: not yet verified.
- `/clock` rate, `/joint_states` rate: not yet verified.
- Legacy loop real tick period (mean / p95 over one trial): not yet measured;
  `scripts/measure_legacy_tick.py` is ready.
- Baseline unit suite + real-NEST smoke (2026-09-29, image
  `aung9htet/ubuntu-20.04:tiago_ring_forward`, package bind-mounted from the git
  checkout): 178 tests, 59 failures on `main` and the identical 59 on `cosim-loop`
  (file-mode inventory subtests in `test_api_compatibility` /
  `test_environment_contracts`, checked-in PNG inventory in
  `test_artifact_contracts`, bytecode cache tree in `test_bytecode_provenance`);
  the real-NEST smoke test passes. The cosim suite (`test/test_cosim_*.py`) passes,
  including the real-NEST parity test against the legacy per-tick loop.
