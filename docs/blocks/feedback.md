# Robot feedback path: encoder modes (plan 5c) and what drives a joint

Date: 2026-09-29, branch `blocks-refactor`, phase 5. Graph
`two_ring_single_joint` (joint 5, collector mapping, calibrated decoder), real
NEST (`HEAD@41892a5`, seed 13579), **fake robot** (`FakeRobotEngine`: first-order
lag τ = 80 ms integrating the commanded horizon). Gazebo was **not run**: the
sweep script takes `--engines full`, the numbers below must be repeated there
before they are used for the robot. Raw data: `feedback_sweep.json`
(`scripts/sweep_feedback.py --out ... --markdown ...`), table
`feedback_sweep_table.md`.

## 1. Sweep: mode, rate `r`, half width `h`, dead band

State encoder (`enc_state`, `Joint.angle → Encoder.angle → r1.stim`), goals
+0.6 and −0.5 rad, one trial each, 400-step budget, settle when the delayed
drive stays under 5 for 10 ticks. Tracking error = mean |r1 centroid − ring index
of the measured joint| over the main ticks; lag = tick shift maximising the
cross-correlation between the r1 centroid and the mapped joint index (positive:
the bump leads the joint).

| mode | r (Hz) | h | dead band | mean abs error (rad) | mean steps | tracking (idx) | lag (ticks) |
|---|---:|---:|---:|---:|---:|---:|---:|
| once | 200 | 5 | – | **0.035** | 225 | 3.29 | (n/a) |
| continuous | 25 | 2 | – | 0.251 | 400 | 1.08 | 3.5 |
| continuous | 25 | 5 | – | 0.470 | 400 | 1.30 | 3 |
| continuous | 25 | 10 | – | 0.731 | 400 | 1.61 | 3 |
| continuous | 50 | 2 | – | 0.173 | 400 | 1.04 | 3 |
| continuous | 50 | 5 | – | 0.574 | 400 | 1.31 | 3 |
| continuous | 50 | 10 | – | 0.715 | 400 | 1.74 | 3 |
| continuous | 100 | 2 | – | 0.267 | 400 | 1.21 | 3 |
| continuous | 100 | 5 | – | 0.618 | 400 | 1.66 | 3 |
| continuous | 100 | 10 | – | 1.053 | 400 | 2.94 | 3 |
| continuous | 200 | 2 | – | 0.526 | 400 | 1.20 | 3 |
| continuous | 200 | 5 | – | 0.516 | 400 | 2.03 | 3 |
| continuous | 200 | 10 | – | 1.562 | 400 | 3.74 | 2 |
| continuous | 400 | 2 | – | 0.793 | 260 | 1.36 | 3 |
| continuous | 400 | 5 | – | 1.210 | 400 | 3.76 | 2.5 |
| continuous | 400 | 10 | – | 1.688 | 400 | 4.22 | 2 |
| corrective | 200 | 5 | 0 | 0.560 | 400 | 2.09 | 3 |
| corrective | 200 | 5 | 2 | 0.243 | 306 | 1.12 | 3 |
| corrective | 200 | 5 | 5 | 0.113 | 400 | 1.70 | 5.5 |
| corrective | 200 | 5 | 10 | 0.035 | 225 | 3.29 | (never triggers = once) |

Reading:

- **`once` is the best closed-loop-free baseline** and stays the example
  graph's default (`Encoder.mode="once"`). The bump is the state *estimate*
  transported by the gain feedback; it runs ahead of the joint (tracking error
  3.3 indices is the bump leading the lagging joint, not an error of the loop).
- **`continuous` at any rate or width degrades the trial**: even 25 Hz on
  ±2 neurons pins the belief to the measured joint (tracking error ≈ 1 index,
  bump leads by 3 ticks = 150 ms), which removes the very displacement the
  comparator turns into drive; the loop never settles within 400 steps and the
  error grows with `r` and `h`. This is the outcome plan 5c anticipated: a
  continuous stimulus competes with the transport.
- **`corrective` with a dead band of 5 indices** is the only feedback variant
  near `once` (error 0.11 rad): the bump is corrected only when it has drifted
  more than 5 indices from the joint, so the transport is free inside the band.
  Dead band 10 never triggers on this robot. This is the candidate for the
  Gazebo repeat, with `r` and `h` to be swept around (200, 5) there.

## 2. Goal distance and the decision pair

Probing the same graph with other goals (real NEST, fake robot, once mode):

| joint | goal (rad) | goal index / start | steps | final (rad) |
|---|---:|---:|---:|---:|
| 5 | +0.25 | 116 / 100 | 10 | 0.000 (no drive) |
| 5 | +0.40 | 126 / 100 | 188 | 0.442 |
| 5 | −0.35 | 76 / 100 | 148 | −0.248 |
| 6 | −0.40 | 81 / 99 | 10 | 0.000 (no drive) |
| 6 | −0.80 | 63 / 99 | 359 | −0.524 |
| 6 | −1.20 | 45 / 99 | 395 | −0.836 |
| 6 | +1.20 | 154 / 99 | 10 | 0.000 (no drive; moved to 0.655 inside the two-joint graph) |

Below about 18 ring indices between the state and goal bumps the warm/cold
comparison does not produce a left/right decision and the trial "settles"
immediately with zero drive; larger distances move the joint but the trial
stops short (the drive decays under the threshold before the goal). Whether a
goal is reached is also seed dependent (joint 6, +1.2: silent alone, moving in
the two-joint graph where the RNG streams differ). These are properties of the
ridge-fitted comparator and the winner-take-all decision pair, unchanged from
the legacy model; they bound what any feedback mode can achieve and belong in
the motor-signal experiment (plan 5d follow-up, `motor-signal.md`, not done).

## 3. Multi-joint (plan 5e, level 1)

`multi_joint_two_ring(joints=(5, 6), goals=(0.6, −1.2))`: two independent
two-ring circuits in one NEST kernel, two decoders, one robot engine, one
trajectory per tick carrying both joints (`arm_velocity_cmd.commands`,
`build_multi_joint_trajectory`). Fake robot, real NEST: 400 steps, final
(0.5245, −0.8191): both joints move in the right direction, joint 5 to 87 % of
its goal and joint 6 to 68 % (the same shortfall as alone). Pinned by
`test_graph_real_nest.MultiJointRealNestTests`. The Gazebo smoke
(`scripts/cosim_gazebo_smoke.py --joints 5 6`) commands two joints with opposite
constant velocities in one trajectory; **not run** (needs the simulation).

## 4. Decisions carried into the example graphs

- `Encoder.mode="once"` stays the default of `two_ring_single_joint` and of the
  per-joint circuits in `multi_joint_two_ring`.
- `three_ring_single_joint` keeps `state_mode="continuous"` on the *actual* ring
  (that ring is meant to mirror the joint; the belief ring is only moved by the
  two transports), with `enc_belief` once at trial start. Its transports produce
  no drive above the threshold in the first ten ticks on the fake robot; the
  trust weights and the sense rate are the sweep to run once the comparator's
  goal-distance behaviour above is understood.
- `Joint.limits_source="urdf"` reads the limits from `/robot_description` when a
  robot transport exists (`resolve_joint_limits`, encoders follow); the
  templates keep the calibration limits so fake runs need no ROS.
