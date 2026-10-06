# Velocity control check and spike → velocity calibration plan

Date: 2026-10-02. Branch `velocity-calibration`, cut from `blocks-refactor`.
Results of the work are recorded in `calibration.md` next to this file.

## Context

The gain-modulation spikes are meant to drive TIAGo by velocity. This plan answers
two questions: whether the velocity control path works, and how the spike → velocity
gain should be calibrated.

The command path was read end to end and then run in throwaway containers from the
pinned image (repository mounted read-only, no network): about 330 real-NEST trials
with the fake robot, and the real Gazebo simulation headless (a velocity-step test on
joints 5 and 1, two full graph trials on joint 5). The numbers in parts 1 and 2 are
from those ad hoc runs on 2026-10-02; the scripts added by part 3 reproduce them.

**Short answer**

- The velocity *transport* works. TIAGo's arm has no velocity interface, so the code
  integrates the decoded velocity into position waypoints for the arm's position
  trajectory controller. In Gazebo the joint follows that to within 2%.
- The spike → velocity *gains* are not calibrated, and several supporting pieces are
  wrong (table 1.3).
- The robot adds almost nothing to the calibration problem. The gain can be fitted in
  NEST alone in minutes, scaled per joint by geometry, then checked on the robot.

## Part 1. Findings: does velocity control work?

### 1.1 How a spike becomes joint motion today

Gain left/right counts per 50 ms tick → `Decoder` (R − L, exponential filter τ, gain)
→ 4-point velocity horizon → `CommandState` integrates it into positions →
`JointTrajectory` on `/arm_controller/command`.

`arm_controller` is a `position_controllers/JointTrajectoryController`, and the Gazebo
hardware plugin exposes only position and effort interfaces for the arm. A true
velocity loop would need `effort_controllers/JointVelocityController`, controller
switching around `play_motion` and PID tuning, and would not carry over to the real
arm. **Decision: keep position streaming as the velocity interface.**

### 1.2 What works (measured)

| Check | Result |
| --- | --- |
| Constant ±0.02, ±0.1, ±0.4 rad/s on joints 5 and 1 (Gazebo) | displacement 98.2–99.8% of commanded; position error at most 14 mrad |
| Full graph trials, joint 5, goals +0.8 and −0.6 (Gazebo) | joint moved 99.8% and 100.5% of the command integral; final angle within 4 mrad of the fake-robot run |
| NEST ↔ robot coupling in `once` mode | gain counts identical tick for tick between the Gazebo run and the fake-robot run |
| Simulation facts (open items in `docs/cosim/README.md`) | `/use_sim_time` true, `/clock` 1000 Hz, `/joint_states` 50 Hz, 59 ms wall per tick |

### 1.3 What is wrong

| # | Problem | Evidence | Where |
| --- | --- | --- | --- |
| 1 | Gains are hand values, one for all joints | Top-level ±0.0011 (0.00105 for joint 0) differ from the calibration's own last result in the same file (joint 5: +0.0046 / −0.0071; joint 4: k_neg = +15.1). Joints 2, 4, 6 have 1.5–1.7× more rad per ring index: joint 4 moved 64–66% of the bump's displacement | `src/config/calibration/velocity_calibration.json`, `graph/templates.py` `decoder_params` |
| 2 | One linear gain cannot be right | Bump speed is ~0 below ≈19 gain spikes per tick, then linear. With the best linear gain (it equals 0.0011 on joint 5) joint/bump displacement still spreads 0.84–1.36 (p10–p90) and leftward moves overshoot the bump by 21% | `blocks/decoder.py` |
| 3 | τ and delay are artefacts | JSON τ = 0.10–0.83 s, joint 6 delay 4 ticks. Measured lag from gain counts to bump motion is 0 ticks, robot lag under 20 ms. Joint 6 simply trails its bump by more than 1 s | same JSON |
| 4 | Measured velocity is unusable on the wrist | `/joint_states` velocity at rest: arm_6 −1.45, arm_7 −0.39, arm_4 +0.27 rad/s. The engine's settle checks never pass and run to their 8 s timeouts, once in reset and once at finish: 16 s of a 38 s trial is waiting. `joint_velocity` in records is wrong | `cosim/gazebo_ros_engine.py` (`_do_reset`, `wait_for_settled`, `stop_and_settle`) |
| 5 | No joint-limit or velocity clamp | In one sweep trial the bump ran to the ring edge and joint 5 was commanded to 2.41 rad (limit 1.417) | `control/trajectory.py`, both robot engines |
| 6 | Records and one tool are misaligned | Graph records pair spike columns of NEST step k+1 with ring columns of step k+4. `sweep_feedback.analyse` returns NaN since commit e4d6884 (reads `tick.outputs`; the data is in `tick.inputs`) | `graph/compile.py` `JointCommandTF`, `scripts/sweep_feedback.py` |
| 7 | A 50 ms tick is 51 ms of Gazebo time | The clock-wait stepper overshoots 1 ms every tick: velocities 2% low, displacement unaffected | `cosim/stepping.py` |
| 8 | Launcher fails on a fresh clone | It bind-mounts `recovered/experiment_results`, which is not in the repo | `run_model_docker.bash:43` |

Model limits seen. They are not velocity-control bugs, but they bound what any
calibration can deliver, and after calibration they are the larger error:

- The bump stops off the goal: final bump − goal index is median +3.5, p10 −10,
  p90 +16 (median absolute error 7.9 indices, about 0.12 rad on joint 5). In the
  Gazebo trial to +0.8 rad the 0.137 rad shortfall was 0.089 from the bump stopping
  short and 0.048 from the decoder gain.
- Short moves often produce no drive: 13 of 96 interior trials did not move; the
  eight inspected had goals 25 indices away.
- Goals above index ≈165: the bump overshoots to ≈190 and stays there while the left
  gain population fires (spikes without bump motion).
- JSON limits differ from the URDF for joint 2 (upper 1.261 vs 1.571 rad).

## Part 2. Calibration plan

### 2.1 Target

In `once` mode the state bump is the model's estimate of the joint, and the gain
spikes move both. The calibration is right when the joint moves as far as the bump:
Δq = ρ · Δindex, where ρ is the rad per ring index of the encoder mapping.
**Decision: calibrate the joint to follow the bump, not to reach the goal.** Reaching
the goal is the comparator's job; folding its error into the gain gives a gain that
is right for no trial.

### 2.2 Why the old procedure is dropped

`legacy/calibrate_single_joint.py` and `training/calibration.py` stay frozen and unused:

- They regress on `dq_desired`, so the comparator's endpoint error goes into the gain.
- Mini-batches of 3 trials from the home pose often move one way only. The other
  direction's gain is then fitted to noise and blended in (β = 0.5). This is where
  +15.1 and −0.0071 come from.
- τ and delay are chosen by a grid on trial integrals, which do not depend on them.
- 100 Gazebo trials per joint, although the spike train does not depend on the robot.

### 2.3 Decoder model

`v = k_j · dz(R − L)`, then the existing filter, with
`dz(x) = sign(x) · max(|x| − θ, 0)` and `k_j = a · ρ_j / dt`.

- `a` (ring indices per spike above θ) and `θ` (spikes per tick) belong to the
  network: one fit serves all joints.
- `ρ_j` comes from the joint limits and the encoder mapping: no fitting.

Held-out error of the predicted bump displacement (83 interior trials, 4 seeds,
leave one seed out):

| Model | Median | p90 | Bias left / right |
| --- | --- | --- | --- |
| Linear (today's form), a = 0.0037 | 10.7% | 47% | +21% / +4% |
| Dead zone, θ ≈ 19, a ≈ 0.0053 | 4.5% | 12.7% | −0.5% / −0.2% |

Separate left and right gains add nothing (the transport is symmetric), and 12–24
trials already give 4–5%. Residuals do not correlate with rate, ring position or
distance.

### 2.4 Procedure

- **A. Transport fit (NEST + fake robot, about 3 min).** Run the graph over a grid of
  start and goal indices in the ring interior (starts 35–135, goals 35–150, at least
  22 apart, 4 seeds). Per tick keep left/right counts and the unwrapped circular
  centroid of the state ring. Drop the three injection ticks; flag, count and report
  trials that never move, leave the interior or show spikes without motion. Fit `a`
  and `θ` on whole-trial displacement, cross-validated by seed. Keep the dead zone
  only if it beats the linear model on held-out p90. Output: `a`, `θ`, CV table,
  speed-versus-rate plot, and a validity map (per start/goal cell: moved, endpoint
  error, tracking error).
- **B. Geometry.** `ρ_j` from the same mapping the encoder uses; `k_j = a · ρ_j / dt`
  for all seven joints.
- **C. Actuator check (Gazebo, no NEST, about 1 min per joint).** Velocity steps
  through the real command path; displacement gain `g_j`, lag and static offset from
  positions and message stamps, never from `/joint_states` velocity. Pass:
  |g_j − 1| ≤ 0.03 and lag under one tick. Otherwise use `k_j / g_j` and record it.
- **D. Smoothing.** τ = 0.15 s and delay 0 for every joint. This is a design value,
  not a fit; step E reports the lag it causes.
- **E. Validation (Gazebo, full loop).** 20 interior trials each on joints 5 and 4,
  legacy gains against new gains. Pass: joint-versus-bump displacement error median
  ≤ 6% and p90 ≤ 15%, direction bias within ±3%, joint/command ratio 0.97–1.03, no
  clamp events. The bump-versus-goal error is reported separately and is not a pass
  criterion.

## Part 3. Implementation

Code paths are relative to `src/tiago_ring_controller/src/tiago_ring_controller/`.
`scripts/`, `test/`, `legacy/` and `src/config/` are relative to the package root
`src/tiago_ring_controller/`; `docs/` and `run_model_docker.bash` to the repo root.

**Phase 0. Branch and unblock**
- `git checkout -b velocity-calibration blocks-refactor`; this plan lives in
  `docs/velocity/plan.md` (repo convention).
- `run_model_docker.bash`: add `recovered/experiment_results` to the existing
  `mkdir -p` line.

**Phase 1. Make the velocity path trustworthy** (each item with a test on the fakes)
- `cosim/gazebo_ros_engine.py`: decide stillness from position change between
  snapshots in `_do_reset`, `wait_for_settled`, `stop_and_settle`; add a
  finite-difference `velocities_fd` (with stamps) to `joint_state`.
  `blocks/joint.py` `Joint.read` and `cosim/runner.py` `legacy_collector_record`
  use it.
- `control/trajectory.py`: `CommandState.build_multi(..., limits=None)` clamps
  integrated positions and zeroes the velocity at a limit. `graph/compile.py`
  `JointCommandTF` passes each joint's limits, applies a new `Joint.max_velocity`
  (`blocks/params.py`) and sets a `clamped` flag in the command datapack.
  `build_receding_trajectory` stays untouched (legacy parity).
- `graph/compile.py` `JointCommandTF`: take ring fields from the NEST step of the
  consumed sample (small buffer keyed by `nest_step`).
  `scripts/sweep_feedback.py`: read `ring_counts` from `tick.inputs`.
- `cosim/stepping.py` `ClockWaitStepper`: step to an absolute schedule so overshoot
  does not accumulate.

**Phase 2. Transport calibration (NEST only)**
- New `training/transport.py` (pure NumPy): unwrapped centroid (reuse
  `math/circular.py` `decode_circular_centroid`), trial container with exclusion
  masks, `dead_zone`, `fit_transport` with CV by seed, `decoder_gains_from_transport`.
- `control/profiles.py`: `LegacyControlProfile.radians_per_index(...)`, the exact
  slope of `joint_to_ring_index`; `blocks/encoder.py` exposes it (2π/N for `circular`).
- `blocks/params.py` + `blocks/decoder.py`: `Decoder.dead_zone` (default 0, today's
  behaviour), applied to the signed count before the filter; thread it through
  `cosim/config.py` and `legacy_collector_record` so `decoder_predicted_dq_rad`
  stays consistent.
- New `scripts/calibrate_transport.py`: stage A and B on any graph with the
  single-joint motif (`find_primary`), start poses through `FakeRobotEngine.home`;
  writes a candidate artifact, per-tick `.npz`, report and plots to `--out` only.
- New artifact `src/config/calibration/transport_calibration.json` (`a`, `θ`, τ,
  delay, dt, CV numbers, dataset, fingerprint of the Ring / FourierReadout /
  Homeostasis / Gain parameters, actuator section); add it to both manifests in
  `test/golden/`. `velocity_calibration.json` is not edited.
- `graph/templates.py`: `decoder_params(..., calibration="legacy" | "transport")`;
  `graph/compile.py` `resolve_joint_limits` rescales the decoder gains when URDF
  limits replace the JSON ones; warn when a graph's network fingerprint differs from
  the artifact's.

**Phase 3. Robot side (Gazebo)**
- New `scripts/identify_actuator.py` (stage C), built on `scripts/cosim_gazebo_smoke.py`
  and `GazeboRosEngine`; fills the actuator section.
- New `scripts/validate_velocity.py` (stage E) on `run_graph_trial`.

**Phase 4. Switch defaults and document**
- Templates default to `calibration="transport"`; regenerate `graph/examples/` with
  `write_examples()` and `docs/blocks/blocks.md` with `scripts/write_block_reference.py`.
  The cosim-runner parity tests pin `calibration="legacy"`.
- `docs/velocity/calibration.md`: method, tables from stages A, C and E, how to rerun.
  Fill "Gazebo checks still open" in `docs/cosim/README.md`; update the README status.

Out of scope, listed in the docs as follow-ups: comparator endpoint error and its
no-drive zone, ring-edge behaviour and the choice of mapping range, closed-loop
encoder modes, the three-ring template, real hardware.

## Verification

- Host, fakes only:
  `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s test -p "test_*.py"`.
  New `test/test_transport_calibration.py`: the fit recovers a known `a` and `θ` from
  synthetic trials; centroid unwrap across the ring seam; `radians_per_index` equals
  the numeric slope of each mapping; gains scale with joint range; `dead_zone=0`
  reproduces today's decoder samples. Additions to `test/test_cosim_engines.py`: a
  transport with a constant bogus velocity still settles; commanded positions never
  pass a limit and the flag is set; mean tick equals dt under overshoot.
- Container, real NEST: the full suite (README section 3.2) plus a 12-trial
  calibration gate with held-out median error < 10%, θ in 10–30 and `a` in
  0.004–0.0065 (ranges from this session).
- Calibration:
  `python3 -B scripts/calibrate_transport.py --template two_ring_single_joint --engines nest --out /tmp/transport_cal`;
  compare its CV table with section 2.3.
- Gazebo (`roslaunch tiago_ring_controller tiago_ring_controller.launch gui:=false`):
  `scripts/cosim_gazebo_smoke.py --joints 5 6`, `scripts/identify_actuator.py`, then
  `scripts/validate_velocity.py --joints 5 4 --trials 20`; pass criteria of stages C
  and E; finish time per trial drops from 8 s to under 1 s. Record the actual output
  in `docs/velocity/calibration.md`.
