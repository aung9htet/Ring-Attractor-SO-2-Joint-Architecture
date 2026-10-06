# Gain spikes to joint velocity: path, calibration, measurements

Date: 2026-10-02, branch `velocity-calibration`. Plan and the findings that led
here: `../velocity-plan.md`. Real NEST `HEAD@41892a5`; Gazebo 11 headless (`gui:=false`,
software rendering) from the pinned image.

> **Where this stands (2026-10-04).** This report is about architecture v3 (two rings, the joint
> read once). Its transport calibration is still what v3 uses. Architecture v4 feeds the joint
> back continuously and needs no transport calibration. How to run each version: the top-level
> `README.md`; what each is: [`../architecture.md`](../../architecture.md); how they compare:
> [`../results.md`](../../results.md).

## 1. The path

```text
Gain left/right counts per 50 ms tick
  -> Decoder: s = right - left, dead zone theta, exponential filter tau, gain k   (rad/s)
  -> JointCommandTF: clip to Joint.max_velocity, attach the joint limits
  -> robot engine: CommandState integrates the 4-point horizon into positions, held at the limits
  -> JointTrajectory on /arm_controller/command
```

TIAGo's arm has no velocity interface: `arm_controller` is a
`position_controllers/JointTrajectoryController` and the Gazebo hardware plugin
exposes position and effort only. Velocity control is therefore position
streaming of the integrated velocity. It is kept: it follows the command to
within 2% (section 4) and carries over to the real arm.

## 2. Calibration target and model

In `once` mode the state bump is the model's estimate of the joint and the gain
spikes move both. The joint must move as far as the bump:
`delta_q = rho * delta_index`, with `rho` the joint angle per ring index of the
encoder mapping. Reaching the goal is the comparator's job and is not folded
into the gain.

The bump's displacement per tick is `a * dz(s)` with
`dz(s) = sign(s) * max(|s| - theta, 0)`. `a` (ring indices per spike above the
dead zone) and `theta` (spikes per tick) belong to the network: in `once` mode
the gain spike train is identical with and without the robot. The decoder gain
of joint `j` is

```text
k_j = a * rho_j / dt        rho_j = (joint_max - joint_min) / (indices the joint range covers)
```

so one fit on NEST serves all seven joints.

## 3. Stage A and B: transport fit and per-joint gains

`scripts/calibrate_transport.py --template two_ring_single_joint --engines nest`
(96 trials, start indices 35-135, goal indices 35-150, at least 22 apart, seeds
13579, 7, 99, 2024; 175 s). Verdicts: 83 `ok`, 13 `no_motion`, 0 `edge`, 0 `stuck`.

| Model | a (indices per spike) | theta (spikes per tick) | held-out median | held-out p90 | bias left | bias right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | 0.003690 | 0 | 10.4% | 47.4% | +21.3% | +3.5% |
| dead zone (chosen) | 0.005332 | 20 | 4.8% | 13.0% | -0.6% | -0.5% |

Errors are of the predicted bump displacement relative to the true one, on the
trials of the seed left out; bias is positive when the prediction travels
further than the bump. The linear gain equals the hand value on joint 5
(0.00369 * 0.01492 / 0.05 = 0.0011), so 0.0011 was the best a linear decoder
could do there.

Bump speed against gain rate (20-tick windows, left and right pooled):

| rate (spikes per tick) | 0-10 | 10-20 | 20-30 | 30-40 | 40-50 | 50-60 | 60-70 | 70-80 | 80-90 | 90-100 | 100-110 | 110-130 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| speed (indices per tick) | 0.011 | 0.035 | 0.054 | 0.086 | 0.135 | 0.184 | 0.245 | 0.289 | 0.344 | 0.389 | 0.453 | 0.523 |

Decoder gains (collector mapping, limits of `velocity_calibration.json`,
dead zone 20, tau 0.15 s, delay 0):

| joint | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rho (rad per index) | 0.01447 | 0.01401 | 0.02524 | 0.01447 | 0.02205 | 0.01492 | 0.02206 |
| gain (rad/s per spike/tick) | 0.001543 | 0.001494 | 0.002692 | 0.001543 | 0.002351 | 0.001591 | 0.002353 |
| legacy gain | 0.00105 | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0011 |

tau and delay are design values, not fits: the lag from gain counts to bump
motion is 0 ticks and the robot's lag is about 10 ms, so there is nothing to
identify. tau = 0.15 s smooths the count noise; the joint trails the bump by
about that much.

The artifact is `src/config/calibration/transport_calibration.json`. It holds
`a`, `theta`, tau, delay, the tick length, the cross-validation numbers and a
fingerprint of the network (ring, readout, comparator and gain parameters,
encoder injection parameters, neuron parameters and fitted comparator weights).
`velocity_calibration.json` is unchanged and still supplies the joint limits.

## 4. Stage C: the robot (Gazebo, no NEST)

`scripts/identify_actuator.py --joints 5 4 1`: velocity steps of +-0.02, +-0.1
and +-0.4 rad/s through `GazeboRosEngine`.

| joint | displacement gain | lag | static offset | max position error | sim time per tick | finish |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.991 | 8 ms | -2.1 mrad | 9.0 mrad | 0.0500 s (0.049-0.051) | 0.1 s |
| 4 | 0.989 | 9 ms | -0.3 mrad | 7.0 mrad | 0.0500 s | 0.1 s |
| 1 | 0.985 | 14 ms | -6.7 mrad | 17.2 mrad | 0.0501 s | 0.1 s |

All within the tolerance (|gain - 1| <= 0.03, lag under one tick), so no gain
correction is applied. Measured velocities equal the commanded ones (0.1005 for
0.1 rad/s) now that the stepper keeps an absolute schedule; before, every tick
was 51 ms and velocities were 2% low. The static offset is the position
controller's steady error under gravity; priming the command state from the
measured pose shifts the joint by that amount once per trial.

## 5. Stage E: full loop in Gazebo

`scripts/validate_velocity.py --engines full --joints 5 4 --trials 20`: 80 trials
from the start pose (joint 5 at index 113, joint 4 at index 28), interior goals at
least 30 indices away, legacy gains against the transport calibration. Tracking
is the joint displacement against `rho` times the bump displacement.

| joint, gains | usable | tracking median | p90 | bias left | bias right | joint / command | clamp ticks | bump - goal (median abs) | goal error (median abs) | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 5, legacy | 18/20 | 8.8% | 28.3% | +16.3% | -2.1% | 0.998-1.009 | 0 | 6.8 idx | 0.053 rad | no |
| 5, transport | 17/20 | 5.4% | 10.7% | -5.8% | +0.9% | 0.998-1.009 | 0 | 10.0 idx | 0.178 rad | no |
| 4, legacy | 20/20 | 31.3% | 39.6% | (none) | -29.9% | 0.999-1.000 | 0 | 4.6 idx | 0.583 rad | no |
| 4, transport | 20/20 | 3.6% | 13.0% | (none) | -2.4% | 0.999-1.000 | 0 | 4.6 idx | 0.128 rad | yes |

Criteria: median <= 6%, p90 <= 15%, direction bias within +-3%, joint / command
0.97-1.03, no clamp events. Joint 4 only moves right from its start pose, so it
has no leftward bias.

Reading:

- **The robot does what it is told**: joint / command 0.998-1.009 in every trial,
  no clamp events, 9 s reset (the `play_motion` itself) and about 25 s per trial.
- **The joint now follows the bump** on both joints: 5.4% and 3.6% median against
  8.8% and 31.3%. Joint 4 moved 69% of the bump's displacement before and moves
  98% now.
- **Joint 5 misses one criterion**: leftward moves under-travel the bump by 5.8%
  (limit 3%). The fit's own held-out bias was -0.6%; these leftward trials all
  start at index 113 and run at low gain rates, where the dead-zone line slightly
  under-predicts the speed (section 3, 20-40 spikes per tick). The +-3% criterion
  was set tighter than a two-parameter model supports on 10 trials of one start.
- **Reaching the goal got worse on joint 5** (0.178 rad against 0.053). Two causes,
  both outside the gain. The legacy decoder over-travelled leftward moves by 16%,
  which happened to cancel the bump stopping short of the goal. And with the dead
  zone the drive is zero below 20 spikes per tick, so the settle check ends the
  trial while the gain populations still fire weakly and the bump still creeps:
  the bump ended 10.0 indices from the goal instead of 6.8. On joint 4 the goal
  error fell from 0.583 to 0.128 rad.

So the calibration does what it was built for, the joint tracks the state bump,
and the remaining error is where the bump stops. That is the comparator and the
settle rule (section 7), and it is the next thing to work on.

Videos of three of these trials, legacy on the left and transport on the right,
with the joint, bump and command values overlaid (`media/videos.json` has the
numbers): `media/joint4_goal110_before_after.mp4`,
`media/joint5_goal55_before_after.mp4`, `media/joint5_goal150_before_after.mp4`.
Raw results: `media/validation.json`, `media/actuator.json`.

## 6. What was fixed in the velocity path

| Problem | Fix | Where |
| --- | --- | --- |
| Settle checks used the `/joint_states` velocity, which reads about -1.45 rad/s for a resting `arm_6_joint`: 8 s timeout in reset and at the end of every trial | Stillness and measured velocity come from position differences between joint-state stamps (`velocities_fd`) | `cosim/gazebo_ros_engine.py`, `blocks/joint.py`, `cosim/runner.py` |
| No limit on commanded positions or velocities | Commands carry the joint limits and are clipped to `Joint.max_velocity`; the engines hold the commanded position at a limit and flag it (`clamped_joints`, `velocity_clamped`) | `control/trajectory.py`, `graph/compile.py`, both robot engines |
| Every tick was 51 ms of Gazebo time | The clock-wait stepper follows an absolute schedule | `cosim/stepping.py` |
| Graph records paired spike columns with ring columns three steps later | Ring fields are taken from the NEST step of the consumed sample | `graph/compile.py` |
| `sweep_feedback.analyse` returned NaN | It reads `ring_counts` from the engine datapacks | `scripts/sweep_feedback.py` |
| Launcher failed on a fresh clone | It creates `recovered/experiment_results` | `run_model_docker.bash` |

## 7. Limits of the model seen on the way

These bound what the decoder can deliver and are not addressed here. The
bounded ring (`bounded-ring.md`, `two_ring_single_joint(bounds="bounded")`) takes
up the limit, seam and settle items below: it has its own transport calibration
(`transport_calibration_bounded.json`), and this page's artifact stays the one
for the legacy ring.

- The bump stops off the goal: final bump minus goal index has a median absolute
  value of about 8 indices on interior trials (0.12 rad on joint 5).
- Goals about 25 indices away often produce no drive (13 of 96 trials).
- Goals above index 165: the bump overshoots to about 190 and stays there while
  the left gain population fires. The calibration and validation trials stay in
  indices 35-150; the limit clamp keeps the robot safe outside.
- The settle rule looks at the decoder's drive. With the dead zone it ends a trial
  while the gain populations still fire below 20 spikes per tick and the bump still
  creeps toward the goal.
- `once` mode only. Closed-loop encoder modes and the three-ring template keep
  the legacy gains until their transport is characterised.

## 8. Rerunning

```bash
# inside the container, package root
python3 -B scripts/calibrate_transport.py --template two_ring_single_joint --engines nest --out /tmp/transport_cal
python3 -B scripts/calibrate_transport.py --trials-from /tmp/transport_cal --out /tmp/refit      # refit without NEST
# with the simulation running (roslaunch tiago_ring_controller tiago_ring_controller.launch gui:=false)
python3 -B scripts/identify_actuator.py --joints 5 4 1 --artifact /tmp/transport_cal/transport_calibration.json
python3 -B scripts/validate_velocity.py --engines full --joints 5 4 --trials 20 --out /tmp/validation
# the bounded ring: bounded-ring.md section 9
```

Recalibrate when a ring, readout, comparator or gain parameter, the tick
length or the encoder injection changes: `run_graph.py` warns when a graph uses
the calibration on a network with a different fingerprint. Installing a new
artifact (`--install`) changes a scientific input: update its two entries in
`test/golden/` and this page.
