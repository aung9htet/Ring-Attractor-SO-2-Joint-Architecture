# Kangaroo arm velocity control (ROS 2)

Moves each joint of PAL Kangaroo's arms individually by velocity, the same way the TIAGo controller does: a
velocity command becomes a short receding-horizon `JointTrajectory` sent to the arm's trajectory controller.
Design decisions are in `docs/kangaroo/DECISIONS.md`.

Everything runs inside the `docker/ros2_kangaroo` container, from the repo root. Start the simulation first:

```bash
docker/ros2_kangaroo/run.bash
ros2 launch /opt/kangaroo_sim.launch.py &      # simulation + left/right arm controllers
```

## Move a joint

```bash
python3 kangaroo/arm_velocity.py left 2 -0.3 1.5   # arm_left_3_joint at -0.3 rad/s for 1.5 s
```

Arguments: `side` (`left` | `right`), `joint` (0-3, meaning `arm_<side>_1..4_joint`), `velocity` (rad/s) and
`seconds`. It prints the start and end position and the expected change.

From Python, the interface mirrors TIAGo's `TiagoPublisher`:

```python
arm = ArmVelocityPublisher(node, 'left')
arm.publish_receding_trajectory(joint_idx, [v] * HORIZON)  # call once per 0.05 s tick
arm.stop_joint(joint_idx)
```

## Drive a joint with the ring attractor

The TIAGo single-joint ring loop, ported: the ring's decoded velocity drives one joint toward a goal. Run from
the ring package directory, because `SingleRingModel` loads `./config/...`:

```bash
cd /work/src/tiago_ring_controller/src
python3 /work/kangaroo/ring_reach.py left 0 -0.8 [--seed 12345]   # arm_left_1_joint toward -0.8 rad
```

Each run writes `kangaroo/results/ring_reach/<side>_j<joint>_seed<seed>_<time>/trial.json`. One ring per process
(NEST's kernel is global): to drive both arms at once, start one process per arm. Without `--calibration` the
decoder is uncalibrated (k = ±1e-4): joints move the right way but only a few percent of the distance
(`docs/experiments/2026-10-13_kangaroo_ring_reach.md`). With calibration (below), left j0 lands within 0.08-0.26 rad
on large moves (`docs/experiments/2026-10-13_kangaroo_calibration.md`). Right j1 (blocked above 1.46 rad) is
calibrated over its measured range from a mid-range start (j1 = 0.161; PAL home leaves only 0.14 rad upward): it
moves the right way in both directions, lands within 0.01-0.28 rad on upward moves, and overshoots downward moves by
up to 0.50 rad (`docs/experiments/2026-10-13_kangaroo_joint_range_recalibration.md`, DECISIONS.md D25-D26, F13).

## Measure a joint's reachable range

The URDF range is not always reachable: right j1 is blocked at about 1.46 rad (URDF 2.443). As TIAGo's
`find_joint_limits_manually`, drive the joint to its stops (from home, ±1 rad/s for 5 s) and read the measured position:

```bash
cd /work/src/tiago_ring_controller/src
python3 /work/kangaroo/joint_limits.py right 1   # writes kangaroo/results/joint_limits/right_j1_<time>/limits.json
```

Pass the file to `calibrate.py` or `ring_reach.py` with `--limits <path>/limits.json` (default: URDF range). A
calibrated reach always uses the range its calibration was fitted on (DECISIONS.md D20).

## Calibrate the decoder

Port of TIAGo's `calibrate_single_joint.py` (100 trials from PAL's home pose toward random goals, decoder refitted
every 3 trials, β = 0.5), under TIAGo's calibration profile:

```bash
cd /work/src/tiago_ring_controller/src
python3 /work/kangaroo/calibrate.py left 0 [--trials 100] [--seed 2026]
python3 /work/kangaroo/calibrate.py right 1 --balanced --start-joint 0.161 --limits <path>/limits.json
python3 /work/kangaroo/ring_reach.py left 0 -0.8 --from-home --calibration <path>/calibration.json
```

Calibration writes `kangaroo/results/calibration/<side>_j<joint>_seed<seed>_<time>/calibration.json` (final gains plus
every trial). `ring_reach.py --calibration` runs under the profile the gains were fitted with (DECISIONS.md D17).
`--balanced` alternates goals above and below the start (D23); `--start-joint` starts the calibrated joint at that
angle instead of home (D25), and `ring_reach.py --from-home --calibration` then resets to that start pose.

## Test

```bash
python3 kangaroo/test_arm_velocity.py -v
cd /work/src/tiago_ring_controller/src && python3 /work/kangaroo/test_ring_reach.py -v
cd /work/src/tiago_ring_controller/src && python3 /work/kangaroo/test_calibrate.py -v   # about 10 min
cd /work/src/tiago_ring_controller/src && python3 /work/kangaroo/test_joint_limits.py -v
```

`test_arm_velocity.py` checks that each of the 8 joints moves by `v·T` (±10%) while every other arm joint stays
within 0.02 rad, and that both arms can be commanded at once. `test_ring_reach.py` checks that the same seed gives
the same ring spikes (and a different seed different ones), and that on the robot the ring moves the joint toward
the goal, by Σv·dt (±10%), without moving any other arm joint.

## Velocity tracking experiment

```bash
python3 kangaroo/velocity_tracking.py   # writes kangaroo/results/velocity_tracking/
```

Results and interpretation: `docs/experiments/2026-10-13_kangaroo_arm_velocity_tracking.md`.

| File | Role |
|---|---|
| `arm_velocity.py` | `ArmVelocityPublisher` plus the CLI. Reuses `build_receding_trajectory` from the TIAGo package |
| `test_arm_velocity.py` | Behaviour test (needs the simulation running) |
| `velocity_tracking.py` | Commanded vs measured velocity, CSV + plot |
| `ring_reach.py` | `RingDrive` (seeded ring) and `run_trial` (ROS 2 port of the TIAGo collector loop), plus the CLI |
| `test_ring_reach.py` | Ring behaviour tests (spikes: NEST only; reach: simulation running) |
| `calibrate.py` | `reset_arm` and `calibrate`: decoder calibration, reusing TIAGo's fit unchanged |
| `test_calibrate.py` | Calibration behaviour tests: reset reaches home; learned decoder is correctly signed, also with balanced goals near a limit and from a mid-range start pose |
| `joint_limits.py` | `measure_limits`: reachable range by driving to the stops, plus the CLI |
| `test_joint_limits.py` | Range test: right j1 range inside the URDF range and stopped by its obstruction |
