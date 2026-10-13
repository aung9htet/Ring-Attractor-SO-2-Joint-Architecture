# 2026-10-13: Kangaroo arm joint velocity tracking

## Question
When one Kangaroo arm joint is commanded at a constant velocity through the TIAGo-style receding-horizon path,
how closely does it follow? This is the arm alone, before any ring-attractor controller is connected.

## Setup
- **Simulator:** `docker/ros2_kangaroo` image (ROS 2 Humble, MuJoCo 3.12.0, PAL apt packages listed in its
  README). PAL 4-DoF model with a floating base, in an empty world.
- **Controller:** PAL's `arm_{left,right}_controller` (JointTrajectoryController, position interface, PAL's
  PID gains unchanged).
- **Command path:** `kangaroo/arm_velocity.py` (`ArmVelocityPublisher`), `dt = 0.05 s`, 4-point horizon, the
  same as TIAGo. See `docs/kangaroo/DECISIONS.md` D1-D5.
- **Protocol:** each of the 8 arm joints, one at a time, gets a velocity of 0.3 rad/s toward the centre of its
  range for 2.0 s. It is then stopped and recorded for a further 1.0 s, then driven back to its start pose.
- **Measured signal:** `/joint_states` velocity, about 50 Hz.
- **Randomness:** none, so there is no seed. Runs are still not bit-reproducible (see Limitations).
- **Code:** branch `kangaroo-migration`, based on commit `0ecde6a`. The code was uncommitted at run time.
- **Command:**
  ```bash
  docker/ros2_kangaroo/run.bash bash -c 'ros2 launch /opt/kangaroo_sim.launch.py & sleep 40; python3 kangaroo/velocity_tracking.py'
  ```
- **Output:** `kangaroo/results/velocity_tracking/tracking.csv` and `tracking.png`. These are from the second run;
  the first run's files were overwritten.

## Results (second run)

The steady state is measured from 0.5 s to 2.0 s after the step starts. `t90` is when the joint first reaches
90% of the commanded speed. "Peak overshoot" is the maximum |measured| / |commanded| − 1 during the step.
"Stop overshoot" is the maximum |velocity| from 0.05 s after the stop command.

| joint | cmd (rad/s) | steady mean | ratio | steady std | t90 (s) | peak overshoot | stop overshoot (rad/s) |
|---|---|---|---|---|---|---|---|
| arm_left_1  | −0.30 | −0.2969 | 0.990 | 0.0070 | 0.04 | +89% | 0.164 |
| arm_left_2  | −0.30 | −0.2973 | 0.991 | 0.0090 | 0.04 | +87% | 0.170 |
| arm_left_3  | −0.30 | −0.2989 | 0.996 | 0.0084 | 0.03 | +29% | 0.093 |
| arm_left_4  | −0.30 | −0.2992 | 0.997 | 0.0087 | 0.04 | +49% | 0.045 |
| arm_right_1 | +0.30 | +0.2961 | 0.987 | 0.0069 | 0.05 | +74% | 0.167 |
| arm_right_2 | −0.30 | −0.2972 | 0.991 | 0.0067 | 0.05 | +88% | 0.146 |
| arm_right_3 | +0.30 | +0.2990 | 0.997 | 0.0085 | 0.04 | +45% | 0.081 |
| arm_right_4 | −0.30 | −0.2985 | 0.995 | 0.0092 | 0.03 | +46% | 0.055 |

Position check, from `kangaroo/test_arm_velocity.py` and the CLI: commanded displacements were reproduced to
within 0.001 rad. For example, `arm_left_3`: commanded −0.450 rad, measured −0.450 rad. Joints that were not
commanded moved less than 0.02 rad (the test passed).

## Interpretation
- **Steady state:** every joint tracks the commanded velocity to within 1.3%, with about 0.008 rad/s ripple.
- **Transients:** the transients are not small.
  - Each step starts with a velocity overshoot of +74% to +89% on joints 1-2 (shoulder) and +29% to +49% on
    joints 3-4. It settles within about 0.3 s (see the plot).
  - Stopping overshoots by up to 0.17 rad/s.
- **Likely cause (a hypothesis, not tested):**
  - The trajectory controller turns each horizon into a position target, and PAL's simulation PID is P-only
    (p = 1000, i = d = 0).
  - With no damping term, a change in the target slope rings.
  - The shoulder joints carry more of the arm's inertia, which would explain their larger overshoot.
- **Relevance to the ring controller:** with a 0.05 s tick, a 0.3 s transient spans about 6 ticks. That is
  comparable to the decoder time constants calibrated on TIAGo (τ = 0.10–0.83 s), so it may matter when the
  ring is connected.

## Limitations
- One run per joint, at one speed (0.3 rad/s), from one start pose. There is no repetition statistic yet.
- Not bit-reproducible. Simulation and controllers run asynchronously at about 1x real time; two identical runs
  differed by up to 0.003 in the steady-state ratio.
- Floating base. Arm motion may slightly disturb the base. Base height was 0.88 m and steady at rest; it was
  not measured during motion.
- Not directly comparable to the TIAGo results: different robot, simulator (MuJoCo vs Gazebo), controller and
  gains. Only the command path (receding-horizon trajectory) is the same.
