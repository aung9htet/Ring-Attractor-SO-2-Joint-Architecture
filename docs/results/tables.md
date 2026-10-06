# Result tables

Written by `scripts/results_report.py` from `data/`; the reading of these numbers is in [`../results.md`](../results.md).
Goal error is |goal - final joint angle|. Settled: the error stays under 0.05 rad for 5 samples (the paper's definition); the settling time is the first time it does.

## Systems

| key | system | what it is |
|---|---|---|
| L | original scripts | The frozen original ring controller (`legacy/single_joint_data_collector.py`) |
| PID | PID baseline | The PID of the draft (`legacy/single_joint_pid_data_collector.py`): three-term PID with fixed gains kp 2.0, ki 0.1, kd 0.05 for every joint, integral limit 2.0, output limit 0.5 rad/s |
| P | upper reference: tuned controller | A controller tuned by grid search for these joints (it came out proportional: kp 8, 60% of the joint's velocity limit), in the same 50 ms loop as the ring controllers (`scripts/run_baselines.py`) |
| N | upper reference: robot's own controller | TIAGo's joint trajectory controller given the goal as one trajectory point (`scripts/run_baselines.py`) |
| G | v3, legacy ring, hand gains | Two rings, joint range on indices 5-195, gains of `velocity_calibration.json` |
| T | v3, legacy ring, transport | Two rings, legacy ring, decoder gains from the transport calibration |
| B1 | v3, bounded ring, 1x | Two rings on the bounded ring, one NEST step per robot tick |
| B3 | v3, bounded ring, 3x | Two rings on the bounded ring, three NEST steps per robot tick (the v3 default) |
| F1 | v4, 1x | Three rings with body feedback, one NEST step per robot tick |
| F3 | v4, 3x | Three rings with body feedback, three NEST steps per robot tick (the v4 default) |

## 1. Every system, Gazebo, joints arm_1 to arm_7, goals from the reset pose

| system | trials | goal error, median / p90 / max (rad) | 95% interval of the median (rad) | within 0.05 rad at the end | settled | settling time, median (s) | time to 90%, median (s) | overshoot, p90 / max (rad) | still when the trial ends, median range over the last 0.5 s (rad) | no drive | wrong way | clamped | base pushed over 1 cm | reset failed | wall time per trial (s) | per tick (ms) |
|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L (original scripts) | 56 | 0.260 / 0.796 / 1.211 | 0.2058 - 0.3685 | 14% | 29% | 13.30 | 12.07 | 0.135 / 0.308 | 0.001 | 2 | 0 | 0 | 0 | - | 53 | 77 |
| PID (PID baseline) | 56 | 0.027 / 0.048 / 0.054 | 0.0191 - 0.0348 | 95% | 100% | 2.23 | 2.10 | 0.093 / 0.093 | 0.001 | 0 | 0 | 0 | 0 | - | 30 | 51 |
| P (upper reference: tuned controller) | 56 | 0.0000 / 0.0002 / 0.0005 | 0.0000 - 0.0001 | 100% | 100% | 0.85 | 0.85 | 0.001 / 0.005 | 0.000 | 0 | 0 | 0 | 0 | 0 | 13 | 52 |
| N (upper reference: robot's own controller) | 56 | 0.001 / 0.005 / 0.006 | 0.0007 - 0.0018 | 100% | 100% | 0.95 | 0.90 | 0.003 / 0.006 | 0.000 | 0 | 0 | 0 | 0 | 0 | 13 | 52 |
| B3 (v3, bounded ring, 3x) | 56 | 0.091 / 0.257 / 0.344 | 0.0733 - 0.1319 | 32% | 54% | 2.85 | 2.33 | 0.206 / 0.344 | 0.004 | 0 | 0 | 2 | 0 | 0 | 15 | 74 |
| F3 (v4, 3x) | 56 | 0.008 / 0.022 / 0.045 | 0.0062 - 0.0110 | 100% | 100% | 2.35 | 1.88 | 0.023 / 0.050 | 0.001 | 0 | 0 | 0 | 0 | 0 | 25 | 92 |

Every system ran the same 54 goals.

## 1a. v4 against each other system on the same goals (paired)

For every goal both systems ran, the difference of their goal errors. 'Closer in' is the share of goals in which that system ended nearer the goal. The sign test asks how likely a split this uneven would be if neither were better; under 0.05 is read as a real difference.

| v4 against | shared trials | median of (v4 error - other error) (rad) | v4 closer in | other closer in | sign test p | reading |
|---|---:|---:|---:|---:|---:|---|
| original scripts | 54 | -0.247 | 100% | 0% | under 0.001 | v4 is closer |
| PID baseline | 54 | -0.014 | 87% | 13% | under 0.001 | v4 is closer |
| upper reference: tuned controller | 54 | +0.008 | 2% | 98% | under 0.001 | upper reference: tuned controller is closer |
| upper reference: robot's own controller | 54 | +0.007 | 17% | 83% | under 0.001 | upper reference: robot's own controller is closer |
| v3, bounded ring, 3x | 54 | -0.076 | 93% | 7% | under 0.001 | v4 is closer |

## 2. Goal error per joint, median / p90 (rad)

| joint | L | PID | P | N | B3 | F3 |
|---|---|---|---|---|---|---|
| arm_1 | 0.231 / 0.344 | 0.041 / 0.048 | 0.0001 / 0.0003 | 0.0000 / 0.0001 | 0.035 / 0.203 | 0.008 / 0.010 |
| arm_2 | 0.139 / 0.310 | 0.016 / 0.032 | 0.0001 / 0.0004 | 0.004 / 0.005 | 0.091 / 0.185 | 0.007 / 0.015 |
| arm_3 | 0.574 / 0.914 | 0.027 / 0.050 | 0.0000 / 0.0000 | 0.001 / 0.002 | 0.114 / 0.165 | 0.006 / 0.021 |
| arm_4 | 0.085 / 0.290 | 0.016 / 0.033 | 0.0000 / 0.0001 | 0.001 / 0.001 | 0.103 / 0.185 | 0.008 / 0.027 |
| arm_5 | 0.704 / 1.080 | 0.047 / 0.053 | 0.0000 / 0.0002 | 0.005 / 0.005 | 0.257 / 0.330 | 0.010 / 0.016 |
| arm_6 | 0.155 / 0.384 | 0.017 / 0.036 | 0.0001 / 0.0002 | 0.002 / 0.003 | 0.050 / 0.098 | 0.009 / 0.022 |
| arm_7 | 0.442 / 0.652 | 0.040 / 0.046 | 0.0000 / 0.0001 | 0.001 / 0.001 | 0.120 / 0.186 | 0.015 / 0.031 |

## 3. Settling time per joint, median in s (trials settled / trials)

| joint | L | PID | P | N | B3 | F3 |
|---|---|---|---|---|---|---|
| arm_1 | 14.6 (2/8) | 2.9 (8/8) | 1.2 (8/8) | 1.3 (8/8) | 3.1 (7/8) | 2.5 (8/8) |
| arm_2 | 9.0 (7/8) | 1.8 (8/8) | 0.7 (8/8) | 0.9 (8/8) | 3.3 (2/8) | 2.1 (8/8) |
| arm_3 | - (0/8) | 2.2 (8/8) | 0.8 (8/8) | 0.9 (8/8) | 2.8 (4/8) | 2.5 (8/8) |
| arm_4 | 12.7 (3/8) | 1.8 (8/8) | 0.6 (8/8) | 0.9 (8/8) | 2.8 (4/8) | 1.5 (8/8) |
| arm_5 | - (0/8) | 4.2 (8/8) | 1.8 (8/8) | 2.0 (8/8) | 2.8 (3/8) | 2.5 (8/8) |
| arm_6 | 15.1 (3/8) | 1.8 (8/8) | 0.8 (8/8) | 0.9 (8/8) | 1.8 (6/8) | 1.9 (8/8) |
| arm_7 | 13.9 (1/8) | 2.7 (8/8) | 1.3 (8/8) | 1.4 (8/8) | 2.6 (4/8) | 2.3 (8/8) |

## 4. Repeatability: range of the final angle over three runs of the same goal (rad)

| system | goals | median range | largest range |
|---|---:|---:|---:|
| L (original scripts) | 8 | 0.0005 | 0.0017 |
| PID (PID baseline) | 8 | 0.0001 | 0.0018 |
| P (upper reference: tuned controller) | 8 | 0.0000 | 0.0001 |
| N (upper reference: robot's own controller) | 8 | 0.0001 | 0.0008 |
| B3 (v3, bounded ring, 3x) | 8 | 0.0003 | 0.0016 |
| F3 (v4, 3x) | 8 | 0.0133 | 0.0272 |

## 5. Disturbances, Gazebo: the PID baseline, v3 and v4 (both at 3x)

| condition | system | trials | goal error, median / p90 / max (rad) | settled | overshoot, p90 / max (rad) | time to 90%, median (s) | bumps lost | wrong way |
|---|---|---:|---|---:|---|---:|---:|---:|
| nominal | PID baseline | 18 | 0.022 / 0.049 / 0.054 | 100% | 0.093 / 0.093 | 1.93 | 0 | 0 |
| nominal | v3 | 18 | 0.069 / 0.202 / 0.286 | 50% | 0.174 / 0.354 | 2.50 | 0 | 0 |
| nominal | v4 | 18 | 0.011 / 0.019 / 0.046 | 100% | 0.019 / 0.022 | 1.98 | 0 | 0 |
| nominal | upper reference: tuned controller | 18 | 0.0000 / 0.0002 / 0.0003 | 100% | 0.001 / 0.002 | 0.88 | 0 | 0 |
| motor gain x0.7 | PID baseline | 18 | 0.033 / 0.056 / 0.064 | 100% | 0.091 / 0.092 | 2.75 | 0 | 0 |
| motor gain x0.7 | v3 | 18 | 0.314 / 0.662 / 1.215 | 6% | 0.000 / 0.000 | 2.48 | 0 | 0 |
| motor gain x0.7 | v4 | 18 | 0.013 / 0.028 / 0.039 | 100% | 0.015 / 0.028 | 2.80 | 0 | 0 |
| motor gain x0.7 | upper reference: tuned controller | 18 | 0.0000 / 0.0001 / 0.0003 | 100% | 0.001 / 0.003 | 1.23 | 0 | 0 |
| motor gain x1.3 | PID baseline | 18 | 0.017 / 0.046 / 0.049 | 100% | 0.094 / 0.094 | 1.50 | 0 | 0 |
| motor gain x1.3 | v3 | 18 | 0.203 / 0.855 / 0.950 | 33% | 0.869 / 0.996 | 2.38 | 0 | 0 |
| motor gain x1.3 | v4 | 18 | 0.010 / 0.022 / 0.024 | 100% | 0.199 / 0.357 | 1.33 | 0 | 0 |
| motor gain x1.3 | upper reference: tuned controller | 18 | 0.0000 / 0.0004 / 0.0033 | 94% | 0.095 / 0.197 | 0.70 | 0 | 0 |
| push 0.2 rad | PID baseline | 18 | 0.022 / 0.048 / 0.053 | 100% | 0.102 / 0.146 | 2.05 | 0 | 0 |
| push 0.2 rad | v3 | 18 | 0.234 / 0.373 / 0.485 | 11% | 0.373 / 0.553 | 2.50 | 0 | 0 |
| push 0.2 rad | v4 | 18 | 0.009 / 0.020 / 0.031 | 100% | 0.156 / 0.176 | 1.83 | 0 | 0 |
| push 0.2 rad | upper reference: tuned controller | 18 | 0.0000 / 0.0004 / 0.0004 | 100% | 0.195 / 0.208 | 0.90 | 0 | 0 |
| held 2 s | PID baseline | 18 | 0.041 / 0.055 / 0.060 | 100% | 0.094 / 0.094 | 3.95 | 0 | 0 |
| held 2 s | v3 | 18 | 0.533 / 1.530 / 2.224 | 0% | 0.000 / 0.000 | 0.98 | 0 | 0 |
| held 2 s | v4 | 18 | 0.007 / 0.023 / 0.038 | 100% | 0.015 / 0.023 | 3.93 | 0 | 0 |
| held 2 s | upper reference: tuned controller | 18 | 0.0001 / 0.0002 / 0.0008 | 100% | 0.001 / 0.002 | 0.85 | 0 | 0 |

## 7. NEST with the fake robot, v3 against v4

| condition | system | trials | goal error, median / p90 / max (rad) | settled | overshoot, p90 / max (rad) | time to 90%, median (s) | bumps lost | wrong way |
|---|---|---:|---|---:|---|---:|---:|---:|
| all joints, 3 starts x 3 seeds x 8 goals | v3 | 504 | 0.063 / 0.194 / 0.531 | 65% | 0.117 / 0.306 | 2.10 | 0 | 0 |
| all joints, 3 starts x 3 seeds x 8 goals | v4 | 504 | 0.011 / 0.027 / 0.050 | 100% | 0.031 / 0.136 | 1.55 | 0 | 0 |
| motor gain x0.7 (arm_2, arm_5, arm_6) | v3 | 48 | 0.245 / 0.636 / 1.088 | 6% | 0.000 / 0.000 | 2.08 | 0 | 0 |
| motor gain x0.7 (arm_2, arm_5, arm_6) | v4 | 48 | 0.010 / 0.027 / 0.045 | 100% | 0.019 / 0.029 | 2.70 | 0 | 0 |
| motor gain x1.3 (arm_2, arm_5, arm_6) | v3 | 48 | 0.151 / 0.458 / 0.789 | 52% | 0.515 / 0.797 | 2.08 | 0 | 0 |
| motor gain x1.3 (arm_2, arm_5, arm_6) | v4 | 48 | 0.014 / 0.036 / 0.056 | 98% | 0.079 / 0.215 | 1.10 | 0 | 0 |
| push 0.2 rad (arm_2, arm_5, arm_6) | v3 | 48 | 0.239 / 0.357 / 0.512 | 12% | 0.306 / 0.467 | 2.10 | 0 | 0 |
| push 0.2 rad (arm_2, arm_5, arm_6) | v4 | 48 | 0.011 / 0.024 / 0.029 | 100% | 0.184 / 0.197 | 1.85 | 0 | 0 |
| held 2 s (arm_2, arm_5, arm_6) | v3 | 48 | 0.330 / 1.126 / 1.903 | 12% | 0.000 / 0.009 | 1.00 | 0 | 0 |
| held 2 s (arm_2, arm_5, arm_6) | v4 | 48 | 0.007 / 0.015 / 0.026 | 100% | 0.025 / 0.058 | 3.83 | 0 | 0 |
| goals exactly at the joint limits (arm_3, arm_6) | v3 | 12 | 0.167 / 0.301 / 0.481 | 17% | 0.000 / 0.000 | 2.20 | 0 | 0 |
| goals exactly at the joint limits (arm_3, arm_6) | v4 | 12 | 0.011 / 0.023 / 0.039 | 100% | 0.000 / 0.000 | 1.85 | 0 | 0 |
| time scale 1 (arm_5, arm_6) | v3 | 48 | 0.074 / 0.311 / 0.628 | 62% | 0.101 / 0.269 | 5.65 | 0 | 0 |
| time scale 1 (arm_5, arm_6) | v4 | 48 | 0.010 / 0.030 / 0.104 | 94% | 0.021 / 0.033 | 5.43 | 0 | 0 |
| time scale 2 (arm_5, arm_6) | v3 | 48 | 0.086 / 0.203 / 0.311 | 65% | 0.117 / 0.267 | 3.03 | 0 | 0 |
| time scale 2 (arm_5, arm_6) | v4 | 48 | 0.011 / 0.035 / 0.060 | 98% | 0.019 / 0.084 | 2.48 | 0 | 0 |

## 8. v4: where the three bumps end

| run | trials | predicted - state, median / p90 (indices) | predicted - goal bump (indices) | goal bump - goal (indices) | bumps lost |
|---|---:|---|---|---|---:|
| Gazebo, comparison set | 56 | 0.31 / 0.95 | 0.38 / 1.23 | 0.50 / 0.98 | 0 |
| NEST, all joints | 504 | 0.36 / 0.87 | 0.44 / 0.99 | 0.61 / 1.06 | 0 |
| Gazebo, motor gain x0.7 | 18 | 0.34 / 0.83 | 0.49 / 1.06 | 0.46 / 1.09 | 0 |
| Gazebo, motor gain x1.3 | 18 | 0.32 / 0.74 | 0.48 / 0.95 | 0.62 / 1.01 | 0 |
| Gazebo, push 0.2 rad | 18 | 0.38 / 0.93 | 0.35 / 1.42 | 0.40 / 1.00 | 0 |
| Gazebo, held 2 s | 18 | 0.33 / 0.88 | 0.68 / 1.12 | 0.59 / 1.05 | 0 |

## 9. Acceptance criteria of v4 (written before the runs)

| criterion | target | v3 | v4 | met |
|---|---|---|---|---|
| bumps lost, wrong-way or clamped trials | 0 | 2 | 0 | yes |
| goal error, all joints, median / p90 | <= 0.045 / 0.12 rad (2 / 4.5 indices) | 0.091 / 0.257 rad | 0.008 / 0.022 rad (0.4 / 1.0 indices) | yes |
| goal error p90 with the motor gain x0.7 / x1.3 | within 1.5x of the nominal p90 (0.019 rad) | x0.7: 0.662, x1.3: 0.855 | x0.7: 0.028, x1.3: 0.022 | yes |
| predicted bump - goal bump at rest, median | <= 1 index | no predictive ring | 0.38 | yes |
| goal bump - goal index at rest, median | <= 1 index | not recorded | 0.50 | yes |
| time to 90% of the move, median | <= 1.5x of v3 at the same time scale | 2.33 s | 1.88 s | yes |
| goal error p90 after a 0.2 rad push | back within the nominal error (0.019 rad p90) | 0.373 | 0.020 | yes |
