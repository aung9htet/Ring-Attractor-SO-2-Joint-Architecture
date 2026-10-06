# Numbers for the paper

Gazebo, joints arm_1 to arm_7, 8 goals per joint from the reset pose (fewer where MoveIt reports a collision). PID: the frozen baseline. v3: two rings on the bounded ring. v4: three rings with body feedback. Both ring systems at 3 NEST steps per 50 ms robot tick. Settled: the error stays under 0.05 rad for 5 samples. Written by `scripts/results_report.py`; figures with the same names are next to this file.

## All joints

| system | trials | goal error, median / p90 / max (rad) | 95% interval of the median (rad) | within 0.05 rad at the end | settled | settling time, median (s) | time to 90%, median (s) | overshoot, p90 / max (rad) | still when the trial ends, median range over the last 0.5 s (rad) | no drive | wrong way | clamped | base pushed over 1 cm | reset failed | wall time per trial (s) | per tick (ms) |
|---|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PID (PID baseline) | 56 | 0.027 / 0.048 / 0.054 | 0.0191 - 0.0348 | 95% | 100% | 2.23 | 2.10 | 0.093 / 0.093 | 0.001 | 0 | 0 | 0 | 0 | - | 30 | 51 |
| B3 (v3, bounded ring, 3x) | 56 | 0.091 / 0.257 / 0.344 | 0.0733 - 0.1319 | 32% | 54% | 2.85 | 2.33 | 0.206 / 0.344 | 0.004 | 0 | 0 | 2 | 0 | 0 | 15 | 74 |
| F3 (v4, 3x) | 56 | 0.008 / 0.022 / 0.045 | 0.0062 - 0.0110 | 100% | 100% | 2.35 | 1.88 | 0.023 / 0.050 | 0.001 | 0 | 0 | 0 | 0 | 0 | 25 | 92 |

## Goal error per joint, median / p90 (rad) (figure `error_by_joint`)

| joint | PID | B3 | F3 |
|---|---|---|---|
| arm_1 | 0.041 / 0.048 | 0.035 / 0.203 | 0.008 / 0.010 |
| arm_2 | 0.016 / 0.032 | 0.091 / 0.185 | 0.007 / 0.015 |
| arm_3 | 0.027 / 0.050 | 0.114 / 0.165 | 0.006 / 0.021 |
| arm_4 | 0.016 / 0.033 | 0.103 / 0.185 | 0.008 / 0.027 |
| arm_5 | 0.047 / 0.053 | 0.257 / 0.330 | 0.010 / 0.016 |
| arm_6 | 0.017 / 0.036 | 0.050 / 0.098 | 0.009 / 0.022 |
| arm_7 | 0.040 / 0.046 | 0.120 / 0.186 | 0.015 / 0.031 |

## Settling time per joint, median in s (trials settled / trials) (figure `settling_time`)

| joint | PID | B3 | F3 |
|---|---|---|---|
| arm_1 | 2.9 (8/8) | 3.1 (7/8) | 2.5 (8/8) |
| arm_2 | 1.8 (8/8) | 3.3 (2/8) | 2.1 (8/8) |
| arm_3 | 2.2 (8/8) | 2.8 (4/8) | 2.5 (8/8) |
| arm_4 | 1.8 (8/8) | 2.8 (4/8) | 1.5 (8/8) |
| arm_5 | 4.2 (8/8) | 2.8 (3/8) | 2.5 (8/8) |
| arm_6 | 1.8 (8/8) | 1.8 (6/8) | 1.9 (8/8) |
| arm_7 | 2.7 (8/8) | 2.6 (4/8) | 2.3 (8/8) |

## Disturbances on arm_2, arm_5, arm_6 (figure `robustness`)

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

## Ratios

- Goal error, v3 over v4: median 11.1x, p90 11.5x, largest 7.7x.
- Trials settled: v3 30 of 56, v4 56 of 56.
