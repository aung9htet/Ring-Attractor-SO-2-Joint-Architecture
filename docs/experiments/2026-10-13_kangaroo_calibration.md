# 2026-10-13: Ring decoder calibration on Kangaroo, and held-out reach evaluation

## Hypothesis
Calibrating the ring decoder on Kangaroo with TIAGo's own procedure lets the ring drive a joint to a goal, rather than
the 1-4 % of the way the uncalibrated decoder achieved (`2026-10-13_kangaroo_ring_reach.md`).

## Config
- Code: `kangaroo/calibrate.py` (port of `calibrate_single_joint.py` as its `main()` runs it), `kangaroo/ring_reach.py`.
  Decisions D16-D19 and findings F4-F9 in `docs/kangaroo/DECISIONS.md`.
- Base commit `0ecde6a` + uncommitted `kangaroo-migration` changes. Image `ros2-kangaroo:humble`, NEST
  `HEAD@41892a50f`, PAL 4-DoF Kangaroo, floating base, MuJoCo 3.12.0.
- Calibration: `CALIBRATION_PROFILE` (owner Q9), 100 trials, goals uniform over the URDF range, reset to PAL home
  before each trial, fit every 3 trials (τ/delay grid + regularised least squares on dq_desired), blend β = 0.5.
  Seed 2026 (NEST seed 2026 + i per trial; goals from `numpy.random.default_rng(2026)`).
- Joints: left j0 (`arm_left_1_joint`) and right j1 (`arm_right_2_joint`), run concurrently (owner Q10).
- Evaluation design, fixed before seeing results: from home, held-out goals left j0 {−2.0, −1.0, −0.3, 0.8, 1.8},
  right j1 {−0.8, 0.0, 0.6, 1.8, 2.3}, seeds 100-104, each goal uncalibrated (collector profile, k = ±1e-4) and
  calibrated (`--calibration`, calibration profile), same seed.

## Commands
```bash
cd /work/src/tiago_ring_controller/src
python3 /work/kangaroo/calibrate.py left 0 &  python3 /work/kangaroo/calibrate.py right 1 &  wait
python3 /work/kangaroo/ring_reach.py <side> <joint> <goal> --seed <s> --from-home [--calibration <calibration.json>]
```
Outputs: `kangaroo/results/calibration/left_j0_seed2026_20261013_100025/`,
`kangaroo/results/calibration/right_j1_seed2026_20261013_100119/` (directory times are container UTC), and
`kangaroo/results/ring_reach/*_seed10[0-4]_*/`.

## Results: calibration
| | left j0 | right j1 |
|---|---|---|
| Final k_pos / k_neg | +5.17e-3 / −5.05e-3 | +2.55e-3 / −5.93e-3 |
| Final τ / delay | 0.376 s / 0 | 0.385 s / 0 |
| Median relative error, goals over 0.6 rad, trials 1-20 / 41-60 / 81-100 | 0.52 / 0.59 / 0.20 | 0.77 / 0.19 / 0.84 |
| max_steps stops | 13 / 100 | 23 / 100 |
| No motion (\|Δq\| < 0.01 rad) | 24 trials, desired up to 0.66 rad | 32 trials, desired up to 0.86 rad |
| Wrong-direction trials | 4 | 5 |
| Unreachable goals (above the right j1 obstruction at about 1.47 rad) | 0 | 22 |

Wall time: about 2 min per trial, 100 trials in about 3.5 h per arm, both arms in parallel.

## Results: held-out evaluation (valid trials, all from home)
| Goal | left uncal \|err\| | left cal \|err\| | | Goal | right uncal \|err\| | right cal \|err\| |
|---|---|---|---|---|---|---|
| −2.0 | 2.158 | **0.083** | | −0.8 | 2.012 | **0.335** (stopped at joint limit) |
| −1.0 | 1.173 | **0.197** | | 0.0 | 1.224 | 1.135 (overshot to joint limit) |
| −0.3 | 0.534 | 0.532 (no motion) | | 0.6 | 0.682 | **0.260** |
| +0.8 | 0.539 | 0.478 (overshoot) | | 1.8 | 0.450 | 0.344 (blocked at 1.456) |
| +1.8 | 1.496 | **0.264** | | 2.3 | 0.935 | 0.843 (blocked at 1.457) |
| **mean \|err\|** | 1.180 | **0.311** | | | 1.061 | **0.583** |
| **median relative error** | 0.96 | **0.17** | | | 0.94 | **0.72** |

Post hoc (not part of the pre-set design): on the right goals reachable below the obstruction (−0.8, 0.0, 0.6),
mean |err| 1.306 uncalibrated vs 0.576 calibrated, median relative error 0.94 vs 0.36.

Two of the first 20 evaluation runs did not start from home (left seed 101 uncalibrated at −2.083, right seed 101
calibrated at 1.224): the reset was ignored. They are kept on disk (`left_j0_seed101_20261013_100553`,
`right_j1_seed101_20261013_100703`), excluded here, and were rerun after the reset fix (D18).

## Interpretation
- **Calibration works on the left arm:** final error fell from 1.18 to 0.31 rad on average; large moves land within
  0.08-0.26 rad. Short moves (−0.3, a 0.54 rad move) still do not happen (dead zone, F8).
- **The right arm is only partly calibrated.** Its calibration did not converge (F7), and two of its five held-out
  goals are physically unreachable: right j1 is blocked at about 1.47 rad (F9). The URDF range used for the ring
  mapping and goal sampling (D11) includes an unreachable 27 % of the range.
- **Sign flips (F6)** cause the wrong-direction trials; the fit's lack of a sign constraint is inherited from TIAGo.
- Plant fidelity holds wherever the joint is not blocked: measured Δq / Σv·dt = 0.98-1.00 (from the trial files).

## Limitations
- One calibration seed per joint, 5 held-out goals per condition: indicative, not statistics.
- Two joints of eight calibrated (owner Q10).
- Robot timing is not bit-reproducible (D4).
- Uncalibrated runs use the collector profile and calibrated runs the calibration profile (Q9), so the comparison
  changes both gains and profile.
