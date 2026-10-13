# 2026-10-13: Measured joint range and right j1 recalibration on Kangaroo

## Hypothesis
Right j1 is blocked at about 1.47 rad, 27 % short of its URDF limit (F9 in `2026-10-13_kangaroo_calibration.md`). If
the ring maps the measured reachable range instead of the URDF range, and the decoder is recalibrated over it with
the same protocol, held-out reaches on right j1 land closer to their goals than with the first calibration.

## Config
- Code: `kangaroo/joint_limits.py` (port of TIAGo's `find_joint_limits_manually`), `--limits` in
  `kangaroo/calibrate.py` and `kangaroo/ring_reach.py`. Decisions D20-D22 in `docs/kangaroo/DECISIONS.md`; owner Q12 (b).
- Base commit `0ecde6a` + uncommitted `kangaroo-migration` changes. Image `ros2-kangaroo:humble`, NEST
  `HEAD@41892a50f`, PAL 4-DoF Kangaroo, floating base, MuJoCo 3.12.0.
- Range: from PAL home, ±1 rad/s for 5 s each way, measured position, arm reset to home before each sweep (D20).
- Calibration: as the first run (seed 2026, 100 trials, fit every 3, β = 0.5, `CALIBRATION_PROFILE`), goals uniform
  over the measured range. Only right j1 ran (the first run calibrated both arms in parallel).
- Evaluation design fixed before the recalibration existed (D22): from home, goals −0.8 / 0.0 / 0.6 / −0.4 / 1.0 with
  seeds 100 / 101 / 102 / 105 / 106; conditions uncalibrated (URDF range), first calibration, new calibration.

## Commands
```bash
cd /work/src/tiago_ring_controller/src
python3 /work/kangaroo/joint_limits.py right 1          # twice
python3 /work/kangaroo/joint_limits.py left 0
python3 /work/kangaroo/calibrate.py right 1 --seed 2026 --limits /work/kangaroo/results/joint_limits/right_j1_20261013_125903/limits.json
python3 /work/kangaroo/ring_reach.py right 1 <goal> --seed <s> --from-home [--calibration <calibration.json>]
```
Outputs (directory times are container UTC): `kangaroo/results/joint_limits/{right_j1_20261013_125903,
right_j1_20261013_125954, left_j0_20261013_125925}/`, `kangaroo/results/calibration/right_j1_seed2026_20261013_140925/`,
`kangaroo/results/ring_reach/right_j1_seed10[0-7]_20261013_1{30810,30857,30925,30953,41047,41123,41159,41247,41320,41410}/`.

## Results: measured range
| Joint | URDF | Measured | |
|---|---|---|---|
| right j1 | [−1.134, +2.443] | [−1.1396, +1.4623] | identical to 4 decimals on 2 runs |
| left j0 | [−2.618, +2.618] | [−2.6735, +2.6714] | past the URDF limit by up to 0.055 (F10) |

## Results: recalibration (right j1, 100 trials)
| | first calibration (URDF range) | new calibration (measured range) |
|---|---|---|
| Final k_pos / k_neg | +2.55e-3 / −5.93e-3 | **−7.80e-3** / −2.85e-3 |
| Final τ / delay | 0.385 s / 0 | 0.709 s / 0 |
| Median relative error, goals over 0.6 rad, trials 1-20 / 41-60 / 81-100 | 0.77 / 0.19 / 0.84 | 0.29 / 0.11 / 0.13 |
| Goals needing a positive move (above home 1.321) | 26 / 100 | 5 / 100 |
| Fits with a wrong-signed gain | 13 of 34 | 23 of 34 |
| Trials run with a wrong-signed decoder | 27 | 58 |
| Wrong-direction trials | 7 | 4 |
| No motion (\|Δq\| < 0.01 rad) | 32 | 23 |
| max_steps stops | 23 | 28 |

Wall time: about 1 h (about 36 s per trial), against about 3.5 h for the first run; cause not investigated (the first
run shared the machine with the left-arm calibration).

## Results: held-out evaluation (D22, all from home)
| Goal (seed) | uncal \|err\| | first cal \|err\| | new cal \|err\| |
|---|---|---|---|
| −0.8 (100) | 2.012 | 0.335 (stopped at lower stop −1.135) | **0.147** |
| −0.4 (105) | 1.612 | 0.735 (overshot to −1.135) | **0.010** |
| 0.0 (101) | 1.224 | 1.135 (overshot to −1.135) | **0.410** |
| 0.6 (102) | 0.682 | 0.260 | **0.115** |
| 1.0 (106) | 0.322 (no motion) | 0.321 (no motion) | **0.051** |
| **mean \|err\|** | 1.170 | 0.557 | **0.147** |
| **median relative error** | 0.943 | 0.427 | **0.158** |

None of the five new-calibration reaches decoded a positive velocity, so the wrong-signed k_pos was not used.
The −0.8, 0.0 and 0.6 first-calibration and uncalibrated runs are those of the first evaluation (the invalid
`right_j1_seed101_20261013_100703` stays excluded); −0.4 and 1.0 were run today before the recalibration.

**Post hoc (not in the design):** a positive move, goal 1.45 (seed 107), with the new calibration went the wrong way,
1.322 → 1.275 (`right_j1_seed107_20261013_141410`).

## Interpretation
- **The measured range fixes right j1 for moves below home:** mean error 0.557 → 0.147 rad, median relative error
  0.427 → 0.158, and the overshoots to the lower stop are gone. k_neg converged: every fit after trial 40 gave
  −2.6e-3 to −3.4e-3, except one at the −1e-2 clip (trial 75), which the blend pulled back within 15 trials.
- **It breaks k_pos (F11).** Home sits 0.14 rad below the measured upper limit, so only 5 of 100 goals needed a
  positive move. Most 3-trial batches then had no positive drive, k_pos was not determined, and the fit returned 0 or
  the −0.01 clip. The final decoder moves the joint the wrong way on positive moves (post hoc check). This is direct
  evidence for the F6 guess (one-sided batches leave the other gain undetermined); it is the Q11 problem, now in the
  final result rather than in passing.
- Confound: the new calibration changes both the ring mapping range and the goal distribution; the design does not
  separate them.

## Limitations
- One calibration seed, 5 held-out goals per condition, all below home: indicative, not statistics.
- No held-out goal tests positive moves; the one positive check is post hoc.
- Robot timing is not bit-reproducible (D4).

---

# Addendum (15:25-16:30 BST): balanced calibration goals (owner Q11 (d), D23-D24)

## Hypothesis
Alternating goals above and below home (`--balanced`) identifies both gains, so k_pos comes out positive and the
positive goal is reached, without losing the accuracy on moves below home.

## Config and command
As above, plus `--balanced` (D23): seed 2026, 100 trials, measured range. Evaluation fixed before the run (D24): the
five D22 goals plus 1.45 (seed 107).
```bash
python3 /work/kangaroo/calibrate.py right 1 --seed 2026 --balanced --limits /work/kangaroo/results/joint_limits/right_j1_20261013_125903/limits.json
```
Outputs: `kangaroo/results/calibration/right_j1_seed2026_20261013_152533/`,
`kangaroo/results/ring_reach/right_j1_seed10[0-7]_20261013_152{630,703,733,816,844,909}/`.

## Results: calibration
| | unbalanced (measured range) | balanced |
|---|---|---|
| Final k_pos / k_neg | −7.80e-3 / −2.85e-3 | **+8.12e-3** / −1.70e-3 |
| Final τ / delay | 0.709 s / 0 | 0.135 s / 0 |
| Goals needing a positive move | 5 | 49 (all +0.00 to +0.135 rad) |
| Fits with a wrong-signed gain | 23 of 34 | 4 of 33 (trials 60, 75, 87, 90) |
| Fits with k_pos at the +0.01 clip | (not counted) | 25 of 33 |
| Trials run with a wrong-signed decoder | 58 | 12 |
| Wrong-direction trials (\|Δq\| > 0.01) | 4 | 31: 26 of them positive goals |
| Median relative error, goals over 0.6 rad, trials 1-20 / 41-60 / 81-100 | 0.29 / 0.11 / 0.13 | 0.51 / 0.10 / 0.79 |

## Results: held-out evaluation (D24, all from home)
| Goal (seed) | unbalanced \|err\| | balanced \|err\| (final position) |
|---|---|---|
| −0.8 (100) | **0.147** | 0.928 (0.128) |
| −0.4 (105) | **0.010** | 0.668 (0.268) |
| 0.0 (101) | **0.410** | 0.775 (0.775) |
| 0.6 (102) | **0.115** | 0.357 (0.957) |
| 1.0 (106) | **0.051** | 0.158 (1.158) |
| 1.45 (107) | 0.175 (**wrong way**, 1.275) | **0.064** (right way, 1.386) |
| **mean \|err\|, six goals** | **0.151** | 0.492 |
| **median relative error, six goals** | **0.159** | 0.492 |

## Interpretation
- **(d) fixed the signs:** the final decoder is correctly signed and the positive goal is approached the right way.
- **It made the reaches worse overall:** every balanced reach stops about halfway (relative error 0.39-0.59 on all six
  goals). The final k_neg (−1.70e-3) is a snapshot mid-recovery: wrong-signed fits at trials 87 and 90 drove the
  decoder to k_neg = +2.93e-3 by trial 93, and the β = 0.5 blend had pulled it back only to −1.70e-3 by trial 100.
  The k_neg fits after trial 40 were −2.2e-3 to −3.7e-3 except trials 75 (0), 78 (−0.01 clip), 87 and 90.
- **F12. Positive goals near home produce downward motion.** In 26 of 49 positive-goal trials the joint moved down
  (up to 0.756 rad); the decoder in use had k_pos > 0 in 22 of those 26. All positive goals are within 0.135 rad of
  home. Hypothesis, unverified: with 200 neurons over the 2.6 rad range (about 0.013 rad per neuron) these moves are
  0-10 neurons, within the overlap of the goal and state bumps (half-width 5), so the ring's drive is not reliably
  signed. These trials feed the fit sign-inconsistent data, which matches the late wrong-signed fits and the k_pos
  clip in 25 of 33 fits.
- **Neither right j1 calibration is good in both directions.** The root cause is the start pose: PAL home sits
  0.14 rad under right j1's reachable maximum, so calibration has almost no room to learn upward moves.

## Limitations (addendum)
- One seed per condition; the final decoder depends strongly on where the last fits fell (unstable process).
- F12's mechanism is a hypothesis; the ring's drive for small moves was not measured directly.

## Addendum: calibration from a mid-range start pose (D25/D26, owner Q13 (c))
- **Hypothesis:** F12 comes from home sitting 0.14 rad under right j1's max; starting at mid-range (0.161 rad)
  gives room both ways, so balanced calibration (D23) should learn both signs without wrong-direction trials.
- **Command:** `calibrate.py right 1 --seed 2026 --balanced --start-joint 0.161 --limits
  results/joint_limits/right_j1_20261013_125903/limits.json` → `right_j1_seed2026_20261013_164230`.
  Commit 0ecde6a + uncommitted kangaroo/ changes. Every trial started at 0.169 rad.

| | balanced, from home | balanced, from 0.161 |
|---|---|---|
| Final k_pos / k_neg / τ | +8.12e-3 / −1.70e-3 / 0.135 | +2.59e-3 / −8.41e-3 / 0.115 |
| Fits with a wrong-signed gain | 4 of 33 | 1 of 33 (trial 60) |
| Wrong-direction trials (\|Δq\| > 0.01) | 31 (26 positive goals) | **2** (both positive goals) |
| Median relative error, trials 1-25 / 26-50 / 51-75 / 76-100 | 0.998 / 0.195 / 0.586 / 1.001 | 0.502 / 0.145 / 0.431 / 0.672 |
| Trials ending on max_steps | 17 | 9 |

**D26 evaluation** (`ring_reach.py right 1 <goal> --seed <s> --from-home --calibration <above>`; starts at 0.169):

| Goal (seed) | final | \|err\| | relative error |
|---|---|---|---|
| −0.8 (100) | −1.135 (overshoot, at the stop) | 0.335 | 0.346 |
| −0.4 (105) | −0.897 (overshoot) | 0.497 | 0.873 |
| 0.0 (101) | +0.156 (moved 0.013) | 0.156 | 0.923 |
| 0.6 (102) | +0.591 | 0.009 | 0.021 |
| 1.0 (106) | +1.080 | 0.080 | 0.096 |
| 1.45 (107) | +1.167 | 0.283 | 0.221 |
| **all six** | 6/6 right way | mean 0.227 | **median 0.283** |

**Pass bar (fixed before the run): met** — every reach the right way, median relative error 0.283 < 0.5.

### Interpretation
- **F12's root cause confirmed:** same balanced goals and seed, only the start moved, and wrong-direction trials fell
  31 → 2. The hypothesis that small moves near home fall inside the bump overlap is still unmeasured.
- **F13. k_neg ends inflated by the fit's clip.** `training/calibration.py:64-65` clips both gains to ±1e-2. The
  k_neg fits at trials 87, 96 and 99 hit −0.0100; with β = 0.5 the decoder moved from about −0.003 to −0.0084. Late
  negative trials were bimodal (6 of 13 overshot, median Δq ratio 0.31). Both negative evaluation goals overshot.
- Upward reaches are good (relative error 0.02-0.22); downward reaches are the weak side.
- Not directly comparable with D22/D24: those started from home, so goal distances differ.

### Limitations
- One calibration seed; 6 evaluation reaches. The final k_neg depends on where the last clipped fits fell.
- The decoder is valid only from the 0.161 start pose it was learned at (`start_pose` in calibration.json).
