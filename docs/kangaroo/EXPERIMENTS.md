# Kangaroo experiments: overview

What was run on the simulated PAL Kangaroo (ROS 2 Humble, MuJoCo), what came out, and why each step followed from
the last. Run output (trials, figures) is not in the repo; every number here can be regenerated with the commands
below. Pass bars were always fixed before the results.

## Where it stands (2026-10-14)
All 8 arm joints (left/right j0-j3) are calibrated and reach their goals. Final evaluation: 320 reaches, all 8
joints, 10 seeds per fixed goal and 20 sampled goals per joint, 0 failed.

| Measure | Result | Bar |
|---|---|---|
| Joint-direction medians of relative error < 0.5 | 16 of 16 (0.07-0.31) | ≥ 10 of 12 judged |
| Sampled goals: median relative error | 0.19 over the six first-judged joints; per joint, all 8: 0.12-0.24 | < 0.35 |
| Sampled goals: moved the right way | 0.95 over the six; left j0 and right j1: 1.00 | ≥ 0.9 |
| Plant: moved / commanded | 0.96-1.00 | — |

Relative error = |goal − q_final| / |goal − q_start|. Known weak spots: large seed-to-seed spread on left j2 up (IQR
0.92; some ring seeds barely drive) and right j2 down (IQR 0.70). Elsewhere the IQR is 0.08-0.23.

## Reproduce
Inside the container (`docker/ros2_kangaroo/`), from `/work`, simulation running; `S=src/kangaroo_ring_controller/scripts`.

```bash
python3 $S/joint_limits.py <side> <joint>                       # measured range -> results/joint_limits/
python3 $S/calibrate.py <side> <joint> --seed 2026 --balanced \
        --limits <limits.json> [--start-joint <rad>]           # 100 trials, ~1 h per joint
python3 $S/refit_calibration.py <calibration.json> ...           # batch refit over the last 50 trials
python3 $S/evaluate.py left --name <run>                        # 160 reaches, ~80 min; run right in parallel
python3 $S/plot_results.py --name <run> --evaluation <left dir> <right dir>   # figures + summary.json
```
Joint-specific settings: right j1 starts at 0.161 rad (`--start-joint`, home is near its upper stop); left j1 at
0.142; left j0 used the URDF range and home start without `--balanced`. The committed calibrations
(`results/calibration/`, the newest per joint) are the ones the controller loads.

## How we got here
| # | Experiment | Question | Result | Consequence |
|---|---|---|---|---|
| 1 | Velocity tracking | Does Kangaroo execute a joint velocity? | Steady state within 1.3 %; start-up overshoot +29-89 % (PAL's P-only gains) | The plant is usable as is |
| 2 | Ring reach, uncalibrated | Does TIAGo's ring loop, ported, drive a joint? | Right direction, plant realises 99.7 % of the command, but only 1-4 % of the way | Decoder must be calibrated on Kangaroo |
| 3 | Calibration (left j0, right j1) | Does TIAGo's calibration transfer? | Left: median rel. error 0.96 → 0.17. Right: 0.72, blocked at 1.47 rad (27 % of URDF range unreachable) | Measure each joint's real range |
| 4 | Measured range, right j1 | Does the measured range fix right j1? | 0.43 → 0.16, but upward gain came out wrong-signed | Balanced goals, mid-range start |
| 5 | Six remaining joints | Do all joints reach? | FAIL, 0.52: three reaches stop at tick 10 | Finding: TIAGo's settle rule fires before a delayed drive arrives |
| 6 | Settle guard | Does starting the settle count only once the drive arrives fix it? | PASS, 0.36; no early stops | Overshoot remains (up to 198 %) |
| 7 | Batch refit | Is overshoot the plant or the fit? | Plant moves 97-100 % of the command; the online 3-trial fit is noisy. One fit over 50 trials: 0.15 | Use the batch refit |
| 8 | Multi-seed evaluation | Does it hold over seeds and new goals? | PASS for the six joints; right j1 down overshoots on every seed (2.7x) | Recalibrate left j0, right j1 with guard + refit |
| 9 | Left j0, right j1 redone | Same treatment for the last two | PASS: right j1 down 0.98 → 0.17 | All 8 joints done |

## Deviations from TIAGo (read before comparing with the paper)
- **Settle guard:** the reach stops only after |delayed drive| has once reached the threshold; TIAGo counts from tick
  1. Once armed the rule is TIAGo's. Without it, slow or delayed decoders stop before moving.
- **Batch refit:** decoders are fitted once over the last 50 calibration trials with TIAGo's own fit function,
  instead of TIAGo's fit every 3 trials blended at β = 0.5.
- **Measured joint range** instead of URDF limits for the ring mapping; **balanced goals** (alternating above/below
  the start) and a **mid-range start** for joints whose home sits near a limit.
- The ring itself (`src/tiago_ring_controller`) is used unchanged.

## Open issues
- Seed spread on left j2 up and right j2 down comes from the ring, not the decoder or the plant.
- A failed arm reset during calibration loses that calibration run.
- Floating base; PAL's P-only gains unchanged; simulation is not bit-reproducible (asynchronous, ~1x real time).
