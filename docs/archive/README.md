# Archive

Reports, data and plans that the current results have superseded. Nothing here
is needed to run or understand the current system: start from the top-level
`README.md`, `docs/architecture.md` and `docs/results.md`. The pages are kept as
the record of how the system got here. Paths written inside these files are the
ones that were valid when they were written and are not maintained.

## Earlier results

| Here | What it is | Superseded by |
| --- | --- | --- |
| `velocity/calibration.md` | How gain spikes become joint velocity in v3; the transport calibration and its measurements | still the method of v3 (README section 5.2); v4 needs no calibration |
| `velocity/bounded-ring.md` | The bounded ring: limits, reading the joint off the ring, all joints, first comparison with the original scripts | `docs/results.md` |
| `velocity/speed.md` | The speed setting `time_scale` and its measurements on v3 | README section 4, `docs/results.md` |
| `velocity/all-joints.md` | v3 on all seven joints from three start poses | `docs/stress-test.md` |
| `velocity/media/` | Figures, data and videos of those four reports | `docs/results/` |
| `blocks/feedback.md`, `blocks/feedback_sweep*.{json,md}` | Feeding the joint back into the state ring of the two-ring model (it made things worse) | architecture v4 |
| `results-data/compare/`, `results-data/repeat/` | Gazebo trials of the intermediate variants: v3 on the legacy ring with hand-set and transport-calibrated gains (`gt.json`), v3 and v4 at one NEST step per tick (`b1.json`, `f1.json`) | the systems in `docs/results.md` |
| `results-data/starts/` | v3 and v4 from three fixed start poses per joint (168 trials each) | `docs/stress-test.md` (random starts and arm poses) |

What those intermediate variants showed, in one table (Gazebo, all seven
joints, 56 goals, measured 2026-10-04 with the same rules as `docs/results.md`):

| Variant | Goal error, median / p90 / max (rad) | Settled |
| --- | --- | ---: |
| v3, legacy ring, hand-set gains | 0.333 / 1.047 / 1.886 | 32% |
| v3, legacy ring, transport calibration | 0.139 / 0.429 / 0.822 | 54% |
| v3, bounded ring, 1 NEST step per tick | 0.088 / 0.231 / 0.363 | 66% |
| v4, 1 NEST step per tick | 0.009 / 0.060 / 0.140 | 88% |
| v3 from three start poses (3 steps per tick) | 0.063 / 0.195 / 0.359 | 62% |
| v4 from three start poses (3 steps per tick) | 0.012 / 0.029 / 0.075 | 98% |

To put these variants back into the generated tables, copy the files of
`results-data/compare/` into `docs/results/data/compare/` and rerun
`scripts/results_report.py`; it reads them when they are there.

## Plans and progress logs

| Here | What it was | Where the result lives now |
| --- | --- | --- |
| `cosim-assessment.md` | Why the original control loop was defective (cites the legacy code line by line) | `docs/legacy.md`, `docs/cosim/README.md` |
| `cosim-plan-a-nrp-core.md` | Option A for the loop: move onto the HBP Neurorobotics Platform (not taken) | - |
| `cosim-plan-b-inhouse-loop.md` | Option B: the lock-stepped loop inside this package (taken) | `docs/cosim/README.md`, `src/.../cosim/` |
| `blocks-plan.md` | Plan of the block and graph refactor, phases 1-7 | `docs/blocks/architecture.md` |
| `blocks-progress.md` | Dated record of each phase and its gate output | `docs/blocks/` |
| `velocity-plan.md` | Plan of the spike-to-velocity calibration | `velocity/calibration.md` |

Scripts that only served these reports are in
`src/tiago_ring_controller/scripts/archive/`.
