# Data behind `docs/results.md` and `docs/stress-test.md`

Everything here was written by the measurement scripts and is read by
`scripts/results_report.py` (writes `../tables.md`, `../figures/`, `../paper/`)
and `scripts/stress_test.py report` (writes `../stress/`). Trials are JSON rows;
a row holds the start, goal and final angle, the motion metrics of
`tiago_ring_controller.evaluation.motion`, how far the robot's base moved during
the trial (`base_shift_m`), and in `compare/` the joint trace per 50 ms tick
(`trace.q`). Joints are 0-based (joint 5 is `arm_6_joint`).

| Folder | Content | Written by |
| --- | --- | --- |
| `compare/` | Every system on the same 8 goals per joint from the reset pose, Gazebo: `legacy_trials_j<n>.json` (original scripts), `pid_trials_j<n>.json` (the PID baseline), `b3.json` (v3), `f3.json` (v4), and the upper references `p_controller.json`, `native.json` | `run_legacy_trials.py --plan 8 --screen-moveit`, `run_baselines.py run --trials 8 --screen-moveit`, `validate_velocity.py --engines full --trials 8 --screen-moveit` |
| `repeat/` | Three runs of two goals on joints 5, 4, 1, 6, same file names | the same with `--plan 2 --repeats 3` / `--trials 2 --repeats 3` |
| `robust/` | Joints 1, 4, 5, six goals each: `nominal.json`, `mg07.json`, `mg13.json` (motor gain x0.7, x1.3), `push.json` (0.2 rad), `hold.json` (2 s) for v3 and v4; `<name>_pid/pid_trials_j<n>.json` for the PID baseline; `<name>_p_controller.json` for the tuned reference controller | `validate_velocity.py --systems B F --trials 6`, `run_legacy_trials.py --controller pid --plan 6`, `run_baselines.py run --controller p --trials 6`, with `--motor-gain-scale`, `--push`, `--hold` |
| `stress/` | The stress test: `gazebo_plan.json`, `nest_plan.json` (the random trials: joint, start, goal, pose of the other joints, seed) and one folder per run, `gazebo_<v4\|v3\|pid\|p_controller\|native>/`, `nest_<v3\|v4>/`, with `results_shard*.json` | `stress_test.py plan`, `stress_test.py run` |
| `nest/` | v3 and v4 on NEST with the simulated joint: `j<n>.json` (3 starts x 3 seeds x 8 goals per joint), `rob_*.json` (the disturbances), `k1.json`, `k2.json` (time scale 1 and 2), `limits.json` (goals exactly at the joint limits) | `validate_velocity.py --engines nest --systems B F` |
| `measure/` | `fine_map.json`, `fine_map_sharp.json` (drive against goal offsets of one index, without and with v4's projection), `drift.json` (free-bump drift) | `fit_comparator.py map --fine [--sharp]`, `fit_comparator.py drift` |
| `pid_tuning.json` | The tuning grid of the tuned reference controller (70 settings) and the setting the rule chose | `run_baselines.py tune` |
| `tuning.json` | The 115 tuning sweeps of v4 on NEST, in the order they were run | `validate_velocity.py --engines nest --systems F --feedback ...` (`docs/v4-tuning.md`) |

Trials of superseded variants (v3 on the legacy ring, runs at one NEST step per
tick, the three-start-pose set) are in `docs/archive/results-data/`. The exact
command lines are in the top-level `README.md`, section 6, and in
`docs/v4-tuning.md`, section "How to reproduce".
