# Progress: block architecture (branch `blocks-refactor`)

Dated entries per phase of `plan.md`. Each entry records what changed, the gate
results with the actual command output, and what was not run.

## 2026-09-29 — Phase 1: legacy folder and test policy

Open questions of plan section 9 were taken as answered by the hand-over brief:
Q1 delete the freeze tests (D1 restated as not to re-open), Q2 accept new pinned
goldens after phase 2 (the brief says "keep that parity test passing until the
phase 2 re-baseline"), Q4 vanilla JS/SVG (D5 restated), Q5 no hardware joints in
this pass (plan non-goal "hardware use"; a per-block interlock is not designed).

### Changes

- `git mv` of every flat script, `builders/`, `builders_analysis/`, `helpers/`,
  `trainer/`, the recovered `__pycache__/` folders and the entrypoint guide
  (`src/README.md`) into `src/tiago_ring_controller/legacy/`. The scripts are
  byte-identical (rename-only diff). `legacy/config` is a symlink to `../src/config`
  so the historical `./config/...` defaults resolve when a script runs from
  `legacy/`; `legacy/README.md` carries the frozen notice and the run recipe
  (`PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH`, appended so NEST's own path survives).
  `src/Neurips_2026.pdf` stays where it is.
- Deleted the freeze tests and their manifests: `test_api_compatibility`,
  `test_environment_contracts`, `test_bytecode_provenance`,
  `test_packaging_contracts`; `api_manifest`, `import_graph_manifest`,
  `bytecode_manifest`, `orphan_bytecode_manifest`, `environment_manifest`,
  `checked_output_manifest`. `test_output_schemas` lost the PNG byte freeze, the
  checked-in `fourier_results.npz` check and the external-world hash; it keeps the
  configuration, calibration, local world and launch contracts. `test_ring_math`
  lost the two import-shim tests that ran `builders/ring_attractor.py` from `src/`
  without the package on the path (the shim now points at `legacy/`, where the
  package is not). `test_artifact_contracts` is scoped to `src/config/**` and
  `worlds/*.world`; the two golden manifests were pruned to those entries
  (34 → 23 hashed files, 32 → 22 schemas), nothing re-hashed.
- Kept tests repointed at `legacy/`: `test_control_ros_contracts`,
  `test_multi_ring_contracts`, `test_ring_math`, `test_evaluation_facade_plots`,
  `test_real_nest_smoke`, `test_cosim_real_nest`.
- Package: `config.legacy_root()`; `cosim/runner.make_nest_engine` imports
  `single_ring` from there; `scripts/measure_legacy_tick.py` likewise.
- Packaging: `setup.py` lists the package only (no `py_modules`); `CMakeLists.txt`
  installs `scripts/*.py` plus `legacy/record_experiments.py` (the wrist FT logger
  that `tiago_ring_controller.launch` starts); the devel-space `env-hooks/`
  PYTHONPATH hook for flat modules is gone. `package.xml` unchanged.
- Untracked generated results: `src/outputs/` (34 MB), `results/`, `outputs/`
  (incl. the nine `fourier_results.npz` copies) and the 23 recovered `.pyc` files
  inside the package. Files stay on disk; `.gitignore` rewritten (package
  bytecode ignored, legacy bytecode kept, legacy working-directory outputs ignored).
  `docker/check_recovery.py` will report these as differences, as documented.
- READMEs: package README describes the layout, the container test command and the
  legacy run recipe; cosim commands now run from the package root.

### Gates

Host (`/usr/bin/python3` 3.13, `-B`, `unittest discover`): before the change
178 tests, 66 failures + 1 error, all in the deleted freeze tests plus
`test_evaluation_facade_plots` (no `colorcet` on the host). After:

```
Ran 162 tests in 6.128s
FAILED (errors=1, skipped=2)      # test_evaluation_facade_plots: No module named 'colorcet' (host only, pre-existing)
```

Container (`aung9htet/ubuntu-20.04:tiago_ring_forward`, package bind-mounted,
`python3 -B -m unittest discover -s test -p "test_*.py"`), previously 178 tests
with 59 pre-existing inventory failures:

```
Ran 170 tests in 73.796s
OK
```

`scripts/run_cosim_trial.py --engines nest --seed 13579 --goal 0.6` from the
package root in the container (imports `SingleRingModel` from `legacy/`):

```
trial 1 timing: reset [nest 12.3 s, robot 0.0 s]; lead 0.06 s; 177 ticks in 1.3 s (7.5 ms/tick)
trial 1: goal=0.6000 start=0.0000 final=0.4628 |err|=0.1372 steps=177 stop=drive_settled tick_wall_ms(mean=7.54 max=17.28)
```

`cd legacy && PYTHONPATH=$PWD:$PWD/../src:$PYTHONPATH python3 -B single_ring.py`
in the container (100 s of simulated time, figures under `legacy/outputs/single_ring/`):

```
exit=0   real 16m29.729s  user 260m43.119s   (one 100 s run, 10 repeat runs, 10x10 goal sweep)
legacy/outputs/single_ring/: goal_distance_mean_std_across_runs.{csv,png}, raster.png,
  relative_goal_error_boxplot.png, relative_goal_error_per_run.csv, relative_goal_error_summary.csv,
  ring_activity_heatmap.png, ring_centroid_change_over_time.png
```

The first attempt failed with `No module named 'nest'` because `export
PYTHONPATH=$PWD:$PWD/../src` replaced the path that `nest_vars.sh` sets; the
READMEs append `:$PYTHONPATH` for that reason.

Not run: Gazebo (needs `./run_model_docker.bash` with the simulation launched; no
robot code changed in this phase) and the browser dashboard (no dashboard code
changed; its end-to-end test with the fakes is part of the suite above).
