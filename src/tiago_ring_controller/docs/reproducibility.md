# Reproducibility baseline

This document records the environment in which the behavior-preserving refactor
was characterized. The executable source is authoritative. The paper and README
files provide scientific context, but they do not override observed code behavior.

## Immutable baseline

`test/golden/artifact_manifest.json` contains the byte size and SHA-256 digest of
all repository JSON, NPY, NPZ, PDF, and SDF world files that existed before the
refactor. The manifest deliberately excludes `docs/` and `test/` so it does not
hash itself.

`test/golden/artifact_schema_manifest.json` independently freezes every tracked
JSON/NPY/NPZ structure: JSON strictness, key order and recursive type structure;
NPZ key order; and array shape, dtype, storage order, and scalar values. The two
source-less bytecode files are protected without import or decompilation by
`test/golden/orphan_bytecode_manifest.json`. The original 28-module static import
graph is captured in `test/golden/import_graph_manifest.json`; explicit builder
package-import replacements are characterized separately from added internals.
Existing checked-in PNG outputs have their bytes, formats, modes, and dimensions
frozen in `test/golden/checked_output_manifest.json`.
`test/golden/api_manifest.json` freezes all 28 original module paths, 34 class
bases, 51 top-level functions, 375 methods (including underscore compatibility
methods), decorators, arguments/defaults/annotations, executable modes, and 20
`__main__` blocks.

Run the read-only checks from the package root:

```bash
cd /tiago_public_ws/src/tiago_ring_controller
python3 -m unittest discover -s test -p 'test_*.py'
```

A hash failure is a review event, not a prompt to regenerate the manifest. Update
the manifest only for an explicitly approved scientific artifact or configuration
change. Training and figure-generation commands must run in a disposable copy or
with output paths redirected outside the package.

The tests additionally freeze:

- NPY/NPZ key order, array shapes, dtypes, and important scalar metadata;
- both legacy and builder ring-distance/weight formulas;
- the intentionally different odd-size ring profiles;
- Fourier positive/negative sine column order;
- signed multi-ring feature order and row-major flattening;
- checked-in output and configuration schemas;
- launch arguments, the camera-less local SDF world, and the hash of the external
  world currently selected by the upstream launcher.

## Captured environment

The machine-readable record is `test/golden/environment_manifest.json`.

| Component | Baseline |
|---|---|
| Python | CPython 3.8.10 at `/usr/bin/python3` |
| ROS | ROS 1 Noetic, Python 3 |
| NEST | `HEAD@41892a5`, built 2026-03-24 |
| NEST resolution | 0.1 ms |
| Default `static_synapse` delay | 1.0 ms |
| NumPy | 1.24.4 |
| Matplotlib | 3.7.5 |
| SciPy | 1.10.1 |
| Pandas | 2.0.3 |
| Build system | Catkin |

The source does not explicitly set NEST resolution or the default static-synapse
delay. Exact spike-event comparisons are therefore valid only for the pinned NEST
environment. A different environment should report the mismatch rather than
silently broaden numerical tolerances.

## Working-directory contract

Many entrypoints resolve `./config`, `./outputs`, and `./experiment_results`
relative to the current working directory. The historical scientific commands
assume the process starts in `src/`, while package-level ROS launch starts nodes
through Catkin. This discrepancy is part of the compatibility contract until all
legacy paths are wrapped and tested.

Source-space lookup and existing-file precedence remain unchanged. In a Catkin
install space only, a missing legacy input containing a `config/` path component
falls back to `share/tiago_ring_controller/config`. Installed logger entrypoints
map their package-adjacent output root to `share/tiago_ring_controller` rather than
the unrelated `lib/` directory. These fallbacks do not activate for source or
devel modules.

The model configuration contains `./config/ring_decoder`, while runtime Fourier
loading uses `./config/ring_decoding_weights`. Both strings are retained. The
homeostasis configuration directory and scalar artifact source path are also
CWD-relative.

## Randomness and ordering

- `SingleRingModel(seed=...)` pins the NEST seed and one local thread. Its default
  scientific entrypoint uses seed 12345.
- Raw ring builders, most trainers, robot goal sampling, and several worker-seed
  paths are not fully seeded.
- Multi-ring random training configurations use NumPy `default_rng(42)` only when
  the full Cartesian grid exceeds the configured sample cap.
- State and target bumps are injected serially. The state ring advances for 50 ms
  before target-ring injection completes.
- Multiprocessing completion order is not a scientific ordering unless a caller
  explicitly sorts results.

Tests may inject deterministic seeds or fakes, but production defaults must remain
unchanged in a behavior-preserving refactor.

## World and launch resolution

The package launch forwards `world=tiago_ring_controller` to `tiago_gazebo`.
No package path is supplied, so the baseline resolves to:

```text
/tiago_public_ws/src/pal_gazebo_worlds/worlds/tiago_ring_controller.world
```

That external world contains the recording camera. The checked-in local world at
`worlds/tiago_ring_controller.world` is camera-less. Do not substitute one for the
other as part of a refactor.

## Validation tiers

1. Run immutable hash, pure-math, schema, fake-NEST, and mocked-ROS tests.
2. Build and test the package in Catkin.
3. Run small, pinned NEST smoke cases in a disposable output directory.
4. Compare old and refactored command traces in Gazebo.
5. Run robot-moving workflows only after explicit hardware authorization and the
   checklist in `docs/robot_safety.md`.

No automated test in this repository should move hardware, overwrite calibration,
or regenerate checked-in weights.
