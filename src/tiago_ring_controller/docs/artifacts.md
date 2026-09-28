# Artifact contracts

Checked-in artifacts are scientific inputs or provenance records. A static search
that finds no loader is not authority to delete one. `test/golden/artifact_manifest.json`
is the byte-level baseline and `test/golden/artifact_schema_manifest.json` is the
complete structural baseline; `test/test_artifact_contracts.py` and
`test/test_output_schemas.py` enforce both.
Checked-in figures are independently frozen by
`test/golden/checked_output_manifest.json`, including image dimensions and mode.

## Active model artifacts

| Artifact | Contract | Consumer |
|---|---|---|
| `N_100_fourier_weights.npy` | float64 `(100,10)`; sine positive/negative pairs | builder experiments |
| `N_200_fourier_weights.npy` | float64 `(200,40)`; sine positive/negative pairs | legacy robot/single-ring path |
| `N_100_homeostasis_weights.npy` | float64 `(20,2)` | N=100 comparator |
| `N_200_homeostasis_weights.npy` | float64 `(80,2)` | N=200 comparator |
| `N_100_J_2_multi_ring_sawtooth_weights.npz` | three float64 `(16384,100)` matrices plus topology metadata | `MultiRingDecode` |
| `velocity_calibration.json` | seven joint records with current and historical values | robot neural controller paths |

Fourier runtime validates only matrix shape and scales stored masks by 200. Its
column order is `sin_pos_k1`, `sin_neg_k1`, then the same pair for increasing
harmonics.

Homeostasis metadata includes fitted warm/cold biases. Runtime loads those values
but does not inject them. The metadata and unused behavior are both preserved.

## Multi-ring schema

The active NPZ key order is:

```text
W_signed_lift
W_signed_pitch
W_signed_yaw
signed_product_feature_order
signed_product_population_size
population_size
num_joints
joint_axes
output_ring_size
signed_product_output_weight_scale
output_weight_matrix_is_target_by_source
```

The 16-feature order is positive then negative for each term:

```text
cos1, sin1, cos2, sin2,
cos1cos2, cos1sin2, sin1cos2, sin1sin2
```

The matrices are target-by-source. Current metadata calibrates the signed-product
DC baseline to 120, input weight to 700, and output scale to 1000. The metadata
contains bare `NaN`; Python's default JSON loader accepts it, strict JSON parsers do
not. Artifact validation must initially report this without rejecting the file.

## Derived and legacy artifacts

| Artifact | Status |
|---|---|
| Multi-ring scalar NPZ/JSON | Analytic `(100,)` ramp from 0 to 100; records DC 200 and a CWD-relative source path. No current runtime reloads the saved scalar NPZ. |
| Compositional NPZ/JSON | Three `(1024,100)` matrices are all zero; no static consumer. Provenance unknown, preserve. |
| Two-joint Fourier NPY/JSON | float64 `(100,20)` with sine/cosine positive/negative columns; no static consumer. Preserve. |
| `fourier_results.npz` copies | `phi_true (30,)`, `phi_est (30,)`; most copies are byte-identical, one has different values with the same schema. |
| `ring_model.pyc`, `calibrate_velocity_model.pyc` | Bytecode has no corresponding source. It is not an approved runtime dependency and must not be deleted or decompiled; exact bytes are frozen in `orphan_bytecode_manifest.json`. |

## Calibration precedence

Each joint record can contain top-level decoder gains/tau/delay, raw legacy values,
and a nested `calibration_progress.current_decoder`. Runtime currently prefers the
top-level decoder fields. Some nested and raw values materially disagree with those
fields. Loaders must retain:

- top-level precedence;
- unknown fields and key types;
- existing insertion order when writing;
- absent PID-key fallback behavior;
- atomic replacement behavior used by calibration.

No schema migration is part of this behavior-preserving refactor.

## Output artifacts

Robot collectors write trial CSV summaries, compressed time-series/raster NPZ,
batch error NPY, aggregate raster NPZ, and batch CSV summaries. Analysis and
calibration add JSON summaries and figures. Sensor logging has two distinct CSV
schemas under `experiment_results/sensor_data` and `experiment_results/torque_data`.

Paths, filenames, CSV column order, NPZ key order, dtype, and raw-versus-smoothed
data choices are scientific compatibility contracts. Tests write only to temporary
directories; they never clean stale experimental output automatically.

## Safe regeneration policy

1. Copy the package to a disposable directory.
2. Record the original manifest and environment.
3. Run one trainer with an explicit temporary output/config location.
4. Compare keys, shapes, dtypes, scalars, and deterministic numerical results.
5. Do not replace a checked-in artifact unless the change is separately reviewed as
   a scientific result change.

Root and nested Fourier trainers historically target the same filenames. That
collision is preserved for compatibility but should never be exercised against the
checked-in artifact directory during validation.
