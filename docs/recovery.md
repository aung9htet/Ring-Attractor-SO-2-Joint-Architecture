# Recovery record

Recovered on 2026-09-28 from a newly created, stopped container using the immutable
image recorded in `recovery-manifest.json`. The original running `ros_container`
was not used as the extraction source and was left running. The temporary
extraction container was removed afterward.

The recovery contains 480 files: the complete controller package, the
workspace-level experiment recordings, and four upstream custom files. Each
file's original container path, SHA-256, byte length and permission mode is in
`recovery-manifest.json`. Files were verified against the bytes streamed from
Docker before adding the launcher. Ownership was set naturally to the host user
by extracting as that user. The repository's existing `.git` directory and remote
were retained.

The recovered recordings are local-only: `recovered/` and video extensions are
ignored by Git. Their extraction hashes remain in the manifest for verification.

The upstream customizations are:

| Upstream package | Recovered customization |
| --- | --- |
| `pal_gazebo_worlds` | `worlds/tiago_ring_controller.world`, including the recording camera |
| `tiago_bringup` | `config/motions/tiago_motions.yaml.em`, defining the experimental starting pose |
| `tiago_bringup` | `config/motions/pmb2/tiago_motions_pal-hey5_schunk-ft.yaml`, containing the same custom motion |
| `tiago_gazebo` | `scripts/tuck_arm.py`, selecting `tiago_experiment_start_1` at simulation startup |

`docker/overrides.list` maps the recovered paths beneath `overrides/` to the same
relative locations under `/tiago_public_ws/src/`. They are mounted read-only in
the container and remain editable from the host. The controller's own local
world is preserved separately: it does not contain the recording camera and is
not substituted for the external world.

All historical Python bytecode was recovered, including the two source-less
files. Explicit `.gitignore` exceptions preserve the recovered cache inventory
while new caches are ignored. `PYTHONDONTWRITEBYTECODE=1` prevents normal runs from
changing these provenance files. Nothing was decompiled or regenerated.

The separate `ring_ari_movement` project remains in the environment image. It is
not a dependency of this controller and was not migrated into this repository.
The full upstream ROS/TIAGo source tree and installed dependencies remain supplied
by the pinned image.

Run `python3 docker/check_recovery.py` from the repository root to verify the
original snapshot. It reports deliberate later edits without replacing them.
