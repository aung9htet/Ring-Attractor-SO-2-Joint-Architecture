#!/usr/bin/env python3
"""Add or refresh files of ``src/config/`` in the two golden manifests.

    python3 -B scripts/update_artifact_manifest.py src/config/calibration/transport_calibration_bounded.json [more files ...]

``test/test_artifact_contracts.py`` pins every scientific input by size, hash
and schema.  After installing a new or re-fitted artifact (comparator weights,
a transport calibration), run this with the package-relative paths; the test
then accepts exactly these files.  The package must be writable.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "test"))

import test_artifact_contracts as contracts  # noqa: E402


def main(argv=None):
    paths = list(sys.argv[1:] if argv is None else argv)
    if not paths or paths[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if paths else 2
    for relative in paths:
        if not (contracts.ROOT / relative).is_file():
            raise SystemExit("not a file under the package: %s" % relative)
    for manifest_path, entry in ((contracts.MANIFEST_PATH, lambda p: {"bytes": p.stat().st_size, "sha256": contracts._sha256(p)}),
                                 (contracts.SCHEMA_MANIFEST_PATH, contracts._artifact_schema)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative in paths:
            manifest["files"][relative] = entry(contracts.ROOT / relative)
        manifest["files"] = dict(sorted(manifest["files"].items()))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("updated %s: %s" % (manifest_path, ", ".join(paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
