#!/usr/bin/env python3
"""Verify the recovered snapshot; intentional research edits will be reported."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/recovery-manifest.json").read_text())
failures = []
for relative, expected in manifest["files"].items():
    path = root / relative
    if not path.is_file():
        failures.append(relative + ": missing")
        continue
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        failures.append(relative + ": content changed")
    if oct(path.stat().st_mode & 0o777) != expected["mode"]:
        failures.append(relative + ": permissions changed")
print("\n".join(failures) if failures else "Verified all %d recovered files." % len(manifest["files"]))
sys.exit(bool(failures))
