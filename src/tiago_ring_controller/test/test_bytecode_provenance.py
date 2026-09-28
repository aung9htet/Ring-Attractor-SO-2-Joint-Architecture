"""Protect opaque source-less bytecode without importing or decompiling it."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BYTECODE_MANIFEST = ROOT / "test/golden/bytecode_manifest.json"
ORPHAN_MANIFEST = ROOT / "test/golden/orphan_bytecode_manifest.json"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OrphanBytecodeProvenanceTests(unittest.TestCase):
    def test_complete_pre_refactor_cache_tree_is_present_and_byte_identical(self):
        manifest = json.loads(BYTECODE_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["files"]), 24)
        expected_paths = set(manifest["files"])
        current_paths = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "src").rglob("*.pyc")
        }
        self.assertEqual(current_paths, expected_paths)
        for relative, expected in manifest["files"].items():
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(_sha256(path), expected["sha256"])
                self.assertEqual(
                    (ROOT / expected["source_path"]).is_file(),
                    expected["source_present"],
                )

    def test_source_less_bytecode_is_present_and_byte_identical(self):
        manifest = json.loads(ORPHAN_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("without importing", manifest["provenance"])
        for relative, expected in manifest["files"].items():
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(_sha256(path), expected["sha256"])
                source = path.parent.parent / (path.name.split(".")[0] + ".py")
                self.assertFalse(source.exists())


if __name__ == "__main__":
    unittest.main()
