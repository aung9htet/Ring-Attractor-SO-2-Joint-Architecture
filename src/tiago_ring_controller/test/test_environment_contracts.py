"""Validate the captured execution environment and legacy import graph."""

import ast
import importlib.metadata
import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PATH = ROOT / "test/golden/environment_manifest.json"
IMPORT_GRAPH_PATH = ROOT / "test/golden/import_graph_manifest.json"


def _imports(path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append("." * node.level + (node.module or ""))
    return set(result)


class EnvironmentManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = json.loads(
            ENVIRONMENT_PATH.read_text(encoding="utf-8")
        )

    def test_python_and_scientific_library_versions_match_capture(self):
        expected = self.environment["python"]
        self.assertEqual(sys.implementation.name, expected["implementation"].lower())
        self.assertEqual(sys.version.split()[0], expected["version"])
        self.assertEqual(Path(sys.executable), Path(expected["executable"]))
        for distribution, version in expected["packages"].items():
            with self.subTest(distribution=distribution):
                self.assertEqual(importlib.metadata.version(distribution), version)

    def test_original_python_inventory_and_executable_modes_are_unchanged(self):
        inventory = self.environment["python_source_inventory"]
        executable = set(inventory["executable"])
        non_executable = set(inventory["non_executable"])
        self.assertEqual(len(executable | non_executable), inventory["count"])
        self.assertFalse(executable & non_executable)
        for relative in sorted(executable | non_executable):
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file())
                mode = os.stat(path).st_mode & 0o777
                self.assertEqual(mode, 0o755 if relative in executable else 0o644)

    def test_captured_tool_paths_remain_present(self):
        for name, path_text in self.environment["tools"].items():
            with self.subTest(tool=name):
                self.assertTrue(Path(path_text).is_file(), path_text)


class LegacyImportGraphTests(unittest.TestCase):
    def test_every_original_module_has_a_captured_graph(self):
        environment = json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
        graph = json.loads(IMPORT_GRAPH_PATH.read_text(encoding="utf-8"))
        inventory = environment["python_source_inventory"]
        expected = set(inventory["executable"] + inventory["non_executable"])
        self.assertEqual(graph["schema_version"], 1)
        self.assertEqual(set(graph["files"]), expected)
        for entry in graph["files"].values():
            self.assertRegex(entry["source_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(entry["imports"], sorted(set(entry["imports"])))

    def test_baseline_imports_remain_or_have_the_explicit_package_rename(self):
        graph = json.loads(IMPORT_GRAPH_PATH.read_text(encoding="utf-8"))
        explicit_renames = {
            ("src/builders/single_joint/ring_component.py", "ring_attractor"):
                "builders.ring_attractor",
            ("src/builders_analysis/ring_attractor_analysis.py", "ring_attractor"):
                "builders.ring_attractor",
            ("src/trainer/train_ring_model.py", "ring_component"):
                "builders.single_joint.ring_component",
        }
        missing = []
        for relative, entry in graph["files"].items():
            current = _imports(ROOT / relative)
            for baseline_import in entry["imports"]:
                if baseline_import in current:
                    continue
                replacement = explicit_renames.get((relative, baseline_import))
                if replacement not in current:
                    missing.append(
                        "%s: %s (replacement %r)"
                        % (relative, baseline_import, replacement)
                    )
        self.assertEqual(missing, [], "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
