"""Freeze the complete public and de-facto-public flat-module API."""

import ast
import hashlib
import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "test/golden/api_manifest.json"


def _dump(value):
    if value is None:
        return None
    return ast.dump(value, annotate_fields=True, include_attributes=False)


def _function_entry(node):
    return {
        "name": node.name,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "arguments": _dump(node.args),
        "returns": _dump(node.returns),
        "decorators": [_dump(item) for item in node.decorator_list],
    }


def _is_main_guard(node):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _module_contract(path):
    tree = ast.parse(path.read_bytes(), filename=str(path))
    functions = [
        _function_entry(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        classes.append(
            {
                "name": node.name,
                "bases": [_dump(item) for item in node.bases],
                "keywords": [
                    {"arg": item.arg, "value": _dump(item.value)}
                    for item in node.keywords
                ],
                "decorators": [_dump(item) for item in node.decorator_list],
                "methods": [
                    _function_entry(item)
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ],
            }
        )
    guards = [node for node in tree.body if _is_main_guard(node)]
    return {
        "mode": oct(os.stat(path).st_mode & 0o777),
        "functions": functions,
        "classes": classes,
        "main_blocks": [
            hashlib.sha256(_dump(node).encode("utf-8")).hexdigest()
            for node in guards
        ],
    }


class FlatModuleApiCompatibilityTests(unittest.TestCase):
    def test_all_original_signatures_bases_modes_and_main_blocks_are_exact(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        totals = {"classes": 0, "functions": 0, "methods": 0, "main": 0}
        for relative, expected in manifest["files"].items():
            with self.subTest(path=relative):
                actual = _module_contract(ROOT / relative)
                self.assertEqual(actual, expected)
            totals["functions"] += len(expected["functions"])
            totals["classes"] += len(expected["classes"])
            totals["methods"] += sum(
                len(item["methods"]) for item in expected["classes"]
            )
            totals["main"] += len(expected["main_blocks"])
        self.assertEqual(
            totals,
            {"classes": 34, "functions": 51, "methods": 375, "main": 20},
        )


if __name__ == "__main__":
    unittest.main()
