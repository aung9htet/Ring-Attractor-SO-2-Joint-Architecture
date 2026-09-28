"""Immutable artifact and configuration characterization tests.

These tests are intentionally read-only.  A changed hash is a review event: update
the golden manifest only when a scientific artifact change is explicitly intended.
"""

import hashlib
import json
import math
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.artifacts import (  # noqa: E402
    atomic_json_dump_legacy,
    inspect_artifact,
    load_json_legacy,
    save_npz_legacy,
)

MANIFEST_PATH = ROOT / "test/golden/artifact_manifest.json"
SCHEMA_MANIFEST_PATH = ROOT / "test/golden/artifact_schema_manifest.json"
TRACKED_SUFFIXES = {".json", ".npy", ".npz", ".pdf", ".world"}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_scope():
    paths = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TRACKED_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"docs", "test"} or "__pycache__" in relative.parts:
            continue
        paths.add(relative.as_posix())
    return paths


def _npz_headers(path):
    """Read NPY headers from an NPZ without materializing large matrices."""
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in {(2, 0), (3, 0)}:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:  # pragma: no cover - NumPy raises before this for unknown formats
                    raise AssertionError("unsupported NPY version %r" % (version,))
            headers[member[:-4]] = (shape, bool(fortran), str(dtype))
    return headers


def _json_value_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _json_structure(value):
    if isinstance(value, dict):
        return [
            "object",
            [[str(key), _json_structure(item)] for key, item in value.items()],
        ]
    if isinstance(value, list):
        return ["array", [_json_structure(item) for item in value]]
    return _json_value_type(value)


def _scalar_manifest_value(array):
    value = array.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"type": "float", "repr": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _array_schema(array, key=None):
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "fortran_order": bool(
            array.flags.f_contiguous and not array.flags.c_contiguous
        ),
    }
    if key is not None:
        result["key"] = key
    if array.shape == ():
        result["scalar_value"] = _scalar_manifest_value(array)
    return result


def _artifact_schema(path):
    suffix = path.suffix.lower()
    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        value = json.loads(text)
        try:
            json.loads(
                text,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(token)
                ),
            )
            strict_json = True
        except ValueError:
            strict_json = False
        encoded_structure = json.dumps(
            _json_structure(value),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        result = {
            "kind": "json",
            "strict_json": strict_json,
            "top_level_type": _json_value_type(value),
            "structure_sha256": hashlib.sha256(encoded_structure).hexdigest(),
        }
        if isinstance(value, dict):
            result["key_order"] = [str(key) for key in value]
            result["top_level_value_types"] = {
                str(key): _json_value_type(item) for key, item in value.items()
            }
        return result
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        return {"kind": "npy", "arrays": [_array_schema(np.asarray(array))]}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            arrays = [
                _array_schema(np.asarray(archive[key]), key=key) for key in keys
            ]
        return {"kind": "npz", "key_order": keys, "arrays": arrays}
    raise AssertionError("unsupported schema artifact: %s" % path)


class ArtifactImmutabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_is_versioned_and_complete(self):
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(self.manifest["hash_algorithm"], "sha256")
        self.assertEqual(set(self.manifest["files"]), _manifest_scope())

    def test_every_file_matches_size_and_sha256(self):
        mismatches = []
        for relative, expected in self.manifest["files"].items():
            path = ROOT / relative
            if not path.is_file():
                mismatches.append("%s: missing" % relative)
                continue
            if path.stat().st_size != expected["bytes"]:
                mismatches.append(
                    "%s: size %d != %d"
                    % (relative, path.stat().st_size, expected["bytes"])
                )
            actual_hash = _sha256(path)
            if actual_hash != expected["sha256"]:
                mismatches.append(
                    "%s: sha256 %s != %s"
                    % (relative, actual_hash, expected["sha256"])
                )
        self.assertEqual(mismatches, [], "\n".join(mismatches))


class CompleteArtifactSchemaManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hash_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.schema_manifest = json.loads(
            SCHEMA_MANIFEST_PATH.read_text(encoding="utf-8")
        )

    def test_schema_manifest_covers_every_json_npy_and_npz(self):
        expected = {
            relative
            for relative in self.hash_manifest["files"]
            if Path(relative).suffix.lower() in {".json", ".npy", ".npz"}
        }
        self.assertEqual(self.schema_manifest["schema_version"], 1)
        self.assertEqual(set(self.schema_manifest["files"]), expected)

    def test_every_artifact_schema_matches_the_archived_baseline(self):
        for relative, expected in self.schema_manifest["files"].items():
            with self.subTest(path=relative):
                self.assertEqual(_artifact_schema(ROOT / relative), expected)


class ArtifactCompatibilityWriterTests(unittest.TestCase):
    def test_atomic_json_preserves_unknown_fields_nan_and_sibling_replace(self):
        value = {
            "known": 1,
            "unknown_history": {"nested": [1, 2, 3]},
            "legacy_nan": float("nan"),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "velocity_calibration.json"
            path.write_text('{"old": true}', encoding="utf-8")
            atomic_json_dump_legacy(str(path), value)
            self.assertFalse(Path(str(path) + ".tmp").exists())
            loaded = load_json_legacy(str(path))
            self.assertEqual(loaded["known"], 1)
            self.assertEqual(
                loaded["unknown_history"], {"nested": [1, 2, 3]}
            )
            self.assertTrue(math.isnan(loaded["legacy_nan"]))

    def test_npz_writer_retains_caller_key_order_shapes_and_dtypes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            save_npz_legacy(
                str(path),
                z=np.array([1, 2], dtype=np.int16),
                a=np.array(3.5, dtype=np.float64),
                flag=np.array(True, dtype=np.bool_),
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive.files, ["z", "a", "flag"])
                self.assertEqual((archive["z"].shape, str(archive["z"].dtype)), ((2,), "int16"))
                self.assertEqual((archive["a"].shape, str(archive["a"].dtype)), ((), "float64"))
                self.assertEqual(bool(archive["flag"]), True)

    def test_report_only_inspection_records_malformed_numpy_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "opaque.npy"
            path.write_bytes(b"not a numpy artifact")
            descriptor = inspect_artifact(str(path))
        self.assertEqual(descriptor.kind, "npy")
        self.assertEqual(descriptor.arrays, ())
        self.assertEqual(len(descriptor.errors), 1)
        self.assertIn("ValueError", descriptor.errors[0])


class NumpyArtifactSchemaTests(unittest.TestCase):
    def test_standalone_weight_array_contracts(self):
        contracts = {
            "src/config/homeostasis/N_100_homeostasis_weights.npy": ((20, 2), "float64"),
            "src/config/homeostasis/N_200_homeostasis_weights.npy": ((80, 2), "float64"),
            "src/config/ring_decoding_weights/N_100_fourier_weights.npy": ((100, 10), "float64"),
            "src/config/ring_decoding_weights/N_200_fourier_weights.npy": ((200, 40), "float64"),
            "src/config/two_joints_ring_decoding_weights/N_100_two_joint_fourier_weights.npy": ((100, 20), "float64"),
        }
        for relative, (shape, dtype) in contracts.items():
            with self.subTest(path=relative):
                array = np.load(ROOT / relative, mmap_mode="r", allow_pickle=False)
                self.assertEqual(array.shape, shape)
                self.assertEqual(str(array.dtype), dtype)
                self.assertTrue(array.flags.c_contiguous)
                self.assertTrue(np.all(np.isfinite(array)))

    def test_multi_ring_npz_key_order_and_headers(self):
        path = ROOT / "src/config/ring_decoding_weights/N_100_J_2_multi_ring_sawtooth_weights.npz"
        expected_keys = [
            "W_signed_lift",
            "W_signed_pitch",
            "W_signed_yaw",
            "signed_product_feature_order",
            "signed_product_population_size",
            "population_size",
            "num_joints",
            "joint_axes",
            "output_ring_size",
            "signed_product_output_weight_scale",
            "output_weight_matrix_is_target_by_source",
        ]
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data.files, expected_keys)
        headers = _npz_headers(path)
        for key in ("W_signed_lift", "W_signed_pitch", "W_signed_yaw"):
            self.assertEqual(headers[key], ((16384, 100), False, "float64"))
        self.assertEqual(headers["signed_product_feature_order"], ((16,), False, "<U12"))
        self.assertEqual(headers["joint_axes"], ((2,), False, "<U1"))
        self.assertEqual(headers["output_weight_matrix_is_target_by_source"], ((), False, "bool"))

    def test_legacy_compositional_artifact_is_all_zero(self):
        path = ROOT / "src/config/ring_decoding_weights/N_100_J_2_compositional_weights.npz"
        expected_keys = [
            "W_prod_lift",
            "W_prod_pitch",
            "W_prod_yaw",
            "joint_axes",
            "output_ring_size",
            "prod_feature_dim",
            "product_population_size",
            "population_size",
            "num_joints",
        ]
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data.files, expected_keys)
            for key in ("W_prod_lift", "W_prod_pitch", "W_prod_yaw"):
                self.assertEqual(data[key].shape, (1024, 100))
                self.assertEqual(str(data[key].dtype), "float64")
                self.assertFalse(np.any(data[key]))
            self.assertEqual(data["joint_axes"].tolist(), ["x", "y"])
            self.assertEqual(int(data["prod_feature_dim"]), 1024)

    def test_scalar_artifact_contract(self):
        path = ROOT / "src/config/ring_decoding_weights/N_100_J_2_multi_ring_sawtooth_scalar_weights.npz"
        expected_keys = [
            "scalar_ramp_weights",
            "scalar_ramp_weight_scale",
            "scalar_dc_baseline",
            "output_ring_size",
            "population_size",
            "num_joints",
            "joint_axes",
            "source_sawtooth_weights_path",
        ]
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data.files, expected_keys)
            self.assertEqual(data["scalar_ramp_weights"].shape, (100,))
            np.testing.assert_array_equal(
                data["scalar_ramp_weights"], np.linspace(0.0, 1.0, 100) * 100.0
            )
            self.assertEqual(float(data["scalar_ramp_weight_scale"]), 100.0)
            self.assertEqual(float(data["scalar_dc_baseline"]), 200.0)
            self.assertEqual(data["joint_axes"].tolist(), ["x", "y"])
            self.assertEqual(
                str(data["source_sawtooth_weights_path"]),
                "./config/ring_decoding_weights/N_100_J_2_multi_ring_sawtooth_weights.npz",
            )

    def test_sawtooth_metadata_is_intentionally_non_strict_json(self):
        path = ROOT / "src/config/ring_decoding_weights/N_100_J_2_multi_ring_sawtooth_metadata.json"
        text = path.read_text(encoding="utf-8")
        self.assertIn("NaN", text)
        parsed = json.loads(text)
        self.assertEqual(parsed["population_size"], 100)
        with self.assertRaises(ValueError):
            json.loads(
                text,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )


if __name__ == "__main__":
    unittest.main()
