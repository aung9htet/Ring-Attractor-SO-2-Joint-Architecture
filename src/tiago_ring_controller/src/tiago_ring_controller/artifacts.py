"""Read-only inspection and legacy-compatible loading of research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import ArtifactArrayDescriptor, ArtifactDescriptor
from .config import resolve_legacy_read_path


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_value(array: np.ndarray) -> Optional[Any]:
    if array.shape != ():
        return None
    value = array.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _array_descriptor(key: Optional[str], array: np.ndarray) -> ArtifactArrayDescriptor:
    return ArtifactArrayDescriptor(
        key=key,
        shape=tuple(int(n) for n in array.shape),
        dtype=str(array.dtype),
        scalar_value=_scalar_value(array),
    )


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError("non-standard JSON constant: {}".format(value))

    return json.loads(text, parse_constant=reject_constant)


def inspect_artifact(path: str) -> ArtifactDescriptor:
    """Describe an artifact without changing or validating it.

    Inspection failures are returned in ``errors``.  They are not raised after
    the file has been opened and hashed, which keeps this suitable for legacy
    manifests containing opaque files.
    """

    path_text = os.fspath(path)
    size = os.path.getsize(path_text)
    digest = sha256_file(path_text)
    suffix = Path(path_text).suffix.lower()
    errors: List[str] = []
    arrays: List[ArtifactArrayDescriptor] = []
    key_order: Tuple[str, ...] = ()
    json_type: Optional[str] = None
    json_keys: Tuple[str, ...] = ()
    strict_json: Optional[bool] = None

    if suffix == ".npy":
        kind = "npy"
        try:
            array = np.load(path_text, allow_pickle=False)
            arrays.append(_array_descriptor(None, np.asarray(array)))
        except Exception as exc:  # report-only by design
            errors.append("{}: {}".format(type(exc).__name__, exc))
    elif suffix == ".npz":
        kind = "npz"
        try:
            with np.load(path_text, allow_pickle=False) as archive:
                key_order = tuple(str(key) for key in archive.files)
                for key in archive.files:
                    try:
                        arrays.append(_array_descriptor(str(key), np.asarray(archive[key])))
                    except Exception as exc:  # one opaque member must not hide others
                        errors.append("{}: {}: {}".format(key, type(exc).__name__, exc))
        except Exception as exc:
            errors.append("{}: {}".format(type(exc).__name__, exc))
    elif suffix == ".json":
        kind = "json"
        try:
            with open(path_text, "r", encoding="utf-8") as stream:
                text = stream.read()
            value = json.loads(text)
            json_type = type(value).__name__
            if isinstance(value, dict):
                json_keys = tuple(str(key) for key in value.keys())
            try:
                _strict_json_loads(text)
                strict_json = True
            except ValueError:
                strict_json = False
        except Exception as exc:
            errors.append("{}: {}".format(type(exc).__name__, exc))
    else:
        kind = suffix[1:] if suffix else "binary"

    return ArtifactDescriptor(
        path=path_text,
        kind=kind,
        size_bytes=int(size),
        sha256=digest,
        arrays=tuple(arrays),
        key_order=key_order,
        json_top_level_type=json_type,
        json_keys=json_keys,
        strict_json=strict_json,
        errors=tuple(errors),
    )


def inspect_artifacts(paths: Iterable[str]) -> Tuple[ArtifactDescriptor, ...]:
    return tuple(inspect_artifact(path) for path in paths)


def inspect_tree(
    root: str,
    suffixes: Sequence[str] = (".json", ".npy", ".npz"),
) -> Tuple[ArtifactDescriptor, ...]:
    wanted = {suffix.lower() for suffix in suffixes}
    paths = sorted(
        str(path)
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in wanted
    )
    return inspect_artifacts(paths)


def descriptor_to_dict(descriptor: ArtifactDescriptor) -> Dict[str, Any]:
    """Return a JSON-serializable manifest entry without writing it."""

    return asdict(descriptor)


def load_numpy_legacy(path: str, allow_pickle: bool = False) -> Any:
    """Thin loader retaining NumPy's exceptions, dtypes, and key order."""

    return np.load(resolve_legacy_read_path(path), allow_pickle=allow_pickle)


def load_json_legacy(path: str) -> Any:
    with open(resolve_legacy_read_path(path), "r", encoding="utf-8") as stream:
        return json.load(stream)


def save_numpy_legacy(path: str, array: Any) -> None:
    """Write one NPY array using NumPy's existing default format selection."""

    np.save(os.fspath(path), array)


def save_npz_legacy(path: str, **arrays: Any) -> None:
    """Write NPZ members in caller keyword order, as the legacy scripts do."""

    np.savez(os.fspath(path), **arrays)


def save_npz_compressed_legacy(path: str, **arrays: Any) -> None:
    np.savez_compressed(os.fspath(path), **arrays)


def save_json_legacy(path: str, value: Any, indent: int = 2) -> None:
    """Write permissive Python JSON, including the current NaN behavior."""

    with open(os.fspath(path), "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=indent)


def atomic_json_dump_legacy(path: str, value: Any, indent: int = 2) -> None:
    """Write then replace via the historical ``<path>.tmp`` sibling."""

    path_text = os.fspath(path)
    temporary_path = path_text + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=indent)
    os.replace(temporary_path, path_text)


def expected_shape_report(array: np.ndarray, expected_shape: Sequence[int]) -> Dict[str, Any]:
    actual = tuple(int(n) for n in np.shape(array))
    expected = tuple(int(n) for n in expected_shape)
    return {"actual": actual, "expected": expected, "matches": actual == expected}
