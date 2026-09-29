"""DataPack: the named, timestamped, JSON-serialisable unit of exchange.

A :class:`DataPack` is what an engine publishes after a step and accepts
before one, and what transceiver functions map between.  Its payload is a
plain mapping of JSON-compatible values so that every tick of a trial can be
logged, compared byte-for-byte across runs, and replayed without either
simulator.  NumPy scalars and arrays are converted on construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..evaluation.serialization import json_safe


class DataPackError(ValueError):
    """Raised for payloads that cannot be represented as JSON."""


def _freeze(data: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise DataPackError("DataPack data must be a mapping, got %r" % type(data))
    converted = json_safe(dict(data))
    try:
        json.dumps(converted)
    except (TypeError, ValueError) as exc:
        raise DataPackError("DataPack data is not JSON-serialisable: %s" % exc)
    return converted


@dataclass(frozen=True)
class DataPack:
    """Immutable named payload stamped with the producing engine's time."""

    name: str
    t_ms: float
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DataPackError("DataPack name must be a non-empty string")
        object.__setattr__(self, "t_ms", float(self.t_ms))
        object.__setattr__(self, "data", _freeze(self.data))

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def replace(self, t_ms: Optional[float] = None, **changes: Any) -> "DataPack":
        """Return a copy with updated fields (shallow merge of ``data``)."""

        merged = dict(self.data)
        merged.update(changes)
        return DataPack(self.name, self.t_ms if t_ms is None else t_ms, merged)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "t_ms": self.t_ms, "data": dict(self.data)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataPack":
        return cls(str(value["name"]), float(value["t_ms"]), dict(value.get("data", {})))

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "DataPack":
        return cls.from_dict(json.loads(text))


def make_datapack(name: str, t_ms: float, **data: Any) -> DataPack:
    return DataPack(name, t_ms, data)


def datapacks_to_dict(packs: Mapping[str, DataPack]) -> Dict[str, Dict[str, Any]]:
    return {key: pack.to_dict() for key, pack in packs.items()}


def datapacks_from_dict(value: Mapping[str, Mapping[str, Any]]) -> Dict[str, DataPack]:
    return {str(key): DataPack.from_dict(item) for key, item in value.items()}


__all__ = [
    "DataPack",
    "DataPackError",
    "datapacks_from_dict",
    "datapacks_to_dict",
    "make_datapack",
]
