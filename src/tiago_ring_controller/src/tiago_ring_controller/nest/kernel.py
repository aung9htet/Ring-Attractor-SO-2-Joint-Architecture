"""Small NEST-kernel and recorder utilities.

The NEST module is always supplied by the caller.  Importing this module does
not initialize PyNEST or modify global kernel state.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np


def configure_kernel(
    backend: Any,
    reset_kernel: bool = True,
    verbosity: Optional[str] = "M_ERROR",
    local_num_threads: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> None:
    if reset_kernel:
        backend.ResetKernel()
    if verbosity is not None:
        backend.set_verbosity(verbosity)
    status: Dict[str, Any] = {}
    if local_num_threads is not None:
        status["local_num_threads"] = local_num_threads
    if rng_seed is not None:
        status["rng_seed"] = int(rng_seed)
    if status:
        backend.SetKernelStatus(status)


def recorder_events(backend: Any, recorder: Any) -> MappingLike:
    """Read events from either legacy recorder access style."""

    getter = getattr(recorder, "get", None)
    if callable(getter):
        try:
            events = getter("events")
            if events is not None:
                return events
        except (KeyError, TypeError, AttributeError):
            pass
    return backend.GetStatus(recorder, "events")[0]


def spike_counts(backend: Any, recorders: Iterable[Any]) -> np.ndarray:
    return np.array(
        [len(recorder_events(backend, recorder).get("times", [])) for recorder in recorders],
        dtype=float,
    )


# Avoid importing typing.Mapping solely at runtime in hot recorder loops.
MappingLike = Dict[str, Any]
