"""Pure schemas and row formatting for the two existing FT loggers."""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence, Tuple

from ..config import installed_package_share
from .transport import JOINT_NAMES


@dataclass(frozen=True)
class CsvLogSchema:
    directory_name: str
    filename_prefix: str
    columns: Tuple[str, ...]


WRENCH_COLUMNS: Tuple[str, ...] = (
    "timestamp",
    "force_x",
    "force_y",
    "force_z",
    "torque_x",
    "torque_y",
    "torque_z",
)

SENSOR_LOG_SCHEMA = CsvLogSchema(
    directory_name="sensor_data",
    filename_prefix="ft_sensor_data",
    columns=WRENCH_COLUMNS + JOINT_NAMES,
)

TORQUE_LOG_SCHEMA = CsvLogSchema(
    directory_name="torque_data",
    filename_prefix="torque_data",
    columns=WRENCH_COLUMNS,
)


def timestamp_token(value: datetime) -> str:
    """Return the filename token currently used by both loggers."""

    return value.strftime("%Y%m%d_%H%M%S")


def log_filename(schema: CsvLogSchema, timestamp: str) -> str:
    return "%s_%s.csv" % (schema.filename_prefix, timestamp)


def log_path(package_dir: str, schema: CsvLogSchema, timestamp: str) -> str:
    """Construct a logger path without creating any directories or files."""

    return os.path.join(
        package_dir,
        "experiment_results",
        schema.directory_name,
        log_filename(schema, timestamp),
    )


def package_root_for_module(module_file: str) -> str:
    """Return the legacy package root in source or Catkin install space.

    Source modules retain their exact two-directory ``__file__`` lookup.  An
    installed executable or Python module instead uses the corresponding
    package-share directory, avoiding the unrelated ``<prefix>/lib`` tree.
    """

    share_dir = installed_package_share(module_file)
    if share_dir is not None:
        return share_dir
    return os.path.dirname(os.path.dirname(os.path.abspath(module_file)))


def format_wrench_row(
    elapsed_s: float,
    force_xyz: Sequence[float],
    torque_xyz: Sequence[float],
) -> Tuple[str, ...]:
    """Format the seven columns common to both logger callbacks."""

    return (
        "%.4f" % elapsed_s,
        "%.6f" % force_xyz[0],
        "%.6f" % force_xyz[1],
        "%.6f" % force_xyz[2],
        "%.6f" % torque_xyz[0],
        "%.6f" % torque_xyz[1],
        "%.6f" % torque_xyz[2],
    )


def format_sensor_row(
    elapsed_s: float,
    force_xyz: Sequence[float],
    torque_xyz: Sequence[float],
    joint_positions: Iterable[float],
) -> Tuple[str, ...]:
    """Format the TiagoSubscriber row, including seven joint positions."""

    return format_wrench_row(elapsed_s, force_xyz, torque_xyz) + tuple(
        "%.6f" % position for position in joint_positions
    )


__all__ = [
    "CsvLogSchema",
    "SENSOR_LOG_SCHEMA",
    "TORQUE_LOG_SCHEMA",
    "WRENCH_COLUMNS",
    "format_sensor_row",
    "format_wrench_row",
    "log_filename",
    "log_path",
    "package_root_for_module",
    "timestamp_token",
]
