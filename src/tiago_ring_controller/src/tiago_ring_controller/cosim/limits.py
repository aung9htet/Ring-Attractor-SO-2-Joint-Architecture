"""Joint limits from the URDF instead of an empirical sweep (plan B.7)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, Optional, Tuple

from ..ros.transport import JOINT_NAMES


def joint_limits_from_urdf(urdf_xml: str, joint_name: str) -> Optional[Tuple[float, float]]:
    """Return ``(lower, upper)`` for a revolute/prismatic joint, else ``None``."""

    root = ET.fromstring(urdf_xml)
    for joint in root.iter("joint"):
        if joint.get("name") != joint_name:
            continue
        limit = joint.find("limit")
        if limit is None:
            return None
        lower = limit.get("lower")
        upper = limit.get("upper")
        if lower is None or upper is None:
            return None
        return float(lower), float(upper)
    return None


def arm_joint_limits_from_urdf(urdf_xml: str) -> Dict[int, Tuple[float, float]]:
    limits: Dict[int, Tuple[float, float]] = {}
    for index, name in enumerate(JOINT_NAMES):
        value = joint_limits_from_urdf(urdf_xml, name)
        if value is not None:
            limits[index] = value
    return limits


def joint_limits_from_transport(transport: Any, joint_index: int) -> Optional[Tuple[float, float]]:
    """Read ``/robot_description`` through the transport and parse it."""

    urdf = transport.get_param("/robot_description")
    if not urdf:
        return None
    return joint_limits_from_urdf(str(urdf), JOINT_NAMES[int(joint_index)])


__all__ = ["arm_joint_limits_from_urdf", "joint_limits_from_transport", "joint_limits_from_urdf"]
