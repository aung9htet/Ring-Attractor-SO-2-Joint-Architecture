#!/usr/bin/env python3
"""Small rostest entrypoint for the declarative ROS interface contract."""

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.ros.transport import (  # noqa: E402
    ARM_COMMAND_TOPIC,
    JOINT_NAMES,
    JOINT_STATES_TOPIC,
    PLAY_MOTION_ACTION,
    WRIST_FT_TOPIC,
)


class RosInterfaceContractSmoke(unittest.TestCase):
    def test_namespaces_and_joint_order(self):
        self.assertEqual(ARM_COMMAND_TOPIC, "/arm_controller/command")
        self.assertEqual(JOINT_STATES_TOPIC, "/joint_states")
        self.assertEqual(WRIST_FT_TOPIC, "/wrist_ft")
        self.assertEqual(PLAY_MOTION_ACTION, "play_motion")
        self.assertEqual(
            JOINT_NAMES,
            tuple("arm_%d_joint" % index for index in range(1, 8)),
        )


if __name__ == "__main__":
    import rostest

    rostest.rosrun(
        "tiago_ring_controller",
        "ros_interfaces_runner",
        RosInterfaceContractSmoke,
    )
