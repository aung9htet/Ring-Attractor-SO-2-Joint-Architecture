"""Reachable range of one Kangaroo arm joint: port of TiagoCalibration.find_joint_limits_manually
(src/tiago_ring_controller/src/calibrate_single_joint.py:1064-1086). Decision: docs/kangaroo/DECISIONS.md D20.

TIAGo drives the joint at -v for T seconds and reads the measured position as joint_min, then at +v for T as
joint_max (v = 1 rad/s, T = 5 s). Here the arm is reset to PAL's home pose before each sweep and at the end, so
the range is measured in the configuration calibration starts from, and a 5 rad sweep need not cross the whole
5.24 rad range of joints 0 and 2. The reading is the measured joint position [rad], so an obstruction (F9) bounds it.
"""

VELOCITY, SECONDS = 1.0, 5.0  # rad/s, s: calibrate_single_joint.py:1067-1068


def measure_limits(arm, joint, velocity=VELOCITY, seconds=SECONDS):
    """Return (joint_min, joint_max) in rad, measured after sweeping each way from home."""
    ends = []
    for v in (-velocity, velocity):
        arm.reset()
        arm.hold(joint, v, seconds)
        ends.append(arm.current_positions[joint])
    arm.reset()
    return tuple(ends)
