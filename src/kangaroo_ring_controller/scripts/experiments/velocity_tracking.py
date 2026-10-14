"""Commanded vs measured joint velocity for every Kangaroo arm joint (work-order step 5).

Each joint in turn gets a constant velocity step toward the centre of its range for STEP_S seconds through
ArmVelocityPublisher, is stopped, and is then driven back. Only the outbound step and stop are recorded.
Writes results/velocity_tracking/both_arms_<time>/{tracking.csv, tracking.png} (a new directory per run)
and prints steady-state metrics.
No random numbers (fixed commands, paced on simulation time), but not bit-reproducible: ROS message timing
against simulation time varies between runs (two runs differed by up to 0.003 in the steady-state ratio).

Run inside the container with the simulation up:
    python3 src/kangaroo_ring_controller/scripts/experiments/velocity_tracking.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.time import Time  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller import new_run_dir  # noqa: E402
from kangaroo_ring_controller.arm import DT, N_JOINTS, connect  # noqa: E402

SPEED = 0.3           # rad/s commanded
STEP_S = 2.0          # s of constant command
SETTLE_S = 1.0        # s recorded after the stop command
STEADY_FROM_S = 0.5   # s; earlier samples are the transient, excluded from steady-state metrics


def drive(node, arm, idx, velocity, seconds):
    arm.hold(idx, velocity, seconds)
    arm.stop_joint(idx)
    arm.sleep(SETTLE_S)


def main():
    node, arms = connect('velocity_tracking')
    arms = list(arms.values())
    rows, rec = [], {'joint': None, 'v': 0.0, 't0': 0.0}  # rows: (joint, t since step start, commanded, measured)

    def on_joint_states(msg):
        if rec['joint'] in msg.name:
            t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9 - rec['t0']
            rows.append((rec['joint'], t, rec['v'] if t < STEP_S else 0.0, msg.velocity[msg.name.index(rec['joint'])]))

    node.create_subscription(JointState, '/joint_states', on_joint_states, 10)

    for arm in arms:
        for idx in range(N_JOINTS):
            v = -SPEED if arm.current_positions[idx] > 0 else SPEED
            rec.update(joint=arm.joint_names[idx], v=v, t0=node.get_clock().now().nanoseconds * 1e-9)
            drive(node, arm, idx, v, STEP_S)
            rec['joint'] = None
            drive(node, arm, idx, -v, STEP_S)  # back to the start pose, not recorded

    out = new_run_dir('velocity_tracking', 'both_arms')
    with open(out / 'tracking.csv', 'w', newline='') as f:
        csv.writer(f).writerows([('joint', 't_s', 'commanded_rad_s', 'measured_rad_s'), *rows])

    joints = [name for arm in arms for name in arm.joint_names]
    fig, axes = plt.subplots(2, N_JOINTS, figsize=(16, 6), sharex=True, sharey=True)
    print(f"{'joint':18} {'cmd':>6} {'steady mean':>11} {'ratio':>6} {'steady std':>10} {'t90 (s)':>7} "
          f"{'peak overshoot':>14} {'stop overshoot':>14}")
    for ax, name in zip(axes.flat, joints):
        t, cmd, meas = np.array([r[1:] for r in rows if r[0] == name]).T
        steady = meas[(t >= STEADY_FROM_S) & (t < STEP_S)]
        v = cmd[0]
        reached = t[(np.abs(meas) >= 0.9 * abs(v)) & (np.sign(meas) == np.sign(v))]
        t90 = f'{reached[0]:.2f}' if reached.size else 'never'
        peak = np.abs(meas[t < STEP_S]).max() / abs(v) - 1  # fraction above |commanded| during the step
        stop = np.abs(meas[t >= STEP_S + DT]).max()  # rad/s still moving after the stop command
        print(f'{name:18} {v:+6.2f} {steady.mean():+11.4f} {steady.mean() / v:6.3f} {steady.std():10.4f} {t90:>7} '
              f'{peak:+13.0%} {stop:11.3f} r/s')
        ax.plot(t, cmd, 'k--', label='commanded')
        ax.plot(t, meas, label='measured')
        ax.set_title(name)
    for ax in axes[:, 0]:
        ax.set_ylabel('rad/s')
    for ax in axes[-1]:
        ax.set_xlabel('s since step start')
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / 'tracking.png', dpi=120)
    print(f'wrote {out}/tracking.csv and tracking.png')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
