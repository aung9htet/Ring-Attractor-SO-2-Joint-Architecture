"""Ring-attractor reach on one Kangaroo arm joint: ROS 2 port of SingleJointDataCollector.run_trial
(src/tiago_ring_controller/src/single_joint_data_collector.py:219-369). Decisions: docs/kangaroo/DECISIONS.md D11-D12.

Ring:   r2 holds the goal, r1 the start position; each 50 ms tick, signed = right - left gain spikes.
Decode: d = a·d + (1-a)·signed, a = exp(-dt/tau);  v = k_pos·d if d >= 0 else k_neg·(-d)   [rad/s]
        (control/controller.py DriveControlCore; math/control.py:37-40).
Map:    q in [joint_min, joint_max] -> ring index, linear, NOT wrapped (profiles.py, collector mapping).
Values: range = Kangaroo URDF limits; gains = collector defaults k = +/-1e-4, tau = 0.3 s, delay 0 (D11).

Run from src/tiago_ring_controller/src (SingleRingModel loads ./config/... relative paths):
    python3 /work/kangaroo/ring_reach.py <left|right> <joint 0-3> <goal rad> [--seed 12345]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nest
import rclpy
from rclpy.parameter import Parameter

from arm_velocity import DT, ArmVelocityPublisher, sleep_sim, wait_ready  # also puts the ring package on sys.path
from single_ring import SingleRingModel
from tiago_ring_controller.control.controller import DecoderParameters, DriveControlCore
from tiago_ring_controller.control.profiles import CALIBRATION_PROFILE, COLLECTOR_PROFILE as PROFILE

LIMITS = [(-2.618, 2.618), (-1.134, 2.443), (-2.618, 2.618), (-1.134, 2.443)]  # PAL 4-DoF URDF, rad, joints 0-3
HALF_WIDTH = 5  # bump half-width, collector set_ring_goal/set_ring_state default


class RingDrive:
    """The collector's ring: one seeded SingleRingModel. One per process: NEST's kernel is global (D12)."""

    def __init__(self, joint_min, joint_max, seed=12345, profile=PROFILE):
        self.joint_min, self.joint_max, self.seed, self.profile = joint_min, joint_max, seed, profile
        self.model = SingleRingModel(seed=seed)  # rng_seed -> configure_kernel (single_ring.py:163-167)

    def index(self, q):
        return self.profile.joint_to_ring_index(q, self.joint_min, self.joint_max,
                                           self.model.population_size, HALF_WIDTH)

    def start(self, q, goal):
        self.goal_idx, self.state_idx = self.index(goal), self.index(q)
        self.model.r2._inject_bump(self.goal_idx, HALF_WIDTH)
        self.model.r1._inject_bump(self.state_idx, HALF_WIDTH)

    def tick(self):
        """One control tick of ring dynamics; returns right - left gain spikes (collector:161-180)."""
        gm = self.model.gain_modulation
        count = lambda recs: sum(len(s) for s in self.model._collect_spikes(recs))  # noqa: E731
        left, right = count(gm.left_gain_spike_recorders), count(gm.right_gain_spike_recorders)
        nest.Simulate(self.profile.time_step_ms)
        return (count(gm.right_gain_spike_recorders) - right) - (count(gm.left_gain_spike_recorders) - left)


def run_trial(node, arm, joint, goal, seed=12345, profile=PROFILE, decoder=None, limits=None):
    """Drive `joint` of `arm` toward `goal` (rad) with the ring; same step order as the collector.

    `profile` sets ring mapping and spike scale (CALIBRATION_PROFILE for calibrated gains, D17); `decoder` defaults
    to the uncalibrated DecoderParameters(); `limits` (joint_min, joint_max) rad defaults to the URDF range (D20).
    """
    limits = tuple(limits or LIMITS[joint])
    ring = RingDrive(*limits, seed=seed, profile=profile)
    arm.prime_command_state(force=True)
    q_start = arm.current_positions[joint]
    ring.start(q_start, goal)
    core = DriveControlCore(DT, decoder or DecoderParameters(),
                            spike_scale=profile.spike_scale(ring.model.population_size),
                            drive_threshold=profile.drive_threshold, n_settle=profile.n_settle)
    buffer = [core.step(ring.tick()) for _ in range(profile.lookahead)]
    out = {'signed_spike': [], 'signed_drive': [], 'decoded_velocity': [], 'joint_position': []}
    stop_reason = 'max_steps'
    for _ in range(profile.max_steps):
        # ponytail: NEST runs between sim-time sleeps, as rospy.sleep did on TIAGo; lock-step if timing matters (D4).
        arm.publish_receding_trajectory(joint, [s.velocity_command for s in buffer])
        out['signed_spike'].append(buffer[0].raw_signed_spike)
        out['signed_drive'].append(buffer[0].signed_drive)  # spike x profile scale: what calibration fits on
        out['decoded_velocity'].append(buffer[0].velocity_command)  # the velocity the command state advanced by
        sleep_sim(node, DT)
        out['joint_position'].append(arm.current_positions[joint])
        if buffer[0].is_settled:
            stop_reason = 'drive_settled'
            break
        buffer = buffer[1:] + [core.step(ring.tick())]
    arm.stop_joint(joint)
    sleep_sim(node, 1.0)
    p = core.parameters
    return dict(out, q_start=q_start, q_goal=goal, q_final=arm.current_positions[joint], stop_reason=stop_reason,
                seed=seed, profile=profile.name, goal_ring_index=ring.goal_idx, initial_ring_index=ring.state_idx,
                joint_limits=limits, dt=DT, decoder_gain_positive=p.gain_positive,
                decoder_gain_negative=p.gain_negative, decoder_tau=p.tau_s, decoder_delay_steps=p.delay_steps,
                nest_version=nest.__version__)


def main():
    parser = argparse.ArgumentParser(description='Ring-attractor reach of one Kangaroo arm joint.')
    parser.add_argument('side', choices=['left', 'right'])
    parser.add_argument('joint', type=int, choices=range(len(LIMITS)), help='joint index 0-3')
    parser.add_argument('goal', type=float, help='goal position, rad')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--calibration', help='calibration.json from calibrate.py; runs under its profile (Q9)')
    parser.add_argument('--from-home', action='store_true', help="reset the arm to PAL's home pose first (D18)")
    parser.add_argument('--limits', help='limits.json from joint_limits.py (D20); default: the calibration\'s range')
    args = parser.parse_args()
    profile, decoder, limits = PROFILE, None, None
    if args.limits:
        lim = json.loads(Path(args.limits).read_text())
        assert (lim['side'], lim['joint']) == (args.side, args.joint), lim
        limits = (lim['joint_min'], lim['joint_max'])
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text())
        assert (cal['side'], cal['joint'], cal['profile']) == (args.side, args.joint, CALIBRATION_PROFILE.name), cal
        assert limits is None or list(limits) == list(cal['joint_limits']), 'ring range must match the calibration'
        limits = tuple(cal['joint_limits'])  # the ring mapping the gains were fitted under
        profile, decoder = CALIBRATION_PROFILE, DecoderParameters(
            cal['decoder_gain_positive'], cal['decoder_gain_negative'], cal['decoder_tau'], cal['decoder_delay_steps'])

    rclpy.init()
    node = rclpy.create_node('ring_reach', parameter_overrides=[Parameter('use_sim_time', value=True)])
    arm = ArmVelocityPublisher(node, args.side)
    if not wait_ready(node, [arm]):
        sys.exit('simulation or arm controller not running')
    if args.from_home:
        from calibrate import reset_arm  # lazy: calibrate imports this module
        reset_arm(node, arm, args.side, pose=cal.get('start_pose') if args.calibration else None)  # D25
    result = run_trial(node, arm, args.joint, args.goal, args.seed, profile, decoder, limits)
    rclpy.shutdown()

    out_dir = Path(__file__).resolve().parent / 'results/ring_reach' / \
        f'{args.side}_j{args.joint}_seed{args.seed}_{time.strftime("%Y%m%d_%H%M%S")}'  # new dir per run (rule 15)
    out_dir.mkdir(parents=True)
    (out_dir / 'trial.json').write_text(json.dumps(
        dict(result, side=args.side, joint=args.joint, calibration=args.calibration, from_home=args.from_home,
             limits_file=args.limits), indent=1))
    print(f'{arm.joint_names[args.joint]}: {result["q_start"]:+.3f} -> {result["q_final"]:+.3f} rad '
          f'(goal {args.goal:+.3f}, {result["stop_reason"]}, {len(result["signed_spike"])} ticks) -> {out_dir}')


if __name__ == '__main__':
    main()
