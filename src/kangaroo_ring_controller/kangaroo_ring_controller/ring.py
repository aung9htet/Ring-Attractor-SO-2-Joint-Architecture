"""Ring-attractor reach of one Kangaroo arm joint: ROS 2 port of SingleJointDataCollector.run_trial
(src/tiago_ring_controller/src/single_joint_data_collector.py:219-369). Decisions: docs/kangaroo/DECISIONS.md D11-D12.

Ring:   r2 holds the goal, r1 the start position; each 50 ms tick, signed = right - left gain spikes.
Decode: d = a·d + (1-a)·signed, a = exp(-dt/tau);  v = k_pos·d if d >= 0 else k_neg·(-d)   [rad/s]
        (control/controller.py DriveControlCore; math/control.py:37-40).
Stop:   settled after 10 ticks of |d_delayed| < 5, counted from the first tick |d_delayed| reaches 5 (settle.py; TIAGo
        counts from tick 1, F14, Q14 (b)); else max_steps.
Map:    q in [joint_min, joint_max] -> ring index, linear, NOT wrapped (profiles.py, collector mapping).
Values: range = Kangaroo URDF limits; gains = collector defaults k = +/-1e-4, tau = 0.3 s, delay 0 (D11).
"""
import nest

from .arm import DT
from .joint_setup import LIMITS  # noqa: F401  (re-exported)
from .ring_core import RING_PARAMS, RING_WEIGHTS
from .settle import GuardedDriveCore
from single_ring import SingleRingModel
from tiago_ring_controller.control.controller import DecoderParameters

HALF_WIDTH = 5  # bump half-width, collector set_ring_goal/set_ring_state default


class RingDrive:
    """The collector's ring: one seeded SingleRingModel. One per process: NEST's kernel is global (D12)."""

    def __init__(self, setup, seed=12345):
        """The ring over `setup`'s ring range, under its profile (joint_setup.JointSetup, D35)."""
        self.setup, self.seed, self.profile = setup, seed, setup.profile
        self.joint_min, self.joint_max = setup.limits
        self.model = SingleRingModel(seed=seed, ring_params_file=str(RING_PARAMS), weights_dir=str(RING_WEIGHTS))

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


def run_trial(arm, setup, goal, seed=12345, on_tick=None, ring=None):
    """Drive `setup.joint` of `arm` toward `goal` (rad) with the ring; same step order as the collector.

    `setup` (joint_setup.JointSetup, D35): its profile sets ring mapping and spike scale (D17), decoder None means
    the uncalibrated DecoderParameters(), limits are the ring range (D20). Raises ValueError, before anything is
    published, if `arm` is not the setup's side or `ring` was built for another setup or seed.
    `on_tick(tick) -> bool` (D27): called after each tick with step, joint_position, signed_drive, velocity,
    state_index, goal_index, neurons; True ends the trial with stop_reason 'stopped'.
    `ring`: a RingDrive built in advance with RingDrive(setup, seed) (the GUI's warm spare, D30); default: build one
    here, which takes 22-30 s.
    """
    if arm.side != setup.side:
        raise ValueError(f'{setup.side} setup on the {arm.side} arm')
    if ring and (ring.setup, ring.seed) != (setup, seed):
        raise ValueError(f'ring built for {ring.setup}, seed {ring.seed}; trial is {setup}, seed {seed}')
    joint, profile, decoder, limits = setup.joint, setup.profile, setup.decoder, tuple(setup.limits)
    ring = ring or RingDrive(setup, seed)
    arm.sleep(DT)  # ROS is not spun while a ring builds (22-30 s): take fresh joint states first
    arm.prime_command_state(force=True)
    q_start = arm.current_positions[joint]
    ring.start(q_start, goal)
    core = GuardedDriveCore(DT, decoder or DecoderParameters(),  # settle armed by the first crossing (F14, Q14 b)
                            spike_scale=profile.spike_scale(ring.model.population_size),
                            drive_threshold=profile.drive_threshold, n_settle=profile.n_settle)
    buffer = [core.step(ring.tick()) for _ in range(profile.lookahead)]
    out = {'signed_spike': [], 'signed_drive': [], 'decoded_velocity': [], 'joint_position': []}
    stop_reason = 'max_steps'
    for step in range(profile.max_steps):
        # ponytail: NEST runs between sim-time sleeps, as rospy.sleep did on TIAGo; lock-step if timing matters (D4).
        arm.publish_receding_trajectory(joint, [s.velocity_command for s in buffer])
        out['signed_spike'].append(buffer[0].raw_signed_spike)
        out['signed_drive'].append(buffer[0].signed_drive)  # spike x profile scale: what calibration fits on
        out['decoded_velocity'].append(buffer[0].velocity_command)  # the velocity the command state advanced by
        arm.sleep(DT)
        q = arm.current_positions[joint]
        out['joint_position'].append(q)
        if on_tick and on_tick(dict(step=step, joint_position=q, signed_drive=float(buffer[0].signed_drive),
                                    velocity=float(buffer[0].velocity_command), state_index=int(ring.index(q)),
                                    goal_index=int(ring.goal_idx), neurons=int(ring.model.population_size))):
            stop_reason = 'stopped'
            break
        if buffer[0].is_settled:
            stop_reason = 'drive_settled'
            break
        buffer = buffer[1:] + [core.step(ring.tick())]
    arm.stop_joint(joint)
    arm.sleep(1.0)
    p = core.parameters
    return dict(out, q_start=q_start, q_goal=goal, q_final=arm.current_positions[joint], stop_reason=stop_reason,
                seed=seed, profile=profile.name, goal_ring_index=ring.goal_idx, initial_ring_index=ring.state_idx,
                joint_limits=limits, dt=DT, decoder_gain_positive=p.gain_positive,
                decoder_gain_negative=p.gain_negative, decoder_tau=p.tau_s, decoder_delay_steps=p.delay_steps,
                nest_version=nest.__version__)
