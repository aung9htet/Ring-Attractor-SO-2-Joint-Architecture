"""Pure controller, trajectory, ROS contract, and facade-equivalence tests."""

import importlib.util
import math
import subprocess
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LEGACY = ROOT / "legacy"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.control.controller import (  # noqa: E402
    DecoderParameters,
    DriveControlCore,
    apply_discrete_delay,
    decode_asymmetric_velocity,
    decoder_features_from_spikes,
    exponential_filter,
    predicted_displacement,
)
from tiago_ring_controller.control.trajectory import (  # noqa: E402
    CommandState,
    build_receding_trajectory,
    build_stop_trajectory,
    tracking_drift,
)
from tiago_ring_controller.ros.camera import (  # noqa: E402
    CAMERA_SUBSCRIPTION_CONTRACT,
    DEFAULT_RECORDING_CAMERA_TOPIC,
    resolve_image_topic,
)
from tiago_ring_controller.ros.logging import (  # noqa: E402
    SENSOR_LOG_SCHEMA,
    TORQUE_LOG_SCHEMA,
    format_sensor_row,
    format_wrench_row,
    log_filename,
    log_path,
    package_root_for_module,
    timestamp_token,
)
from tiago_ring_controller.ros.transport import (  # noqa: E402
    ARM_COMMAND_CONTRACT,
    GRAVITY_COMPENSATION_CONTRACTS,
    GRAVITY_COMPENSATION_TOPICS,
    JOINT_NAMES,
    JOINT_STATES_CONTRACT,
    NODE_CONTRACTS,
    PLAY_MOTION_CONTRACT,
    TORQUE_RECORDER_FT_CONTRACT,
    WRIST_FT_SUBSCRIBER_CONTRACT,
    joint_state_vectors,
    subscriber_joint_positions,
)


class ControlPreprocessingTests(unittest.TestCase):
    def test_filter_matches_the_zero_initialized_legacy_loop(self):
        spikes = np.array([10.0, -5.0, 2.0, 0.0])
        dt_s = 0.05
        tau_s = 0.3
        alpha = np.exp(-dt_s / tau_s)
        expected = []
        state = 0.0
        for spike in spikes:
            state = alpha * state + (1.0 - alpha) * spike
            expected.append(state)
        actual = exponential_filter(spikes, dt_s, tau_s)
        np.testing.assert_array_equal(actual, np.asarray(expected))
        self.assertEqual(str(actual.dtype), "float64")

    def test_delay_zero_fill_copy_and_long_delay_contract(self):
        values = np.array([1.0, 2.0, 3.0])
        no_delay = apply_discrete_delay(values, 0)
        np.testing.assert_array_equal(no_delay, values)
        self.assertIsNot(no_delay, values)
        np.testing.assert_array_equal(apply_discrete_delay(values, -2), values)
        np.testing.assert_array_equal(apply_discrete_delay(values, 2), [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(apply_discrete_delay(values, 3), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(apply_discrete_delay(values, 9), [0.0, 0.0, 0.0])

    def test_asymmetric_decode_preserves_sign_split_and_zero_branch(self):
        self.assertEqual(decode_asymmetric_velocity(5.0, 0.002, -0.003), 0.01)
        self.assertEqual(decode_asymmetric_velocity(-5.0, 0.002, -0.003), -0.015)
        self.assertEqual(decode_asymmetric_velocity(0.0, 0.002, -0.003), 0.0)

    def test_decoder_features_preserve_filter_before_delay_and_integrals(self):
        spikes = np.array([5.0, -8.0, 4.0, -1.0])
        filtered = exponential_filter(spikes, 0.1, 0.25)
        delayed = apply_discrete_delay(filtered, 1)
        features = decoder_features_from_spikes(spikes, 0.1, 0.25, 1)
        np.testing.assert_array_equal(features["filtered_drive"], filtered)
        np.testing.assert_array_equal(features["delayed_drive"], delayed)
        np.testing.assert_array_equal(features["drive_pos"], np.maximum(delayed, 0.0))
        np.testing.assert_array_equal(features["drive_neg"], np.maximum(-delayed, 0.0))
        self.assertEqual(features["S_pos"], float(np.sum(np.maximum(delayed, 0.0)) * 0.1))
        self.assertEqual(features["S_neg"], float(np.sum(np.maximum(-delayed, 0.0)) * 0.1))

    def test_stateful_core_matches_batch_processing_and_strict_settle_threshold(self):
        parameters = DecoderParameters(
            gain_positive=0.2,
            gain_negative=-0.3,
            tau_s=0.5,
            delay_steps=2,
        )
        raw = [8.0, -2.0, 0.0, 0.0]
        core = DriveControlCore(
            dt_s=0.1,
            parameters=parameters,
            spike_scale=0.5,
            drive_threshold=1.0,
            n_settle=2,
        )
        samples = core.prefill(raw)
        batch = decoder_features_from_spikes(np.asarray(raw) * 0.5, 0.1, 0.5, 2)
        np.testing.assert_allclose(
            [sample.filtered_drive for sample in samples],
            batch["filtered_drive"],
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            [sample.delayed_drive for sample in samples],
            batch["delayed_drive"],
            rtol=0.0,
            atol=1e-15,
        )
        self.assertFalse(samples[0].is_settled)
        self.assertTrue(samples[1].is_settled)
        for sample in samples:
            expected_velocity = decode_asymmetric_velocity(
                sample.delayed_drive, 0.2, -0.3
            )
            self.assertEqual(sample.velocity_command, expected_velocity)
        core.reset()
        self.assertEqual(core.consecutive_settled, 0)
        self.assertEqual(core.filtered_drive, 0.0)

    def test_advance_matches_the_legacy_loop_without_decoding_or_coercion(self):
        raw = [7, np.float64(-3.5), 2, 0]
        dt_s = 0.05
        tau_s = 0.3
        delay_steps = 2
        alpha = np.exp(-dt_s / tau_s)
        core = DriveControlCore(
            dt_s,
            DecoderParameters(tau_s=tau_s, delay_steps=delay_steps),
            alpha=alpha,
        )

        legacy_filtered = 0.0
        legacy_queue = [0.0] * delay_steps
        delay_schedule = [delay_steps, 0, delay_steps, 1]
        expected = []
        actual = []
        for signed_spike, current_delay in zip(raw, delay_schedule):
            legacy_filtered = (
                alpha * legacy_filtered + (1.0 - alpha) * signed_spike
            )
            if current_delay > 0:
                legacy_queue.append(legacy_filtered)
                legacy_delayed = legacy_queue.pop(0)
            else:
                legacy_delayed = legacy_filtered
            expected.append((signed_spike, legacy_filtered, legacy_delayed))
            actual.append(core.advance(signed_spike, current_delay))

        for observed, wanted in zip(actual, expected):
            self.assertIs(observed[0], wanted[0])
            self.assertEqual(observed[1], wanted[1])
            self.assertEqual(observed[2], wanted[2])

    def test_predicted_displacement_uses_positive_and_negative_fit_features(self):
        parameters = DecoderParameters(0.1, -0.25, 0.3, 1)
        spikes = np.array([4.0, -5.0, 2.0])
        features = decoder_features_from_spikes(spikes, 0.05, 0.3, 1)
        expected = 0.1 * features["S_pos"] - 0.25 * features["S_neg"]
        self.assertEqual(predicted_displacement(spikes, 0.05, parameters), expected)


class TrajectoryContractTests(unittest.TestCase):
    def test_horizon_integrates_all_points_but_advances_state_one_step(self):
        base = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        trajectory = build_receding_trajectory(base, 2, [1.0, -0.5, 0.25], 0.1)
        self.assertIsNotNone(trajectory)
        self.assertEqual(len(trajectory.points), 3)
        self.assertEqual(
            [point.time_from_start_s for point in trajectory.points],
            [0.1, 0.2, 0.30000000000000004],
        )
        self.assertEqual(
            [point.positions[2] for point in trajectory.points],
            [2.1, 2.0500000000000003, 2.075],
        )
        self.assertEqual(
            [point.velocities[2] for point in trajectory.points],
            [1.0, -0.5, 0.25],
        )
        for point in trajectory.points:
            self.assertEqual(point.positions[:2] + point.positions[3:], tuple(base[:2] + base[3:]))
            self.assertEqual(sum(value != 0.0 for value in point.velocities), 1)
        self.assertEqual(trajectory.next_commanded_positions[2], 2.1)
        self.assertEqual(trajectory.next_commanded_velocities[2], 1.0)
        self.assertEqual(base, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_empty_and_stop_horizons(self):
        base = [0.0] * 7
        self.assertIsNone(build_receding_trajectory(base, 0, [], 0.05))
        stop = build_stop_trajectory(base, 4)
        self.assertEqual(len(stop.points), 3)
        self.assertEqual([point.time_from_start_s for point in stop.points], [0.05, 0.1, 0.15000000000000002])
        self.assertTrue(all(point.velocities == (0.0,) * 7 for point in stop.points))
        self.assertEqual(stop.next_commanded_positions, (0.0,) * 7)

    def test_command_state_prime_rebase_and_consumption(self):
        state = CommandState()
        with self.assertRaises(RuntimeError):
            state.build(0, [1.0], 0.1)
        state.prime([0.0] * 7)
        commanded_positions = state.commanded_positions
        first = state.build(1, [0.5, 0.5], 0.1)
        self.assertEqual(first.next_commanded_positions[1], 0.05)
        self.assertEqual(state.commanded_positions[1], 0.05)
        self.assertIs(state.commanded_positions, commanded_positions)
        # Strict comparison: equality does not rebase.
        measured = [0.0] * 7
        measured[1] = 0.15
        self.assertFalse(state.maybe_rebase(measured, None, 1, 0.1))
        measured[1] = 0.1500001
        self.assertTrue(state.maybe_rebase(measured, None, 1, 0.1))
        self.assertEqual(state.commanded_positions[1], 0.1500001)
        self.assertEqual(tracking_drift([1.0], [0.25], 0), 0.75)


class RosContractTests(unittest.TestCase):
    def test_topics_action_nodes_and_queue_sizes_are_exact(self):
        self.assertEqual(JOINT_NAMES, tuple("arm_%d_joint" % index for index in range(1, 8)))
        self.assertEqual(ARM_COMMAND_CONTRACT.name, "/arm_controller/command")
        self.assertEqual(ARM_COMMAND_CONTRACT.message_type, "trajectory_msgs/JointTrajectory")
        self.assertEqual(ARM_COMMAND_CONTRACT.queue_size, 1)
        self.assertEqual(JOINT_STATES_CONTRACT.name, "/joint_states")
        self.assertEqual(JOINT_STATES_CONTRACT.queue_size, 1)
        self.assertEqual(WRIST_FT_SUBSCRIBER_CONTRACT.name, "/wrist_ft")
        self.assertEqual(WRIST_FT_SUBSCRIBER_CONTRACT.queue_size, 1)
        self.assertIsNone(TORQUE_RECORDER_FT_CONTRACT.queue_size)
        self.assertEqual(
            GRAVITY_COMPENSATION_TOPICS,
            tuple("/gravity_compensation/arm_%d_joint/command" % index for index in range(1, 8)),
        )
        self.assertTrue(all(contract.queue_size == 1 for contract in GRAVITY_COMPENSATION_CONTRACTS))
        self.assertEqual(PLAY_MOTION_CONTRACT.name, "play_motion")
        self.assertEqual(PLAY_MOTION_CONTRACT.motion_name, "tiago_experiment_start_1")
        self.assertFalse(PLAY_MOTION_CONTRACT.skip_planning)
        self.assertEqual(PLAY_MOTION_CONTRACT.server_wait_s, 5.0)
        self.assertEqual(NODE_CONTRACTS["tiago_controller.py"].requested_name, None)
        self.assertEqual(NODE_CONTRACTS["record_experiments.py"].requested_name, "torque_recorder")
        self.assertTrue(NODE_CONTRACTS["record_experiments.py"].anonymous)

    def test_joint_state_ordering_missing_values_and_velocity_zero_fill(self):
        names = list(reversed(JOINT_NAMES))
        positions = list(range(7))
        velocities = [10.0, 11.0]
        ordered = joint_state_vectors(names, positions, velocities)
        self.assertEqual(ordered[0], (6, 5, 4, 3, 2, 1, 0))
        self.assertEqual(ordered[1], (0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 10.0))
        self.assertIsNone(joint_state_vectors(names[:-1], positions[:-1], velocities))
        self.assertEqual(joint_state_vectors(JOINT_NAMES, positions, None)[1], (0.0,) * 7)
        self.assertEqual(subscriber_joint_positions(names, positions), (6, 5, 4, 3, 2, 1, 0))
        self.assertIsNone(subscriber_joint_positions(names[:-1], positions[:-1]))

    def test_logger_schemas_paths_and_numeric_formatting(self):
        self.assertEqual(
            SENSOR_LOG_SCHEMA.columns,
            (
                "timestamp", "force_x", "force_y", "force_z",
                "torque_x", "torque_y", "torque_z",
            ) + JOINT_NAMES,
        )
        self.assertEqual(TORQUE_LOG_SCHEMA.columns, SENSOR_LOG_SCHEMA.columns[:7])
        token = timestamp_token(datetime(2026, 8, 4, 2, 3, 5))
        self.assertEqual(token, "20260804_020305")
        self.assertEqual(log_filename(TORQUE_LOG_SCHEMA, token), "torque_data_20260804_020305.csv")
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "experiment_results/torque_data/torque_data_20260804_020305.csv"
            self.assertEqual(Path(log_path(directory, TORQUE_LOG_SCHEMA, token)), expected)
            self.assertFalse(expected.parent.exists())
        wrench = format_wrench_row(1.23456, [1, -2, 3.25], [4, 5.5, -6])
        self.assertEqual(wrench, ("1.2346", "1.000000", "-2.000000", "3.250000", "4.000000", "5.500000", "-6.000000"))
        sensor = format_sensor_row(0.0, [0, 0, 0], [0, 0, 0], range(7))
        self.assertEqual(len(sensor), 14)
        self.assertEqual(sensor[-7:], tuple("%0.6f" % value for value in range(7)))

    def test_logger_package_root_is_exact_in_source_and_catkin_install_space(self):
        source_module = LEGACY / "record_experiments.py"
        self.assertEqual(Path(package_root_for_module(str(source_module))), ROOT)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            installed_program = prefix / "lib/tiago_ring_controller/record_experiments.py"
            installed_module = (
                prefix / "lib/python3/dist-packages/record_experiments.py"
            )
            expected = prefix / "share/tiago_ring_controller"
            self.assertEqual(
                Path(package_root_for_module(str(installed_program))), expected
            )
            self.assertEqual(
                Path(package_root_for_module(str(installed_module))),
                expected,
            )

    def test_camera_resolution_preference_and_exact_type_filtering(self):
        published = [
            ("/z/image", "sensor_msgs/Image"),
            ("/xtion/rgb/image_raw", "sensor_msgs/Image"),
            (DEFAULT_RECORDING_CAMERA_TOPIC, "sensor_msgs/Image"),
            ("/not_image", "sensor_msgs/CompressedImage"),
        ]
        self.assertEqual(resolve_image_topic("/explicit", []), "/explicit")
        self.assertEqual(resolve_image_topic(None, published), DEFAULT_RECORDING_CAMERA_TOPIC)
        self.assertEqual(
            resolve_image_topic(None, [("/zrgb", "sensor_msgs/Image"), ("/argb", "sensor_msgs/Image")]),
            "/argb",
        )
        self.assertEqual(
            resolve_image_topic(None, [("/z", "sensor_msgs/Image"), ("/a", "sensor_msgs/Image")]),
            "/a",
        )
        with self.assertRaisesRegex(RuntimeError, "No sensor_msgs/Image"):
            resolve_image_topic(None, [("/compressed", "sensor_msgs/CompressedImage")])
        self.assertEqual(CAMERA_SUBSCRIPTION_CONTRACT.queue_size, 1)
        self.assertEqual(CAMERA_SUBSCRIPTION_CONTRACT.buffer_size, 2 ** 24)
        self.assertEqual(CAMERA_SUBSCRIPTION_CONTRACT.loop_hz, 60)


class _Duration:
    def __init__(self, seconds=0.0):
        self.seconds = seconds

    @classmethod
    def from_sec(cls, seconds):
        return cls(seconds)


class _JointTrajectory:
    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None, frame_id="")
        self.joint_names = []
        self.points = []


class _JointTrajectoryPoint:
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.time_from_start = None


class _PublishedMessages:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _module_with_attributes(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_tiago_controller_with_ros_stubs():
    fake_rospy = _module_with_attributes(
        "rospy",
        Time=types.SimpleNamespace(now=lambda: "now-token"),
        Duration=_Duration,
        logwarn=lambda *args, **kwargs: None,
        loginfo=lambda *args, **kwargs: None,
        is_shutdown=lambda: False,
        sleep=lambda _duration: None,
        Publisher=lambda *args, **kwargs: _PublishedMessages(),
        Subscriber=lambda *args, **kwargs: object(),
    )
    fake_actionlib = _module_with_attributes(
        "actionlib", SimpleActionClient=lambda *args, **kwargs: object()
    )
    modules = {
        "rospy": fake_rospy,
        "actionlib": fake_actionlib,
        "std_msgs": _module_with_attributes("std_msgs"),
        "std_msgs.msg": _module_with_attributes("std_msgs.msg", Float64=type("Float64", (), {})),
        "trajectory_msgs": _module_with_attributes("trajectory_msgs"),
        "trajectory_msgs.msg": _module_with_attributes(
            "trajectory_msgs.msg",
            JointTrajectory=_JointTrajectory,
            JointTrajectoryPoint=_JointTrajectoryPoint,
        ),
        "sensor_msgs": _module_with_attributes("sensor_msgs"),
        "sensor_msgs.msg": _module_with_attributes("sensor_msgs.msg", JointState=type("JointState", (), {})),
        "geometry_msgs": _module_with_attributes("geometry_msgs"),
        "geometry_msgs.msg": _module_with_attributes("geometry_msgs.msg", WrenchStamped=type("WrenchStamped", (), {})),
        "play_motion_msgs": _module_with_attributes("play_motion_msgs"),
        "play_motion_msgs.msg": _module_with_attributes(
            "play_motion_msgs.msg",
            PlayMotionAction=type("PlayMotionAction", (), {}),
            PlayMotionGoal=type("PlayMotionGoal", (), {}),
        ),
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "facade_tiago_controller_contract_test", LEGACY / "tiago_controller.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return module


def _load_analysis_with_dependency_stubs():
    modules = {
        "nest": _module_with_attributes("nest"),
        "rospy": _module_with_attributes("rospy"),
        "tiago_controller": _module_with_attributes(
            "tiago_controller",
            TiagoPublisher=type("TiagoPublisher", (), {}),
            TiagoSubscriber=type("TiagoSubscriber", (), {}),
        ),
        "single_ring": _module_with_attributes(
            "single_ring", SingleRingModel=type("SingleRingModel", (), {})
        ),
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "facade_analysis_single_joint_contract_test",
            LEGACY / "analysis_single_joint.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return module


def _load_neural_control_facade(filename):
    modules = {
        "nest": _module_with_attributes("nest"),
        "rospy": _module_with_attributes("rospy"),
        "tiago_controller": _module_with_attributes(
            "tiago_controller",
            TiagoPublisher=type("TiagoPublisher", (), {}),
            TiagoSubscriber=type("TiagoSubscriber", (), {}),
        ),
        "single_ring": _module_with_attributes(
            "single_ring", SingleRingModel=type("SingleRingModel", (), {})
        ),
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        module_name = "facade_%s_control_trace_test" % Path(filename).stem
        spec = importlib.util.spec_from_file_location(module_name, LEGACY / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return module


def _load_record_experiments_with_ros_stubs():
    fake_rospy = _module_with_attributes(
        "rospy",
        init_node=lambda *args, **kwargs: None,
        Subscriber=lambda *args, **kwargs: object(),
        get_time=lambda: 0.0,
        loginfo=lambda *args, **kwargs: None,
        logerr=lambda *args, **kwargs: None,
        on_shutdown=lambda *args, **kwargs: None,
        spin=lambda: None,
    )
    modules = {
        "rospy": fake_rospy,
        "geometry_msgs": _module_with_attributes("geometry_msgs"),
        "geometry_msgs.msg": _module_with_attributes(
            "geometry_msgs.msg", WrenchStamped=type("WrenchStamped", (), {})
        ),
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "facade_record_experiments_contract_test",
            LEGACY / "record_experiments.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return module


def _load_experiment_camera_with_ros_stubs():
    fake_rospy = _module_with_attributes(
        "rospy",
        get_published_topics=lambda: [],
        loginfo=lambda *args, **kwargs: None,
        Subscriber=lambda *args, **kwargs: object(),
        Rate=lambda _hz: types.SimpleNamespace(sleep=lambda: None),
        is_shutdown=lambda: False,
        init_node=lambda *args, **kwargs: None,
    )
    modules = {
        "rospy": fake_rospy,
        "sensor_msgs": _module_with_attributes("sensor_msgs"),
        "sensor_msgs.msg": _module_with_attributes(
            "sensor_msgs.msg", Image=type("Image", (), {})
        ),
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "facade_experiment_camera_contract_test",
            LEGACY / "experiment_camera.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return module


class RosFacadeEquivalenceTests(unittest.TestCase):
    def test_analysis_json_safe_recursion_redispatches_through_override(self):
        module = _load_analysis_with_dependency_stubs()

        class TrackingAnalysis(module.TiagoDecoderAnalysis):
            def _json_safe(self, obj):
                self.json_safe_calls.append(obj)
                return super()._json_safe(obj)

        analysis = object.__new__(TrackingAnalysis)
        analysis.json_safe_calls = []
        value = {"nested": (np.int64(3),)}
        self.assertEqual(analysis._json_safe(value), {"nested": [3]})
        self.assertEqual(len(analysis.json_safe_calls), 3)
        self.assertIs(analysis.json_safe_calls[0], value)

    def test_neural_facades_share_filter_delay_state_but_redispatch_decode(self):
        workflows = (
            (
                "analysis_single_joint.py",
                "TiagoDecoderAnalysis",
                "run_trial",
                "_decode_velocity_from_drive",
            ),
            (
                "calibrate_single_joint.py",
                "TiagoCalibration",
                "calibration_step",
                "_decode_velocity_from_drive",
            ),
            (
                "single_joint_data_collector.py",
                "SingleJointDataCollector",
                "run_trial",
                "_decode_velocity",
            ),
        )
        signed_spikes = [8, np.float64(-3.0), 5]
        dt_s = 0.05
        tau_s = 0.3
        delay_steps = 1
        alpha = np.exp(-dt_s / tau_s)
        filtered = 0.0
        delay_queue = [0.0] * delay_steps
        expected_delayed = []
        for signed_spike in signed_spikes:
            filtered = alpha * filtered + (1.0 - alpha) * signed_spike
            delay_queue.append(filtered)
            expected_delayed.append(delay_queue.pop(0))

        class HorizonCaptured(Exception):
            def __init__(self, velocities):
                super().__init__("horizon captured")
                self.velocities = list(velocities)

        for filename, class_name, trial_method_name, decoder_name in workflows:
            with self.subTest(filename=filename):
                module = _load_neural_control_facade(filename)
                facade = object.__new__(getattr(module, class_name))
                facade.time_step = dt_s * 1000.0
                facade.decoder_tau = tau_s
                facade.decoder_delay_steps = delay_steps
                facade.target_joint = 0
                facade.lookahead = len(signed_spikes)

                class Publisher:
                    current_positions = [0.0] * 7
                    current_velocities = [0.0] * 7

                    def prime_command_state(self, force=False):
                        return True

                    def publish_receding_trajectory(
                        self, joint_idx, velocities, dt
                    ):
                        self.call = (joint_idx, list(velocities), dt)
                        raise HorizonCaptured(velocities)

                facade.publisher = Publisher()
                facade.get_joint_position = lambda: [0.0] * 7
                facade.set_ring_goal = lambda goal: None
                facade.set_ring_state = lambda: None
                facade.get_ring_state = lambda: 0
                facade._compute_centroid = lambda: 0.0
                facade.left_count = 0
                facade.right_count = 0
                facade.state_count = 0.0
                spike_iter = iter(signed_spikes)
                facade.run_ring_simulation = lambda: next(spike_iter)
                decoder_inputs = []

                def decoder(delayed_drive):
                    decoder_inputs.append(delayed_drive)
                    return delayed_drive + 0.125

                setattr(facade, decoder_name, decoder)
                with self.assertRaises(HorizonCaptured) as raised:
                    getattr(facade, trial_method_name)(0.5)

                self.assertEqual(decoder_inputs, expected_delayed)
                self.assertEqual(
                    raised.exception.velocities,
                    [value + 0.125 for value in expected_delayed],
                )

    def test_publisher_constructor_preserves_all_ros_names_types_and_queues(self):
        module = _load_tiago_controller_with_ros_stubs()
        publishers = []
        subscribers = []
        action_clients = []

        def publisher(name, message_type, **kwargs):
            publishers.append((name, message_type, kwargs))
            return _PublishedMessages()

        def subscriber(name, message_type, callback, **kwargs):
            subscribers.append((name, message_type, callback, kwargs))
            return object()

        def action_client(name, action_type):
            marker = object()
            action_clients.append((name, action_type, marker))
            return marker

        module.rospy.Publisher = publisher
        module.rospy.Subscriber = subscriber
        module.actionlib.SimpleActionClient = action_client
        facade = module.TiagoPublisher()

        self.assertEqual(len(publishers), 8)
        self.assertEqual(
            (publishers[0][0], publishers[0][1], publishers[0][2]),
            ("/arm_controller/command", module.JointTrajectory, {"queue_size": 1}),
        )
        self.assertEqual(
            [call[0] for call in publishers[1:]], list(GRAVITY_COMPENSATION_TOPICS)
        )
        self.assertTrue(
            all(
                call[1] is module.Float64 and call[2] == {"queue_size": 1}
                for call in publishers[1:]
            )
        )
        self.assertEqual(
            [(call[0], call[1], call[3]) for call in subscribers],
            [("/joint_states", module.JointState, {"queue_size": 1})],
        )
        self.assertEqual(
            action_clients[0][:2], ("play_motion", module.PlayMotionAction)
        )
        self.assertIs(facade.play_motion_client, action_clients[0][2])

    def test_subscriber_constructor_retains_hidden_logging_side_effect_and_rows(self):
        module = _load_tiago_controller_with_ros_stubs()
        subscriber_calls = []
        clock = [10.0]
        module.rospy.Subscriber = lambda *args, **kwargs: (
            subscriber_calls.append((args, kwargs)) or object()
        )
        module.rospy.get_time = lambda: clock[0]

        with tempfile.TemporaryDirectory() as directory:
            module.package_root_for_module = lambda _module_file: directory
            facade = module.TiagoSubscriber()
            self.assertEqual(
                [(call[0][0], call[0][1], call[1]) for call in subscriber_calls],
                [
                    ("/wrist_ft", module.WrenchStamped, {"queue_size": 1}),
                    ("/joint_states", module.JointState, {"queue_size": 1}),
                ],
            )
            facade.joint_state_callback(
                types.SimpleNamespace(
                    name=list(JOINT_NAMES),
                    position=[float(index) for index in range(7)],
                )
            )
            clock[0] = 11.25
            facade.ft_callback(
                types.SimpleNamespace(
                    wrench=types.SimpleNamespace(
                        force=types.SimpleNamespace(x=1.0, y=2.0, z=3.0),
                        torque=types.SimpleNamespace(x=4.0, y=5.0, z=6.0),
                    )
                )
            )
            facade.shutdown()
            paths = list(
                Path(directory).glob("experiment_results/sensor_data/*.csv")
            )
            self.assertEqual(len(paths), 1)
            rows = paths[0].read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0].split(","), list(SENSOR_LOG_SCHEMA.columns))
            self.assertEqual(
                rows[1].split(",")[:7],
                [
                    "1.2500", "1.000000", "2.000000", "3.000000",
                    "4.000000", "5.000000", "6.000000",
                ],
            )
            self.assertEqual(
                rows[1].split(",")[-7:],
                ["%0.6f" % value for value in range(7)],
            )

    def test_torque_recorder_keeps_node_subscription_and_distinct_csv_schema(self):
        module = _load_record_experiments_with_ros_stubs()
        init_calls = []
        subscriber_calls = []
        clock = [20.0]
        module.rospy.init_node = lambda *args, **kwargs: init_calls.append(
            (args, kwargs)
        )
        module.rospy.Subscriber = lambda *args, **kwargs: (
            subscriber_calls.append((args, kwargs)) or object()
        )
        module.rospy.get_time = lambda: clock[0]
        with tempfile.TemporaryDirectory() as directory:
            module.package_root_for_module = lambda _module_file: directory
            facade = module.TorqueRecorder()
            self.assertEqual(
                init_calls, [(('torque_recorder',), {"anonymous": True})]
            )
            self.assertEqual(
                [(call[0][0], call[0][1], call[1]) for call in subscriber_calls],
                [("/wrist_ft", module.WrenchStamped, {})],
            )
            clock[0] = 20.5
            facade.ft_callback(
                types.SimpleNamespace(
                    wrench=types.SimpleNamespace(
                        force=types.SimpleNamespace(x=-1.0, y=-2.0, z=-3.0),
                        torque=types.SimpleNamespace(x=-4.0, y=-5.0, z=-6.0),
                    )
                )
            )
            facade.shutdown()
            paths = list(
                Path(directory).glob("experiment_results/torque_data/*.csv")
            )
            self.assertEqual(len(paths), 1)
            rows = paths[0].read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0].split(","), list(TORQUE_LOG_SCHEMA.columns))
            self.assertEqual(len(rows[1].split(",")), 7)

    def test_reset_pose_serializes_action_goal_and_timeout_contract(self):
        module = _load_tiago_controller_with_ros_stubs()

        class Client:
            def __init__(self):
                self.server_waits = []
                self.goals = []
                self.result_waits = []

            def wait_for_server(self, duration):
                self.server_waits.append(duration.seconds)
                return True

            def send_goal(self, goal):
                self.goals.append(goal)

            def wait_for_result(self, duration):
                self.result_waits.append(duration.seconds)

        publisher = object.__new__(module.TiagoPublisher)
        publisher.play_motion_client = Client()
        reset_calls = []
        publisher.reset_command_state = lambda: reset_calls.append(True)
        publisher.reset_pose(timeout=7.5)
        self.assertEqual(publisher.play_motion_client.server_waits, [5.0])
        self.assertEqual(publisher.play_motion_client.result_waits, [7.5])
        self.assertEqual(len(publisher.play_motion_client.goals), 1)
        goal = publisher.play_motion_client.goals[0]
        self.assertEqual(goal.motion_name, "tiago_experiment_start_1")
        self.assertFalse(goal.skip_planning)
        self.assertEqual(reset_calls, [True])

    def test_camera_callback_conversion_failure_run_contract_and_cli(self):
        module = _load_experiment_camera_with_ros_stubs()
        stream = object.__new__(module.CameraStream)
        stream.frame_count = 0
        stream.last_stamp = None
        stream.latest_frame = None

        class Bridge:
            def __init__(self):
                self.calls = []
                self.fail = False

            def imgmsg_to_cv2(self, message, desired_encoding):
                self.calls.append((message, desired_encoding))
                if self.fail:
                    raise RuntimeError("conversion failed")
                return "converted-frame"

        bridge = Bridge()
        stream.bridge = bridge
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(to_sec=lambda: 12.5)
            )
        )
        stream._image_callback(message)
        self.assertEqual(stream.frame_count, 1)
        self.assertEqual(stream.last_stamp, 12.5)
        self.assertEqual(stream.latest_frame, "converted-frame")
        self.assertEqual(bridge.calls, [(message, "bgr8")])
        bridge.fail = True
        stream._image_callback(message)
        self.assertEqual(stream.frame_count, 2)
        self.assertIsNone(stream.latest_frame)

        subscriber_calls = []

        class Subscription:
            def __init__(self):
                self.unregistered = False

            def unregister(self):
                self.unregistered = True

        subscription = Subscription()
        module.rospy.Subscriber = lambda *args, **kwargs: (
            subscriber_calls.append((args, kwargs)) or subscription
        )
        module.rospy.Rate = lambda hz: types.SimpleNamespace(
            hz=hz, sleep=lambda: None
        )
        module.rospy.is_shutdown = lambda: False
        stream.topic = "/explicit/image"
        stream.headless = True
        stream.window_name = "fixture"
        stream._cv2 = None
        stream.frame_count = 0
        times = iter([100.0, 100.2])
        original_time = module.time.time
        module.time.time = lambda: next(times)
        try:
            stream.run(duration=0.1, fps_log_interval=1.0)
        finally:
            module.time.time = original_time
        self.assertEqual(
            [(call[0][0], call[0][1], call[1]) for call in subscriber_calls],
            [
                (
                    "/explicit/image",
                    module.Image,
                    {"queue_size": 1, "buff_size": 2 ** 24},
                )
            ],
        )
        self.assertTrue(subscription.unregistered)

        previous_argv = sys.argv
        try:
            sys.argv = [
                "experiment_camera.py",
                "--topic", "/camera/test",
                "--headless",
                "--duration", "2.5",
                "--fps-log-interval", "0.25",
                "--window-name", "Window",
            ]
            args = module._parse_args()
        finally:
            sys.argv = previous_argv
        self.assertEqual(args.topic, "/camera/test")
        self.assertTrue(args.headless)
        self.assertEqual(args.duration, 2.5)
        self.assertEqual(args.fps_log_interval, 0.25)
        self.assertEqual(args.window_name, "Window")

    def test_joint_state_and_settle_timeouts_and_stop_horizon_are_unchanged(self):
        module = _load_tiago_controller_with_ros_stubs()

        class RosTimeValue:
            def __init__(self, seconds):
                self.seconds = float(seconds)

            def __add__(self, duration):
                return RosTimeValue(self.seconds + duration.seconds)

            def __gt__(self, other):
                return self.seconds > other.seconds

        clock = iter([RosTimeValue(0.0), RosTimeValue(0.2)])
        module.rospy.Time = types.SimpleNamespace(now=lambda: next(clock))
        sleeps = []
        module.rospy.sleep = lambda duration: sleeps.append(duration)
        publisher = object.__new__(module.TiagoPublisher)
        publisher.current_positions = None
        self.assertFalse(publisher.wait_for_joint_state(timeout=0.1))
        self.assertEqual(sleeps, [])

        clock = iter([RosTimeValue(1.0), RosTimeValue(1.2)])
        module.rospy.Time = types.SimpleNamespace(now=lambda: next(clock))
        publisher.current_velocities = [1.0] * 7
        publisher.wait_for_settled(
            joint_idx=2, vel_threshold=0.01, timeout=0.1, poll_dt=0.05
        )
        self.assertEqual(sleeps, [])

        stop_calls = []
        publisher.publish_receding_trajectory = lambda *args, **kwargs: (
            stop_calls.append((args, kwargs))
        )
        publisher.stop_joint(4, dt=0.07)
        self.assertEqual(stop_calls, [((4, [0.0, 0.0, 0.0], 0.07), {})])

    def test_tiago_publisher_serializes_the_pure_horizon_without_drift(self):
        module = _load_tiago_controller_with_ros_stubs()
        publisher = object.__new__(module.TiagoPublisher)
        publisher.joint_names = list(JOINT_NAMES)
        publisher.current_positions = [0.0] * 7
        publisher.current_velocities = [0.0] * 7
        publisher.commanded_positions = [0.0] * 7
        publisher.commanded_velocities = [0.0] * 7
        publisher.command_state_initialized = True
        publisher.pub = _PublishedMessages()
        publisher.prime_command_state = lambda force=False: True
        commanded_positions = publisher.commanded_positions

        expected = build_receding_trajectory([0.0] * 7, 3, [0.2, -0.1], 0.05)
        publisher.publish_receding_trajectory(3, [0.2, -0.1], 0.05)
        self.assertEqual(len(publisher.pub.messages), 1)
        message = publisher.pub.messages[0]
        self.assertEqual(message.header.stamp, "now-token")
        self.assertEqual(message.header.frame_id, "")
        self.assertEqual(message.joint_names, list(JOINT_NAMES))
        self.assertEqual([point.positions for point in message.points], [list(point.positions) for point in expected.points])
        self.assertEqual([point.velocities for point in message.points], [list(point.velocities) for point in expected.points])
        self.assertEqual([point.time_from_start.seconds for point in message.points], [point.time_from_start_s for point in expected.points])
        self.assertEqual(publisher.commanded_positions, list(expected.next_commanded_positions))
        self.assertEqual(publisher.commanded_velocities, list(expected.next_commanded_velocities))
        self.assertIs(publisher.commanded_positions, commanded_positions)

    def test_empty_facade_horizon_does_not_publish_or_advance(self):
        module = _load_tiago_controller_with_ros_stubs()
        publisher = object.__new__(module.TiagoPublisher)
        publisher.joint_names = list(JOINT_NAMES)
        publisher.current_positions = [0.0] * 7
        publisher.current_velocities = [0.0] * 7
        publisher.commanded_positions = [0.0] * 7
        publisher.commanded_velocities = [0.0] * 7
        publisher.command_state_initialized = True
        publisher.pub = _PublishedMessages()
        publisher.prime_command_state = lambda force=False: True
        publisher.publish_receding_trajectory(1, [], 0.05)
        self.assertEqual(publisher.pub.messages, [])
        self.assertEqual(publisher.commanded_positions, [0.0] * 7)

    def test_all_internal_modules_import_when_external_ros_and_nest_are_blocked(self):
        code = r'''
import importlib
import importlib.abc
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(src))

class BlockExternal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"nest", "rospy", "actionlib", "cv_bridge"}:
            raise ImportError("blocked external dependency: " + fullname)
        if fullname.startswith(("sensor_msgs", "geometry_msgs", "trajectory_msgs", "std_msgs", "play_motion_msgs")):
            raise ImportError("blocked external dependency: " + fullname)
        return None

sys.meta_path.insert(0, BlockExternal())
root = src / "tiago_ring_controller"
modules = ["tiago_ring_controller"]
for path in sorted(root.rglob("*.py")):
    if path.name == "__init__.py":
        relative = path.parent.relative_to(src)
    else:
        relative = path.with_suffix("").relative_to(src)
    name = ".".join(relative.parts)
    if name not in modules:
        modules.append(name)
for name in modules:
    importlib.import_module(name)
assert "nest" not in sys.modules
assert "rospy" not in sys.modules
print(len(modules))
'''
        completed = subprocess.run(
            [sys.executable, "-c", code, str(SRC)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertGreaterEqual(int(completed.stdout.strip()), 20)


if __name__ == "__main__":
    unittest.main()
