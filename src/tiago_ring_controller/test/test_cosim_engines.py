"""Engine, stepper, config, limits and runner tests with fake NEST and ROS."""

import ast
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support_fake_nest import RecordingNest  # noqa: E402
from tiago_ring_controller.config import load_joint_calibration  # noqa: E402
from tiago_ring_controller.control.profiles import CALIBRATION_PROFILE, COLLECTOR_PROFILE  # noqa: E402
from tiago_ring_controller.control.trajectory import TrajectoryPointData  # noqa: E402
from tiago_ring_controller.cosim import (  # noqa: E402
    CosimConfig,
    DataPack,
    EngineError,
    FakeNestEngine,
    FakeRobotEngine,
    NestEngine,
    RingModelPorts,
    StepTimeoutError,
    StimulusNotSupportedError,
    TrialRecord,
)
from tiago_ring_controller.cosim.gazebo_ros_engine import GazeboRosEngine, RosTransport  # noqa: E402
from tiago_ring_controller.cosim.limits import arm_joint_limits_from_urdf, joint_limits_from_urdf  # noqa: E402
from tiago_ring_controller.cosim.runner import (  # noqa: E402
    build_loop,
    legacy_collector_record,
    make_fake_engines,
    run_session,
    run_trial,
    trial_raster,
)
from tiago_ring_controller.cosim.stepping import ClockWaitStepper, PluginStepper  # noqa: E402
from tiago_ring_controller.cosim.tf import bump_datapack  # noqa: E402
from tiago_ring_controller.evaluation.serialization import (  # noqa: E402
    RING_RASTER_FIELDS,
    RING_TIMESERIES_FIELDS,
    RING_TRIAL_SCALAR_FIELDS,
)


CALIBRATION_PATH = SRC / "config/calibration/velocity_calibration.json"
COSIM_DIR = SRC / "tiago_ring_controller/cosim"


# ---------------------------------------------------------------------------
# Fake NEST that emits scripted spikes on every Simulate/Run
# ---------------------------------------------------------------------------
class ScriptedNest(RecordingNest):
    """``plan[step] = {"left": n, "right": n, "r1": {index: n}}`` spikes per step."""

    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self.sim_step = 0
        self.clock_ms = 0.0
        self.recorders = {}

    def _emit(self, duration):
        entry = self.plan.get(self.sim_step, {})
        self.sim_step += 1
        self.clock_ms += duration
        for group in ("left", "right"):
            recorders = self.recorders.get(group, [])
            for _ in range(int(entry.get(group, 0))):
                recorder = recorders[self.sim_step % max(len(recorders), 1)]
                recorder.events.setdefault("times", []).append(self.clock_ms)
                recorder.events.setdefault("senders", []).append(recorder.ids[0] - 1)
        for index, count in entry.get("r1", {}).items():
            recorder = self.recorders["r1"][index]
            for _ in range(int(count)):
                recorder.events.setdefault("times", []).append(self.clock_ms)
                recorder.events.setdefault("senders", []).append(recorder.ids[0] - 1)

    def Simulate(self, duration):
        super().Simulate(duration)
        self._emit(duration)

    def Run(self, duration):
        super().Run(duration)
        self._emit(duration)


def scripted_model_factory(backend, population=8, hidden_ms=50.0):
    def factory():
        groups = {}
        for group in ("r1", "r2", "left", "right"):
            recorders = []
            for _ in range(population):
                neuron = backend.Create("iaf_psc_alpha", params={"V_th": 1.0})
                recorder = backend.Create("spike_recorder")
                backend.Connect(neuron, recorder)
                recorders.append(recorder)
            groups[group] = recorders
        backend.recorders = groups

        def inject(center, half_width):
            # Same shape as nest.ring.inject_stimulus: create, connect, Simulate.
            generator = backend.Create("poisson_generator", params={"rate": 200.0})
            backend.Simulate(hidden_ms)
            backend.SetStatus(generator, {"rate": 0.0})
            backend.injected.append((center, half_width))

        backend.injected = []
        return RingModelPorts(
            population_size=population,
            r1_recorders=groups["r1"], r2_recorders=groups["r2"],
            left_recorders=groups["left"], right_recorders=groups["right"],
            inject_state=inject, inject_goal=inject,
        )

    return factory


PLAN = {
    0: {"left": 9, "right": 9, "r1": {0: 5}},        # hidden: goal injection
    1: {"left": 9, "right": 9, "r1": {1: 5}},        # hidden: state injection
    2: {"left": 1, "right": 4, "r1": {2: 3, 3: 1}},  # tick 1
    3: {"left": 2, "right": 0, "r1": {}},            # tick 2
    4: {"left": 0, "right": 7, "r1": {5: 2}},        # tick 3
}


class NestEngineTests(unittest.TestCase):
    def make_engine(self, step_mode="run", backend=None):
        backend = backend or ScriptedNest(PLAN)
        engine = NestEngine(backend, scripted_model_factory(backend), step_mode=step_mode)
        engine.initialize()
        engine.reset()
        engine.set_datapacks({
            "goal_bump": bump_datapack("goal_bump", 0.0, 6, 1, "goal"),
            "state_bump": bump_datapack("state_bump", 0.0, 1, 1, "proprioception"),
        })
        return engine, backend

    def test_run_mode_prepares_once_and_excludes_hidden_stimulus_spikes(self):
        engine, backend = self.make_engine("run")
        self.assertEqual(engine.get_datapacks(), {})
        engine.advance(50.0)
        counts = engine.get_datapacks()["ring_counts"]
        self.assertEqual((counts["left"], counts["right"]), (1, 4))
        self.assertEqual(counts["r1_delta"], [0, 0, 3, 1, 0, 0, 0, 0])
        self.assertEqual((counts["r1_spike_count"], counts["r1_bump_index"]), (4.0, 2))
        self.assertAlmostEqual(counts["r1_centroid"], (2 * 3 + 3 * 1) / 4.0)
        self.assertEqual((counts["nest_step"], counts["t_nest_ms"], counts["hidden_ms"]), (1, 50.0, 100.0))
        self.assertEqual(counts["readout_mode"], "node_collection")
        engine.advance(50.0)
        second = engine.get_datapacks()["ring_counts"]
        self.assertEqual((second["left"], second["right"], second["r1_bump_index"]), (2, 0, None))
        self.assertTrue(np.isnan(second["r1_centroid"]))
        engine.finish_trial()
        self.assertEqual(backend.injected, [(6, 1), (1, 1)])
        self.assertEqual(backend.simulate_calls, [50.0, 50.0])
        self.assertEqual(backend.lifecycle_calls, ["Prepare", ("Run", 50.0), ("Run", 50.0), "Cleanup"])

    def test_simulate_mode_matches_the_legacy_call_sequence(self):
        engine, backend = self.make_engine("simulate")
        engine.advance(50.0)
        engine.advance(50.0)
        self.assertEqual(backend.simulate_calls, [50.0, 50.0, 50.0, 50.0])
        self.assertEqual(backend.lifecycle_calls, [])
        left_right = [(p["left"], p["right"]) for p in [engine.get_datapacks()["ring_counts"]]]
        self.assertEqual(left_right, [(2, 0)])

    def test_mid_trial_bumps_are_rejected_by_the_legacy_stimulus_port(self):
        engine, backend = self.make_engine("run")
        engine.advance(50.0)
        engine.set_datapacks({"state_bump": bump_datapack("state_bump", 50.0, 3, 1, "proprioception")})
        with self.assertRaises(StimulusNotSupportedError):
            engine.advance(50.0)
        self.assertFalse(engine.stimulus_port.supports_mid_trial)

    def test_per_recorder_fallback_when_node_collections_are_unavailable(self):
        class NoCollections(ScriptedNest):
            NodeCollection = None

        engine, backend = self.make_engine("run", backend=NoCollections(PLAN))
        engine.advance(50.0)
        counts = engine.get_datapacks()["ring_counts"]
        self.assertEqual(counts["readout_mode"], "events")
        self.assertEqual((counts["left"], counts["right"]), (1, 4))

    def test_reset_rebuilds_and_raster_has_the_collector_schema(self):
        engine, backend = self.make_engine("run")
        engine.advance(50.0)
        raster = engine.raster()
        self.assertEqual(tuple(raster), RING_RASTER_FIELDS)
        self.assertEqual(len(raster["r1_times"]), 5 + 5 + 4)
        self.assertEqual(raster["r1_senders"].dtype.kind, "i")
        engine.reset()
        self.assertEqual(backend.lifecycle_calls[-1], "Cleanup")
        self.assertEqual(engine.get_datapacks(), {})
        self.assertEqual(engine.step_index, 0)
        self.assertEqual(engine.population_size, 8)
        self.assertIsNone(NestEngine(backend, scripted_model_factory(backend)).population_size)
        self.assertEqual(NestEngine(backend, scripted_model_factory(backend), population_size=200).population_size, 200)


# ---------------------------------------------------------------------------
# Fake ROS transport for the Gazebo engine
# ---------------------------------------------------------------------------
class FakeTransport(RosTransport):
    def __init__(self, overshoot_s=0.002, play_motion_state="SUCCEEDED", clock_stalls=False):
        self.overshoot_s = overshoot_s
        self.play_motion_state = play_motion_state
        self.clock_stalls = clock_stalls
        self.sim_time = 10.0
        self.paused = False
        self.positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.velocities = [0.0] * 7
        self.calls = []
        self.published = []
        self.params = {}
        self.initialized = False

    def initialize(self):
        self.initialized = True
        self.calls.append("initialize")

    def shutdown(self):
        self.calls.append("shutdown")

    def latest_joint_state(self):
        return tuple(self.positions), tuple(self.velocities), self.sim_time

    def sim_time_s(self):
        return self.sim_time

    def wait_until_sim_time(self, target_s, timeout_s):
        self.calls.append(("wait", round(target_s, 6)))
        if self.paused or self.clock_stalls:
            return False
        self.sim_time = target_s + self.overshoot_s
        if self.published:
            joint_names, points, _ = self.published[-1]
            first = points[0]
            self.velocities = [(p - q) / 0.05 for p, q in zip(first.positions, self.positions)]
            self.positions = list(first.positions)
        return True

    def pause(self):
        self.paused = True
        self.calls.append("pause")

    def unpause(self):
        self.paused = False
        self.calls.append("unpause")

    def publish_trajectory(self, joint_names, points, stamp_s):
        self.calls.append(("publish", stamp_s))
        self.published.append((tuple(joint_names), tuple(points), stamp_s))

    def play_motion(self, motion_name, timeout_s):
        self.calls.append(("play_motion", motion_name))
        self.velocities = [0.0] * 7
        return self.play_motion_state

    def get_param(self, name):
        return self.params.get(name)


class GazeboEngineTests(unittest.TestCase):
    def make_engine(self, **kwargs):
        transport = FakeTransport(**kwargs)
        engine = GazeboRosEngine(transport, settle_timeout_s=0.2, step_timeout_s=0.05)
        engine.initialize()
        return engine, transport

    def test_reset_runs_unpaused_checks_play_motion_and_pauses(self):
        engine, transport = self.make_engine()
        engine.reset()
        self.assertEqual(transport.calls[:3], ["initialize", "unpause", ("play_motion", "tiago_experiment_start_1")])
        self.assertEqual(transport.calls[-1], "pause")
        self.assertTrue(engine.paused)
        self.assertEqual(engine.command_state.commanded_positions, transport.positions)

        engine, transport = self.make_engine(play_motion_state="ABORTED")
        with self.assertRaises(EngineError):
            engine.reset()

    def test_advance_publishes_stamped_horizon_then_steps_exactly_dt(self):
        engine, transport = self.make_engine(overshoot_s=0.002)
        engine.reset()
        del transport.calls[:]
        before = transport.sim_time
        engine.set_datapacks({
            "arm_velocity_cmd": DataPack("arm_velocity_cmd", 0.0, {
                "joint_index": 5, "velocities": [0.1, 0.2, 0.3, 0.4], "dt_s": 0.05,
            })
        })
        engine.advance(50.0)
        self.assertEqual(transport.calls, [("publish", before), "unpause", ("wait", round(before + 0.05, 6)), "pause"])
        joint_names, points, stamp = transport.published[0]
        self.assertEqual(len(joint_names), 7)
        np.testing.assert_allclose([p.time_from_start_s for p in points], [0.05, 0.1, 0.15, 0.2])
        self.assertAlmostEqual(points[0].positions[5], 0.6 + 0.1 * 0.05)
        self.assertAlmostEqual(engine.command_state.commanded_positions[5], 0.6 + 0.1 * 0.05)
        self.assertAlmostEqual(transport.sim_time - before, 0.052)
        packs = engine.get_datapacks()
        self.assertAlmostEqual(packs["sim_clock"]["overshoot_ms"], 2.0, places=5)
        self.assertEqual(packs["sim_clock"]["stepper"], "clock_wait")
        self.assertAlmostEqual(packs["joint_state"]["positions"][5], 0.6 + 0.1 * 0.05)
        self.assertEqual(engine.t_ms, 50.0)
        # Advancing without a command holds the commanded state, still lock-stepped.
        engine.advance(50.0)
        self.assertEqual(len(transport.published), 1)
        self.assertAlmostEqual(transport.sim_time - before, 0.104)

    def test_step_timeout_leaves_physics_paused_and_raises(self):
        engine, transport = self.make_engine(clock_stalls=True)
        engine.reset()
        with self.assertRaises(StepTimeoutError):
            engine.advance(50.0)
        self.assertTrue(transport.paused)
        # dt must be a whole number of physics iterations.
        engine_bad = GazeboRosEngine(FakeTransport(), max_step_size_s=0.003)
        engine_bad.initialize()
        engine_bad.reset()
        with self.assertRaises(EngineError):
            engine_bad.advance(50.0)

    def test_finish_trial_stops_and_settles_under_stepping(self):
        engine, transport = self.make_engine()
        engine.reset()
        engine.set_datapacks({
            "arm_velocity_cmd": DataPack("arm_velocity_cmd", 0.0, {"joint_index": 5, "velocities": [0.5], "dt_s": 0.05})
        })
        engine.advance(50.0)
        engine.finish_trial()
        names, points, _ = transport.published[-1]
        self.assertEqual([p.velocities[5] for p in points], [0.0, 0.0, 0.0])
        self.assertLess(abs(transport.velocities[5]), 0.01)
        self.assertTrue(transport.paused)
        engine.shutdown()
        self.assertEqual(transport.calls[-2:], ["unpause", "shutdown"])

    def test_plugin_stepper_is_a_hook_until_the_world_plugin_exists(self):
        transport = FakeTransport()
        with self.assertRaises(EngineError):
            PluginStepper(transport).step(50)
        stepper = ClockWaitStepper(transport, timeout_s=0.01)
        with self.assertRaises(ValueError):
            stepper.step(0)

    def test_cosim_sources_never_block_on_ros_time(self):
        # Acceptance A4: no rospy.sleep / rospy.Rate anywhere under cosim/.
        # Inspect the AST so docstrings that *mention* them do not count.
        def offenders(path):
            tree = ast.parse(path.read_bytes(), filename=str(path))
            found = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute) or node.attr not in ("sleep", "Rate"):
                    continue
                owner = node.value
                if isinstance(owner, ast.Name) and owner.id == "rospy":
                    found.append(node.lineno)
                if isinstance(owner, ast.Attribute) and owner.attr == "_rospy":
                    found.append(node.lineno)
            return found

        for path in sorted(COSIM_DIR.glob("*.py")):
            with self.subTest(path=path.name):
                self.assertEqual(offenders(path), [])


class LimitsAndConfigTests(unittest.TestCase):
    URDF = """<robot name="tiago">
      <joint name="arm_6_joint" type="revolute"><limit lower="-1.39" upper="1.39" effort="1" velocity="1"/></joint>
      <joint name="arm_7_joint" type="revolute"><limit lower="-2.07" upper="2.07" effort="1" velocity="1"/></joint>
      <joint name="fixed" type="fixed"/>
    </robot>"""

    def test_urdf_limits(self):
        self.assertEqual(joint_limits_from_urdf(self.URDF, "arm_6_joint"), (-1.39, 1.39))
        self.assertIsNone(joint_limits_from_urdf(self.URDF, "fixed"))
        self.assertIsNone(joint_limits_from_urdf(self.URDF, "missing"))
        self.assertEqual(arm_joint_limits_from_urdf(self.URDF), {5: (-1.39, 1.39), 6: (-2.07, 2.07)})

    def test_config_from_profile_calibration_and_round_trip(self):
        calibration = load_joint_calibration(str(CALIBRATION_PATH), 5)
        config = CosimConfig.from_profile(COLLECTOR_PROFILE, 5, calibration, rng_seed=7)
        self.assertEqual((config.dt_ms, config.nest_lead_steps, config.max_steps), (50.0, 4, 400))
        self.assertEqual(config.joint_min, calibration.joint_min)
        self.assertEqual(config.decoder_gain_positive, 0.0011)
        self.assertEqual(config.decoder.tau_s, calibration.decoder.tau)
        self.assertEqual(config.physics_iterations_per_tick, 50)
        self.assertEqual(CosimConfig.from_json(config.to_json()), config)
        self.assertEqual(CosimConfig.from_profile(CALIBRATION_PROFILE, 3).control_profile, CALIBRATION_PROFILE)
        with self.assertRaises(ValueError):
            CosimConfig(nest_step_mode="tick")
        with self.assertRaises(ValueError):
            CosimConfig(joint_min=0.0)
        with self.assertRaises(ValueError):
            CosimConfig().require_limits()
        with self.assertRaises(ValueError):
            CosimConfig(dt_ms=0.0)
        with self.assertRaises(ValueError):
            CosimConfig(dt_ms=0.0015 * 1000, max_step_size_s=0.001).physics_iterations_per_tick


class RunnerTests(unittest.TestCase):
    def make_config(self, **overrides):
        calibration = load_joint_calibration(str(CALIBRATION_PATH), 5)
        return CosimConfig.from_profile(COLLECTOR_PROFILE, 5, calibration, max_steps=30, **overrides)

    def test_run_trial_with_fakes_produces_the_collector_record_schema(self):
        config = self.make_config()
        engines = make_fake_engines(config)
        loop = build_loop(config, engines)
        record = run_trial(loop, 0.6, config)
        self.assertEqual(record.meta["goal_rad"], 0.6)
        self.assertEqual(record.meta["config"], config.to_dict())
        self.assertIsInstance(record.meta["goal_ring_index"], int)
        self.assertIsInstance(record.meta["initial_ring_index"], int)

        legacy = legacy_collector_record(record, config, trial_raster(loop))
        expected_scalars = tuple(f for f in RING_TRIAL_SCALAR_FIELDS if f not in ("joint_index", "batch_idx", "iteration_idx"))
        self.assertEqual(tuple(legacy["scalars"]), expected_scalars)
        self.assertEqual(tuple(legacy["timeseries"]), RING_TIMESERIES_FIELDS)
        self.assertEqual(tuple(legacy["raster"]), RING_RASTER_FIELDS)
        n = legacy["scalars"]["n_steps"]
        self.assertEqual(n, record.n_steps)
        self.assertEqual(len(legacy["timeseries"]["time"]), n)
        np.testing.assert_allclose(legacy["timeseries"]["time"], np.arange(1, n + 1) * 0.05)
        self.assertEqual(legacy["scalars"]["q_goal"], 0.6)
        self.assertEqual(legacy["scalars"]["goal_ring_index"], record.meta["goal_ring_index"])
        self.assertIn(legacy["scalars"]["stop_reason"], ("drive_settled", "max_steps"))
        # Row k pairs the consumed sample of tick k with the state after that tick.
        first_state = record.main_ticks[1].inputs["joint_state"]["positions"][5]
        self.assertEqual(legacy["timeseries"]["joint_position"][0], first_state)
        self.assertEqual(legacy["timeseries"]["r1_bump_index"].dtype, np.float64)
        self.assertEqual(legacy["timeseries"]["position_error"][0], 0.6 - first_state)

    def test_run_session_writes_the_collector_layout(self):
        config = self.make_config()
        loop = build_loop(config, make_fake_engines(config))
        with tempfile.TemporaryDirectory() as tmp:
            results = run_session(loop, config, [0.4, -0.2], out_dir=tmp)
            self.assertEqual(len(results), 2)
            with open(os.path.join(tmp, "trials_summary.csv"), newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(tuple(rows[0]), RING_TRIAL_SCALAR_FIELDS)
            self.assertEqual([r["iteration_idx"] for r in rows], ["1", "2"])
            timeseries = np.load(os.path.join(tmp, "trials", "trial_0001_timeseries.npz"))
            self.assertEqual(tuple(timeseries.files), RING_TIMESERIES_FIELDS)
            raster = np.load(os.path.join(tmp, "trials", "trial_0002_raster.npz"))
            self.assertEqual(tuple(raster.files), RING_RASTER_FIELDS)
            with open(os.path.join(tmp, "trials", "trial_0001_cosim.json"), encoding="utf-8") as stream:
                restored = TrialRecord.from_json(stream.read())
            self.assertEqual(restored.to_dict(), results[0]["record"].to_dict())
            self.assertTrue(os.path.exists(os.path.join(tmp, "session_meta.json")))

    def test_sessions_are_deterministic_across_loops(self):
        config = self.make_config()
        a = run_session(build_loop(config, make_fake_engines(config)), config, [0.4, -0.2])
        b = run_session(build_loop(config, make_fake_engines(config)), config, [0.4, -0.2])
        self.assertEqual(
            [r["record"].to_dict(include_timing=False) for r in a],
            [r["record"].to_dict(include_timing=False) for r in b],
        )
        self.assertEqual([r["row"] for r in a], [r["row"] for r in b])


if __name__ == "__main__":
    unittest.main()
