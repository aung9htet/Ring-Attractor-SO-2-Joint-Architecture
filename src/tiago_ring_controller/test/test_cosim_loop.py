"""Loop-level tests for the co-simulation layer: order, delay, lead, stops."""

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.control.controller import (  # noqa: E402
    DecoderParameters,
    DriveControlCore,
)
from tiago_ring_controller.control.profiles import (  # noqa: E402
    CALIBRATION_PROFILE,
    COLLECTOR_PROFILE,
)
from tiago_ring_controller.cosim import (  # noqa: E402
    AnyOf,
    DataPack,
    DataPackError,
    Engine,
    EngineError,
    EngineStateError,
    FTILoop,
    FakeNestEngine,
    FakeRobotEngine,
    GoalTF,
    MaxSteps,
    MotorTF,
    ProprioceptionTF,
    SettledFlag,
    TickContext,
    TickRecord,
    TransceiverFunction,
    TrialRecord,
)
from tiago_ring_controller.cosim.loop import PHASE_LEAD, PHASE_MAIN  # noqa: E402
from tiago_ring_controller.math.control import decode_velocity  # noqa: E402


DT_MS = 50.0
LIMITS = (-1.4172695167359413, 1.4171452891568528)


class RecordingEngine(Engine):
    """Engine that logs every call into a shared list."""

    def __init__(self, name, log, inputs=(), outputs=()):
        super().__init__(name)
        self.log = log
        self.inputs = frozenset(inputs)
        self.outputs = frozenset(outputs)
        self.received = None

    def _do_reset(self):
        self.received = None
        self.log.append(("reset", self.name))

    def _do_get_datapacks(self):
        self.log.append(("get", self.name, self.t_ms))
        return {
            name: DataPack(name, self.t_ms, {"step": self.step_index, "received": self.received})
            for name in self.outputs
        }

    def _do_set_datapacks(self, packs):
        self.log.append(("set", self.name, tuple(sorted(packs)), self.t_ms))
        self.received = {key: dict(pack.data) for key, pack in packs.items()}

    def _do_advance(self, dt_ms):
        self.log.append(("advance", self.name, dt_ms))


class ForwardTF(TransceiverFunction):
    name = "forward"

    def __init__(self, source, target, log):
        self.inputs = frozenset({source})
        self.outputs = frozenset({target})
        self.source = source
        self.target = target
        self.log = log

    def __call__(self, inputs, ctx):
        self.log.append(("tf", ctx.t_ms, ctx.phase))
        source = inputs.get(self.source)
        if source is None:
            return {}
        return {self.target: DataPack(self.target, ctx.t_ms, {"from_step": source["step"]})}


def make_motor_tf(profile=COLLECTOR_PROFILE, lead=4, decoder=None, thr=5.0, n_settle=10):
    return MotorTF(
        profile, 5, 100, DT_MS,
        decoder=decoder or DecoderParameters(tau_s=0.3, delay_steps=0),
        nest_lead_steps=lead, drive_threshold=thr, n_settle=n_settle,
    )


def make_loop(nest, robot, tfs, lead, max_steps=400):
    return FTILoop(
        [nest, robot], tfs, DT_MS, nest_lead_steps=lead,
        lead_engine="nest" if lead else None,
        stop_condition=AnyOf(SettledFlag(), MaxSteps(max_steps)),
    )


def make_tfs(lead=4, profile=COLLECTOR_PROFILE, goal=0.3, proprio="once", **motor):
    return [
        GoalTF(profile, LIMITS[0], LIMITS[1], 100, goal_rad=goal),
        ProprioceptionTF(profile, 5, LIMITS[0], LIMITS[1], 100, mode=proprio),
        make_motor_tf(profile, lead, **motor),
    ]


class DataPackTests(unittest.TestCase):
    def test_numpy_payloads_are_frozen_and_round_trip_through_json(self):
        pack = DataPack("x", np.float64(12.5), {"a": np.arange(3), "b": np.float32(1.5), "c": (1, 2)})
        self.assertEqual(pack.t_ms, 12.5)
        self.assertEqual(pack["a"], [0, 1, 2])
        self.assertEqual(pack["c"], [1, 2])
        self.assertEqual(DataPack.from_json(pack.to_json()), pack)
        self.assertEqual(pack.replace(t_ms=1.0, b=2.0).data["b"], 2.0)

    def test_non_json_payloads_are_rejected_early(self):
        with self.assertRaises(DataPackError):
            DataPack("x", 0.0, {"f": object()})
        with self.assertRaises(DataPackError):
            DataPack("", 0.0, {})


class EngineLifecycleTests(unittest.TestCase):
    def test_guards_on_initialization_inputs_and_outputs(self):
        log = []
        engine = RecordingEngine("a", log, inputs={"in"}, outputs={"out"})
        with self.assertRaises(EngineStateError):
            engine.advance(DT_MS)
        engine.initialize()
        engine.reset()
        with self.assertRaises(EngineError):
            engine.set_datapacks({"other": DataPack("other", 0.0, {})})
        with self.assertRaises(ValueError):
            engine.advance(0.0)
        engine.advance(DT_MS)
        self.assertEqual((engine.t_ms, engine.step_index), (DT_MS, 1))
        engine.reset()
        self.assertEqual((engine.t_ms, engine.step_index), (0.0, 0))

        class Leaky(RecordingEngine):
            def _do_get_datapacks(self):
                return {"undeclared": DataPack("undeclared", 0.0, {})}

        leaky = Leaky("b", log)
        leaky.initialize()
        with self.assertRaises(EngineStateError):
            leaky.get_datapacks()


class TickOrderTests(unittest.TestCase):
    def test_tick_is_collect_then_tf_then_set_then_advance_for_every_engine(self):
        log = []
        a = RecordingEngine("a", log, outputs={"a_out"})
        b = RecordingEngine("b", log, inputs={"b_in"}, outputs={"b_out"})
        loop = FTILoop([a, b], [ForwardTF("a_out", "b_in", log)], DT_MS)
        loop.reset()
        del log[:]
        loop.tick()
        self.assertEqual(
            log,
            [
                ("get", "a", 0.0),
                ("get", "b", 0.0),
                ("tf", 0.0, PHASE_MAIN),
                ("set", "b", ("b_in",), 0.0),
                ("advance", "a", DT_MS),
                ("advance", "b", DT_MS),
            ],
        )
        self.assertEqual((loop.t_ms, a.t_ms, b.t_ms), (DT_MS, DT_MS, DT_MS))

    def test_data_produced_at_t_is_consumed_during_t_to_t_plus_dt(self):
        log = []
        a = RecordingEngine("a", log, outputs={"a_out"})
        b = RecordingEngine("b", log, inputs={"b_in"}, outputs={"b_out"})
        loop = FTILoop([a, b], [ForwardTF("a_out", "b_in", log)], DT_MS)
        record = loop.run_trial(max_ticks=4)
        ticks = record.main_ticks
        for k, tick in enumerate(ticks):
            self.assertEqual(tick.inputs["a_out"]["step"], k)
            self.assertEqual(tick.outputs["b_in"]["from_step"], k)
            if k > 0:
                # B's state at t reflects the input it was given for [t-dt, t).
                self.assertEqual(tick.inputs["b_out"]["received"]["b_in"]["from_step"], k - 1)
        self.assertEqual(record.stop_reason, "max_ticks")
        self.assertEqual(record.end_state["a_out"]["step"], 4)

    def test_duplicate_outputs_and_missing_lead_engine_are_configuration_errors(self):
        log = []
        with self.assertRaises(ValueError):
            FTILoop([RecordingEngine("a", log, outputs={"x"}), RecordingEngine("b", log, outputs={"x"})], [], DT_MS)
        with self.assertRaises(ValueError):
            FTILoop([RecordingEngine("a", log, outputs={"x"})], [], DT_MS, nest_lead_steps=2)
        with self.assertRaises(ValueError):
            FTILoop([RecordingEngine("a", log, outputs={"x"})], [], DT_MS).run_trial()


class LeadAndHorizonTests(unittest.TestCase):
    def test_lead_phase_advances_only_nest_and_keeps_a_fixed_clock_offset(self):
        nest = FakeNestEngine("nest", population_size=100)
        robot = FakeRobotEngine("robot")
        loop = make_loop(nest, robot, make_tfs(lead=4), lead=4, max_steps=6)
        record = loop.run_trial()

        self.assertEqual([t.phase for t in record.ticks[:4]], [PHASE_LEAD] * 4)
        self.assertEqual([t.advanced for t in record.lead_ticks], [("nest",)] * 4)
        self.assertEqual([t.t_ms for t in record.lead_ticks], [0.0] * 4)
        self.assertTrue(all(t.advanced == ("nest", "robot") for t in record.main_ticks))
        self.assertEqual(nest.t_ms, robot.t_ms + 4 * DT_MS)
        self.assertEqual(loop.t_ms, robot.t_ms)
        # Bumps go in at the first lead sub-step, before NEST has run.
        self.assertEqual(
            [(b["name"], b["applied_at_step"]) for b in nest.applied_bumps],
            [("goal_bump", 0), ("state_bump", 0)],
        )
        # The first horizon carries four points from NEST steps 1..4.
        first = record.main_ticks[0].outputs["arm_velocity_cmd"]
        self.assertEqual(first["horizon_len"], 4)
        self.assertEqual(first["consumed"]["nest_step"], 1)
        self.assertEqual(record.main_ticks[1].outputs["arm_velocity_cmd"]["consumed"]["nest_step"], 2)
        self.assertEqual(len(robot.published[0]["velocities"]), 4)
        self.assertEqual(record.stop_reason, "max_steps")
        self.assertEqual(record.n_steps, 6)

    def test_pure_lock_step_has_one_tick_of_latency_and_one_point_horizons(self):
        nest = FakeNestEngine("nest", population_size=100)
        robot = FakeRobotEngine("robot")
        loop = make_loop(nest, robot, make_tfs(lead=0), lead=0, max_steps=3)
        record = loop.run_trial()
        self.assertEqual(record.lead_ticks, [])
        self.assertNotIn("arm_velocity_cmd", record.main_ticks[0].outputs)
        cmd = record.main_ticks[1].outputs["arm_velocity_cmd"]
        self.assertEqual(cmd["horizon_len"], 1)
        self.assertEqual(cmd["consumed"]["nest_step"], 1)
        self.assertEqual(nest.t_ms, robot.t_ms)
        self.assertEqual(len(robot.published), 2)


def legacy_loop(script, lead, decoder, thr, n_settle, max_steps, profile=COLLECTOR_PROFILE, population=100):
    """The collector's loop body, reduced to the arithmetic it performs."""

    dt_s = DT_MS / 1000.0
    samples = iter(script)
    core = DriveControlCore(dt_s, decoder, alpha=math.exp(-dt_s / decoder.tau_s))

    def run_ring_simulation():
        left, right = next(samples)
        return profile.signed_drive(right, left, population)

    def sample():
        sig = run_ring_simulation()
        _, filtered, delayed = core.advance(sig, decoder.delay_steps)
        return {"sig": sig, "filtered": filtered, "delayed": delayed,
                "vcmd": decode_velocity(delayed, decoder.gain_positive, decoder.gain_negative)}

    buffer = [sample() for _ in range(lead)]
    published = [[s["vcmd"] for s in buffer]]
    rows = []
    consecutive = 0
    stop_reason = "max_steps"
    for _ in range(max_steps):
        s = buffer[0]
        rows.append(dict(s))
        if abs(s["delayed"]) < thr:
            consecutive += 1
            if consecutive >= n_settle:
                stop_reason = "drive_settled"
                break
        else:
            consecutive = 0
        buffer = buffer[1:] + [sample()]
        published.append([s["vcmd"] for s in buffer])
    return published, rows, stop_reason


class LegacyParityTests(unittest.TestCase):
    SCRIPT = [(3, 40), (2, 35), (1, 30), (0, 28), (5, 20), (7, 12), (9, 9), (10, 8), (12, 7), (11, 6)] + [(3, 3)] * 40

    def run_cosim(self, script, lead, decoder, thr, n_settle, max_steps, profile=COLLECTOR_PROFILE):
        nest = FakeNestEngine("nest", population_size=100, script=script)
        robot = FakeRobotEngine("robot")
        tfs = make_tfs(lead=lead, profile=profile, decoder=decoder, thr=thr, n_settle=n_settle)
        loop = make_loop(nest, robot, tfs, lead=lead, max_steps=max_steps)
        return loop.run_trial(), robot

    def assert_parity(self, lead, decoder, thr, n_settle, max_steps, profile=COLLECTOR_PROFILE):
        published, rows, stop_reason = legacy_loop(
            self.SCRIPT, lead, decoder, thr, n_settle, max_steps, profile
        )
        record, robot = self.run_cosim(self.SCRIPT, lead, decoder, thr, n_settle, max_steps, profile)
        self.assertEqual(record.stop_reason, stop_reason)
        self.assertEqual(record.n_steps, len(rows))
        cosim_published = [p["velocities"] for p in robot.published]
        # Legacy publishes one horizon *after* its last NEST step even when the
        # step budget ends the trial; the loop's stop takes effect first.
        self.assertEqual(cosim_published, published[: len(cosim_published)])
        self.assertGreaterEqual(len(published), len(cosim_published))
        consumed = [t.outputs["arm_velocity_cmd"]["consumed"] for t in record.main_ticks]
        np.testing.assert_allclose([c["signed_spike"] for c in consumed], [r["sig"] for r in rows], rtol=0, atol=0)
        np.testing.assert_allclose([c["filtered_drive"] for c in consumed], [r["filtered"] for r in rows], rtol=0, atol=0)
        np.testing.assert_allclose([c["delayed_drive"] for c in consumed], [r["delayed"] for r in rows], rtol=0, atol=0)
        np.testing.assert_allclose([c["decoded_velocity"] for c in consumed], [r["vcmd"] for r in rows], rtol=0, atol=0)

    def test_collector_profile_settles_like_the_legacy_loop(self):
        self.assert_parity(4, DecoderParameters(tau_s=0.3, delay_steps=0), 5.0, 10, 400)

    def test_delayed_decoder_and_max_steps_budget(self):
        self.assert_parity(4, DecoderParameters(1e-3, -2e-3, tau_s=0.5, delay_steps=2), 0.5, 10, 8)

    def test_calibration_profile_spike_scale(self):
        self.assert_parity(4, DecoderParameters(tau_s=0.1, delay_steps=1), 5.0, 3, 400, CALIBRATION_PROFILE)

    def test_zero_lead_consumes_every_sample_once(self):
        published, rows, stop_reason = legacy_loop(
            self.SCRIPT, 1, DecoderParameters(tau_s=0.3), 5.0, 10, 400
        )
        record, robot = self.run_cosim(self.SCRIPT, 0, DecoderParameters(tau_s=0.3), 5.0, 10, 400)
        # lead=0 emits the same one-point horizons as lead=1, one tick later.
        self.assertEqual([p["velocities"] for p in robot.published], published[: len(robot.published)])
        self.assertEqual(record.stop_reason, stop_reason)
        self.assertEqual(record.n_steps, len(rows) + 1)


class StopConditionTests(unittest.TestCase):
    def _tick(self, tick, settled):
        cmd = DataPack("arm_velocity_cmd", 0.0, {"settled": settled})
        return TickRecord(tick, PHASE_MAIN, 0.0, {}, {"arm_velocity_cmd": cmd}, ("nest", "robot"))

    def test_settle_has_precedence_over_the_step_budget(self):
        condition = AnyOf(SettledFlag(), MaxSteps(3))
        self.assertIsNone(condition(self._tick(0, False)))
        self.assertEqual(condition(self._tick(2, True)), "drive_settled")
        self.assertEqual(condition(self._tick(2, False)), "max_steps")
        self.assertEqual(condition(TickRecord(2, PHASE_MAIN, 0.0, {}, {}, ())), "max_steps")
        with self.assertRaises(ValueError):
            MaxSteps(0)


class TransceiverModeTests(unittest.TestCase):
    def test_goal_and_proprioception_modes(self):
        state = {"joint_state": DataPack("joint_state", 0.0, {"positions": [0.0] * 5 + [0.2] + [0.0]})}
        ctx = TickContext(0.0, 0)
        goal = GoalTF(COLLECTOR_PROFILE, LIMITS[0], LIMITS[1], 100, goal_rad=0.5)
        self.assertEqual(set(goal(state, ctx)), {"goal_bump"})
        self.assertEqual(goal(state, ctx), {})
        goal.reset()
        self.assertEqual(goal(state, ctx)["goal_bump"]["center_index"], goal.ring_index(0.5))
        self.assertEqual(goal(state, ctx), {})

        once = ProprioceptionTF(COLLECTOR_PROFILE, 5, LIMITS[0], LIMITS[1], 100, mode="once")
        self.assertEqual(once(state, ctx)["state_bump"]["joint_position"], 0.2)
        self.assertEqual(once(state, ctx), {})
        continuous = ProprioceptionTF(COLLECTOR_PROFILE, 5, LIMITS[0], LIMITS[1], 100, mode="continuous")
        self.assertIn("state_bump", continuous(state, ctx))
        self.assertIn("state_bump", continuous(state, ctx))
        self.assertEqual(ProprioceptionTF(COLLECTOR_PROFILE, 5, *LIMITS, 100, mode="off")(state, ctx), {})
        self.assertEqual(continuous({}, ctx), {})
        with self.assertRaises(ValueError):
            ProprioceptionTF(COLLECTOR_PROFILE, 5, *LIMITS, 100, mode="sometimes")

    def test_analysis_profile_doubles_the_effective_half_width(self):
        from tiago_ring_controller.control.profiles import ANALYSIS_PROFILE

        goal = GoalTF(ANALYSIS_PROFILE, LIMITS[0], LIMITS[1], 100, goal_rad=0.5)
        pack = goal({}, TickContext(0.0, 0))["goal_bump"]
        self.assertEqual((pack["half_width"], pack["requested_half_width"]), (10, 5))

    def test_continuous_proprioception_reaches_the_fake_nest_every_tick(self):
        nest = FakeNestEngine("nest", population_size=100)
        robot = FakeRobotEngine("robot")
        loop = make_loop(nest, robot, make_tfs(lead=2, proprio="continuous"), lead=2, max_steps=3)
        loop.run_trial()
        names = [b["name"] for b in nest.applied_bumps]
        self.assertEqual(names.count("goal_bump"), 1)
        self.assertEqual(names.count("state_bump"), 2 + 3)


class DeterminismAndSerializationTests(unittest.TestCase):
    def run_once(self):
        nest = FakeNestEngine("nest", population_size=100)
        robot = FakeRobotEngine("robot")
        loop = make_loop(nest, robot, make_tfs(lead=4), lead=4, max_steps=20)
        return loop.run_trial()

    def test_two_runs_produce_identical_records(self):
        first = self.run_once().to_dict(include_timing=False)
        second = self.run_once().to_dict(include_timing=False)
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))

    def test_trial_record_json_round_trip_and_repeated_trials_reset_state(self):
        record = self.run_once()
        text = record.to_json()
        restored = TrialRecord.from_json(text)
        self.assertEqual(restored.to_dict(), record.to_dict())
        self.assertEqual(restored.main_ticks[0].outputs["arm_velocity_cmd"]["horizon_len"], 4)

        nest = FakeNestEngine("nest", population_size=100)
        robot = FakeRobotEngine("robot")
        loop = make_loop(nest, robot, make_tfs(lead=4), lead=4, max_steps=5)
        a = loop.run_trial().to_dict(include_timing=False)
        b = loop.run_trial().to_dict(include_timing=False)
        self.assertEqual(a, b)
        self.assertEqual((nest.reset_count, robot.reset_count), (2, 2))

    def test_fake_robot_moves_towards_the_goal(self):
        nest = FakeNestEngine("nest", population_size=100)
        robot = FakeRobotEngine("robot")
        loop = make_loop(nest, robot, make_tfs(lead=4, goal=0.8), lead=4, max_steps=60)
        record = loop.run_trial()
        start = record.main_ticks[0].inputs["joint_state"]["positions"][5]
        final = record.final_state["joint_state"]["positions"][5]
        self.assertGreater(final - start, 0.0)
        self.assertLess(abs(record.final_state["joint_state"]["velocities"][5]), 0.01)


if __name__ == "__main__":
    unittest.main()
