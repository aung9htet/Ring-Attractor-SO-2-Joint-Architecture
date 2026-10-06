"""Architecture v4 on the fakes: the reference projection, the three-ring template, disturbances.

docs/architecture.md, docs/v4-tuning.md.  The real-NEST gate is
``test_feedback_v4_real_nest.py``.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from support_fake_nest import RecordingNest, connection_rows, label_nodes, relabel_rows  # noqa: E402
from tiago_ring_controller.blocks import BlockError, BuildContext, Gain, Homeostasis, FourierReadout, Ring  # noqa: E402
from tiago_ring_controller.cosim.disturbance import Disturbance  # noqa: E402
from tiago_ring_controller.cosim.fakes import FakeRobotEngine  # noqa: E402
from tiago_ring_controller.graph import Graph, compile_graph, run_graph_trial, three_ring_single_joint, two_ring_single_joint  # noqa: E402
from tiago_ring_controller.graph.calibration import network_fingerprint, transport_motif  # noqa: E402
from tiago_ring_controller.graph.schema import dumps, loads  # noqa: E402
from tiago_ring_controller.graph.templates import FEEDBACK, feedback_settings  # noqa: E402
from tiago_ring_controller.nest.populations import BuildError, build_gain_populations, connect_gain_reference, create_population  # noqa: E402


class ReferenceProjectionTests(unittest.TestCase):
    def rows(self, size=12, window=3, exc=2.0, inh=5.0):
        backend = RecordingNest()
        gain = build_gain_populations(backend, "gain", size, {})
        reference = create_population(backend, "ref", size)
        before = len(backend.connect_calls)
        count = connect_gain_reference(backend, gain, reference, exc, inh, window)
        labels = {}
        for name, nodes in (("ref", reference.neurons), ("left", gain.left.neurons), ("right", gain.right.neurons)):
            label_nodes(labels, name, nodes)
        return count, relabel_rows(connection_rows(backend, backend.connect_calls[before:]), labels)

    def test_pattern_is_the_shifted_window_without_wrap(self):
        size, window, exc, inh = 12, 3, 2.0, 5.0
        count, rows = self.rows(size, window, exc, inh)
        expected = []
        for i in range(size):
            for j in range(size):
                if 1 <= j - i <= window:        # reference neuron above gain neuron i: move up
                    expected += [(("ref", j), ("right", i), exc), (("ref", j), ("left", i), -inh)]
                if 1 <= i - j <= window:        # reference neuron below: move down
                    expected += [(("ref", j), ("left", i), exc), (("ref", j), ("right", i), -inh)]
        self.assertEqual(rows, sorted(expected))
        self.assertEqual(count, len(expected) // 2)

    def test_inhibition_sign_and_bad_arguments(self):
        _, rows = self.rows(exc=1.0, inh=-4.0)          # the sign of inh_weight does not matter
        self.assertEqual({w for _, _, w in rows}, {1.0, -4.0})
        backend = RecordingNest()
        gain = build_gain_populations(backend, "gain", 10, {})
        with self.assertRaisesRegex(BuildError, "reference ring ref has 8 neurons"):
            connect_gain_reference(backend, gain, create_population(backend, "ref", 8), 1.0, 1.0, 2)
        with self.assertRaisesRegex(BuildError, "window must be in 1..9"):
            connect_gain_reference(backend, gain, create_population(backend, "ref", 10), 1.0, 1.0, 10)

    def test_gain_reference_port_is_optional_and_edge_parameters_are_checked(self):
        def graph(reference, **edge):
            g = Graph("ref")
            r1, r2 = g.add(Ring("r1")), g.add(Ring("r2"))
            f1, f2 = g.add(FourierReadout("f1")), g.add(FourierReadout("f2"))
            cmp, gain = g.add(Homeostasis("cmp")), g.add(Gain("gain"))
            for a, b in ((r1.spikes, f1.ring), (r2.spikes, f2.ring), (f1.features, cmp.state_features), (f2.features, cmp.target_features),
                         (cmp.left, gain.left_in), (cmp.right, gain.right_in), (r1.spikes, gain.ring), (gain.feedback, r1.stim)):
                g.connect(a, b)
            if reference:
                g.connect(r2.spikes, gain.reference, **edge)
            return g

        plain, sharp = graph(False), graph(True, exc_weight=1.0e4, inh_weight=3.0e4, window=4)
        self.assertEqual(plain.problems(), [])
        backend_plain, backend_sharp = RecordingNest(), RecordingNest()
        plain.build(BuildContext(backend_plain))
        sharp.build(BuildContext(backend_sharp))
        self.assertEqual(plain.blocks["gain"].references, {})
        self.assertEqual(sharp.blocks["gain"].references, {"r2": 2 * sum(200 - s for s in range(1, 5))})
        self.assertEqual(len(backend_sharp.connect_calls) - len(backend_plain.connect_calls), 4 * 4)   # four one_to_one calls per shift
        self.assertIn("unknown edge parameters ['strength']", " ".join(graph(True, strength=1.0).problems()))
        # an edge survives the file round trip with its parameters
        again = loads(dumps(sharp))
        edge = [e for e in again.edges if e.target.port.name == "reference"][0]
        self.assertEqual(edge.params, {"exc_weight": 1.0e4, "inh_weight": 3.0e4, "window": 4})


class ThreeRingTemplateTests(unittest.TestCase):
    def test_wiring_is_two_gain_pairs_on_the_predictive_ring(self):
        g = three_ring_single_joint(joint=5)
        self.assertEqual(g.problems(), [])
        motif = transport_motif(g)
        # the decoder reads the goal pair, which gates on the predictive ring and compares it with the goal ring
        self.assertEqual((motif["gain"].id, motif["state_ring"].id, motif["goal_ring"].id), ("jt__ct_goal__gain", "jt__P", "jt__G"))
        into_p = sorted(e.source.block.id for e in g.edges_into(g.blocks["jt__P"], "stim"))
        self.assertEqual(into_p, ["jt__ct_body__gain", "jt__ct_goal__gain", "jt__enc_pred"])
        references = {e.target.block.id: (e.source.block.id, e.params) for e in g.edges if e.target.port.name == "reference"}
        self.assertEqual(references["jt__ct_goal__gain"], ("jt__G", FEEDBACK["goal_reference"]))
        self.assertEqual(references["jt__ct_body__gain"], ("jt__S", FEEDBACK["body_reference"]))
        body_cmp = g.blocks["jt__ct_body__cmp"]
        self.assertEqual(sorted(e.source.block.id for e in g.edges_into(body_cmp)), ["jt__fP", "jt__fS"])
        dec, sim = g.blocks["dec"], g.simulation
        self.assertEqual((dec.params["dead_zone"], dec.params["horizon"], sim.nest_lead_steps, sim.time_scale),
                         (FEEDBACK["dead_zone"] * FEEDBACK["time_scale"], 1, 1, FEEDBACK["time_scale"]))
        rho = g.blocks["jt__enc_state"].radians_per_index(200)
        self.assertAlmostEqual(dec.params["gain_positive"], FEEDBACK["motor_gain"] * rho / 0.05)
        self.assertEqual(dec.params["drive_threshold"], FEEDBACK["drive_threshold"] * sim.time_scale)
        for ring in ("jt__G", "jt__P", "jt__S"):
            self.assertEqual(g.blocks[ring].params["wall"], [15, 185])

    def test_settings_override_and_unknown_keys_fail(self):
        g = three_ring_single_joint(feedback={"body_reference": {"inh_weight": 9.0e4}, "goal_mode": "once"}, time_scale=1)
        edge = [e for e in g.edges if e.target.block.id == "jt__ct_body__gain" and e.target.port.name == "reference"][0]
        self.assertEqual(edge.params, {"exc_weight": 2.0e4, "inh_weight": 9.0e4, "window": 10})
        self.assertEqual((g.blocks["jt__enc_goal"].params["mode"], g.simulation.time_scale), ("once", 1))
        self.assertEqual(feedback_settings()["body_reference"], FEEDBACK["body_reference"])
        with self.assertRaisesRegex(ValueError, "unknown v4 setting 'body_pull'"):
            three_ring_single_joint(feedback={"body_pull": 1.0})

    def test_v3_fingerprint_does_not_know_about_the_reference_port(self):
        from tiago_ring_controller.graph import load_transport_calibration

        self.assertEqual(load_transport_calibration()["network_fingerprint"], network_fingerprint(two_ring_single_joint()))

    def test_builds_round_trips_and_runs_on_the_fakes(self):
        g = three_ring_single_joint(joint=5, goal_rad=0.6)
        again = loads(dumps(g))
        self.assertEqual(json.loads(dumps(again)), json.loads(dumps(g)))
        compiled = compile_graph(again, engines="fake")
        compiled.loop.initialize()
        try:
            record = run_graph_trial(compiled, 0.6)
        finally:
            compiled.loop.shutdown()
        self.assertGreater(record.n_steps, 0)
        self.assertIn("jt__P_state", record.main_ticks[-1].outputs)


class DisturbanceTests(unittest.TestCase):
    def run_engine(self, **kwargs):
        from tiago_ring_controller.cosim.datapack import DataPack

        engine = FakeRobotEngine()
        engine.initialize()
        engine.reset()
        disturbance = Disturbance(engine, 5, at_s=0.1, **kwargs)
        trace = []
        for _ in range(8):
            engine.set_datapacks({"arm_velocity_cmd": DataPack("arm_velocity_cmd", engine.t_ms, {
                "joint_index": 5, "velocities": [0.2], "dt_s": 0.05})})
            engine.advance(50.0)
            trace.append(engine.get_datapacks()["joint_state"]["commanded_positions"][5])
        disturbance.remove()
        return np.array(trace), disturbance

    def test_push_shifts_the_commanded_position_once(self):
        plain, _ = self.run_engine()
        pushed, disturbance = self.run_engine(push_rad=0.2)
        self.assertTrue(disturbance.pushed)
        np.testing.assert_allclose(pushed[:2], plain[:2])
        np.testing.assert_allclose(pushed[2:] - plain[2:], 0.2)

    def test_hold_drops_the_commands_for_its_duration(self):
        plain, _ = self.run_engine()
        held, disturbance = self.run_engine(hold_s=0.2)
        self.assertEqual(disturbance.dropped, 4)
        np.testing.assert_allclose(held[2:6], held[1])
        np.testing.assert_allclose(held[-1] - held[5], plain[-1] - plain[5])


class ValidationScriptTests(unittest.TestCase):
    def test_settings_parse_dotted_keys(self):
        from validate_velocity import parse_settings

        self.assertEqual(parse_settings(["a=1", "b.c=2e4", "b.d=x", "mode=corrective"]),
                         {"a": 1, "b": {"c": 2e4, "d": "x"}, "mode": "corrective"})


if __name__ == "__main__":
    unittest.main()
