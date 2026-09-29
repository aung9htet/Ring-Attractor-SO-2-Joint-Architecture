"""Blocks and Graph on the fake NEST: topology equivalence, validation, composites, signal blocks."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from support_fake_nest import RecordingNest, connection_rows, label_nodes, relabel_rows  # noqa: E402
from tiago_ring_controller.blocks import (  # noqa: E402
    REGISTRY,
    BlockError,
    BuildContext,
    Decoder,
    Encoder,
    FeatureEncoder,
    FourierReadout,
    Gain,
    Goal,
    Homeostasis,
    Joint,
    JointTriple,
    OutputRing,
    Probe,
    ProfileDecoder,
    Ring,
    SignedProduct,
    TaskGain,
    Transport,
    block_from_type,
    describe_types,
    schema_for,
)
from tiago_ring_controller.control.profiles import COLLECTOR_PROFILE  # noqa: E402
from tiago_ring_controller.cosim.datapack import DataPack  # noqa: E402
from tiago_ring_controller.cosim.graph_engine import GraphNestEngine  # noqa: E402
from tiago_ring_controller.cosim.tf import MotorTF, TickContext  # noqa: E402
from tiago_ring_controller.graph import Graph, GraphError  # noqa: E402
from tiago_ring_controller.nest.single_ring import build_single_ring_network  # noqa: E402


MODEL_PARAMS = SRC / "config/model_params"


def single_joint_graph(**sim):
    """The two-ring single-joint architecture, hand-built (plan 5b)."""

    g = Graph("two ring single joint", **sim)
    r1, r2 = g.add(Ring("r1")), g.add(Ring("r2"))
    f1, f2 = g.add(FourierReadout("f1")), g.add(FourierReadout("f2"))
    cmp, gain = g.add(Homeostasis("cmp")), g.add(Gain("gain"))
    enc_state = g.add(Encoder("enc_state", joint_min=-1.0, joint_max=1.0))
    enc_goal = g.add(Encoder("enc_goal", joint_min=-1.0, joint_max=1.0))
    dec, j6, goal = g.add(Decoder("dec")), g.add(Joint("j6", index=5)), g.add(Goal("goal", angle_rad=0.6))
    g.connect(r1.spikes, f1.ring)
    g.connect(r2.spikes, f2.ring)
    g.connect(f1.features, cmp.state_features)
    g.connect(f2.features, cmp.target_features)
    g.connect(cmp.left, gain.left_in)
    g.connect(cmp.right, gain.right_in)
    g.connect(r1.spikes, gain.ring)
    g.connect(gain.feedback, r1.stim)
    g.connect(enc_state.stim, r1.stim)
    g.connect(enc_goal.stim, r2.stim)
    g.connect(j6.angle, enc_state.angle)
    g.connect(goal.angle, enc_goal.angle)
    g.connect(gain.left_counts, dec.left_counts)
    g.connect(gain.right_counts, dec.right_counts)
    g.connect(dec.velocity, j6.velocity)
    return g


def graph_rows(backend, graph):
    labels = {}
    for name in ("r1", "r2", "f1", "f2"):
        pop = graph.blocks[name].population
        label_nodes(labels, name, pop.neurons)
        label_nodes(labels, name + "rec", pop.recorders)
    label_nodes(labels, "f1dc", graph.blocks["f1"].population.dc_generator)
    label_nodes(labels, "f2dc", graph.blocks["f2"].population.dc_generator)
    decision = graph.blocks["cmp"].population
    for index, label in enumerate(decision.labels):
        labels[decision.neurons.ids[index]] = ("dec", label)
        labels[decision.recorders.ids[index]] = ("drec", label)
    gain = graph.blocks["gain"].populations
    label_nodes(labels, "left", gain.left.neurons)
    label_nodes(labels, "right", gain.right.neurons)
    label_nodes(labels, "leftrec", gain.left.recorders)
    label_nodes(labels, "rightrec", gain.right.recorders)
    label_nodes(labels, "stim_r1", graph.blocks["enc_state"].stimulus.generators)
    label_nodes(labels, "stim_r2", graph.blocks["enc_goal"].stimulus.generators)
    return relabel_rows(connection_rows(backend), labels)


def network_rows(backend, network):
    labels = {}
    for name in ("r1", "r2", "f1", "f2"):
        pop = getattr(network, name)
        label_nodes(labels, name, pop.neurons)
        label_nodes(labels, name + "rec", pop.recorders)
    label_nodes(labels, "f1dc", network.f1.dc_generator)
    label_nodes(labels, "f2dc", network.f2.dc_generator)
    for index, label in enumerate(network.decision.labels):
        labels[network.decision.neurons.ids[index]] = ("dec", label)
        labels[network.decision.recorders.ids[index]] = ("drec", label)
    label_nodes(labels, "left", network.gain.left.neurons)
    label_nodes(labels, "right", network.gain.right.neurons)
    label_nodes(labels, "leftrec", network.gain.left.recorders)
    label_nodes(labels, "rightrec", network.gain.right.recorders)
    label_nodes(labels, "stim_r1", network.stimulus["r1"].generators)
    label_nodes(labels, "stim_r2", network.stimulus["r2"].generators)
    return relabel_rows(connection_rows(backend), labels)


class SingleJointGraphEquivalenceTests(unittest.TestCase):
    def test_hand_built_graph_creates_the_vectorised_network_exactly(self):
        graph_backend, net_backend = RecordingNest(), RecordingNest()
        graph = single_joint_graph(rng_seed=13579)
        built = graph.build(BuildContext(graph_backend, rng_seed=13579))
        network = build_single_ring_network(net_backend, seed=13579)
        # Same node creation sequence (so the same global ids, the same RNG streams) ...
        self.assertEqual(
            [(c["model"], c["n"], c["params"]) for c in graph_backend.create_calls],
            [(c["model"], c["n"], c["params"]) for c in net_backend.create_calls],
        )
        self.assertEqual(graph_backend.kernel_calls, net_backend.kernel_calls)
        # ... the same synapse set, even the same Connect calls in the same order.
        self.assertEqual(graph_rows(graph_backend, graph), network_rows(net_backend, network))
        self.assertEqual(
            [(c["source"], c["target"], c["conn_spec"]) for c in graph_backend.connect_calls],
            [(c["source"], c["target"], c["conn_spec"]) for c in net_backend.connect_calls],
        )
        self.assertEqual(built.order, ["r1", "r2", "f1", "f2", "cmp", "gain", "enc_state", "enc_goal"])
        self.assertEqual(sorted(built.counts_sources()), ["cmp", "gain", "r1", "r2"])
        self.assertEqual(graph.blocks["gain"].feedback_targets, {"r1": 190})
        self.assertEqual(graph.blocks["cmp"].feature_names[:2], ["sin_pos_k1", "sin_neg_k1"])
        self.assertEqual(len(built.describe()["log"]), 9)

    def test_transport_composite_builds_the_same_synapses(self):
        composite_backend, primitive_backend = RecordingNest(), RecordingNest()
        g = Graph("with composite")
        r1, r2 = g.add(Ring("r1")), g.add(Ring("r2"))
        f1, f2 = g.add(FourierReadout("f1")), g.add(FourierReadout("f2"))
        ct = g.add(Transport("ct", gain={"gain_to_ring_weight": -0.6}))
        g.connect(r1.spikes, f1.ring)
        g.connect(r2.spikes, f2.ring)
        g.connect(f1.features, ct.state_features)
        g.connect(f2.features, ct.target_features)
        g.connect(r1.spikes, ct.ring)
        g.connect(ct.feedback, r1.stim)
        g.build(BuildContext(composite_backend, rng_seed=1))
        self.assertEqual(list(g.blocks), ["r1", "r2", "f1", "f2", "ct__cmp", "ct__gain"])
        self.assertEqual(list(g.composites), ["ct"])
        self.assertEqual([e.key for e in g.edges][:2], ["ct__cmp.left -> ct__gain.left_in", "ct__cmp.right -> ct__gain.right_in"])

        h = single_joint_graph(rng_seed=1)
        h.build(BuildContext(primitive_backend, rng_seed=1))
        # Ignore the encoders' generators (the composite graph has none) and compare.
        def is_generator_edge(call):
            weight = (call["syn_spec"] or {}).get("weight")
            return call["conn_spec"] == "one_to_one" and np.isscalar(weight) and weight == 4500.0

        primitive = [c for c in primitive_backend.connect_calls if not is_generator_edge(c)]
        self.assertEqual(
            [(c["source"], c["target"], c["conn_spec"]) for c in composite_backend.connect_calls],
            [(c["source"], c["target"], c["conn_spec"]) for c in primitive],
        )


class GraphValidationTests(unittest.TestCase):
    def test_all_problems_are_reported_at_once(self):
        g = Graph("bad")
        r1, f1, f2 = g.add(Ring("r1")), g.add(FourierReadout("f1")), g.add(FourierReadout("f2"))
        dec, j = g.add(Decoder("dec")), g.add(Joint("j"))
        gain = g.add(Gain("gain"))
        g.connect(f1.features, f2.ring)                       # build-bound cycle with the next edge
        g.connect(f2.features, f1.ring)
        g.connect(r1.spikes, gain.ring, bogus=1)              # unknown edge parameter
        g.connect(r1.counts, gain.left_in)                    # signal -> spikes
        g.connect(dec.velocity, j.velocity)
        g.connect(j.angle, dec.left_counts)                    # fine
        g.connect(j.angle, dec.right_counts)
        g.connect(dec.velocity, dec.centroid)                  # signal self-cycle
        g.connect(gain.left_counts, dec.left_counts)           # second producer on a signal input
        g.connect(gain.left_counts, r1.stim)                   # output kind mismatch, but multi allowed
        with self.assertRaises(GraphError) as raised:
            g.validate()
        text = str(raised.exception)
        self.assertEqual(text.count("signal cycle within a tick: dec -> j -> dec"), 1)
        for needle in (
            "unknown edge parameters ['bogus']",
            "kinds differ (signal -> spikes)",
            "gain.right_in: required input is not connected",
            "dec.left_counts: 2 producers, at most one allowed",
            "signal cycle within a tick: dec -> dec",
            "cycle among build-bound spike edges involving ['f1', 'f2']",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)
        self.assertGreaterEqual(len(raised.exception.problems), 6)

    def test_connect_direction_and_unknown_blocks(self):
        g = Graph("dir")
        r1 = g.add(Ring("r1"))
        stray = Ring("stray")
        g.connect(r1.stim, stray.spikes)
        problems = g.problems()
        self.assertIn("edge r1.stim -> stray.spikes: target block 'stray' is not in the graph", problems)
        self.assertIn("edge r1.stim -> stray.spikes: source 'stim' is an input port", problems)
        self.assertIn("edge r1.stim -> stray.spikes: target 'spikes' is an output port", problems)
        with self.assertRaises(GraphError):
            g.add(Ring("r1"))
        with self.assertRaises(GraphError):
            g.connect(r1, r1.stim)
        with self.assertRaises(AttributeError):
            r1.nope
        self.assertEqual(Graph("s", dt_ms=0).problems(), ["simulation.dt_ms must be positive"])

    def test_block_errors_name_the_block(self):
        with self.assertRaisesRegex(BlockError, "^Ring 'r1': population_size: 2 is below the minimum 3"):
            Ring("r1", population_size=2)
        with self.assertRaisesRegex(BlockError, "^Gain 'g': unknown parameter 'weight'"):
            Gain("g", weight=1)
        with self.assertRaisesRegex(BlockError, "block id 'a.b' must be"):
            Ring("a.b")
        with self.assertRaisesRegex(BlockError, "unknown block type 'Nope'"):
            block_from_type("Nope", "x")
        # Homeostasis with mismatched feature sizes
        g = Graph("mismatch")
        r1, r2 = g.add(Ring("r1", population_size=20)), g.add(Ring("r2", population_size=20))
        f1, f2 = g.add(FourierReadout("f1", num_fourier_k=2)), g.add(FourierReadout("f2", num_fourier_k=3))
        cmp = g.add(Homeostasis("cmp"))
        g.connect(r1.spikes, f1.ring)
        g.connect(r2.spikes, f2.ring)
        g.connect(f1.features, cmp.state_features)
        g.connect(f2.features, cmp.target_features)
        with self.assertRaisesRegex(BlockError, "^Homeostasis 'cmp': state features \\(4\\) and target features \\(6\\) differ"):
            g.build(BuildContext(RecordingNest()))
        # An encoder drives exactly one ring
        g = Graph("two rings")
        r1, r2, enc = g.add(Ring("r1", population_size=12)), g.add(Ring("r2", population_size=12)), g.add(Encoder("enc"))
        g.connect(g.add(Goal("goal")).angle, enc.angle)
        g.connect(enc.stim, r1.stim)
        g.connect(enc.stim, r2.stim)
        with self.assertRaisesRegex(BlockError, "^Encoder 'enc': already drives ring 'r1'"):
            g.build(BuildContext(RecordingNest()))
        # Feedback into a ring of another size
        g = Graph("sizes")
        r1, r2 = g.add(Ring("r1", population_size=12)), g.add(Ring("r2", population_size=14))
        cmp = Homeostasis("cmp")
        with self.assertRaisesRegex(BlockError, "^TaskGain 'tg': TaskGain is a research stub"):
            h = Graph("stub")
            a = h.add(Ring("a", population_size=12))
            tg = h.add(TaskGain("tg"))
            h.connect(a.spikes, tg.configuration)
            h.connect(a.spikes, tg.left_in)
            h.connect(a.spikes, tg.right_in)
            h.build(BuildContext(RecordingNest()))


class SchemaAndRegistryTests(unittest.TestCase):
    def test_block_defaults_equal_the_config_files(self):
        homeostasis = json.loads((MODEL_PARAMS / "homeostasis_params.json").read_text())
        gain = json.loads((MODEL_PARAMS / "gain_modulation_params.json").read_text())
        defaults = schema_for("Homeostasis").defaults()
        for key in ("warm_exc_weight", "warm_inh_weight", "cold_exc_weight", "cold_inh_weight", "decision_lateral_weight"):
            self.assertEqual(defaults[key], homeostasis[key])
        self.assertEqual(defaults["weight_scale"], homeostasis["warm_cold_weight_scale"])
        defaults = schema_for("Gain").defaults()
        for key in ("left_homeostasis_gain_weight", "right_homeostasis_gain_weight", "ring_to_gain_weight",
                    "gain_to_ring_weight", "cross_inhibition_weight"):
            self.assertEqual(defaults[key], gain[key])
        ring = json.loads((MODEL_PARAMS / "ring_params.json").read_text())
        readout = schema_for("FourierReadout").defaults()
        self.assertEqual((readout["num_fourier_k"], readout["weight_scale"], readout["dc_baseline"]),
                         (ring["num_fourier_k"], ring["readout_weight_scale"], ring["output_dc_baseline"]))
        self.assertEqual(schema_for("Ring").defaults()["population_size"], ring["population_size"])

    def test_registry_and_type_descriptions(self):
        self.assertEqual(list(REGISTRY), list(describe_types()))
        gain = describe_types()["Gain"]
        self.assertEqual([p["name"] for p in gain["ports"]], ["ring", "left_in", "right_in", "feedback", "left_counts", "right_counts"])
        self.assertTrue(gain["neural"])
        self.assertFalse(describe_types()["Decoder"]["neural"])
        ring = describe_types()["Ring"]
        self.assertEqual([p for p in ring["ports"] if p["name"] == "stim"][0]["late"], True)
        self.assertEqual(block_from_type("Goal", "g", {"angle_rad": 0.2}).params["angle_rad"], 0.2)
        for name, cls in REGISTRY.items():
            with self.subTest(block=name):
                self.assertEqual(cls.schema.block_type, name)
                self.assertEqual(cls(name.lower() + "_x").describe()["type"], name)


class EncoderBehaviourTests(unittest.TestCase):
    def _encoder(self, **params):
        backend = RecordingNest()
        g = Graph("enc")
        ring = g.add(Ring("ring", population_size=20))
        enc = g.add(Encoder("enc", joint_min=-1.0, joint_max=1.0, **params))
        g.connect(g.add(Goal("goal")).angle, enc.angle)
        g.connect(enc.stim, ring.stim)
        g.build(BuildContext(backend))
        return backend, enc

    def _rates(self, backend):
        return [entry["rate"] for entry in backend.status_calls[-1][1]]

    def test_once_mode_emits_one_window_that_expires(self):
        backend, enc = self._encoder(mode="once", half_width=1, duration_ticks=2)
        index = COLLECTOR_PROFILE.joint_to_ring_index(0.0, -1.0, 1.0, 20, requested_half_width=1)
        self.assertTrue(enc.drive(0.0, 0.0, 50.0))
        self.assertEqual(enc.last_index, index)
        rates = self._rates(backend)
        self.assertEqual(sum(1 for r in rates if r == 200.0), 3)
        self.assertFalse(enc.drive(0.5, 50.0, 50.0))          # once: ignored
        self.assertFalse(enc.expire(50.0))                    # 2 ticks: still on
        self.assertTrue(enc.expire(100.0))
        self.assertEqual(sum(self._rates(backend)), 0.0)
        enc.reset()
        self.assertTrue(enc.drive(0.5, 100.0, 50.0))

    def test_continuous_and_corrective_modes(self):
        backend, enc = self._encoder(mode="continuous", half_width=0, rate_hz=80.0)
        self.assertTrue(enc.drive(0.0, 0.0, 50.0))
        self.assertTrue(enc.drive(0.0, 50.0, 50.0))
        self.assertEqual(max(self._rates(backend)), 80.0)
        backend, enc = self._encoder(mode="corrective", half_width=0, dead_band=2)
        index = enc.ring_index(0.0)
        self.assertFalse(enc.drive(0.0, 0.0, 50.0, centroid=index + 1.5))   # within the dead band
        self.assertTrue(enc.drive(0.0, 0.0, 50.0, centroid=index + 4.0))
        self.assertTrue(enc.drive(0.0, 50.0, 50.0, centroid=None))          # no centroid: stimulate
        self.assertTrue(enc.drive_index(7, 100.0, 50.0, rate_hz=10.0))
        self.assertEqual(self._rates(backend)[7], 10.0)
        with self.assertRaisesRegex(BlockError, "is not connected to a ring"):
            Encoder("lonely").drive(0.0, 0.0, 50.0)

    def test_feature_encoder_rates_are_push_pull_sines(self):
        backend = RecordingNest()
        g = Graph("fe")
        fe = g.add(FeatureEncoder("fe", num_fourier_k=2, population_size=8, rate_scale=10.0))
        g.connect(g.add(Goal("goal")).angle, fe.angle)
        g.build(BuildContext(backend))
        self.assertEqual(backend.create_calls[-1]["n"], 4)
        self.assertEqual(fe.source_ring_size, 8)
        self.assertTrue(fe.drive(0.0, 0.0, 50.0))
        rates = fe.rates
        self.assertEqual(rates.shape, (4,))
        self.assertTrue(np.all(rates >= 0.0))
        self.assertFalse(fe.drive(0.0, 50.0, 50.0))           # unchanged rates: no SetStatus
        self.assertTrue(fe.clear())


class SignalBlockTests(unittest.TestCase):
    def test_decoder_reproduces_motor_tf(self):
        dec = Decoder("dec", tau_s=0.51, gain_positive=0.0011, gain_negative=-0.0011, drive_threshold=5.0, n_settle=3, horizon=4)
        dec.configure(50.0)
        tf = MotorTF(COLLECTOR_PROFILE, 5, 200, 50.0, decoder=dec.parameters(), nest_lead_steps=4, drive_threshold=5.0, n_settle=3)
        rng = np.random.default_rng(3)
        for step in range(1, 30):
            left, right = int(rng.integers(0, 40)), int(rng.integers(0, 40))
            if step > 20:
                left = right = 0
            is_lead = step <= 4
            pack = DataPack("ring_counts", 50.0 * step, {"left": left, "right": right, "nest_step": step})
            expected = tf({"ring_counts": pack}, TickContext(t_ms=50.0 * step, tick=step, phase="lead" if is_lead else "main"))["arm_velocity_cmd"]
            actual = dec.step(left, right, is_lead=is_lead, nest_step=step)
            self.assertEqual(actual["velocity"], expected["velocities"])
            self.assertEqual(actual["settled"], expected["settled"])
            self.assertEqual(actual["consumed"]["decoded_velocity"], expected["consumed"]["decoded_velocity"])
        self.assertTrue(actual["settled"])
        centroid = Decoder("c", source="centroid_velocity", spike_scale=2.0)
        centroid.configure(50.0)
        self.assertEqual(centroid.signed_input(0, 0, centroid=10.0), 0.0)
        self.assertEqual(centroid.signed_input(0, 0, centroid=12.5), 5.0)

    def test_goal_joint_probe_and_profile_decoder(self):
        goal = Goal("goal", angle_rad=0.3)
        self.assertEqual(goal.step(0.0), {"angle": 0.3})
        goal.set(0.5)
        self.assertEqual(goal.value(), 0.5)
        goal.set_schedule([(100.0, 0.9), (0.0, 0.1)])
        self.assertEqual((goal.value(0.0), goal.value(99.0), goal.value(100.0)), (0.1, 0.1, 0.9))
        joint = Joint("j", index=2, joint_min=-0.5, joint_max=0.5)
        self.assertEqual(joint.limits(), (-0.5, 0.5))
        self.assertEqual(joint.read({"positions": [0, 1, 2.5], "velocities": [0, 0, -1]}), {"angle": 2.5, "velocity_measured": -1.0})
        self.assertEqual(joint.read({"positions": [0, 1, 2.5]})["velocity_measured"], 0.0)
        probe = Probe("p", keep=2)
        for t in range(4):
            probe.record(t, t * 2)
        self.assertEqual([s["value"] for s in probe.samples], [4, 6])
        decoder = ProfileDecoder("pd")
        profile = np.zeros(100)
        profile[25] = 3.0
        self.assertAlmostEqual(decoder.decode(profile), math.pi / 2, places=12)
        self.assertAlmostEqual(ProfileDecoder("c", method="centroid").decode(profile), math.pi / 2, places=12)
        self.assertAlmostEqual(ProfileDecoder("s", method="scalar_ramp").decode(profile), math.pi / 2, places=12)
        self.assertTrue(math.isnan(decoder.decode(np.zeros(4))))


class MultiRingBlockTests(unittest.TestCase):
    def test_signed_product_and_output_ring_build(self):
        backend = RecordingNest()
        g = Graph("fk")
        a, b = g.add(Ring("a", population_size=8)), g.add(Ring("b", population_size=8))
        sp = g.add(SignedProduct("sp", grid_size=4, dc_baseline=1.0, input_weight=2.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            weights = np.round(np.random.default_rng(5).standard_normal((16 * 16, 10)), 3)
            np.savez(path, W_signed_lift=weights, W_signed_pitch=weights * 2)
            lift = g.add(OutputRing("lift", size=10, weights_artifact=str(path), weights_key="W_signed_lift", weight_scale=3.0))
            pd = g.add(ProfileDecoder("pd"))
            g.connect(a.spikes, sp.ring_a)
            g.connect(b.spikes, sp.ring_b)
            g.connect(sp.features, lift.features)
            g.connect(lift.profile, pd.profile)
            built = g.build(BuildContext(backend))
            self.assertEqual(built.order, ["a", "b", "sp", "lift"])
            self.assertEqual(sp.n_features, 256)
            self.assertEqual(sp.ring_size, 8)
            self.assertEqual(len(sp.outputs["features"]), 256)
            call = backend.connect_calls[-1]
            self.assertEqual(call["conn_spec"], {"rule": "all_to_all"})
            np.testing.assert_array_equal(call["syn_spec"]["weight"], (3.0 * weights).T)
            self.assertEqual(len(lift.counts_sources()["profile"]), 10)
            bad = OutputRing("bad", size=7, weights_artifact=str(path))
            h = Graph("bad")
            for block in (Ring("a", population_size=8), Ring("b", population_size=8), SignedProduct("sp", grid_size=4)):
                h.add(block)
            h.add(bad)
            h.connect(h.blocks["a"].spikes, h.blocks["sp"].ring_a)
            h.connect(h.blocks["b"].spikes, h.blocks["sp"].ring_b)
            h.connect(h.blocks["sp"].features, bad.features)
            with self.assertRaisesRegex(BlockError, "^OutputRing 'bad': .*W_signed_lift\\] has shape \\(256, 10\\), expected \\(256, 7\\)"):
                h.build(BuildContext(RecordingNest()))

    def test_joint_triple_expands_and_builds(self):
        g = Graph("triple")
        jt = g.add(JointTriple("j", population_size=20, num_fourier_k=2))
        goal, joint = g.add(Goal("goal")), g.add(Joint("j6"))
        g.connect(goal.angle, jt.goal_angle)
        g.connect(joint.angle, jt.measured_angle)
        self.assertEqual(len(g.blocks), 3 + 3 + 2 + 2 * 2 + 2)
        self.assertEqual(sorted(b.type_name for b in g.blocks.values()).count("Ring"), 3)
        # Comparator artifacts exist only for N=100/200; a 20-ring cannot load them.
        with self.assertRaisesRegex(BlockError, "^Homeostasis 'j__ct_goal__cmp': cannot load fitted weights"):
            g.build(BuildContext(RecordingNest()))
        g = Graph("triple200")
        jt = g.add(JointTriple("j"))
        g.connect(g.add(Goal("goal")).angle, jt.goal_angle)
        g.connect(g.add(Joint("j6")).angle, jt.measured_angle)
        backend = RecordingNest()
        built = g.build(BuildContext(backend))
        self.assertEqual(built.order[:3], ["j__T", "j__B", "j__A"])
        self.assertEqual(g.blocks["j__ct_goal__gain"].feedback_targets, {"j__B": 190})
        self.assertEqual(g.blocks["j__ct_sense__gain"].feedback_targets, {"j__B": 190})
        self.assertEqual(g.blocks["j__enc_state"].params["mode"], "continuous")
        self.assertEqual(len(built.encoders()), 2)


class GraphEngineFakeTests(unittest.TestCase):
    def test_engine_datapacks_bumps_and_recalibration(self):
        backend = RecordingNest()
        graph = single_joint_graph(rng_seed=3, local_num_threads=1)
        engine = GraphNestEngine(backend, graph)
        self.assertEqual(engine.inputs, {"enc_state.angle", "enc_goal.angle"})
        self.assertEqual(engine.outputs, {"r1", "r2", "cmp", "gain"})
        engine.initialize()
        engine.reset()
        self.assertEqual(engine.rebuild_count, 1)
        engine.set_datapacks({
            "enc_goal.angle": DataPack("enc_goal.angle", 0.0, {"angle": 0.6}),
            "enc_state.angle": DataPack("enc_state.angle", 0.0, {"angle": 0.0}),
        })
        engine.advance(50.0)
        packs = engine.get_datapacks()
        self.assertEqual(sorted(packs), ["cmp", "gain", "r1", "r2"])
        self.assertEqual(len(packs["r1"]["counts"]), 200)
        self.assertEqual(packs["r1"]["total"], 0.0)
        self.assertIsNone(packs["r1"]["bump_index"])
        self.assertEqual(packs["gain"]["left_counts"], 0)
        self.assertEqual(packs["cmp"]["counts"], {"warm": 0, "cold": 0, "left": 0, "right": 0})
        self.assertEqual(packs["r1"]["nest_step"], 1)
        self.assertEqual([a["encoder"] for a in engine.applied], ["enc_state", "enc_goal"])
        self.assertEqual(backend.lifecycle_calls, ["Prepare", ("Run", 50.0), "Cleanup", "Prepare"])  # windows expired
        self.assertEqual(engine.recalibrations, 1)
        engine.advance(50.0)
        self.assertEqual(engine.recalibrations, 1)
        engine.set_datapacks({"enc_state.angle": DataPack("enc_state.angle", 100.0, {"angle": 0.3})})
        engine.advance(50.0)                                   # once mode: ignored, no recalibration
        self.assertEqual(engine.recalibrations, 1)
        self.assertEqual(engine.hidden_ms, 0.0)
        self.assertEqual(sorted(engine.raster()), ["r1_senders", "r1_times", "r2_senders", "r2_times"])
        engine.finish_trial()
        engine.reset("continue")
        self.assertEqual(engine.rebuild_count, 1)
        self.assertEqual(engine.describe()["encoders"], ["enc_state", "enc_goal"])
        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
