"""Schemas consumed by existing analysis and visualization entrypoints."""

import json
import hashlib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_MANIFEST = ROOT / "test/golden/environment_manifest.json"
CHECKED_OUTPUT_MANIFEST = ROOT / "test/golden/checked_output_manifest.json"


class CheckedFigureOutputTests(unittest.TestCase):
    def test_all_checked_png_bytes_and_dimensions_match_the_baseline(self):
        manifest = json.loads(CHECKED_OUTPUT_MANIFEST.read_text(encoding="utf-8"))
        current = {
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*.png")
            if "test" not in path.relative_to(ROOT).parts
            and "docs" not in path.relative_to(ROOT).parts
        }
        self.assertEqual(set(manifest["files"]), current)
        for relative, expected in manifest["files"].items():
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected["sha256"],
                )
                with Image.open(path) as image:
                    self.assertEqual(image.format, expected["format"])
                    self.assertEqual(image.mode, expected["mode"])
                    self.assertEqual(image.size, (expected["width"], expected["height"]))


class DecoderOutputSchemaTests(unittest.TestCase):
    def test_all_checked_in_fourier_results_have_the_same_schema(self):
        paths = sorted((ROOT / "outputs/train/single_joint_ring_component").glob("results_*/fourier_results.npz"))
        paths.append(ROOT / "src/outputs/ring_decoding/fourier_results.npz")
        self.assertEqual(len(paths), 10)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(data.files, ["phi_true", "phi_est"])
                    self.assertEqual(data["phi_true"].shape, (30,))
                    self.assertEqual(data["phi_est"].shape, (30,))
                    self.assertEqual(str(data["phi_true"].dtype), "float64")
                    self.assertEqual(str(data["phi_est"].dtype), "float64")
                    self.assertTrue(np.all(np.isfinite(data["phi_true"])))
                    self.assertTrue(np.all(np.isfinite(data["phi_est"])))

    def test_homeostasis_metadata_feature_order(self):
        for population_size, harmonics in ((100, 5), (200, 20)):
            path = ROOT / (
                "src/config/homeostasis/N_%d_homeostasis_metadata.json" % population_size
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            expected = []
            for harmonic in range(1, harmonics + 1):
                expected.extend(("sin_pos_k%d" % harmonic, "sin_neg_k%d" % harmonic))
            with self.subTest(population_size=population_size):
                self.assertEqual(data["population_size"], population_size)
                self.assertEqual(data["feature_names"], expected)
                self.assertEqual(data["num_positions"], 32)
                self.assertEqual(data["settle_ms"], 300.0)
                self.assertEqual(data["stimulus_half_width"], 5)
                self.assertEqual(data["ridge_lambda"], 1e-6)
                self.assertEqual(data["warm_cold_weight_scale"], 5_000_000.0)
                self.assertEqual(data["warm_bias"], data["cold_bias"])


class ConfigurationSchemaTests(unittest.TestCase):
    def test_ring_and_multi_ring_configuration_contract(self):
        ring = json.loads((ROOT / "src/config/model_params/ring_params.json").read_text())
        self.assertEqual(
            ring,
            {
                "population_size": 200,
                "num_positions": 30,
                "num_fourier_k": 20,
                "config_dir": "./config/ring_decoder",
                "samples_per_position": 1,
                "sim_settle_ms": 300.0,
                "stimulus_half_width": 5,
                "ridge_lambda": 1e-6,
                "output_dir": "./outputs/ring_decoding",
                "harmonic_index": 0,
                "readout_weight_scale": 200.0,
                "output_dc_baseline": 200.0,
            },
        )
        multi = json.loads((ROOT / "src/config/model_params/multi_ring_params.json").read_text())
        self.assertEqual(multi["population_size"], 100)
        self.assertEqual(multi["num_joints"], 2)
        self.assertEqual(multi["joint_axes"], ["x", "y"])
        self.assertEqual(multi["product_population_size"], 32)
        self.assertEqual(multi["num_positions"], 16)
        self.assertEqual(multi["max_train_samples"], 256)
        self.assertEqual(multi["sim_settle_ms"], 300.0)
        self.assertEqual(multi["stimulus_half_width"], 5)

    def test_neuron_parameter_contract(self):
        params = json.loads((ROOT / "src/config/model_params/neuron_params.json").read_text())
        self.assertEqual(list(params), ["ring", "homeostasis", "gain"])
        expected_common = {
            "V_m": 0.0,
            "E_L": 0.0,
            "V_reset": 0.0,
            "tau_syn_in": 50.0,
            "tau_m": 7.04,
            "C_m": 3960000.0,
            "t_ref": 0.0,
        }
        for name in ("ring", "homeostasis", "gain"):
            for key, value in expected_common.items():
                self.assertEqual(params[name][key], value)
        self.assertEqual((params["ring"]["V_th"], params["ring"]["tau_syn_ex"], params["ring"]["I_e"]), (0.8, 27.0, 450000.0))
        self.assertEqual((params["homeostasis"]["V_th"], params["homeostasis"]["tau_syn_ex"], params["homeostasis"]["I_e"]), (0.8, 75.0, 900000.0))
        self.assertEqual((params["gain"]["V_th"], params["gain"]["tau_syn_ex"], params["gain"]["I_e"]), (1.35, 27.0, 0.0))

    def test_velocity_calibration_joint_limits_and_runtime_keys(self):
        path = ROOT / "src/config/calibration/velocity_calibration.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(list(data), ["3", "2", "0", "1", "4", "5", "6"])
        expected_limits = {
            "0": (-8.094148823012404e-05, 2.748970474380153),
            "1": (-1.5714253397332918, 1.0911721972233304),
            "2": (-3.5348136904851106, 1.2613053141677515),
            "3": (-0.39271339638354874, 2.356205376381638),
            "4": (-2.0945012771678044, 2.0944328056118007),
            "5": (-1.4172695167359413, 1.4171452891568528),
            "6": (-2.095829059115535, 2.095868598302337),
        }
        for joint, limits in expected_limits.items():
            with self.subTest(joint=joint):
                entry = data[joint]
                self.assertEqual((entry["joint_min"], entry["joint_max"]), limits)
                for key in (
                    "decoder_gain_positive",
                    "decoder_gain_negative",
                    "decoder_tau",
                    "decoder_delay_steps",
                    "raw_drive_velocity_gain",
                    "decoder_model",
                    "calibration_settings",
                    "calibration_progress",
                ):
                    self.assertIn(key, entry)


class WorldAndLaunchSchemaTests(unittest.TestCase):
    def test_current_external_world_resolution_matches_environment_manifest(self):
        manifest = json.loads(ENVIRONMENT_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest["ros"]
        path = Path(expected["resolved_world"])
        self.assertTrue(path.is_file())
        self.assertEqual(path.stat().st_size, expected["resolved_world_bytes"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, expected["resolved_world_sha256"])

    def test_local_world_is_camera_less_sdf_1_4(self):
        root = ET.parse(ROOT / "worlds/tiago_ring_controller.world").getroot()
        self.assertEqual(root.tag, "sdf")
        self.assertEqual(root.attrib["version"], "1.4")
        world = root.find("world")
        self.assertEqual(world.attrib["name"], "default")
        self.assertEqual(world.findtext("physics/real_time_update_rate"), "1000")
        self.assertEqual(world.findtext("physics/max_step_size"), "0.001")
        self.assertEqual(
            [item.findtext("uri") for item in world.findall("include")],
            ["model://ground_plane", "model://table_0m8", "model://aruco_cube"],
        )
        self.assertEqual(world.findall(".//sensor"), [])

    def test_launch_arguments_and_active_node(self):
        root = ET.parse(ROOT / "launch/tiago_ring_controller.launch").getroot()
        args = {node.attrib["name"]: node.attrib["default"] for node in root.findall("arg")}
        self.assertEqual(
            args,
            {
                "world": "tiago_ring_controller",
                "gui": "true",
                "public_sim": "true",
                "arm": "true",
                "end_effector": "pal-hey5",
                "ft_sensor": "schunk-ft",
            },
        )
        active_nodes = root.findall("node")
        self.assertEqual(len(active_nodes), 1)
        self.assertEqual(
            active_nodes[0].attrib,
            {
                "name": "record_experiments",
                "pkg": "tiago_ring_controller",
                "type": "record_experiments.py",
                "output": "screen",
            },
        )


if __name__ == "__main__":
    unittest.main()
