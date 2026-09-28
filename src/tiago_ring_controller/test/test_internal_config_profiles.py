"""Characterize typed configuration and the three distinct control profiles."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tiago_ring_controller.config import (  # noqa: E402
    decoder_from_entry,
    joint_calibration_from_document,
    joint_limits_from_document,
    load_gain_spec,
    load_homeostasis_spec,
    load_joint_calibration,
    load_json,
    load_multi_ring_spec,
    load_neuron_specs,
    load_ring_spec,
    module_config_path,
    pid_from_entry,
    resolve_legacy_read_path,
    source_config_path,
    source_root,
)
from tiago_ring_controller.contracts import DecoderSpec, PIDSpec  # noqa: E402
from tiago_ring_controller.control.profiles import (  # noqa: E402
    ANALYSIS_PROFILE,
    CALIBRATION_PROFILE,
    COLLECTION_PROFILE,
    COLLECTOR_PROFILE,
    LEGACY_CONTROL_PROFILES,
    get_legacy_control_profile,
)


MODEL_PARAMS = SRC / "config/model_params"
CALIBRATION_PATH = SRC / "config/calibration/velocity_calibration.json"


class TypedConfigurationTests(unittest.TestCase):
    def test_source_paths_remain_flat_source_space_paths(self):
        self.assertEqual(Path(source_root()), SRC)
        self.assertEqual(
            Path(source_config_path("model_params", "ring_params.json")),
            MODEL_PARAMS / "ring_params.json",
        )

    def test_current_specs_keep_types_values_and_unknown_raw_fields(self):
        ring = load_ring_spec(str(MODEL_PARAMS / "ring_params.json"))
        self.assertEqual((ring.population_size, ring.num_fourier_k), (200, 20))
        self.assertEqual(ring.stimulus_half_width, 5)
        self.assertEqual(ring.raw["output_dir"], "./outputs/ring_decoding")

        neurons = load_neuron_specs(str(MODEL_PARAMS / "neuron_params.json"))
        self.assertEqual(tuple(neurons), ("ring", "homeostasis", "gain"))
        self.assertEqual(neurons["ring"].model, "iaf_psc_alpha")
        self.assertEqual(neurons["ring"].parameters["I_e"], 450000.0)

        homeostasis = load_homeostasis_spec(
            str(MODEL_PARAMS / "homeostasis_params.json")
        )
        self.assertEqual(homeostasis.warm_cold_weight_scale, 5_000_000.0)
        self.assertEqual(homeostasis.raw["label"], {"left": 0, "right": 1})

        gain = load_gain_spec(str(MODEL_PARAMS / "gain_modulation_params.json"))
        self.assertEqual(gain.gain_to_ring_weight, -0.6)
        self.assertIn("previous_params", gain.raw)

        multi = load_multi_ring_spec(str(MODEL_PARAMS / "multi_ring_params.json"))
        self.assertEqual(multi.feature_grid_size, 32)
        self.assertEqual(multi.joint_axes, ("x", "y"))
        self.assertEqual(multi.num_joints, 2)
        # This key is absent today; the loader must retain its compatibility default.
        self.assertEqual(multi.signed_product_output_weight_scale, 1.0)

    def test_multi_ring_aliases_and_defaults_are_legacy_permissive(self):
        document = {
            "population_size": 80,
            "num_positions": 7,
            "num_fourier_k": 3,
            "sim_settle_ms": 12,
            "stimulus_half_width": 2,
            "product_population_size": 9,
            "sfp_output_weight_scale": 44,
            "unrecognized": {"preserved": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multi.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            spec = load_multi_ring_spec(str(path))
        self.assertEqual(spec.feature_grid_size, 9)
        self.assertEqual(spec.signed_product_output_weight_scale, 44.0)
        self.assertEqual(spec.ridge_lambda, 1.0e-3)
        self.assertEqual(spec.output_ring_size, 100)
        self.assertEqual(spec.joint_axes, ("x", "y"))
        self.assertEqual(spec.raw["unrecognized"], {"preserved": True})

    def test_json_paths_follow_the_callers_current_working_directory(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relative.json"
            path.write_text('{"value": NaN}', encoding="utf-8")
            try:
                os.chdir(directory)
                loaded = load_json("relative.json")
            finally:
                os.chdir(previous)
        self.assertTrue(loaded["value"] != loaded["value"])

    def test_source_missing_paths_do_not_silently_search_other_locations(self):
        missing = "./config/not-a-real-contract.json"
        self.assertEqual(resolve_legacy_read_path(missing), missing)
        with self.assertRaises(FileNotFoundError) as context:
            load_json(missing)
        self.assertEqual(context.exception.filename, missing)

    def test_install_only_config_fallback_preserves_existing_path_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            installed_module = (
                prefix
                / "lib/python3/dist-packages/tiago_ring_controller/config.py"
            )
            installed_config = (
                prefix
                / "share/tiago_ring_controller/config/model_params/ring.json"
            )
            installed_config.parent.mkdir(parents=True)
            installed_config.write_text('{"installed": true}', encoding="utf-8")

            missing_legacy = "./config/model_params/ring.json"
            self.assertEqual(
                Path(resolve_legacy_read_path(missing_legacy, str(installed_module))),
                installed_config,
            )
            self.assertEqual(
                Path(
                    module_config_path(
                        str(installed_module), "model_params", "ring.json"
                    )
                ),
                installed_config,
            )

            existing = prefix / "config/model_params/ring.json"
            existing.parent.mkdir(parents=True)
            existing.write_text('{"cwd": true}', encoding="utf-8")
            self.assertEqual(
                resolve_legacy_read_path(str(existing), str(installed_module)),
                str(existing),
            )

    def test_catkin_program_config_path_maps_to_the_same_installed_share(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            program = prefix / "lib/tiago_ring_controller/analysis_single_joint.py"
            expected = (
                prefix
                / "share/tiago_ring_controller/config/calibration/velocity_calibration.json"
            )
            self.assertEqual(
                Path(
                    module_config_path(
                        str(program), "calibration", "velocity_calibration.json"
                    )
                ),
                expected,
            )

    def test_top_level_decoder_fields_override_conflicting_nested_history(self):
        entry = {
            "decoder_gain_positive": 0.001,
            "decoder_gain_negative": -0.002,
            "decoder_tau": 0.7,
            "decoder_delay_steps": 3,
            "raw_drive_velocity_gain": 0.009,
            "calibration_progress": {
                "decoder_gain_positive": 99.0,
                "decoder_tau": 99.0,
            },
        }
        decoder = decoder_from_entry(entry)
        self.assertEqual(
            (decoder.gain_positive, decoder.gain_negative, decoder.tau, decoder.delay_steps),
            (0.001, -0.002, 0.7, 3),
        )
        self.assertEqual(decoder.source, "decoder")

    def test_decoder_legacy_fallback_invalid_values_and_absent_pid_defaults(self):
        defaults = DecoderSpec(0.1, -0.2, 0.4, 2, "fixture")
        legacy = decoder_from_entry({"raw_drive_velocity_gain": "0.003"}, defaults)
        self.assertEqual(
            (legacy.gain_positive, legacy.gain_negative, legacy.tau, legacy.delay_steps),
            (0.003, -0.003, 0.4, 2),
        )
        self.assertEqual(legacy.source, "raw_drive_velocity_gain")
        self.assertIs(
            decoder_from_entry(
                {
                    "decoder_gain_positive": "bad",
                    "decoder_gain_negative": -1,
                    "decoder_tau": 1,
                    "decoder_delay_steps": 0,
                },
                defaults,
            ),
            defaults,
        )
        self.assertEqual(pid_from_entry({}), PIDSpec())
        custom_pid = pid_from_entry({"pid_kp": 3, "pid_output_limit": 0.25})
        self.assertEqual(custom_pid, PIDSpec(3.0, 0.1, 0.05, 0.25, 2.0))

    def test_joint_limits_failure_and_unknown_fields_round_trip_in_raw(self):
        document = {
            "4": {
                "joint_min": "-1.5",
                "joint_max": 2,
                "decoder_gain_positive": 1e-4,
                "decoder_gain_negative": -2e-4,
                "decoder_tau": 0.5,
                "decoder_delay_steps": 1,
                "unknown_history": [1, 2, 3],
            },
            "5": "malformed legacy value",
        }
        self.assertEqual(joint_limits_from_document(document, 4), (-1.5, 2.0))
        self.assertIsNone(joint_limits_from_document(document, 5))
        calibration = joint_calibration_from_document(document, 4)
        self.assertEqual(calibration.raw["unknown_history"], [1, 2, 3])
        self.assertEqual((calibration.joint_min, calibration.joint_max), (-1.5, 2.0))

    def test_checked_in_calibration_uses_runtime_top_level_values(self):
        document = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        for joint_index in range(7):
            with self.subTest(joint_index=joint_index):
                loaded = load_joint_calibration(str(CALIBRATION_PATH), joint_index)
                entry = document[str(joint_index)]
                self.assertEqual(loaded.decoder.gain_positive, float(entry["decoder_gain_positive"]))
                self.assertEqual(loaded.decoder.gain_negative, float(entry["decoder_gain_negative"]))
                self.assertEqual(loaded.decoder.tau, float(entry["decoder_tau"]))
                self.assertEqual(loaded.decoder.delay_steps, int(entry["decoder_delay_steps"]))

    def test_non_strict_missing_document_returns_defaults_but_strict_raises(self):
        missing = str(ROOT / "test" / "does_not_exist.json")
        loaded = load_joint_calibration(missing, 2)
        self.assertIsNone(loaded.joint_min)
        self.assertEqual(loaded.decoder, DecoderSpec())
        with self.assertRaises(FileNotFoundError):
            load_joint_calibration(missing, 2, strict=True)


class LegacyControlProfileTests(unittest.TestCase):
    def test_profiles_are_three_named_immutable_presets(self):
        self.assertEqual(tuple(LEGACY_CONTROL_PROFILES), ("calibration", "analysis", "collector"))
        self.assertIs(get_legacy_control_profile("calibration"), CALIBRATION_PROFILE)
        self.assertIs(get_legacy_control_profile("analysis"), ANALYSIS_PROFILE)
        self.assertIs(get_legacy_control_profile("collector"), COLLECTOR_PROFILE)
        self.assertIs(COLLECTION_PROFILE, COLLECTOR_PROFILE)
        with self.assertRaises(AttributeError):
            ANALYSIS_PROFILE.max_steps = 1

    def test_profile_constants_capture_current_workflow_differences(self):
        for profile in (CALIBRATION_PROFILE, ANALYSIS_PROFILE, COLLECTOR_PROFILE):
            self.assertEqual(profile.time_step_ms, 50.0)
            self.assertEqual(profile.lookahead, 4)
            self.assertEqual(profile.drive_threshold, 5.0)
            self.assertEqual(profile.n_settle, 10)
            self.assertEqual(profile.max_steps, 400)
        self.assertEqual(CALIBRATION_PROFILE.effective_half_width(5), 5)
        self.assertEqual(ANALYSIS_PROFILE.effective_half_width(5), 10)
        self.assertEqual(COLLECTOR_PROFILE.effective_half_width(5), 5)
        self.assertEqual(CALIBRATION_PROFILE.spike_scale(200), 0.5)
        self.assertEqual(ANALYSIS_PROFILE.spike_scale(200), 1.0)
        self.assertEqual(COLLECTOR_PROFILE.spike_scale(200), 1.0)

    def test_mapping_profiles_remain_numerically_distinct(self):
        kwargs = dict(joint_min=0.0, joint_max=1.0, population_size=200)
        calibration = [
            CALIBRATION_PROFILE.joint_to_ring_index(value, **kwargs)
            for value in (-1.0, 0.0, 0.5, 1.0, 2.0)
        ]
        analysis = [
            ANALYSIS_PROFILE.joint_to_ring_index(value, **kwargs)
            for value in (-1.0, 0.0, 0.5, 1.0, 2.0)
        ]
        collector = [
            COLLECTOR_PROFILE.joint_to_ring_index(value, **kwargs)
            for value in (-1.0, 0.0, 0.5, 1.0, 2.0)
        ]
        self.assertEqual(calibration, [20, 20, 100, 180, 180])
        self.assertEqual(analysis, [10, 10, 100, 189, 189])
        self.assertEqual(collector, [5, 5, 100, 194, 194])

    def test_right_minus_left_sign_and_calibration_scaling(self):
        self.assertEqual(CALIBRATION_PROFILE.signed_drive(13, 7, 200), 3.0)
        self.assertEqual(ANALYSIS_PROFILE.signed_drive(13, 7, 200), 6.0)
        self.assertEqual(COLLECTOR_PROFILE.signed_drive(13, 7, 200), 6.0)


if __name__ == "__main__":
    unittest.main()
