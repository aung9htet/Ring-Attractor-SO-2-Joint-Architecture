"""Static source/devel/install packaging compatibility contracts."""

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CatkinProgramInstallTests(unittest.TestCase):
    def test_every_installed_python_program_has_a_python3_shebang(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        programs = []
        for block in re.findall(
            r"catkin_install_python\((.*?)\n\)", cmake, flags=re.DOTALL
        ):
            programs.extend(
                token
                for token in re.findall(r"(?:^|\s)(src/[^\s)]+\.py)", block)
            )
        self.assertEqual(len(programs), 20)
        self.assertEqual(len(set(programs)), 20)
        for relative in sorted(programs):
            with self.subTest(path=relative):
                first_line = (ROOT / relative).read_bytes().splitlines()[0]
                self.assertEqual(first_line, b"#!/usr/bin/env python3")

    def test_config_launch_and_world_install_destinations_remain_registered(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("DIRECTORY src/config/", cmake)
        self.assertIn("${CATKIN_PACKAGE_SHARE_DESTINATION}/config", cmake)
        self.assertRegex(cmake, r"foreach\(dir launch worlds\)")
        self.assertIn("catkin_python_setup()", cmake)


class PythonSetupContractTests(unittest.TestCase):
    def test_setup_keeps_all_legacy_flat_modules_and_internal_packages(self):
        tree = ast.parse((ROOT / "setup.py").read_bytes(), filename="setup.py")
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "LEGACY_MODULES"
        }
        modules = assignments["LEGACY_MODULES"]
        self.assertEqual(len(modules), 21)
        self.assertEqual(len(set(modules)), 21)
        for required in (
            "analysis_single_joint",
            "ring_attractor",
            "ring_component",
            "tiago_controller",
            "train_ring_model",
        ):
            self.assertIn(required, modules)

        setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
        for package in (
            '"tiago_ring_controller"',
            '"tiago_ring_controller.math"',
            '"tiago_ring_controller.nest"',
            '"tiago_ring_controller.training"',
            '"tiago_ring_controller.control"',
            '"tiago_ring_controller.ros"',
            '"tiago_ring_controller.evaluation"',
            '"builders"',
            '"builders.single_joint"',
            '"builders_analysis"',
            '"trainer"',
        ):
            self.assertIn(package, setup_text)


if __name__ == "__main__":
    unittest.main()
