#!/usr/bin/env python3

from catkin_pkg.python_setup import generate_distutils_setup
from distutils.core import setup


# The flat research scripts live in ``legacy/`` (frozen, not installed).  Only
# the importable package is distributed; executables come from ``scripts/``.
setup_args = generate_distutils_setup(
    packages=[
        "tiago_ring_controller",
        "tiago_ring_controller.math",
        "tiago_ring_controller.nest",
        "tiago_ring_controller.training",
        "tiago_ring_controller.control",
        "tiago_ring_controller.ros",
        "tiago_ring_controller.evaluation",
        "tiago_ring_controller.cosim",
    ],
    package_dir={"": "src"},
)

setup(**setup_args)
