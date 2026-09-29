#!/usr/bin/env python3

from catkin_pkg.python_setup import generate_distutils_setup
from distutils.core import setup


LEGACY_MODULES = [
    "analysis_single_joint",
    "calibrate_single_joint",
    "compositional_fourier_decoder",
    "demo_graphs",
    "experiment_camera",
    "gain_modulation",
    "gain_modulation_analysis",
    "homeostasis",
    "multi_ring_component",
    "multi_ring_sawtooth_scalar_decoder",
    "readout_mean_std_analysis",
    "record_experiments",
    "ring_attractor",
    "ring_component",
    "single_joint_data_collector",
    "single_joint_data_visualizer",
    "single_joint_pid_data_collector",
    "single_ring",
    "tiago_controller",
    "train_homeostasis",
    "train_ring_model",
]


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
        "helpers",
        "builders",
        "builders.single_joint",
        "builders_analysis",
        "trainer",
    ],
    # The explicit module mappings work around catkin's Noetic devel-space
    # interrogation treating each ``py_module`` name like a package path.
    # Distutils still resolves every legacy module from the common ``src``
    # directory for install-space builds.
    package_dir=dict({"": "src"}, **{name: "src" for name in LEGACY_MODULES}),
    py_modules=LEGACY_MODULES,
)

setup(**setup_args)
