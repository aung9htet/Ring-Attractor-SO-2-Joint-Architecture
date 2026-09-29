"""Legacy-compatible, side-effect-free configuration access.

Source-space paths are used exactly as supplied.  In an installed Catkin space,
a missing legacy path containing a ``config`` component may fall back to the
package's installed share directory.  This install-only fallback does not alter
the historical source/CWD lookup or its exception behavior.  Unknown fields are
retained on the returned contracts through ``raw``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple

from .contracts import (
    DecoderSpec,
    GainSpec,
    HomeostasisSpec,
    JointCalibration,
    MultiRingSpec,
    NeuronSpec,
    PIDSpec,
    RingSpec,
)


_PACKAGE_NAME = "tiago_ring_controller"


def installed_package_share(module_file: Optional[str] = None) -> Optional[str]:
    """Return the Catkin install-space share directory, if detectable.

    Detection is deliberately structural and local: source/devel modules do not
    opt into the install fallback merely because a workspace happens to contain
    an ``install`` directory.
    """

    module_path = os.path.realpath(os.fspath(module_file or __file__))
    package_dir = os.path.dirname(module_path)

    # Installed Python package:
    #   <prefix>/lib/pythonX/{dist,site}-packages/tiago_ring_controller/*.py
    site_packages_dir = os.path.dirname(package_dir)
    python_dir = os.path.dirname(site_packages_dir)
    lib_dir = os.path.dirname(python_dir)
    if (
        os.path.basename(package_dir) == _PACKAGE_NAME
        and os.path.basename(site_packages_dir) in ("dist-packages", "site-packages")
        and os.path.basename(python_dir).startswith("python")
        and os.path.basename(lib_dir) in ("lib", "lib64")
    ):
        return os.path.join(
            os.path.dirname(lib_dir), "share", _PACKAGE_NAME
        )

    # Installed legacy flat ``py_module``:
    #   <prefix>/lib/pythonX/{dist,site}-packages/<legacy_module>.py
    flat_site_packages_dir = os.path.dirname(module_path)
    flat_python_dir = os.path.dirname(flat_site_packages_dir)
    flat_lib_dir = os.path.dirname(flat_python_dir)
    if (
        os.path.basename(flat_site_packages_dir) in ("dist-packages", "site-packages")
        and os.path.basename(flat_python_dir).startswith("python")
        and os.path.basename(flat_lib_dir) in ("lib", "lib64")
    ):
        return os.path.join(
            os.path.dirname(flat_lib_dir), "share", _PACKAGE_NAME
        )

    # Catkin-installed executable, including nested compatibility entrypoints:
    #   <prefix>/lib/tiago_ring_controller[/subdir]/*.py
    candidate_dir = os.path.dirname(module_path)
    while candidate_dir and candidate_dir != os.path.dirname(candidate_dir):
        if (
            os.path.basename(candidate_dir) == _PACKAGE_NAME
            and os.path.basename(os.path.dirname(candidate_dir)) in ("lib", "lib64")
        ):
            prefix = os.path.dirname(os.path.dirname(candidate_dir))
            return os.path.join(prefix, "share", _PACKAGE_NAME)
        candidate_dir = os.path.dirname(candidate_dir)

    return None


def module_config_path(module_file: str, *parts: str) -> str:
    """Build a flat-module config path in source and installed layouts."""

    share_dir = installed_package_share(module_file)
    if share_dir is not None:
        return os.path.join(share_dir, "config", *parts)
    return os.path.join(
        os.path.dirname(os.path.abspath(os.fspath(module_file))),
        "config",
        *parts
    )


def resolve_legacy_read_path(
    path: str, module_file: Optional[str] = None
) -> str:
    """Resolve a missing config input only when this package is installed.

    Existing paths always win, preserving CWD precedence.  If neither the
    supplied path nor a matching installed-share file exists, the original path
    is returned so the legacy loader raises the same exception for the same
    filename.
    """

    path_text = os.fspath(path)
    if os.path.exists(path_text):
        return path_text

    share_dir = installed_package_share(module_file)
    if share_dir is None:
        return path_text

    normalized_parts = os.path.normpath(path_text).split(os.sep)
    try:
        config_index = len(normalized_parts) - 1 - normalized_parts[::-1].index(
            "config"
        )
    except ValueError:
        return path_text

    candidate = os.path.join(
        share_dir,
        "config",
        *normalized_parts[config_index + 1 :]
    )
    return candidate if os.path.exists(candidate) else path_text


def source_root() -> str:
    """Return the historical flat ``src`` directory without reading it."""

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source_config_path(*parts: str) -> str:
    return os.path.join(source_root(), "config", *parts)


def legacy_root() -> str:
    """Return the frozen ``legacy/`` folder beside ``src`` (flat research scripts)."""

    return os.path.join(os.path.dirname(source_root()), "legacy")


def load_json(path: str) -> Any:
    """Load JSON using the same permissive parser as the legacy modules."""

    resolved_path = resolve_legacy_read_path(path)
    with open(resolved_path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def load_json_object(path: str) -> Dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object in {!r}".format(os.fspath(path)))
    return value


def load_ring_spec(path: str) -> RingSpec:
    p = load_json_object(path)
    return RingSpec(
        population_size=int(p["population_size"]),
        num_positions=int(p["num_positions"]),
        num_fourier_k=int(p["num_fourier_k"]),
        config_dir=str(p["config_dir"]),
        samples_per_position=int(p["samples_per_position"]),
        sim_settle_ms=float(p["sim_settle_ms"]),
        stimulus_half_width=int(p["stimulus_half_width"]),
        ridge_lambda=float(p["ridge_lambda"]),
        output_dir=str(p["output_dir"]),
        harmonic_index=int(p.get("harmonic_index", 0)),
        readout_weight_scale=float(p["readout_weight_scale"]),
        output_dc_baseline=float(p["output_dc_baseline"]),
        raw=dict(p),
    )


def load_neuron_specs(path: str) -> Dict[str, NeuronSpec]:
    document = load_json_object(path)
    return {
        str(name): NeuronSpec(str(name), dict(parameters))
        for name, parameters in document.items()
    }


def load_neuron_parameters(path: str, population: str) -> Dict[str, Any]:
    """Return a mutable dict, matching the value passed to ``nest.Create``."""

    return dict(load_json_object(path)[population])


def load_homeostasis_spec(path: str) -> HomeostasisSpec:
    p = load_json_object(path)
    return HomeostasisSpec(
        config_dir=str(p["config_dir"]),
        tie_epsilon=float(p["tie_epsilon"]),
        warm_cold_weight_scale=float(p["warm_cold_weight_scale"]),
        warm_cold_bias_scale=float(p.get("warm_cold_bias_scale", 0.0)),
        warm_exc_weight=float(p["warm_exc_weight"]),
        warm_inh_weight=float(p["warm_inh_weight"]),
        cold_exc_weight=float(p["cold_exc_weight"]),
        cold_inh_weight=float(p["cold_inh_weight"]),
        decision_lateral_weight=float(p["decision_lateral_weight"]),
        left_label=int(p["left_label"]),
        right_label=int(p["right_label"]),
        raw=dict(p),
    )


def load_gain_spec(path: str) -> GainSpec:
    p = load_json_object(path)
    return GainSpec(
        left_homeostasis_gain_weight=float(p["left_homeostasis_gain_weight"]),
        right_homeostasis_gain_weight=float(p["right_homeostasis_gain_weight"]),
        ring_to_gain_weight=float(p["ring_to_gain_weight"]),
        gain_to_ring_weight=float(p["gain_to_ring_weight"]),
        cross_inhibition_weight=float(p["cross_inhibition_weight"]),
        raw=dict(p),
    )


def load_multi_ring_spec(path: str) -> MultiRingSpec:
    p = load_json_object(path)
    population_size = int(p["population_size"])
    feature_grid_size = int(
        p.get(
            "signed_product_population_size",
            p.get("product_population_size", min(population_size, 32)),
        )
    )
    output_scale = float(
        p.get("signed_product_output_weight_scale", p.get("sfp_output_weight_scale", 1.0))
    )
    return MultiRingSpec(
        population_size=population_size,
        num_positions=int(p["num_positions"]),
        num_fourier_k=int(p["num_fourier_k"]),
        sim_settle_ms=float(p["sim_settle_ms"]),
        stimulus_half_width=int(p["stimulus_half_width"]),
        ridge_lambda=float(p.get("ridge_lambda", 1.0e-3)),
        output_ring_size=int(p.get("output_ring_size", 100)),
        output_rate_baseline=float(p.get("output_rate_baseline", 50.0)),
        output_rate_amplitude=float(p.get("output_rate_amplitude", 40.0)),
        num_joints=int(p.get("num_joints", 2)),
        joint_axes=tuple(str(axis) for axis in p.get("joint_axes", ["x", "y"])),
        max_train_samples=int(p.get("max_train_samples", 256)),
        max_test_samples=int(p.get("max_test_samples", 64)),
        n_vis_points=int(p.get("n_vis_points", 20)),
        feature_grid_size=feature_grid_size,
        signed_product_dc_baseline=float(p.get("signed_product_dc_baseline", 180.0)),
        signed_product_input_weight=float(p.get("signed_product_input_weight", 120.0)),
        signed_product_output_weight_scale=output_scale,
        output_dc_baseline=float(p.get("output_dc_baseline", 200.0)),
        scalar_ramp_weight_scale=float(p.get("scalar_ramp_weight_scale", 100.0)),
        scalar_dc_baseline=float(p.get("scalar_dc_baseline", 200.0)),
        raw=dict(p),
    )


def load_velocity_calibration_document(path: str) -> Dict[str, Any]:
    return load_json_object(path)


def joint_limits_from_document(
    document: Mapping[str, Any], joint_index: int
) -> Optional[Tuple[float, float]]:
    """Mirror the legacy loaders: malformed or absent limits yield ``None``."""

    try:
        entry = document[str(joint_index)]
        return float(entry["joint_min"]), float(entry["joint_max"])
    except (KeyError, TypeError, ValueError):
        return None


def load_joint_limits(path: str, joint_index: int) -> Optional[Tuple[float, float]]:
    try:
        return joint_limits_from_document(load_velocity_calibration_document(path), joint_index)
    except (OSError, ValueError, TypeError):
        return None


def decoder_from_entry(
    entry: Mapping[str, Any], defaults: Optional[DecoderSpec] = None
) -> DecoderSpec:
    """Apply current top-level decoder precedence and legacy gain fallback."""

    base = defaults if defaults is not None else DecoderSpec()
    required = (
        "decoder_gain_positive",
        "decoder_gain_negative",
        "decoder_tau",
        "decoder_delay_steps",
    )
    try:
        if all(key in entry for key in required):
            return DecoderSpec(
                gain_positive=float(entry["decoder_gain_positive"]),
                gain_negative=float(entry["decoder_gain_negative"]),
                tau=float(entry["decoder_tau"]),
                delay_steps=int(entry["decoder_delay_steps"]),
                source="decoder",
            )
        if "raw_drive_velocity_gain" in entry:
            gain = float(entry["raw_drive_velocity_gain"])
            return DecoderSpec(
                gain_positive=gain,
                gain_negative=-gain,
                tau=base.tau,
                delay_steps=base.delay_steps,
                source="raw_drive_velocity_gain",
            )
    except (TypeError, ValueError, OverflowError):
        # Legacy wrappers leave their constructor defaults in place.
        return base
    return base


def pid_from_entry(entry: Mapping[str, Any], defaults: Optional[PIDSpec] = None) -> PIDSpec:
    base = defaults if defaults is not None else PIDSpec()
    try:
        return PIDSpec(
            kp=float(entry.get("pid_kp", base.kp)),
            ki=float(entry.get("pid_ki", base.ki)),
            kd=float(entry.get("pid_kd", base.kd)),
            output_limit=float(entry.get("pid_output_limit", base.output_limit)),
            windup_limit=base.windup_limit,
        )
    except (TypeError, ValueError, OverflowError):
        return base


def joint_calibration_from_document(
    document: Mapping[str, Any],
    joint_index: int,
    decoder_defaults: Optional[DecoderSpec] = None,
    pid_defaults: Optional[PIDSpec] = None,
) -> JointCalibration:
    raw_value = document.get(str(joint_index), {})
    entry = raw_value if isinstance(raw_value, Mapping) else {}
    limits = joint_limits_from_document(document, joint_index)
    return JointCalibration(
        joint_index=int(joint_index),
        joint_min=None if limits is None else limits[0],
        joint_max=None if limits is None else limits[1],
        decoder=decoder_from_entry(entry, decoder_defaults),
        pid=pid_from_entry(entry, pid_defaults),
        raw=dict(entry),
    )


def load_joint_calibration(
    path: str,
    joint_index: int,
    decoder_defaults: Optional[DecoderSpec] = None,
    pid_defaults: Optional[PIDSpec] = None,
    strict: bool = False,
) -> JointCalibration:
    try:
        document = load_velocity_calibration_document(path)
    except Exception:
        if strict:
            raise
        document = {}
    return joint_calibration_from_document(
        document,
        joint_index,
        decoder_defaults=decoder_defaults,
        pid_defaults=pid_defaults,
    )
