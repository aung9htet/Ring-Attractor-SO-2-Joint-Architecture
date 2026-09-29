"""Assemble engines, TFs and the loop; map records to the legacy schemas.

``run_trial`` / ``run_session`` are the entry points the three legacy
workflows will delegate to once parity is shown (plan B phase 4).  Until
then the flat scripts are untouched; ``legacy_collector_record`` already
produces the collector's ``{"scalars", "timeseries", "raster"}`` dictionary
with the exact field sets frozen in ``evaluation.serialization``.

Engines that need NEST or ROS are built lazily in ``make_nest_engine`` /
``make_gazebo_engine``; importing this module needs neither.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..artifacts import save_json_legacy, save_npz_compressed_legacy
from ..config import load_ring_spec, source_config_path, source_root
from ..control.controller import decoder_features_from_spikes
from ..evaluation.serialization import (
    RING_RASTER_FIELDS,
    RING_TIMESERIES_FIELDS,
    RING_TRIAL_SCALAR_FIELDS,
    json_safe,
)
from .config import CosimConfig
from .engine import Engine
from .fakes import FakeNestEngine, FakeRobotEngine
from .loop import AnyOf, FTILoop, LoopObserver, MaxSteps, SettledFlag, StopCondition, TrialRecord
from .nest_engine import NestEngine, RingModelPorts
from .tf import GoalTF, MotorTF, ProprioceptionTF, TransceiverFunction


# -- assembly ---------------------------------------------------------------
def ring_population_size(config: CosimConfig) -> int:
    path = config.ring_params_file or source_config_path("model_params", "ring_params.json")
    return int(load_ring_spec(path).population_size)


def population_size_for(engines: Sequence[Engine], config: CosimConfig) -> int:
    for engine in engines:
        value = getattr(engine, "population_size", None)
        if value:
            return int(value)
    return ring_population_size(config)


def build_transceivers(
    config: CosimConfig, population_size: int, goal_rad: Optional[float] = None
) -> List[TransceiverFunction]:
    profile = config.control_profile
    joint_min, joint_max = config.require_limits()
    return [
        GoalTF(
            profile, joint_min, joint_max, population_size,
            requested_half_width=config.stimulus_half_width,
            goal_rad=goal_rad, mode=config.goal_mode,
        ),
        ProprioceptionTF(
            profile, config.joint_index, joint_min, joint_max, population_size,
            requested_half_width=config.stimulus_half_width,
            mode=config.proprioception_mode, rate_gain=config.proprioception_rate_gain,
        ),
        MotorTF(
            profile, config.joint_index, population_size, config.dt_ms,
            decoder=config.decoder, nest_lead_steps=config.nest_lead_steps,
            drive_threshold=config.drive_threshold, n_settle=config.n_settle,
        ),
    ]


def build_stop_condition(config: CosimConfig) -> StopCondition:
    # Legacy precedence: the settle check runs before the step budget.
    return AnyOf(SettledFlag(), MaxSteps(config.max_steps))


def build_loop(
    config: CosimConfig,
    engines: Sequence[Engine],
    population_size: Optional[int] = None,
    goal_rad: Optional[float] = None,
    observers: Optional[Sequence[LoopObserver]] = None,
) -> FTILoop:
    if population_size is None:
        population_size = population_size_for(engines, config)
    return FTILoop(
        engines,
        build_transceivers(config, population_size, goal_rad),
        dt_ms=config.dt_ms,
        nest_lead_steps=config.nest_lead_steps,
        lead_engine="nest" if config.nest_lead_steps > 0 else None,
        stop_condition=build_stop_condition(config),
        observers=observers,
    )


def make_fake_engines(
    config: CosimConfig,
    home: Optional[Sequence[float]] = None,
    script: Optional[Sequence[Any]] = None,
    population_size: Optional[int] = None,
) -> List[Engine]:
    size = population_size if population_size is not None else 100
    return [
        FakeNestEngine("nest", population_size=size, script=script),
        FakeRobotEngine("robot", home=home),
    ]


def make_nest_engine(
    config: CosimConfig,
    backend: Any = None,
    model_factory: Optional[Callable[[], RingModelPorts]] = None,
) -> NestEngine:
    """Real PyNEST engine around the unchanged ``SingleRingModel``."""

    if backend is None:
        import nest as backend  # noqa: WPS433 - lazy by design
    population_size = ring_population_size(config)
    if model_factory is None:
        src = source_root()
        if src not in sys.path:
            sys.path.insert(0, src)
        from single_ring import SingleRingModel  # noqa: WPS433 - legacy facade

        ring_params = config.ring_params_file or source_config_path("model_params", "ring_params.json")
        weights_dir = config.weights_dir or source_config_path("ring_decoding_weights")

        def model_factory() -> RingModelPorts:  # type: ignore[no-redef]
            model = SingleRingModel(
                ring_params_file=ring_params,
                weights_dir=weights_dir,
                seed=config.rng_seed,
                local_num_threads=config.local_num_threads,
            )
            return RingModelPorts.from_single_ring_model(model)

    return NestEngine(
        backend, model_factory, step_mode=config.nest_step_mode, population_size=population_size
    )


def make_gazebo_engine(config: CosimConfig, transport: Any = None) -> Engine:
    from .gazebo_ros_engine import GazeboRosEngine, RospyTransport

    if transport is None:
        transport = RospyTransport()
    return GazeboRosEngine(
        transport,
        stepper_name=config.stepper,
        max_step_size_s=config.max_step_size_s,
        step_timeout_s=config.step_timeout_s,
    )


# -- trials -----------------------------------------------------------------
def _find(loop: FTILoop, kind: type) -> Optional[Any]:
    for item in list(loop.engines) + list(loop.tfs):
        if isinstance(item, kind):
            return item
    return None


def run_trial(
    loop: FTILoop,
    goal_rad: float,
    config: Optional[CosimConfig] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> TrialRecord:
    goal_tf = _find(loop, GoalTF)
    if goal_tf is None:
        raise ValueError("loop has no GoalTF")
    goal_tf.set_goal(goal_rad)

    trial_meta: Dict[str, Any] = {"goal_rad": float(goal_rad)}
    if config is not None:
        trial_meta["config"] = config.to_dict()
    if meta:
        trial_meta.update(dict(meta))

    record = loop.run_trial(reset=True, meta=trial_meta)

    record.meta["goal_ring_index"] = goal_tf.last_index
    proprio = _find(loop, ProprioceptionTF)
    record.meta["initial_ring_index"] = None if proprio is None else proprio.last_index
    nest_engine = _find(loop, NestEngine)
    if nest_engine is not None:
        record.meta["nest_hidden_ms"] = nest_engine.hidden_ms
        record.meta["nest_step_mode"] = nest_engine.step_mode
        record.meta["nest_readout_mode"] = nest_engine.readout_mode
    robot = _find(loop, Engine)
    step_log = getattr(robot, "step_log", None)
    for engine in loop.engines:
        step_log = getattr(engine, "step_log", None)
        if step_log:
            overshoot = np.array([step.overshoot_s for step in step_log]) * 1000.0
            record.meta["gazebo_overshoot_ms"] = {
                "mean": float(overshoot.mean()),
                "p95": float(np.percentile(overshoot, 95)),
                "max": float(overshoot.max()),
                "n": int(len(overshoot)),
            }
    return record


def trial_raster(loop: FTILoop) -> Dict[str, np.ndarray]:
    nest_engine = _find(loop, NestEngine)
    if nest_engine is None:
        return {key: np.array([], dtype=float if key.endswith("times") else int) for key in RING_RASTER_FIELDS}
    return nest_engine.raster()


# -- legacy artifact mapping ----------------------------------------------
def legacy_collector_record(
    record: TrialRecord,
    config: CosimConfig,
    raster: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Map a :class:`TrialRecord` to the collector's ``run_trial`` result.

    Row ``k`` pairs the sample consumed by the horizon emitted at tick ``k``
    with the robot state measured after that tick's advance, which is exactly
    what the legacy loop logs after its ``rospy.sleep``.
    """

    joint = config.joint_index
    dt_s = record.dt_ms / 1000.0
    goal = float(record.meta["goal_rad"])
    main = record.main_ticks
    if not main:
        raise ValueError("trial record has no main ticks")

    q_start = float(main[0].inputs["joint_state"]["positions"][joint])
    dq_desired = goal - q_start
    movement_direction = float(np.sign(dq_desired)) if abs(dq_desired) > 1e-9 else 0.0

    hist: Dict[str, List[float]] = {key: [] for key in RING_TIMESERIES_FIELDS}
    for index, tick in enumerate(main):
        cmd = tick.outputs.get("arm_velocity_cmd")
        if cmd is None:
            continue
        sample = cmd["consumed"]
        after = main[index + 1].inputs if index + 1 < len(main) else record.end_state
        state = after["joint_state"]
        position = float(state["positions"][joint])
        velocity = float(state["velocities"][joint])
        bump = sample["r1_bump_index"]
        hist["time"].append(tick.t_ms / 1000.0 + dt_s)
        hist["joint_position"].append(position)
        hist["joint_velocity"].append(velocity)
        hist["signed_spike"].append(float(sample["signed_spike"]))
        hist["left_gain_spikes"].append(int(sample["left"]))
        hist["right_gain_spikes"].append(int(sample["right"]))
        hist["r1_spike_count"].append(float(sample["r1_spike_count"]))
        hist["r1_bump_index"].append(-1 if bump is None else int(bump))
        hist["r1_centroid"].append(float(sample["r1_centroid"]))
        hist["filtered_drive"].append(float(sample["filtered_drive"]))
        hist["delayed_drive"].append(float(sample["delayed_drive"]))
        hist["decoded_velocity"].append(float(sample["decoded_velocity"]))
        hist["position_error"].append(goal - position)

    q_final = float(record.final_state["joint_state"]["positions"][joint])
    raw_err = goal - q_final
    features = decoder_features_from_spikes(
        np.asarray(hist["signed_spike"], dtype=float), dt_s,
        config.decoder_tau_s, config.decoder_delay_steps,
    )
    S_pos = float(features["S_pos"])
    S_neg = float(features["S_neg"])
    dq_pred = config.decoder_gain_positive * S_pos + config.decoder_gain_negative * S_neg

    scalars = {
        "q_start": q_start, "q_goal": goal, "q_final": q_final,
        "dq_desired": float(dq_desired), "dq_actual": float(q_final - q_start),
        "decoder_predicted_dq_rad": float(dq_pred),
        "raw_position_error_rad": float(raw_err),
        "abs_position_error_rad": float(abs(raw_err)),
        "aligned_error_rad": float(raw_err * movement_direction),
        "S_raw": float(np.sum(hist["signed_spike"]) * dt_s),
        "S_filtered": float(np.sum(hist["filtered_drive"]) * dt_s),
        "S_delayed": float(np.sum(hist["delayed_drive"]) * dt_s),
        "S_pos": S_pos, "S_neg": S_neg,
        "S_vel": float(np.sum(hist["decoded_velocity"]) * dt_s),
        "n_steps": len(hist["time"]),
        "stop_reason": record.stop_reason,
        "goal_ring_index": record.meta.get("goal_ring_index"),
        "initial_ring_index": record.meta.get("initial_ring_index"),
        "decoder_gain_positive": config.decoder_gain_positive,
        "decoder_gain_negative": config.decoder_gain_negative,
        "decoder_tau": config.decoder_tau_s,
        "decoder_delay_steps": config.decoder_delay_steps,
    }

    timeseries = {
        "time": np.array(hist["time"]),
        "joint_position": np.array(hist["joint_position"]),
        "joint_velocity": np.array(hist["joint_velocity"]),
        "signed_spike": np.array(hist["signed_spike"]),
        "left_gain_spikes": np.array(hist["left_gain_spikes"]),
        "right_gain_spikes": np.array(hist["right_gain_spikes"]),
        "r1_spike_count": np.array(hist["r1_spike_count"]),
        "r1_bump_index": np.array(hist["r1_bump_index"], dtype=float),
        "r1_centroid": np.array(hist["r1_centroid"], dtype=float),
        "filtered_drive": np.array(hist["filtered_drive"]),
        "delayed_drive": np.array(hist["delayed_drive"]),
        "decoded_velocity": np.array(hist["decoded_velocity"]),
        "position_error": np.array(hist["position_error"]),
    }
    if raster is None:
        raster = {
            key: np.array([], dtype=float if key.endswith("times") else int)
            for key in RING_RASTER_FIELDS
        }
    raster_out = {key: np.asarray(raster[key]) for key in RING_RASTER_FIELDS}
    return {"scalars": scalars, "timeseries": timeseries, "raster": raster_out}


# -- sessions ---------------------------------------------------------------
class TrialWriter:
    """Writes the collector's artifact layout plus one cosim JSON per trial."""

    def __init__(self, out_dir: str, config: CosimConfig, num_iterations: Optional[int] = None) -> None:
        self.out_dir = out_dir
        self.trials_dir = os.path.join(out_dir, "trials")
        os.makedirs(self.trials_dir, exist_ok=True)
        save_json_legacy(
            os.path.join(out_dir, "session_meta.json"),
            {"joint_index": config.joint_index, "num_iterations": num_iterations,
             "cosim": True, "config": config.to_dict()},
        )
        self._file = open(os.path.join(out_dir, "trials_summary.csv"), "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=RING_TRIAL_SCALAR_FIELDS)
        self._writer.writeheader()

    def write(self, index: int, record: TrialRecord, legacy: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        self._writer.writerow({key: row[key] for key in RING_TRIAL_SCALAR_FIELDS})
        self._file.flush()
        save_npz_compressed_legacy(
            os.path.join(self.trials_dir, "trial_%04d_timeseries.npz" % index), **legacy["timeseries"]
        )
        save_npz_compressed_legacy(
            os.path.join(self.trials_dir, "trial_%04d_raster.npz" % index), **legacy["raster"]
        )
        with open(os.path.join(self.trials_dir, "trial_%04d_cosim.json" % index), "w", encoding="utf-8") as stream:
            stream.write(record.to_json(indent=1))

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def trial_row(config: CosimConfig, index: int, legacy: Mapping[str, Any]) -> Dict[str, Any]:
    row = {"joint_index": config.joint_index, "batch_idx": 1, "iteration_idx": index}
    row.update(legacy["scalars"])
    return row


def run_session(
    loop: FTILoop,
    config: CosimConfig,
    goals: Sequence[float],
    out_dir: Optional[str] = None,
    on_trial: Optional[Callable[[int, TrialRecord, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Run several trials, writing the collector's artifact layout if asked."""

    results: List[Dict[str, Any]] = []
    writer = TrialWriter(out_dir, config, len(goals)) if out_dir is not None else None
    try:
        for index, goal in enumerate(goals, start=1):
            record = run_trial(loop, float(goal), config, meta={"iteration_idx": index})
            legacy = legacy_collector_record(record, config, trial_raster(loop))
            row = trial_row(config, index, legacy)
            if writer is not None:
                writer.write(index, record, legacy, row)
            results.append({"record": record, "legacy": legacy, "row": json_safe(row)})
            if on_trial is not None:
                on_trial(index, record, legacy)
    finally:
        if writer is not None:
            writer.close()
    return results


__all__ = [
    "build_loop",
    "build_stop_condition",
    "build_transceivers",
    "legacy_collector_record",
    "make_fake_engines",
    "make_gazebo_engine",
    "make_nest_engine",
    "population_size_for",
    "ring_population_size",
    "run_session",
    "run_trial",
    "trial_raster",
    "trial_row",
    "TrialWriter",
]
