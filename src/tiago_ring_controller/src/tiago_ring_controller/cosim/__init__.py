"""NRP-style fixed-time-increment co-simulation loop (plan B).

Engines own simulated time (:mod:`nest_engine`, :mod:`gazebo_ros_engine`,
:mod:`fakes`), datapacks carry state between them (:mod:`datapack`),
transceiver functions map datapacks (:mod:`tf`), and :class:`FTILoop`
(:mod:`loop`) advances everything in lock step with a documented one-step
delay.  Importing this package imports neither NEST nor ROS.
"""

from .config import CosimConfig
from .datapack import DataPack, DataPackError, make_datapack
from .engine import Engine, EngineClock, EngineError, EngineStateError, StepTimeoutError
from .fakes import FakeNestEngine, FakeRobotEngine
from .loop import AnyOf, FTILoop, LoopObserver, MaxSteps, SettledFlag, TickRecord, TrialRecord
from .nest_engine import (
    LegacyInjectStimulusPort,
    NestEngine,
    RingModelPorts,
    SpikeCountReader,
    StimulusNotSupportedError,
    StimulusPort,
)
from .stepping import ClockWaitStepper, GazeboStepper, PluginStepper, StepResult
from .tf import GoalTF, MotorSample, MotorTF, ProprioceptionTF, TickContext, TransceiverFunction

__all__ = [
    "AnyOf",
    "ClockWaitStepper",
    "CosimConfig",
    "DataPack",
    "DataPackError",
    "Engine",
    "EngineClock",
    "EngineError",
    "EngineStateError",
    "FTILoop",
    "FakeNestEngine",
    "FakeRobotEngine",
    "GazeboStepper",
    "GoalTF",
    "LegacyInjectStimulusPort",
    "LoopObserver",
    "MaxSteps",
    "MotorSample",
    "MotorTF",
    "NestEngine",
    "PluginStepper",
    "ProprioceptionTF",
    "RingModelPorts",
    "SettledFlag",
    "SpikeCountReader",
    "StepResult",
    "StepTimeoutError",
    "StimulusNotSupportedError",
    "StimulusPort",
    "TickContext",
    "TickRecord",
    "TransceiverFunction",
    "TrialRecord",
    "make_datapack",
]
