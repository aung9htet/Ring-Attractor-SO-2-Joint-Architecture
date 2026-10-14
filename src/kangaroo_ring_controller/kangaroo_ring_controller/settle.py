"""TIAGo's decoder core with settle counting armed by the first threshold crossing (DECISIONS.md F14, Q14 (b)).

TIAGo's rule (control/controller.py:193): settled after n_settle consecutive ticks of |d_delayed| < threshold, counted
from tick 1. A zero-filled delay queue and a slow filter (d = a·d + (1-a)·signed, a = exp(-dt/tau); tau 0.93 s gives
a = 0.948) keep |d_delayed| under the threshold for the first ticks of a reach, so it "settles" before moving.
Here the count starts only once |d_delayed| has reached the threshold; a reach whose drive never does runs to max_steps.
Filter, delay and decoded velocity are TIAGo's, unchanged.
"""
from dataclasses import replace

from . import ring_core  # noqa: F401  (TIAGo package on sys.path)
from tiago_ring_controller.control.controller import DriveControlCore


class GuardedDriveCore(DriveControlCore):
    def reset(self):
        super().reset()
        self.armed = False  # has |delayed drive| reached the threshold yet

    def step(self, raw_signed_spike):
        sample = super().step(raw_signed_spike)
        self.armed = self.armed or abs(sample.delayed_drive) >= self.drive_threshold
        if self.armed:
            return sample
        self.consecutive_settled = 0
        return replace(sample, consecutive_settled=0, is_settled=False)
