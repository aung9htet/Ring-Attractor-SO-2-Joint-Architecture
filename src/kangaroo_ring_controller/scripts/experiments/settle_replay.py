"""Replay saved reaches' spikes through TIAGo's DriveControlCore and report the tick each settles (F14).

    python3 src/kangaroo_ring_controller/scripts/experiments/settle_replay.py <ring_reach/.../trial.json> ...

Settle rule (control/controller.py:193): |delayed drive| < 5 for n_settle = 10 consecutive ticks, counted from tick 1.
Prints the recorded tick count, the replayed settle tick, and the replay with delay 0 and with delay 0 and tau 0.05 s:
if only the decoder's delay/filter makes a reach settle early, the last two do not settle at 10.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the kangaroo_ring_controller package
from kangaroo_ring_controller.arm import DT  # noqa: E402
from tiago_ring_controller.control.controller import DecoderParameters, DriveControlCore  # noqa: E402


def settle_tick(spikes, scale, gain_positive, gain_negative, tau, delay):
    """1-based tick at which the core reports settled, or None."""
    core = DriveControlCore(DT, DecoderParameters(gain_positive, gain_negative, tau, delay), spike_scale=scale)
    return next((i + 1 for i, s in enumerate(spikes) if core.step(s).is_settled), None)


for path in sys.argv[1:]:
    t = json.loads(Path(path).read_text())
    spikes = t['signed_spike']
    scale = next((d / s for d, s in zip(t['signed_drive'], spikes) if s), 1.0)
    gains = t['decoder_gain_positive'], t['decoder_gain_negative']
    print(f"{Path(path).parent.name}: {len(spikes)} ticks, replay settles at "
          f"{settle_tick(spikes, scale, *gains, t['decoder_tau'], t['decoder_delay_steps'])}, "
          f"delay 0: {settle_tick(spikes, scale, *gains, t['decoder_tau'], 0)}, "
          f"delay 0 tau 0.05: {settle_tick(spikes, scale, *gains, 0.05, 0)}")
