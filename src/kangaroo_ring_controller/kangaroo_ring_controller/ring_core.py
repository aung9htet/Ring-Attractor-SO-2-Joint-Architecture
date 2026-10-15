"""Where the shared ring-attractor core lives: inside the TIAGo package (ADR-0001, decision 3).

Importing this module puts src/tiago_ring_controller/src on sys.path, so `single_ring` and `tiago_ring_controller.*`
import unchanged. RING_PARAMS and RING_WEIGHTS are the absolute forms of SingleRingModel's default `./config/...`
paths (single_ring.py:154-155), so the ring builds from any working directory.
"""
import sys
from pathlib import Path

TIAGO_SRC = Path(__file__).resolve().parents[2] / 'tiago_ring_controller' / 'src'
RING_PARAMS = TIAGO_SRC / 'config' / 'model_params' / 'ring_params.json'
RING_WEIGHTS = TIAGO_SRC / 'config' / 'ring_decoding_weights'

if str(TIAGO_SRC) not in sys.path:
    sys.path.insert(0, str(TIAGO_SRC))
