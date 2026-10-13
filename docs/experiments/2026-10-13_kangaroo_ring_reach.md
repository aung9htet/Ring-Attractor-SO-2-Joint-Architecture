# 2026-10-13: Ring-attractor reach on Kangaroo, both arms at once (uncalibrated decoder)

## Hypothesis
The TIAGo single-joint ring loop (`single_joint_data_collector.py:219-369`), ported to ROS 2 on Kangaroo, drives
each arm joint toward a goal, and Kangaroo executes the decoded velocity faithfully. Both arms can be ring-driven
at the same time, each by its own ring.

## Config
- Code: `kangaroo/ring_reach.py`, `kangaroo/arm_velocity.py` (with the D13 reference-priming fix). Decisions
  D11-D15 in `docs/kangaroo/DECISIONS.md`.
- Base commit `0ecde6a` + uncommitted `kangaroo-migration` changes. Image `ros2-kangaroo:humble`, NEST
  `HEAD@41892a50f`, Python 3.10, PAL 4-DoF Kangaroo, floating base, MuJoCo 3.12.0.
- Ring: `SingleRingModel` N = 200, `COLLECTOR_PROFILE` (50 ms tick, lookahead 4, settle |d| < 5 for 10 ticks,
  max 400). Decoder uncalibrated: k = ±1e-4, τ = 0.3 s, delay 0 (owner decision Q2 = Kangaroo-native).
- Seed: NEST `rng_seed` 12345 (both runs).

## Command
Inside the container, simulation running, two processes started together:
```bash
cd /work/src/tiago_ring_controller/src
python3 /work/kangaroo/ring_reach.py left 0 -0.8 &
python3 /work/kangaroo/ring_reach.py right 1 0.5 &
wait
```
Outputs: `kangaroo/results/ring_reach/left_j0_seed12345_20261013_080903/trial.json`,
`kangaroo/results/ring_reach/right_j1_seed12345_20261013_080915/trial.json`.

## Results
| Run | Ring index start → goal | q start → final (rad) | Goal | Δq | Σv·dt | Δq / Σv·dt | max abs v (rad/s) | Ticks | Stop |
|---|---|---|---|---|---|---|---|---|---|
| left j0 | 109 → 70 | +0.273 → +0.266 | −0.800 | −0.0067 | −0.0067 | 0.997 | 0.0046 | 69 | drive_settled |
| right j1 | 135 → 91 | +1.321 → +1.292 | +0.500 | −0.0288 | −0.0288 | 0.997 | 0.0074 | 233 | drive_settled |

Supporting test run (`kangaroo/test_ring_reach.py`): 3/3 OK, 175.7 s. The arm velocity test still passes 2/2.

## Interpretation
- **Direction:** both joints moved toward their goals, with both rings running concurrently in separate
  processes (D12).
- **Plant fidelity:** Kangaroo realised the decoded commands to 0.997 of Σv·dt. Before the D13 fix the same check
  was 0.0086 rad short (about 80 %), because of the P-only position loop's gravity offset.
- **Reach:** the joints covered only 0.6 % (left) and 3.5 % (right) of the distance to the goal before the drive
  settled. With k = 1e-4 the decoded speed never exceeded 0.0074 rad/s. This is the expected result of an
  uncalibrated decoder, not a failure of the loop. Reaching goals needs decoder calibration on Kangaroo (Q8).
- The left start (+0.273) is not the home pose (0.239): earlier test and diagnosis runs had moved it.

## Limitations
- One trial per arm, one seed: a demonstration, not statistics.
- Robot timing is not bit-reproducible (D4); ring spikes are, for a given seed and start/goal ring index.
- Floating base (D6) and PAL's unchanged P-only gains (Q7).
