# Kangaroo migration: decisions for review

Made overnight on 2026-10-13 while the owner was away. The goal: control each Kangaroo arm joint individually by
velocity, the same way the TIAGo controller does. Each entry gives the decision, why it was made, the
alternatives, and the evidence. Anything marked **REVIEW** is a judgement call to confirm.

## D1. Velocity is realised through PAL's trajectory controllers, as on TIAGo (REVIEW)
- **Decision:** a velocity command for one joint becomes a short `JointTrajectory` (positions integrated as
  `q += v·dt`). It is published to PAL's own `arm_{left,right}_controller` (`JointTrajectoryController`, position
  interface).
- **Why:**
  - This is exactly how TIAGo did it: `TiagoPublisher.publish_receding_trajectory` published to
    `/arm_controller/command`.
  - It runs on PAL's controllers and gains unchanged (position PID p=1000, `kangaroo_mujoco/config/pids.yaml`).
- **Alternative rejected for now:** the joints' `velocity` command interface. PAL ships all arm velocity PID
  gains as 0, so commanding velocity directly produces zero torque, and using it would mean changing PAL's
  parameters. Revisit if the research needs true velocity-level actuation.

## D2. The existing receding-horizon code is reused, not copied
`tiago_ring_controller.control.trajectory.build_receding_trajectory` is pure Python and imports no ROS. I
verified that it imports on the container's Python 3.10 without loading nest, rospy, matplotlib or scipy. The
new ROS 2 code only adds the transport.

## D3. Joints are addressed by Kangaroo's own index (REVIEW)
- `joint_idx` 0..3 means `arm_<side>_<idx+1>_joint`, the same convention TIAGo used for its own arm.
- Mapping the TIAGo calibration JSON joints 0-6 onto Kangaroo (open question Q2) is deferred to the step where
  the ring controller is connected. That is when the mapping changes behaviour.

## D4. Timing matches TIAGo, but is paced by simulation time
- `dt = 0.05 s` and a 4-point horizon, the same as TIAGo (`profiles.py`).
- The loop waits on `/clock` (simulation time) instead of `rospy.sleep`'s wall clock.
- This does **not** make runs bit-reproducible. Simulation and controllers run asynchronously at about 1x real
  time, so ROS message timing varies. Two identical tracking runs differed by up to 0.003 in the steady-state
  velocity ratio. Exact reproducibility would need lock-stepping the simulator with the controller, which is a
  later decision.

## D5. Trajectories are stamped 0, meaning "start on receipt"
- With `header.stamp = 0` the controller starts each horizon as soon as it arrives.
- A wall-clock or simulation stamp risks starting in the past or the future relative to the controller's
  clock.
- TIAGo used `rospy.Time.now()`. The behaviour is the same, because each horizon replaces the previous one.

## D6. The floating base is kept (REVIEW)
- PAL's 4-DoF variant has a floating base. The measured base height was a steady 0.88 m over 3666 samples, so
  the robot stands.
- A fixed base (open question Q5) would mean regenerating the MuJoCo model from the URDF, which PAL's
  pregenerated models do not cover.

## D7. The arm controllers start with the simulation
`docker/ros2_kangaroo/kangaroo_sim.launch.py` includes PAL's unchanged
`kangaroo_controller_configuration/arm_controller.launch.py` for both sides, the way TIAGo's simulation brings
up `arm_controller`.

## D8. Code lives in `kangaroo/` at the repo root
Plain `rclpy` scripts, run inside the container from `/work`. There is no colcon package, because nothing needs
building or installing yet.

## D10. Known limit kept on purpose: no timeout if the simulator dies
`sleep_sim` waits for `/clock`. If the simulator crashes mid-run, the CLI, test or tracker waits until Ctrl-C.
A wall-clock guard is a few lines, to add if unattended runs need it.

## D9. The test checks behaviour in the running simulation
Expected values come from kinematics (`Δq = v·T`), not from the code under test:
- Commanding velocity `v` on one joint for time `T` moves that joint by `v·T` (±10%).
- It moves no other arm joint, on either arm, by more than 0.02 rad.
- This holds for all 8 arm joints individually, and for both arms commanded at once with different
  velocities.

## Overnight findings (evidence for the next decisions)

**F1. The ring model reproduces exactly on the ROS 2 image's Python.**
- Throwaway spike, not in the repo: NEST built from commit `41892a50f95d…` (the pinned `HEAD@41892a5`) on top
  of `ros2-kangaroo:humble` (Ubuntu 22.04, Python 3.10).
- `test_real_nest_smoke.py` passed 2/2. That covers the exact pinned spike events and the full
  comparator/gain/feedback dynamics.
- Caveats:
  - NEST reports the hash as `HEAD@41892a50f` (newer git abbreviates longer). The test's pinned-version
    constant was set to the full string in memory only; the test file was not edited.
  - `colorcet` had to be pip-installed, because `single_ring.py` imports it.

**F2. The existing suite on Python 3.10.** 139 tests ran. Every failure is environmental or pre-existing:
- File modes are `0o664` vs the pinned `0o644`, from this checkout's umask.
- Python 3.10 `ast` output differs from the captured 3.8.
- ROS 1 tool paths are absent.
- `src/outputs/` contains files that the golden manifest does not list. This was already true; nothing in
  this work touched `src/`.

**F3. Incident, fixed.**
- `run.bash` was missing `PYTHONDONTWRITEBYTECODE=1`, which the original launcher sets. Container runs wrote 13
  Python 3.10 `.pyc` files into the tracked `__pycache__` trees. They were git-ignored but would trip the
  bytecode tests.
- All 13 were removed, and the flag was added to `run.bash`. Verified: a fresh run writes 0 `.pyc` files.

## Ring controller connected (2026-10-13, morning)

Owner answers: **Q4** "yes, connect the ring controller now"; **Q2** "Kangaroo-native" (chosen in session);
test seams: all four offered. Everything below follows from those; **REVIEW** marks my own calls.

## D11. Ring parameters are Kangaroo-native (owner decision, Q2)
- Joint range = Kangaroo URDF limits (`ring_reach.py` `LIMITS`), mapped linearly onto ring indices 5..194
  (N = 200) with the collector's mapping (`COLLECTOR_PROFILE`, `profiles.py`).
- Decoder = `DecoderParameters()` defaults, k = ±1e-4, τ = 0.3 s, delay 0, which are exactly the collector's
  uncalibrated defaults (`single_joint_data_collector.py:47-50`).
- TIAGo's `velocity_calibration.json` is not read: its gains were fitted on TIAGo + Gazebo.

## D12. One ring per process; two arms = two processes
- `SingleRingModel.__init__` resets NEST's global kernel (`single_ring.py:163-167`), so one process can hold one
  ring. Both arms are driven at once by running two `ring_reach.py` processes (verified, see the experiment note).
- Rejected: two rings in one kernel. It would need changes to the existing model code (rule 5).

## D13. Commands start from the controller's reference, not the measured pose (REVIEW)
- **Found by the ring test** (red): measured Δq was 0.0086 rad short of Σv·dt.
- **Cause, measured:** PAL's position loop is P-only (p = 1000), so under gravity `arm_left_1_joint` sits
  +0.0085 rad off its reference (`/arm_left_controller/controller_state`: reference 0.25286, feedback 0.26135).
  Priming the command state from the measured pose re-introduced that offset on every `prime_command_state(force)`;
  holding v = 0 after a re-prime drifted +0.0082..+0.0085 rad, three times out of three.
- **Fix:** `ArmVelocityPublisher` primes from the JTC's `reference.positions`. One change covers every caller (CLI,
  tracking, both tests, `run_trial`).
- **Deviation from TIAGo:** `TiagoPublisher` primes from the measured pose. Whether TIAGo had the same offset is
  unknown (not checked against its Gazebo gains); no claim is made about the TIAGo code.
- **Evidence after the fix:** ring test 3/3, velocity test 2/2 (no regression); Δq / Σv·dt = 0.997 on both arms.
- The overnight velocity test passed with the bug because 0.0085 rad is 2 % of its 0.45 rad moves.

## D14. The robot loop keeps the collector's step order and stop rules
- Step order: publish horizon, sleep 50 ms of sim time, check settled, simulate one ring tick, shift
  (`single_joint_data_collector.py:262-339`). Lookahead 4, settle |d| < 5 for 10 ticks, max 400 ticks
  (`COLLECTOR_PROFILE`).
- `decoded_velocity` is recorded at publish time, i.e. the velocity the command state actually advanced by
  (`trajectory.py:62-64`), so Σv·dt is the exact commanded displacement.
- Seed: NEST `rng_seed` = 12345 by default (`--seed`), as `single_ring.py:1120`; no numpy seed because nothing in
  this path draws numpy randoms (the goal is given, not sampled as in collector:437). Seed and NEST version are
  saved in each `trial.json`.
- NEST runs between sim-time sleeps, as it did between `rospy.sleep`s on TIAGo, so robot timing is not
  bit-reproducible (D4). Ring spikes are: the ring sees the robot only once, at trial start.

## D15. Tests for the ring loop (owner chose the seams)
- Same seed + same start/goal gives identical spikes; a different seed gives different spikes (guards against a
  silently ignored seed).
- On the robot: the joint moves toward the goal by more than 0.01 rad; measured Δq = Σv·dt within ±10 % (floor
  0.005 rad); no other arm joint on either arm moves more than 0.02 rad.
- Mutants killed: the spike sign flipped (joint moved +0.073 rad, away from the goal) and D13 reverted (0.0086 rad
  short). Red first: `ModuleNotFoundError: No module named 'ring_reach'`.

## Decoder calibration on Kangaroo (2026-10-13, late morning)

Owner answers: **Q8** "go on", interpreted as my recommendation (a), port TIAGo's calibration (**REVIEW**: say
if you meant otherwise); **Q9** "Calibration profile for both"; **Q10** "One joint per arm".

## D16. Calibration is a port of TIAGo's, reusing its fit unchanged
- `kangaroo/calibrate.py` reproduces `TiagoCalibration.calibration` as its `main()` runs it
  (`calibrate_single_joint.py:1331-1386`): 100 trials, one region (the full joint range), a fit every 3 trials,
  blend β = 0.5, pose reset before every trial.
- Fit and blend are the existing pure functions `fit_decoder_from_trials` and `update_decoder_parameters`
  (`tiago_ring_controller/training/calibration.py`), not copies: τ/delay grid search, then regularised least
  squares dq_desired ≈ k_pos·S_pos + k_neg·S_neg, best by desired-displacement MSE.
- Not ported: plots, per-trial config writes, the "full-region refit if no mini-batch fit succeeded" fallback
  (unreachable with 100 trials and fits every 3), manual joint-limit search (URDF limits are used, D11).

- **Run length (owner decision):** about 4 min per trial (rough: 4 trials in about 17 min, two arms in parallel),
  so about 6-7 h for 100 trials; my earlier estimate of 1.5 h left out the per-trial network rebuild and reset.
  Owner: "Let it run", keeping TIAGo's 100-trial protocol unchanged.
- **Observed:** the robot looks still early on because the first trials use the uncalibrated gains (moves of
  0.000-0.018 rad); after the first fit, moves are 0.13-0.27 rad. The right arm's first goals all lay below home,
  so its first fit had no positive-drive data and returned k_pos = 0 (the existing fit's behaviour, unchanged).

## D17. Calibration and calibrated reaches use the same profile (owner decision, Q9)
- **Finding (F4):** TIAGo's three workflows use different profiles (`profiles.py`): calibration scales the signed
  drive by 100/N (0.5 at N = 200) and maps with a 10 % edge margin; the collector and analysis use scale 1 and the
  full range, and analysis doubles the bump half-width. Gains fitted under one profile drive about 2× faster under
  another. Reported, not changed; it may be deliberate and may bear on KB Q6.
- On Kangaroo, `calibrate.py` fits under `CALIBRATION_PROFILE`, and `ring_reach.py --calibration <file>` runs under
  the profile recorded in that file (asserted). Without `--calibration`, `ring_reach.py` keeps the collector
  profile and uncalibrated gains, so earlier results and tests are unchanged.
- `run_trial` now records `signed_drive` (spike × profile scale), which is what the fit consumes, as TIAGo's
  `spike_history` was already scaled (`calibrate_single_joint.py:452`).

## D18. Pose reset and seeds (REVIEW)
- TIAGo reset with play_motion `tiago_experiment_start_1`. Kangaroo's simulation has no arm play_motion, so
  `reset_arm` sends one trajectory point to PAL's home pose (2 s) through the same arm controller and waits 1 s.
  Only the calibrated arm is reset. New helper: `ArmVelocityPublisher.move_to`.
- Rule 14: trial i uses NEST seed `seed + i`, goals come from `numpy.random.default_rng(seed)` (default seed 2026).
  TIAGo's calibration was unseeded (fresh `SingleRingModel()` per trial, `np.random.uniform`).
- dq_desired uses the measured start position, as on TIAGo (`calibration_step`); the command state is primed from
  the controller reference (D13).
- **Reset fix (found in evaluation):** 2 of 20 evaluation resets were ignored, in fresh processes right after a trial
  that ended on max_steps (reproduced 1 of 3; cause unknown). `reset_arm` now checks the controller reference reached
  home (within 1e-3 rad) and resends, up to 3 attempts, else raises. With it: 4 of 4. Calibration was not affected:
  all 200 calibration trials started from home (q_start constant to 0.001 rad).

## D19. Calibration test
- Oracles: PAL's home pose (reference within 1e-3 rad, measured within 0.02 rad after reset), and physics of the
  sign convention: positive drive must move the joint up, so a calibrated decoder has k_pos > 0 > k_neg; also
  fitted τ and delay inside TIAGo's grid, goals inside the joint limits.
- Red first: `ModuleNotFoundError: No module named 'calibrate'`. Green: 2/2 (343 s, 6 trials, seed 7).
- **Insight (F5):** the fit regresses on ring features only. The ring sees the robot once per trial, and the pose is
  reset to the same home each trial, so the robot enters the fit only through q_start; it matters for evaluating
  the gains (actual error), not for fitting them.

## Calibration results (100 trials per arm, seed 2026): findings
Full numbers: `docs/experiments/2026-10-13_kangaroo_calibration.md`.
- **F6. Decoder sign flips.** 9 of 200 trials moved the wrong way. In every one, the decoder in use at the time had
  a wrong-signed gain (for example right trial 60 with k_pos = −3.22e-3; left trial 92 with k_neg = +2.11e-3).
  The ring's drive had the right sign (S_pos > 0, S_neg = 0 for positive goals), and the robot followed the
  prediction to within 0.006 rad in 8 of 9. My earlier hypothesis (|k_neg| ≫ k_pos) was **wrong**. The existing
  fit does not constrain sign; it only clips to ±1e-2 (`training/calibration.py:64-65`), and TIAGo's script only
  logs a warning (`calibrate_single_joint.py:751-761`). Unverified guess at the cause: 3-trial batches with goals
  on one side only leave the other gain undetermined.
- **F7. Convergence.** Left j0 converged partly: median relative error on goals over 0.6 rad fell from 0.52
  (trials 1-20) to 0.20 (81-100). Right j1 did not: 0.77 → 0.19 (41-60) → 0.84 (81-100).
- **F9. Right j1 is blocked at about 1.47 rad.** Commanded to 1.9 from home, the reference rose to 1.9 but the joint
  stopped at 1.473-1.477 with effort at the actuator limit (+43 Nm), then the controller aborted (`goal_time` 0.6 s,
  goal tolerance 0.02) and reset its reference to 1.477. The URDF and MJCF limit is 2.443, so this is an obstruction,
  most likely self-collision (body not identified). 22 of 100 calibration goals and 2 of 5 evaluation goals were
  above it. TIAGo found limits by driving to the stops (`find_joint_limits_manually`, `calibrate_single_joint.py:1064`)
  when none were configured; D11 used URDF limits instead.
- **F8. Dead zone.** No motion (|Δq| < 0.01 rad) in 24 (left) and 32 (right) of 100 trials, with desired moves
  up to 0.66 and 0.86 rad. Not a clean threshold: some moves of 0.09-0.16 rad did happen.

## Measured joint range (2026-10-13, afternoon; owner chose Q12 (b))
Done while the owner was away ("continue working and document decsion which I will audit"); all items below are
for audit. The machine rebooted at 13:51 BST; results on disk were intact and the simulation was restarted.

## D20. Joint range is measured by driving to the stops, as on TIAGo (REVIEW)
- **What:** new `kangaroo/joint_limits.py`, a port of `find_joint_limits_manually`
  (`calibrate_single_joint.py:1064-1086`): drive at −1 rad/s for 5 s, read the **measured** position (min), then
  +1 rad/s for 5 s (max). Writes `kangaroo/results/joint_limits/<side>_j<joint>_<time>/limits.json`.
  `ring_reach.py` and `calibrate.py` take `--limits limits.json`; without it they keep the URDF range (D11), so
  earlier runs are reproducible unchanged. A calibrated reach uses the range stored in its calibration file (the ring
  mapping the gains were fitted under) and refuses a different `--limits`.
- **Deviation from TIAGo (REVIEW):** TIAGo sweeps down then up from wherever the joint is. Here the arm is reset to
  PAL home before each sweep and at the end. Why: (1) the range is then the one calibration actually sees, since
  every calibration trial starts at home; (2) j0/j2 span 5.24 rad, which a 5 rad up-sweep from the lower stop
  cannot cross. Rejected: longer sweeps (changes TIAGo's numbers for no gain); TIAGo's order unchanged (would
  under-measure j0/j2 max).
- **Measured raw value, no margin** (as TIAGo). Rejected: shrinking by a safety margin; TIAGo used none.
- **Results** (`results/joint_limits/`, UTC times): right j1 [−1.1396, +1.4623] rad, twice, identical to 4 decimals
  (`right_j1_20261013_125903`, `_125954`). Left j0 [−2.6735, +2.6714] (`left_j0_20261013_125925`).
- **F10. Measured limits pass the URDF limits by up to 0.055 rad** (left j0 ±2.618 URDF; right j1 −1.134 URDF).
  Probably MuJoCo's soft joint-limit constraint yielding under the 1 rad/s drive (unverified). Not acted on: left j0
  is not recalibrated, and on right j1 the excess is 0.006 rad.
- **Test** (`kangaroo/test_joint_limits.py`): right j1 measured from home satisfies min < home < max, both within the
  URDF range ±0.02, and max < 1.6 (independent oracle: the F9 stall at 1.473-1.477). Red first:
  `ModuleNotFoundError: No module named 'joint_limits'`. Green 1/1 (22 s). Mutant (read the commanded instead of the
  measured position) killed: min −3.679.

## D21. Right j1 recalibrated over the measured range
- Same protocol, seed and trial count as the first run (D16-D18: seed 2026, 100 trials, β = 0.5, fit every 3,
  `CALIBRATION_PROFILE`), only the range changes: `--limits right_j1_20261013_125903/limits.json`. Why the same seed:
  the only intended difference to the first calibration is the range. Note: the same seed gives the same uniform
  draws, rescaled onto the new range, so goals differ in value but not in their position within the range.
- Q11 (sign flips) is still open; this run keeps TIAGo's protocol, so sign flips can recur.
- Left j0 is not recalibrated: its measured range is the URDF range to within 0.055 rad (F10).

## D22. Held-out evaluation of the right j1 recalibration (design fixed before the run)
- Goals, all inside the measured range, from home: the three reachable goals of the first evaluation with their
  seeds (−0.8 seed 100, 0.0 seed 101, 0.6 seed 102), plus −0.4 (seed 105) and 1.0 (seed 106), replacing the
  unreachable 1.8 and 2.3. Chosen to cover the reachable range below home (1.321); none tuned on results.
- Conditions: new calibration (primary), first calibration (URDF range) on the two new goals (the other three
  already exist), uncalibrated (URDF range, as the first evaluation's baseline) on the two new goals.
- Primary measure: mean |error| and median relative error, new vs first calibration, over the 5 goals.

## Recalibration results (D21/D22): findings
Full numbers: `docs/experiments/2026-10-13_kangaroo_joint_range_recalibration.md`.
- **Held-out (D22, design fixed beforehand), right j1:** mean |err| uncalibrated 1.170 / first calibration 0.557 /
  new calibration **0.147** rad; median relative error 0.943 / 0.427 / **0.158**. All five goals are below home.
- **F11. The new calibration's k_pos is wrong-signed (−7.80e-3).** Home (1.321) is 0.14 rad below the measured upper
  limit, so only 5 of 100 goals needed a positive move; 23 of 34 fits returned k_pos = 0 or the −0.01 clip. A post hoc
  positive reach (goal 1.45) went the wrong way (1.322 → 1.275). This confirms the F6 guess with direct evidence and
  makes Q11 blocking for right j1's positive direction. k_neg converged (−2.6e-3 to −3.4e-3 after trial 40).
- **Not done, on purpose:** I did not patch k_pos, refit with a sign constraint or rerun with different batches;
  each is a Q11 option and your call.

## D23. Balanced calibration goals (owner decision, Q11 (d))
- **What:** `calibrate(..., balanced=True)` / `--balanced`: trial i draws its goal uniformly from (home, max) when i is
  even and from (min, home) when i is odd, with the same seeded rng. With fit_every = 3, every fit batch contains
  both directions. Recorded as `balanced` in `calibration.json`. Default False keeps TIAGo's uniform draw, so all
  earlier runs reproduce unchanged. **Deviation from TIAGo** (owner-approved).
- **Why alternate, not shuffle:** alternation guarantees both signs in every 3-trial batch; a shuffled 50/50 split
  does not. Rejected: per-batch quotas (more code, same effect at fit_every = 3).
- **Why it works despite the dead zone (F8):** the fit regresses on desired Δq (F5), so a positive goal identifies
  k_pos from the ring's drive even if the joint barely moves.
- **Test** (`test_calibrate.py`, added): right j1 over the measured range, 6 trials, seed 7, balanced → k_pos > 0 > k_neg,
  3 goals above home, all inside the range. Red: `TypeError: ... 'balanced'`. Green 1/1 (189 s). Mutant (balanced
  ignored) killed: k_pos = −0.0057, i.e. F11 reproduced. **Caveat seen in the green run:** the first fit still had
  both gains wrong-signed (k_pos −0.01, k_neg +0.002) from three small moves (+0.087, −0.254, +0.108 rad); the
  second fit corrected it. Balancing makes both gains identifiable; it does not by itself prevent a bad early fit.
- **Run:** right j1, seed 2026, 100 trials, `--balanced --limits right_j1_20261013_125903/limits.json`.

## D24. Held-out evaluation of the balanced calibration (design fixed before the run)
- Same as D22 (goals −0.8 / 0.0 / 0.6 / −0.4 / 1.0, seeds 100 / 101 / 102 / 105 / 106, from home), plus one positive
  goal, 1.45 (seed 107), so the k_pos direction is now part of the design. Comparison: balanced vs the unbalanced
  measured-range calibration (`right_j1_seed2026_20261013_140925`, whose 1.45 run already exists, post hoc in D22).
- Primary measure: mean |error| and median relative error over the six goals; also over D22's five for comparison.

## Balanced calibration results (D23/D24): findings
Full numbers: addendum of `docs/experiments/2026-10-13_kangaroo_joint_range_recalibration.md`.
- Final decoder correctly signed (k_pos +8.12e-3, k_neg −1.70e-3, τ 0.135); positive goal 1.45 approached the right
  way (1.386). But six-goal mean |err| 0.492 vs 0.151 for the unbalanced calibration; every reach stops about halfway.
- The low final k_neg is a mid-recovery snapshot after wrong-signed fits at trials 87 and 90.
- **F12. Positive goals near home produce downward motion** (26 of 49, up to 0.756 rad; k_pos > 0 in 22 of them).
  Hypothesis (unverified): moves under 0.135 rad are 0-10 ring neurons, inside the goal/state bump overlap, so the
  drive is not reliably signed. Root cause for right j1: PAL home is 0.14 rad under its reachable maximum.
- **Not done, on purpose:** no further reruns, no gain patching, no choice of which calibration is "the" right j1
  decoder. Both are on disk; see Q13.

## D25. Right j1 calibrates from a mid-range start pose (owner decision, Q13 (c); REVIEW)
- **Owner intent:** right j1 must be driven in both directions. From PAL home it has 0.14 rad upward (F12).
- **What:** `calibrate(..., start=pose)` / `--start-joint <rad>` replaces the calibrated joint's home value; the
  other joints stay at PAL home. `reset_arm(..., pose=)` resets to it, the balanced split is taken at the start
  value, and `calibration.json` records `start_pose`. `ring_reach.py --from-home --calibration` resets to the
  calibration's `start_pose` (calibrated reaches start where their gains were learned); files without it use home.
- **Value:** right j1 = 0.161 rad, the midpoint of the measured range [−1.1396, 1.4623] (about 1.3 rad each way).
  Reachable: the range sweep passed through it from home. `--balanced` kept on (both signs in every batch).
- **Deviation from TIAGo (REVIEW):** TIAGo calibrated every joint from one pose (`tiago_experiment_start_1`); here
  right j1 starts from its own pose. Left j0 keeps PAL home (room both ways: −2.9 / +2.4 rad).
- **Test** (`test_calibrate.py`, added): right j1, start 0.161, balanced, 6 trials, seed 7 → k_pos > 0 > k_neg, every
  q_start within 0.02 of 0.161, 3 goals above it. Red: `TypeError: ... 'start'`. Green 1/1 (188 s), both fits
  correctly signed. No mutant run: ignoring `start` puts q_start at 1.32, which the q_start check rejects directly.
- **Run:** seed 2026, 100 trials, `--balanced --start-joint 0.161 --limits right_j1_20261013_125903/limits.json`.

## D26. Held-out evaluation from the mid-range start (design and pass bar fixed before the run)
- Start 0.161 (reset via the calibration's `start_pose`); goals −0.8 / −0.4 / 0.0 / 0.6 / 1.0 / 1.45 with seeds
  100 / 105 / 101 / 102 / 106 / 107: moves −0.96 / −0.56 / −0.16 / +0.44 / +0.84 / +1.29 rad, three each way.
- Condition: the new calibration only, judged against the pass bar. Uncalibrated baseline dropped (decided before
  results): `ring_reach.py` can start uncalibrated reaches only from home, and uncalibrated right j1 reaches covered under 8 %
  of the way in every earlier evaluation (relative error 0.926-0.999); a start flag for it is not worth the code.
- **Pass bar:** every calibrated reach moves the right way, and median relative error < 0.5.

## Mid-range start results (D25/D26): findings
Full numbers: last addendum of `docs/experiments/2026-10-13_kangaroo_joint_range_recalibration.md`.
- Calibration `right_j1_seed2026_20261013_164230`: k_pos +2.59e-3, k_neg −8.41e-3, τ 0.115; 1 of 33 fits
  wrong-signed; wrong-direction trials 2 of 100 (31 from home, same goals and seed) → F12's root cause confirmed.
- **D26 pass bar met:** 6/6 reaches the right way, median relative error 0.283 (mean |err| 0.227). Upward reaches
  relative error 0.02-0.22; both downward reaches overshot (−0.8 → −1.135 at the stop; −0.4 → −0.897); goal 0.0
  moved only 0.013 rad.
- **F13. The fit's gain clip inflates k_neg.** `training/calibration.py:64-65` clips k to ±1e-2; k_neg fits at
  trials 87, 96, 99 hit −0.0100 and the β = 0.5 blend moved the decoder from about −0.003 to −0.0084. Reported, not
  changed (rule 6); relevant to Q11 (b)/(c).
- **Current right j1 decoder:** this calibration, used from its `start_pose` (j1 = 0.161).

## Decisions waiting for you (asked when the step needs them)
- **Q11** answered (d), see D23-D24. **Q13** answered (c), see D25-D26.
- **Q13. Right j1: which calibration, and what next?** Neither works in both directions. Options:
  (a) use the unbalanced measured-range calibration (accurate below home, 0.151; upward range from home is only
  0.14 rad anyway) and document upward moves as unsupported; (b) use the balanced one (both directions, about half
  distance); (c) calibrate right j1 from a start pose with room both ways (for example j1 at mid-range), deviating
  from PAL home only for this joint; (d) rerun balanced with more trials/seeds to see whether it settles (cost about
  1 h per run). Recommended: (a) now, (c) if upward right j1 moves matter for your experiments. Your call.
- **Q11. Fix the calibration's sign flips?** Options: (a) keep TIAGo's protocol exactly (current; results above);
  (b) reject fits with k_pos ≤ 0 or k_neg ≥ 0 (TIAGo only warns); (c) larger batches (fit_every > 3). (b) and (c)
  deviate from the TIAGo method, so they change what the Kangaroo results mean; your call.
- **Q12. Joint range for the ring mapping.** (answered (b), see D20-D22) D11 (owner Q2) uses URDF limits, but right j1 cannot reach 27 % of its
  URDF range from home (F9). Options: (a) keep URDF limits; (b) measure each joint's reachable range by driving to
  the stops, as TIAGo's `find_joint_limits_manually` did, store it in a new Kangaroo limits file, and recalibrate
  right j1 (about 3.5 h). Recommended (b): it is TIAGo's method and removes unreachable goals.
- **Q8. Calibrate the decoder on Kangaroo?** (answered "go on", see D16-D19) With uncalibrated gains the ring drives each joint the right way
  but only 1–4 % of the way to the goal before the drive settles (experiment note 2026-10-13_kangaroo_ring_reach).
  On TIAGo, `calibrate_single_joint.py` (1386 lines) fitted k_pos/k_neg (and τ/delay) by least squares,
  y ≈ k_pos·S_pos + k_neg·S_neg (`:701-716`), with its own `CALIBRATION_PROFILE` (different mapping and spike
  scale). Options:
  - (a) Port that calibration as a new Kangaroo module writing a new calibration file (recommended: it is the
    TIAGo method, and the paper's numbers depend on it).
  - (b) Stay uncalibrated: the plumbing is validated, but goals are not reached.
  - Note: because the ring sees the robot only at trial start, S_pos/S_neg do not depend on the robot, so a fit
    could even be done offline from spikes alone. Whether that is legitimate depends on what y is (desired or
    actual Δq) in your method; that is your call.
- **Q5. Fixed base?** The floating base stands steadily (D6). Only needed if base motion contaminates
  results.
- **Q7. Overshoot.** Steps overshoot by +29% to +89% (see the experiment note). Accept PAL's P-only gains
  (current, unchanged), or allow tuning in our own parameter file? Tuning would be a deliberate deviation
  from PAL defaults.
