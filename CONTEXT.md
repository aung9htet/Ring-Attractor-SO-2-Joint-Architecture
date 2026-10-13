# Ring-attractor joint control

A spiking SO(2) ring-attractor model that drives a robot arm joint toward a goal, first on TIAGo and now on PAL
Kangaroo. This glossary fixes the words used across the code, decisions and experiment notes.

## Ring

**State ring (r1)**:
The ring whose bump encodes the joint's position at the start of a trial.
_Avoid_: current ring, position ring

**Goal ring (r2)**:
The ring whose bump encodes the goal position.
_Avoid_: target ring

**Ring index**:
A neuron index on a ring, onto which a joint position is mapped linearly over the joint's range (not wrapped).
_Avoid_: angle, bin

**Signed drive**:
Right minus left gain-modulation spike counts in one control tick.
_Avoid_: error signal, spike difference

## Decoding

**Decoded velocity**:
The joint velocity (rad/s) obtained from the filtered, delayed signed drive through the decoder gains.
_Avoid_: commanded speed, ring velocity

**Decoder gains**:
The per-joint constants k_pos and k_neg (rad/s per spike) that turn drive into decoded velocity.
_Avoid_: velocity gain, scale

**Calibration**:
Fitting a joint's decoder gains (and τ, delay) to a specific robot and simulator.
_Avoid_: tuning

**Settled**:
The drive has stayed below its threshold for the required number of consecutive ticks; the trial stops.
_Avoid_: converged, reached

## Robot

**Velocity control**:
Moving one joint at a decoded or given velocity by sending a short receding-horizon position trajectory
integrated from that velocity.
_Avoid_: velocity interface, torque control

**Command state**:
The joint positions the publisher integrates its trajectories from.
_Avoid_: target, setpoint

**Reference**:
The positions the arm's trajectory controller is currently holding.
_Avoid_: desired, command

**Measured position**:
The joint position reported in joint states; under gravity it differs slightly from the reference.
_Avoid_: actual, feedback

**Trial**:
One ring-driven reach of one joint from its measured start toward one goal, with one seed.
_Avoid_: episode, run
