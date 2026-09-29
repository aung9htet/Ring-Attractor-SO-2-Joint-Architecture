# Executable architecture

## Compatibility rule

The current script paths, module names, classes, method signatures, private members
used across modules, ROS interfaces, configuration precedence, artifact contracts,
and output schemas are compatibility APIs. Refactored implementation may live in an
internal `tiago_ring_controller` package, but the existing modules remain facades.

Known inconsistencies are modeled as named behavior profiles. They are not fixed by
this refactor.

## Ring implementations

Two independent ring stacks exist:

| Stack | Ring | Readout | Main callers |
|---|---|---|---|
| Legacy/robot | `ring_attractor.Ring_Attractor` | `ring_component.RingAttractorComponent` | single-ring control, robot control, homeostasis, gain, multi-ring |
| Builder/analysis | `builders.ring_attractor.RingAttractor` | `builders.single_joint.ring_component` | builder analyses and nested trainer |

Both create individual `iaf_psc_alpha` neurons and spike recorders and include all
`N²` recurrent connections, including self-connections. They are not interchangeable:

- legacy recurrence keeps Gaussian widths 10 and 5 for every population size;
- builder recurrence divides those widths by `N/100`;
- odd-size distance arrays differ at their final element.

The characterization suite freezes each implementation separately.

## Single-ring neural control flow

```text
state angle ──> state ring r1 ───────────────┐
                                             │ same-index excitation
target angle ─> target ring r2               v
       │              │                 left/right gain populations
       └─ Fourier push-pull ─> warm/cold ─> decision neurons
                                             │ shifted inhibitory feedback
                                             └──────────────> state ring r1
```

`RingAttractorComponent` loads analytic positive/negative sine masks and creates
`2K` readout neurons. `HomeostasisModel` embeds fitted phase-ramp weights and drives
warm/cold/left/right comparator neurons. `GainModulationModel` combines decision and
state activity. `SingleRingModel` is the path that explicitly connects shifted gain
feedback to the state ring; `gain_modulation_analysis.py` intentionally measures a
feedforward-only model.

Exact decision wiring is:

- cold to right +20000 and cold to left -20000;
- warm to left +20000 and warm to right -20000;
- left and right inhibit one another with -20000.

Gain feedback is left[i] to ring[i+1] and right[i] to ring[i-1], both -0.6, only for
indices 5 through `N-6`.

## Robot control flow

The analysis, calibration, and collection entrypoints each couple `SingleRingModel`
to `TiagoPublisher` and `TiagoSubscriber`. A step reads cumulative recorder counts,
computes a window delta, forms `right-left`, applies an exponential filter, applies
an integer delay, splits by sign, applies asymmetric gains, and publishes a
receding trajectory.

Three legacy profiles must remain distinct:

| Profile | Spike scale | Joint/ring mapping | Injected half-width |
|---|---:|---|---:|
| Calibration | `100/N` | 10% edge margin | 5 |
| Analysis | 1 | full range with doubled effective margin | 10 |
| Collection | 1 | full range | 5 |

Trajectory messages always contain all seven arm joints. A four-point default
horizon advances the internal command state by only its first point.

On branch `cosim-loop`, `tiago_ring_controller.cosim` provides a lock-stepped
alternative harness around the same model and trajectory contracts (engines,
datapacks, transceiver functions, `FTILoop`; see the repository's `docs/cosim/`).
The three entrypoints are unchanged until the parity report in plan B phase 4.

## Training and fitting boundaries

- Ring Fourier masks are analytically derived, despite trainer naming.
- Homeostasis uses closed-form ridge regression with an intercept over measured
  NEST count features.
- Multi-ring readout uses ordinary primal/dual ridge regression over signed NEST
  grid features.
- Scalar ramp weights are analytic.
- Joint velocity calibration grid-searches filter/delay and solves clipped
  positive/negative gains, then applies an exponential moving update.
- There is no gradient optimizer, epoch loop, backpropagation, checkpoint resume,
  Nengo network, or NEF solver.

“NEF-inspired feature/readout construction” is the accurate architectural term.

## Multi-ring orientation flow

Two 100-neuron joint rings represent `q1` and `q2`, ordered as x then y rotations.
Sixteen signed feature populations encode positive/negative forms of:

```text
cos1, sin1, cos2, sin2,
cos1cos2, cos1sin2, sin1cos2, sin1sin2
```

Each feature population is a row-major 32×32 grid. Concatenation is feature-major
in the order above, positive then negative per term, yielding 16,384 features.
Three `(16384,100)` matrices drive lift, pitch, and yaw output rings. Stored matrices
are target-by-source and are transposed at the NEST connection boundary.

Kinematic targets use `Rx(q1) @ Ry(q2)`, then:

```text
pitch = -asin(R[2,0])
lift  = atan2(R[2,1], R[2,2])
yaw   = atan2(R[1,0], R[0,0])
```

The executable implementation supports exactly two joints.

## Side-effect boundaries

Compatibility scripts still combine NEST kernel resets, topology construction,
simulation, fitting, plotting, serialization, ROS transport, and logging. The
refactor provides the following internal, side-effect-controlled boundaries behind
those permanent facades:

- pure ring/circular/Fourier/kinematics/control mathematics;
- explicit NEST topology builders;
- pure analytic/ridge/calibration fitting;
- pure control-step and trajectory construction;
- ROS transport, logger, and camera adapters;
- data-only evaluation records, metrics, serializers, and plot consumers.

Legacy classes remain wrappers around these boundaries, including constructor side
effects and source/CWD-relative output behavior. Install-only path resolution is a
compatibility fallback and does not change source-space precedence.
