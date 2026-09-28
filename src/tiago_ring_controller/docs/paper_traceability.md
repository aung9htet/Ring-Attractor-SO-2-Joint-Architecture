# Paper-to-code traceability

The paper `src/Neurips_2026.pdf`, “A Spiking Ring-Attractor Architecture for
Representation and Control of Revolute Robot Joints,” describes the scientific
intent. This table maps concepts to executable source. “Qualified” means the code
implements a discrete or partial analogue; it is not evidence that a particular
checked-in figure was produced by that script.

| Paper concept | Executable path | Assessment |
|---|---|---|
| Localized SO(2) state bump | `ring_attractor.py` | Strong ring-attractor correspondence. |
| Bump transport | shifted opponent feedback in `gain_modulation.py` and `single_ring.py` | Qualified discrete transport; no explicit Lie-algebra operator. |
| Positional Fourier phase features | `train_ring_model.py`, `ring_component.py` | Analytic positive/negative sine masks; evaluator uses clipped first-harmonic arcsine rather than a full Fourier reconstruction. |
| Push-pull representation | paired readout neurons and signed subtraction | Strong. |
| Homeostatic comparator | `train_homeostasis.py`, `homeostasis.py` | Ridge-fitted absolute phase ramp; shortest-arc wrap behavior is not established. |
| Opponent gain modulation | `gain_modulation.py` | Strong, with a code-specific ten-index feedback exclusion. |
| Signed control drive | robot workflows | Code uses `right-left`, matching the paper equation; the pre-refactor README text that stated the opposite sign is retained only as a documented historical discrepancy. |
| Product Fourier features | `compositional_fourier_decoder.py` | Thresholded/conjunctive signed NEST grids, not literal numerical multiplication. |
| Orientation output | multi-ring trainer and inference | Strong for exactly two rings with `Rx(q1) @ Ry(q2)` and lift/pitch/yaw output rings. |
| Receding-horizon Tiago control | `tiago_controller.py` | Strong. |
| Three-ring generality | appendix discussion | Not implemented by the current runtime, which rejects joint counts other than two. |

## Training terminology

The code contains no Nengo dependency and no implemented NEF ensemble, encoder,
intercept, gain distribution, evaluation-point object, NEF solver, transform, or
synapse object. It uses NEST spiking populations plus analytic masks or ordinary
NumPy ridge regression. Documentation therefore uses:

- “analytically derived” for ring Fourier and scalar ramp weights;
- “ridge-fitted” for homeostasis and multi-ring output weights;
- “calibrated” for DC/input/output selections and robot velocity gains;
- “fixed” or “manually selected” for neuron, recurrence, gain, timing, and PID
  constants;
- “NEF-inspired” only for the general feature/readout pattern.

No checked-in file is a gradient-training checkpoint.

## Figure provenance

The following associations are plausible from execution flow and mathematics, but
the repository contains no generation manifest proving them:

- ring/readout training and repeatability analyses resemble paper Figure 2;
- `demo_graphs.py` resembles Figure 3-style visualizations;
- `single_ring.py` and gain analyses resemble Figure 4-style dynamics;
- neural/PID collectors and `single_joint_data_visualizer.py` resemble robot-control
  result figures;
- compositional training, multi-ring inference, and scalar analysis resemble later
  orientation-decoder figures.

These associations must remain labelled as inferred until raw experiment IDs,
script versions, parameters, seeds, and hashes are recovered.

## Material paper/code qualifications

- The paper neuron table agrees with the checked-in ring/homeostasis/gain parameter
  JSON.
- The paper signed drive `R-L` agrees with executable robot code.
- Root and nested “training” scripts build analytic sine masks; they do not solve the
  ridge objective described in older documentation.
- Homeostasis biases and tie-label configuration are loaded but do not influence the
  current NEST topology.
- The multi-ring trainer ridge-fits measured spike features, an option allowed by
  the paper, but only for two x/y-ordered joint rings.
- Existing experiment data is insufficient to claim complete reproduction of the
  submitted robot figures.

Behavior-preserving work must characterize these differences. Resolving them is a
separate scientific change requiring new baselines.
