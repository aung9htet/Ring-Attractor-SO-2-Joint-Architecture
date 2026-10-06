# Technical reference: one ring attractor per joint (SO(2))

The mathematics behind every part of the controller, with the derivations. How to run
things is in [`README.md`](README.md); which part is which block and parameter is in
[`docs/architecture.md`](docs/architecture.md); what was measured is in
[`docs/results.md`](docs/results.md).

Two kinds of statement appear below and are kept apart:

- **Built**: what the code does, with the file it is in.
- **Model**: a simplified description (a few equations) that explains why the built thing
  behaves as it does. The network that runs is the spiking one; the models are for
  understanding and were checked against measurements where a number is given.

Paths are relative to `src/tiago_ring_controller/src/tiago_ring_controller/`.

## Contents

1. [The task and the notation](#1-the-task-and-the-notation)
2. [The neuron](#2-the-neuron)
3. [The ring attractor](#3-the-ring-attractor)
4. [Joint angle and ring index](#4-joint-angle-and-ring-index)
5. [Writing an angle: the encoder](#5-writing-an-angle-the-encoder)
6. [Reading an angle: the population vector](#6-reading-an-angle-the-population-vector)
7. [The Fourier readout](#7-the-fourier-readout)
8. [The ramp comparator](#8-the-ramp-comparator)
9. [The gain pair: moving a bump](#9-the-gain-pair-moving-a-bump)
10. [The decoder: spikes to joint velocity](#10-the-decoder-spikes-to-joint-velocity)
11. [Architecture v3: two rings, open loop](#11-architecture-v3-two-rings-open-loop)
12. [Architecture v4: three rings, closed loop](#12-architecture-v4-three-rings-closed-loop)
13. [Time: ticks, delay, time scale](#13-time-ticks-delay-time-scale)
14. [The PID baseline](#14-the-pid-baseline)
15. [How a trial is scored](#15-how-a-trial-is-scored)
16. [Symbols](#16-symbols)

## 1. The task and the notation

One revolute joint has an angle $q \in [q_{\min}, q_{\max}]$. Its orientation is a rotation
about one axis: an element of SO(2), a point on a circle. A ring of neurons is the natural
place to hold such a number: neuron $i$ of $N$ stands for the ring angle

```math
\theta_i = \frac{2\pi i}{N}, \qquad i = 0, \dots, N-1, \qquad N = 200 ,
```

and the number is held as a *bump*: a group of neighbouring neurons that fire while the
rest are silent. The position of the bump is the stored value.

The controller has to move the joint to a goal angle $q^\ast$. It does so by holding the
goal and the joint's angle as bumps on rings, comparing where the bumps are, moving a bump
toward the other, and commanding the joint with the same spikes that move the bump.

## 2. The neuron

**Built.** Every neuron is NEST's `iaf_psc_alpha`: a leaky integrator with a threshold and
synaptic currents in the shape of an alpha function
(`src/config/model_params/neuron_params.json`).

```math
C_m \frac{dV}{dt} = -\frac{C_m}{\tau_m}\,(V - E_L) + I_e + I_{\text{syn}}(t),
\qquad V \ge V_{th} \;\Rightarrow\; \text{spike},\; V \leftarrow V_{\text{reset}}
```

```math
I_{\text{syn}}(t) = \sum_{\text{spikes } s} w_s \,\alpha(t - t_s), \qquad
\alpha(t) = \frac{e}{\tau_{\text{syn}}}\; t\; e^{-t/\tau_{\text{syn}}} \quad (t \ge 0)
```

A spike through a synapse of weight $w$ gives a current that rises to $w$ (in pA) at
$t = \tau_{\text{syn}}$ and decays. Positive weights use $\tau_{\text{syn,ex}}$, negative ones
$\tau_{\text{syn,in}}$.

**Charge of one spike.** Integrating the alpha function,

```math
\int_0^\infty w\,\frac{e}{\tau}\,t\,e^{-t/\tau}\,dt = w\,e\,\tau ,
```

so a sender firing at rate $r$ delivers the mean current $\bar I = r\,w\,e\,\tau_{\text{syn}}$.
This is how every weight in the three repos is turned into a current and back.

**Rate for a constant current.** With $E_L = V_{\text{reset}} = 0$ and no refractory time,
$V(t) = \frac{I\tau_m}{C_m}\,(1 - e^{-t/\tau_m})$ after a spike. It reaches $V_{th}$ only if
$I > I_{th} = C_m V_{th} / \tau_m$ (the *rheobase*), after the time

```math
T = \tau_m \ln\frac{I}{I - I_{th}}, \qquad f = \frac{1}{T} .
```

**The ring neuron rests exactly at threshold.** Its parameters are $C_m = 3.96\cdot10^6$ pF,
$\tau_m = 7.04$ ms, $V_{th} = 0.8$ mV and a bias current $I_e = 4.5\cdot10^5$ pA, so

```math
I_{th} = \frac{C_m V_{th}}{\tau_m} = \frac{3.96\cdot10^6 \cdot 0.8}{7.04} = 4.5\cdot10^5\ \text{pA} = I_e .
```

The bias holds the neuron at the threshold without crossing it. Writing a small extra
current as $I = I_{th}(1 + \varepsilon)$ gives

```math
f = \frac{1}{\tau_m \ln\!\big((1+\varepsilon)/\varepsilon\big)} \approx \frac{1}{\tau_m \ln(1/\varepsilon)} .
```

Consequences that matter everywhere below:

- Any net positive input makes the neuron fire; any net negative input silences it.
- The rate grows only with the logarithm of the input. A bump's rate is therefore almost
  independent of how strong its input is (about 35 Hz per neuron in a bump).
- A ring neuron whose bias is lowered a little (the *wall* of section 4) can never fire on
  its own.

Other populations: the gain neurons have $I_e = 0$ and $V_{th} = 1.35$ mV (they fire only
when driven); the decision neurons have twice the bias; the Fourier readout neurons are
NEST's default `iaf_psc_alpha` with a 200 pA bias.

## 3. The ring attractor

**Built** (`math/ring.py`, block `Ring`). Every pair of ring neurons is connected. The
weight depends only on their distance along the ring. Distances are scaled so that the far
side of the ring is at $d = 50$: one index is $\Delta d = 50/(N/2) = 0.5$. With
$\sigma_1 = 10$ and $\sigma_2 = 5$ the code's formula

```math
w(d) = 3\;\frac{\sigma_2\, e^{-d^2/2\sigma_1^2} - \sigma_1\, e^{-d^2/2\sigma_2^2}}{\sigma_2 - \sigma_1}
     = 3\left(2\,e^{-d^2/50} - e^{-d^2/200}\right)
```

is a narrow excitation minus a wide inhibition:

| distance $d$ | 0 | 6.8 | 13.6 | 30 | 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ring indices | 0 | 13.6 | 27 | 60 | 100 |
| $w(d)$ | +3.0 | 0 | −1.04 | −0.03 | 0 |

- **Zero crossing.** $w(d) = 0$ where $2e^{-d^2/50} = e^{-d^2/200}$, that is
  $d^2\left(\tfrac{1}{50} - \tfrac{1}{200}\right) = \ln 2$, $d = \sqrt{200 \ln 2 / 3} = 6.8$
  (13.6 indices). Neurons closer than that excite each other.
- **Strongest inhibition** at $d = \sqrt{(200/3)\ln 8} \approx 11.8$ (about 24 indices), where
  $w \approx -1.13$.
- **Far away the weight is zero.** Two bumps more than about 60 indices apart do not feel
  each other. This is why a bump cannot *jump*: a new stimulus far away makes a second bump
  instead of moving the first (measured; the encoders therefore only ever move a bump in
  small steps).

**Model: why a bump forms and stays.** Neurons inside a group excite each other
($w > 0$ for $|d| < 6.8$) and, with every neuron resting at threshold, that excitation is
enough to keep them firing. Just outside the group the net input is negative (the
inhibition of all bump neurons at distance 7 to 30), and a neuron at threshold with a
negative input is silent. So the group is stable and its neighbours cannot join. The width
settles where a neuron at the edge receives as much excitation as inhibition; measured,
about 28 active neurons.

**Model: why the bump can be anywhere.** $w$ depends only on the distance, so the network
is the same after a rotation of the ring. If a bump at one position is a stable state, so is
the same bump at every other position: a line (here a circle) of stable states, a
*continuous attractor*. Nothing in the ring prefers one position, which has two sides:

- a small asymmetric input moves the bump and it stays where it was moved (this is how
  section 9 moves it);
- noise moves it too. Measured drift of a free bump: about 2 indices in 20 s.

## 4. Joint angle and ring index

**Built** (`control/mapping.py`, `math/control.py`).

*Bounded mapping* (v4 and the bounded v3). The joint range lies on an arc of the ring,
centred on index $N/2$, that covers the fraction $f_{\text{arc}} = 0.7$ of it:

```math
i(q) = i_0 + \frac{q - q_{\min}}{q_{\max} - q_{\min}}\; S, \qquad
S = f_{\text{arc}}\,N = 140, \qquad i_0 = \frac{N - S}{2} = 30 ,
```

rounded to the nearest index when an encoder writes it. One index is

```math
\rho = \frac{q_{\max} - q_{\min}}{S}\quad \text{rad}
```

(0.020 rad for arm_6, 0.034 rad for the joints with the widest range). The inverse is
$q(i) = q_{\min} + (i - i_0)\,\rho$.

The 60 indices that are left over lie around the seam (index 0 = index $N$). Two things keep
a bump out of them:

- **The wall.** Neurons outside the indices 15 to 185 get the bias $0.9\,I_e$. By section 2
  they are then below threshold by 10% of the rheobase and no bump can rest there.
- **Fading at the limits.** The gain population that moves the bump toward a limit is driven
  less and less over the last 10 indices before that limit (section 9), so the bump cannot be
  pushed past it.

*Legacy mapping* (the draft's architecture). The range lies on the indices 5 to 195, so the
two joint limits are neighbours across the seam and a bump near one limit can slip to the
other. The bounded mapping exists because of that.

## 5. Writing an angle: the encoder

**Built** (block `Encoder`). One Poisson generator per ring neuron. To write the angle $q$,
the generators of the 11 neurons $i(q) - 5, \dots, i(q) + 5$ fire at 200 Hz with weight 4500;
the others are silent. By section 2 the mean current into a stimulated neuron is

```math
\bar I = r\,w\,e\,\tau_{\text{syn,ex}} = 200\ \text{Hz} \cdot 4500\ \text{pA} \cdot e \cdot 0.027\ \text{s} \approx 6.6\cdot10^4\ \text{pA}
       = 0.147\, I_{th} ,
```

which makes those neurons fire at about $1/(7.04\ \text{ms}\cdot\ln(1.147/0.147)) \approx 70$ Hz,
twice the rate of a free bump. The ring's own weights then shape the bump around them.

Three modes say *when* the encoder stimulates:

| Mode | When | Used for |
| --- | --- | --- |
| `once` | the first tick of a trial | v3: both rings; v4: the predictive ring |
| `continuous` | every tick | (measured to bias the comparator, see below) |
| `corrective` | only while the bump is more than `dead_band` indices from $i(q)$ | v4: goal ring (2) and state ring (1) |

**Why corrective.** A stimulated bump fires harder than a free one. The comparator of
section 8 reads firing rates, so a harder bump looks like a bump at a different position.
Stimulating only when the bump is off keeps a resting bump at its free amplitude, and still
cancels drift.

## 6. Reading an angle: the population vector

**Built** (block `RingState`, `math/circular.py`). From the spike counts $c_i$ of the ring
neurons over the last ticks:

```math
z = \sum_{i} c_i\, e^{\,\mathrm{i}\theta_i}, \qquad
\hat\imath = \frac{N}{2\pi}\,\arg z \pmod N, \qquad
\text{strength} = \frac{|z|}{\sum_i c_i} .
```

$\hat\imath$ is the centre of the activity as a fractional index, and $q(\hat\imath)$ of
section 4 the angle the ring holds. The strength is 1 for a single tight bump and near 0 for
activity spread around the ring.

*Why this is the centre.* For counts that are symmetric about a position $\varphi$,
$c(\theta) = b(\theta - \varphi)$ with $b$ even,
$z = e^{\mathrm{i}\varphi}\sum b(u)\,e^{\mathrm{i}u} = e^{\mathrm{i}\varphi}\sum b(u)\cos u$: the
sine part cancels by symmetry, so $\arg z = \varphi$ exactly.

## 7. The Fourier readout

**Built** (block `FourierReadout`, `math/fourier.py`). $2K = 40$ feature neurons read a ring.
For each harmonic $k = 1, \dots, K$ there is a pair with the weights from ring neuron $i$

```math
m^{+}_{k,i} = \max(\sin k\theta_i,\, 0), \qquad m^{-}_{k,i} = \max(-\sin k\theta_i,\, 0) ,
```

times a scale of 200, plus a bias of 200 pA. Weights of one sign only can be realised by
excitatory synapses; the *signed* feature is the difference of the pair:

```math
s_k = \sum_i \left(m^+_{k,i} - m^-_{k,i}\right) r_i = \sum_i \sin(k\theta_i)\, r_i .
```

**What the features say about the bump.** Let the ring's rates be a bump of even shape $b$
at position $\varphi$: $r_i = b(\theta_i - \varphi)$. Substituting $u = \theta_i - \varphi$,

```math
s_k = \sum_u \sin\big(k(u + \varphi)\big)\, b(u)
    = \sin k\varphi \underbrace{\sum_u b(u)\cos ku}_{B_k} + \cos k\varphi \underbrace{\sum_u b(u)\sin ku}_{=\,0} .
```

So $s_k = B_k \sin k\varphi$: the features are the sine series of the bump's position, each
scaled by the corresponding Fourier coefficient $B_k$ of the bump's shape. A wider bump has
smaller $B_k$ at high $k$, which limits how sharp a function of the position the features
can express; a stronger bump scales all $B_k$, which is the amplitude sensitivity mentioned
in section 5.

(The feature neurons turn their input into a rate through their own threshold, so the
counts are a monotone, not exactly linear, function of these sums. The fit of the next
section is made on measured counts and absorbs that.)

## 8. The ramp comparator

The comparator has to say on which side of the reference bump the moved bump is.

**Built** (blocks `FourierReadout` + `Homeostasis`, `training/comparator.py`,
`scripts/fit_comparator.py`).

**Step 1: a number that rises along the ring.** A weight vector $a \in \mathbb{R}^K$ on the
signed features gives a function of the bump position,

```math
g(\varphi) = \sum_{k=1}^{K} a_k\, s_k(\varphi) \approx \sum_k a_k B_k \sin k\varphi .
```

The target is a ramp that rises from 0 to 1 across the arc and falls back across the seam
sector (`bounded_ramp`, rising over the fraction 0.8 of the ring). Being odd about the seam
after removing its mean, it has a sine series, which is why sine features suffice. The
weights are a ridge fit to *measured* feature counts $S$ (one row per bump position, the
bump placed at every index):

```math
a = \arg\min_a \; \lVert S a - g \rVert^2 + \lambda \lVert a \rVert^2
  = (S^\top S + \lambda I)^{-1} S^\top g .
```

**Step 2: two ramps and a decision.** Two neurons carry the ramp: *warm* is driven by the
features of the ring being moved, *cold* by the features of the reference ring, each with
the weights $a$ (positive parts on the `sin_pos` neuron, negative parts on the `sin_neg`
neuron of a pair). Their rates rise with the position of their ring. Two decision neurons
compare them:

```math
\text{cold} \xrightarrow{+} \text{right}, \quad \text{cold} \xrightarrow{-} \text{left}, \quad
\text{warm} \xrightarrow{+} \text{left}, \quad \text{warm} \xrightarrow{-} \text{right}, \quad
\text{left} \overset{-}{\longleftrightarrow} \text{right} .
```

If the reference is further along the ring than the moved bump, cold fires more than warm,
*right* wins, and the bump is moved up. With mutual inhibition only one side fires.

**Model: why there is a blind band.** The decision is driven by the difference of the two
ramps, $g(\varphi_{\text{ref}}) - g(\varphi) \approx g'(\varphi)\,\Delta\varphi$. The ramp rises by
1 over 160 indices, so an offset of 5 indices is a difference of 3% of the ramp's range,
which is below the spike-count noise of the two ramp neurons and the threshold of the
decision neurons. Measured: no drive in a band 10 to 21 indices wide around the reference.
The comparator sees the direction from far away and is blind nearby. Section 12 adds the
mechanism that works nearby.

## 9. The gain pair: moving a bump

**Built** (block `Gain`, `nest/populations.py`). Two populations of $N$ neurons, *left* and
*right*, with $I_e = 0$ and a threshold of 1.35 mV.

**An AND of two inputs.** Gain neuron $i$ of each population gets

- the spikes of ring neuron $i$ (one-to-one, weight $4.5\cdot10^5$), and
- the spikes of its decision neuron (weight $10^4$), and inhibition from the other decision
  neuron (weight $-3\cdot10^4$).

Its threshold is set so that neither input alone makes it fire. So *left* neuron $i$ fires
only where the bump is **and** the decision is "left": the gain populations carry a copy of
the bump, on the side that won.

**The copy moves the bump.** Each gain neuron inhibits one ring neuron next to its own:

```math
\text{right}_i \xrightarrow{\;w_g\;} \text{ring}_{i-1}, \qquad
\text{left}_i \xrightarrow{\;w_g\;} \text{ring}_{i+1}, \qquad w_g = -0.6 .
```

With *right* active, every bump neuron inhibits its lower neighbour. Inside the bump this is
a small uniform reduction. At the lower edge it is decisive: the edge neuron is at the
balance between excitation and inhibition (section 3), and the extra inhibition silences it.
At the upper edge nothing is taken away. The bump loses neurons at the bottom and, keeping
its width, gains them at the top: it moves up.

**Model: how fast.** Measured on NEST over 96 trials (`scripts/calibrate_transport.py`), the
displacement of the bump per tick is proportional to the gain spike count above a dead zone:

```math
\Delta i = a\cdot \operatorname{dz}_\theta(s), \qquad s = R - L, \qquad
\operatorname{dz}_\theta(s) = \operatorname{sign}(s)\,\max(|s| - \theta,\, 0),
```

with $R$, $L$ the spikes of the two populations in the tick, $a = 0.005332$ indices per
spike and $\theta = 20$ spikes per tick. Below $\theta$ the push is too weak to silence the
edge neuron and the bump stays (it is pinned to the lattice of neurons). A linear model
without dead zone predicts the displacement to 10.4% in the median on held-out trials, the
dead-zone model to 4.8%.

## 10. The decoder: spikes to joint velocity

**Built** (block `Decoder`, `math/control.py`). Per tick of length $dt$:

```math
x_t = \operatorname{dz}_{\theta}(R_t - L_t), \qquad
y_t = \alpha\, y_{t-1} + (1 - \alpha)\, x_t, \quad \alpha = e^{-dt/\tau_s}, \qquad
v_t = k\, y_t ,
```

clipped to the joint's velocity limit. The filter is a first-order low-pass with time
constant $\tau_s$: for a constant input it approaches that input as $1 - e^{-t/\tau_s}$.

**Position streaming.** TIAGo's arm takes positions, not velocities. The velocity is
integrated into a position, $q^{\text{cmd}}_{t+1} = \operatorname{clip}(q^{\text{cmd}}_t + v_t\,dt)$,
held at the joint limits, and sent as a short trajectory. Measured in Gazebo: the joint
follows to within 2% with a lag of 5 to 14 ms.

**The gain $k$ in v3: a calibration.** In v3 the bump *is* the model's belief about the
joint, and the same spikes move both. For the joint to move as far as the bump,

```math
\Delta q = \rho\, \Delta i
\;\;\Longrightarrow\;\;
v\, dt = \rho\, a \operatorname{dz}_\theta(s)
\;\;\Longrightarrow\;\;
k = \frac{a\,\rho}{dt} .
```

$a$ and $\theta$ belong to the network and $\rho$ to the joint, so one fit on NEST serves
all joints (`graph/calibration.py`). In v4 the same $k$ is only a loop gain (section 12).

## 11. Architecture v3: two rings, open loop

**Built** (`graph/templates.py: two_ring_single_joint`). A goal ring G and a state ring S.

1. The encoder writes $i(q^\ast)$ on G and $i(q_0)$ on S, once.
2. The comparator compares S (warm) with G (cold).
3. The winning gain population moves S toward G; the decoder turns the same spikes into
   joint velocity.
4. When the bumps are within the blind band, the comparator falls silent; the trial ends.

**Model.** With $p$ the index of the state bump and $q$ the joint angle,

```math
p_{t+1} = p_t + a\,\operatorname{dz}_\theta(s_t), \qquad
q_{t+1} = q_t + \gamma\, k\, dt\; \tilde s_t ,
```

where $\tilde s$ is the filtered $\operatorname{dz}_\theta(s)$ and $\gamma$ is how much the
joint really moves per commanded unit (1 for a perfect actuator). Summing over a trial, with
$k = a\rho/dt$,

```math
q_{\text{end}} - q_0 = \gamma\,\rho\,(p_{\text{end}} - p_0) .
```

The joint ends where the bump ends, scaled by $\gamma$. The joint is never looked at after
step 1, so every error stays in the result:

```math
q_{\text{end}} - q^\ast =
\underbrace{\rho\,(p_{\text{end}} - i(q^\ast))}_{\text{bump stops off the goal}}
\;+\; \underbrace{(\gamma - 1)\,\rho\,(p_{\text{end}} - p_0)}_{\text{actuator gain}}
\;+\; \text{rounding, drift, disturbances} .
```

Measured: the bump stops a median of 3.1 indices from the goal (the blind band), the joint a
median of 1.5 indices from the bump. A motor gain that is 30% low ($\gamma = 0.7$) leaves the
joint 30% of the move short, as the second term says.

## 12. Architecture v4: three rings, closed loop

**Built** (`graph/templates.py: three_ring_single_joint`, `blocks/composites.py: JointTriple`).

| Ring | Holds | Written by |
| --- | --- | --- |
| G, goal | the goal $i(q^\ast)$ | the goal encoder (corrective) |
| S, state | where the joint is, $i(q)$ | the joint's encoder, all the time (corrective) |
| P, predictive | where the joint is expected to be | the measured angle once, then only two gain pairs |

Two comparisons act on P, each with its own gain pair:

- **goal pair** (P against G): moves P toward the goal, and its spikes are the motor signal;
- **body pair** (P against S): moves P toward the body, and commands nothing.

### 12.1 The projection: a comparison that works nearby

**Built** (`connect_gain_reference`). The reference ring projects straight onto the gain
pair. Gain neuron $i$ of the *right* population receives from the reference ring

```math
I^{R}_i = w_e \sum_{j=i+1}^{i+W} r^{\text{ref}}_j \;-\; w_h \sum_{j=i-W}^{i-1} r^{\text{ref}}_j ,
\qquad W = 10,
```

and the *left* population the mirror image: excitation from the $W$ neurons just below,
inhibition from the $W$ just above. For *right*, excitation comes from just above and
inhibition from just below.

**Model: the drive is proportional to the offset.** The gain neurons that can fire are those
under the P bump (the AND of section 9). Take both bumps as blocks of the same width with
rate $r$, the reference block shifted up by $\Delta$ indices ($0 < \Delta < W$). For a gain
neuron in the middle of the P bump, the window above it contains $\Delta$ more reference
neurons than a centred window would, and the window below $\Delta$ fewer:

```math
I^R_i \approx r\,\big[(w_e - w_h)\,n_0 + (w_e + w_h)\,\Delta\big], \qquad
I^L_i \approx r\,\big[(w_e - w_h)\,n_0 - (w_e + w_h)\,\Delta\big],
```

with $n_0$ the reference neurons in a window when the bumps coincide. The first term is the
same on both sides and negative, because $w_h > w_e$ (inhibition 3 to 7.5 times the
excitation): with the bumps on top of each other neither side is driven. The second term
has opposite signs: the side that leads toward the reference is driven in proportion to
$\Delta$. Measured blind band: 0 to 1 index, against 10 to 21 for the ramp comparator.

The projection only reaches $W$ indices. The ramp comparator covers the rest. Each
comparison of v4 uses both: the ramp from far, the projection near.

### 12.2 The loop

**Model.** Let $g = i(q^\ast)$, $p$ the predictive bump, $s = i(q) = i_0 + (q - q_{\min})/\rho$
the state bump (it follows the joint), all in ring indices. Let $u_g$ and $u_b$ be the signed
spike counts of the goal pair and the body pair, each a rising, odd function of its offset:

```math
u_g = \phi_g(g - p), \qquad u_b = \phi_b(s - p) .
```

Per tick,

```math
\begin{aligned}
p_{t+1} &= p_t + a_g\,u_g + a_b\,u_b && \text{both pairs move P}\\
q_{t+1} &= q_t + \gamma\,k\,dt\;\widetilde{\operatorname{dz}}(u_g) && \text{only the goal pair moves the joint}
\end{aligned}
```

$a_g$ and $a_b$ are the bump displacements per spike; they are set by the feedback weights
of the two pairs ($-0.6$ and $-3.0$), so $a_b \approx 5\,a_g$.

**Where it comes to rest.** At rest both right-hand sides vanish:

```math
\widetilde{\operatorname{dz}}(u_g) = 0 \;\Rightarrow\; u_g \approx 0 \;\Rightarrow\; p \approx g,
\qquad\text{then}\qquad a_b u_b = 0 \;\Rightarrow\; s = p .
```

So $s = p = g$: **the body is at the goal**, to within the dead zones. The actuator gain
$\gamma$ does not appear in this. That is the difference from v3, where
$q_{\text{end}}$ carries the factor $\gamma$.

**What a wrong motor gain does.** Suppose $\gamma < 1$: the joint does less than commanded.
The goal pair moves P by $a_g u_g$ per tick as before, the joint moves S by only
$\gamma\,k\,dt\,u_g/\rho$. P runs ahead of S, the body pair sees $s - p < 0$ and pulls P
back. P is now further from the goal than it would have been, so the goal pair keeps firing
and keeps commanding. The move takes longer; it ends in the same place. Measured in Gazebo
with the motor gain at 0.7 and at 1.3 of its value: median goal error 0.013 and 0.010 rad
for v4, 0.314 and 0.203 rad for v3 (without the change, over all joints: 0.008 and
0.091 rad).

**Why only the goal pair drives the robot.** If one merged signal moved both the bump and
the joint, bump and joint would always move together and a difference between them could
never shrink. With two pairs, the difference between prediction and body is exactly what
the body pair acts on. Measured: a robot driven from a gain pair on the state ring
overshoots by 0.37 to 0.64 rad.

**Why the body pair is the stronger.** Linearise near the goal: $u_g \approx c_g\,(g - p)$,
$u_b \approx c_b\,(s - p)$, ignore the filter, and write $e = g - s$ (the body's error) and
$d = p - s$ (prediction ahead of body). With $\beta = \gamma k\,dt\,c_g/\rho$ (indices the
joint moves per tick per index of goal offset),

```math
\begin{aligned}
e_{t+1} &= e_t - \beta\,(e_t - d_t) \\
d_{t+1} &= d_t + a_g c_g\,(e_t - d_t) - a_b c_b\, d_t - \beta\,(e_t - d_t)
\end{aligned}
```

If the joint follows exactly as the bump is moved ($\beta = a_g c_g$), $d$ only decays:
$d_{t+1} = (1 - a_b c_b)\,d_t$. If not, $d$ settles where the body pair's pull balances the
mismatch,

```math
d \approx \frac{(a_g c_g - \beta)}{a_b c_b}\; e ,
```

small when $a_b c_b$ is large. The prediction can run ahead of the body only by a fraction
of the remaining error, and that fraction shrinks with the strength of the body pair. The
error then decays as $e_{t+1} \approx (1 - \beta(1 - d/e))\,e_t$: a first-order approach
whose rate, not whose end point, depends on $\gamma$.

**The decoder gain is a loop gain.** In v4 `motor_gain` $= a$ of section 10 enters only
through $\beta$. ±30% leaves the goal error where it is and changes the speed. This is why
v4 needs no transport calibration.

**The dead zone.** Without a dead zone in the decoder the joint hunts ±1 index around the
goal with a period of about 8 s (measured): the goal pair always produces a few spikes, and
a few spikes always move the joint a little. A dead zone of 3 spikes per NEST step removes
it at the cost of stopping within about an index.

## 13. Time: ticks, delay, time scale

**Built** (`cosim/loop.py`). NEST and the robot advance in ticks of $dt = 50$ ms, locked
together: the network runs one tick with the joint angle of the previous tick, the robot
then gets the command computed from that tick's spikes. There is one tick of delay in each
direction.

`time_scale` $= K$ runs $K$ NEST steps per robot tick. The network does spike for spike
what it would do anyway; the robot gets $K$ times the spikes per tick, so by section 10 it
moves $K$ times as fast, while the bump covers $K$ times the distance per robot tick. The
loop of section 12 is unchanged in units of NEST steps. The decoder's dead zone and settle
threshold are counted per NEST step and scaled by $K$. Default: $K = 3$.

## 14. The PID baseline

**Built** (`control/controller.py: PIDControllerCore`, run by
`legacy/single_joint_pid_data_collector.py`). The controller of the draft, the same for
every joint:

```math
e_t = q^\ast - q_t, \qquad
I_t = \operatorname{clip}\!\big(I_{t-1} + e_t\,dt,\ \pm I_{\max}\big), \qquad
v_t = \operatorname{clip}\!\Big(k_p e_t + k_i I_t + k_d \frac{e_t - e_{t-1}}{dt},\ \pm 0.5\ \text{rad/s}\Big)
```

with $k_p = 2.0$, $k_i = 0.1$, $k_d = 0.05$. It reads the joint every tick; it has no model
of the goal as a state and no neurons.

## 15. How a trial is scored

One scoring function for every system (`evaluation/`):

- **Goal error**: $|q^\ast - q_{\text{final}}|$ in rad, reported as median and 90th percentile
  over trials.
- **Settled**: the error stays within 0.05 rad.
- **Settling time**: the time after which it stays there.
- **Time to 90%**: the time to cover 90% of the move.
- **Repeatability**: the range of the end point over repeated runs of one goal.

## 16. Symbols

| Symbol | Meaning | Value |
| --- | --- | --- |
| $N$ | neurons per ring | 200 |
| $\theta_i$ | ring angle of neuron $i$ | $2\pi i/N$ |
| $q$, $q^\ast$ | joint angle, goal | rad |
| $i(q)$, $\rho$ | ring index of an angle, radians per index | section 4 |
| $w(d)$ | recurrent ring weight at distance $d$ | section 3 |
| $I_{th}$ | rheobase $C_m V_{th}/\tau_m$ | $4.5\cdot10^5$ pA (ring) |
| $s_k$, $B_k$ | signed Fourier feature, Fourier coefficient of the bump shape | section 7 |
| $g(\varphi)$, $a_k$ | ramp, its fitted weights | section 8 |
| $R$, $L$, $s$ | right and left gain spikes per tick, $s = R - L$ | |
| $a$, $\theta$ | bump displacement per spike, dead zone | 0.005332, 20 (v3 fit) |
| $k$, $\tau_s$ | decoder gain, filter time constant | $a\rho/dt$, 0.05 s (v4) |
| $\gamma$ | real over commanded joint motion | 1 if perfect |
| G, P, S | goal, predictive, state ring | section 12 |
| $u_g$, $u_b$ | signed spikes of the goal pair, the body pair | |
| $dt$, $K$ | tick, NEST steps per tick | 50 ms, 3 |
