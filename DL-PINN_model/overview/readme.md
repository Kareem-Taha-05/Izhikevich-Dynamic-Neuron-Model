# Physics-Informed Neural Network (PINN) for Non-Linear Izhikevich Neuron Dynamics

This specific subdirectory focuses on bypassing traditional numerical solvers by encoding the underlying biophysical laws of neuronal spiking directly into the loss function of a neural network. 

Standard neural networks struggle with the stiff, discontinuous nature of neuronal action potentials. This implementation addresses that limitation using multi-scale feature embedding, adaptive collocation scheduling, and custom loss regularizers designed for discontinuous system transitions.

---

## 1. Problem Statement & Mathematical Model

The objective is to accurately model the membrane potential `v(t)` and the recovery variable `w(t)` of a single neuron subjected to an external step-current injection `I(t)`. The system is governed by the two-dimensional non-linear Izhikevich ODE system:

$$C \frac{dv}{dt} = k(v - v_r)(v - v_t) - w + I(t)$$

$$\frac{dw}{dt} = a(b(v - v_r) - w)$$

### Auxiliary Reset Conditions
When the membrane potential reaches its apex value, an explicit reset condition is enforced numerically:

If `v(t) >= v_peak`, then:
* `v` becomes `c`
* `w` becomes `w + d`

### Experimental Constants
The parameters configured in this codebase correspond to standard cortical neuron dynamics:
* **C** = 100.0 pF (Membrane capacitance)
* **v_r** = -60.0 mV (Resting potential)
* **v_t** = -40.0 mV (Threshold potential)
* **k** = 0.7, **a** = 0.03, **b** = -2.0 (Dimensionless scaling constants)
* **c** = -50.0 mV, **d** = 100.0 (Post-spike reset values)
* **I(t)** = 70.0 pA for `t >= 100 ms` (0.0 pA otherwise)

---

## 2. Neural Network Architecture

To capture both the low-frequency sub-threshold oscillations and high-frequency spike configurations, the model departs from a standard dense architecture in favor of a **Multi-Scale Fourier Feature PINN**.

```text
Input (t) ──> [Normalization & Multi-Scale Features] ──> 6x Dense Layers (128 units, Tanh) ──> [Physics Scaling] ──> Outputs (v, w)
```

### Input Feature Embedding
Time input `t` is normalized (`t_norm = t / 200`) and mapped into a 5-dimensional coordinate space using pairs of trigonometric basis functions to capture fast discontinuities without spectral bias:
`[ t_norm, sin(2π*t_norm), cos(2π*t_norm), sin(4π*t_norm), cos(4π*t_norm) ]`

### Hidden Layers & Scaling
* **Depth & Width:** 6 fully connected layers with 128 hidden neurons each.
* **Initialization:** Xavier Normal distribution (`gain = 1.0`) with zeroed biases.
* **Physics-Based Output Layer:** Network outputs are explicitly scaled to physically plausible regimes before loss evaluation to prevent optimization drift:
    * `v(t) = (out_0 * 100) + v_r`
    * `w(t) = out_1 * 300`

---

## 3. Code Implementation & Optimization Strategy

The training script leverages a **Curriculum Learning Policy** paired with an adaptive point-distribution scheme to successfully navigate the highly non-convex loss landscape.

### Collocation Point Allocation
Rather than standard uniform sampling, `create_adaptive_training_points` builds an asymmetric grid across the 200 ms interval. Out of 2,000 points, high-density clusters are allocated around critical phase shifts:
* **Stimulus onset region:** 95 - 105 ms
* **Primary action potential window:** 100 - 120 ms
* **Secondary action potential window:** 120 - 140 ms

### Multi-Objective Loss Formulation
The network optimizes a compound loss function: `Loss_total = L_pde + L_ic + L_bounds + L_spike + L_cont`

* **PDE Residual (`L_pde`):** Enforces the governing differential equations. The active weight scales up by 5x during the active stimulus window (`t >= 90 ms`).
* **Initial Condition (`L_ic`):** Anchors the system at the correct physiological steady state `[v_r, 0]`.
* **Soft Bounds (`L_bounds`):** Penalizes non-biological network behavior (e.g., voltages `> 50 mV` or `< -100 mV`).
* **Spike-Aware (`L_spike`):** Forces sharp gradients (`dv/dt`) when `v` approaches threshold boundaries using a smooth indicator function.
* **Continuity (`L_cont`):** Imposes a smoothness constraint on discrete derivatives to suppress artificial high-frequency oscillations outside true spike regions.

### Two-Phase Optimization Protocol
The model is optimized over a total of **12,000 epochs** via PyTorch's automatic differentiation engine (`torch.autograd`):

1.  **Phase 1: Macro-Dynamics (Epochs 0–5,000)**
    * **Optimizer:** Adam (`lr = 5e-4`, `weight_decay = 1e-6`).
    * **Objective:** Fits basic resting potentials and smooth trends. Spike and continuity weights are scaled up linearly from 0 to introduce rigid structural limits smoothly.
2.  **Phase 2: Fine-Tuning (Epochs 5,000–12,000)**
    * **Optimizer:** Adam (`lr = 1e-4` coupled to a `ReduceLROnPlateau` scheduler, `patience = 1000`).
    * **Objective:** Resolves sharp peaks, precise spike intervals, and alignment errors. Gradient clipping is tightened to `max_norm = 0.5` to prevent exploding gradients during rapid state changes.

---

## 4. Verification and Baselines

The code block includes two distinct numerical solvers to validate PINN output integrity:
* **Forward Euler Baseline:** Standard discrete integration step (`h = 0.05 ms`) configured with condition triggers for checking spike boundaries.
* **Classical Runge-Kutta (RK4) Ground Truth:** A robust 4th-order solver used as the scientific baseline to calculate absolute Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) against the trained neural model.

---

## 5. Analysis & Visualization Dashboard

To examine the training convergence and computational efficiency of this model in detail, refer to the **[Interactive PINN Analysis Dashboard](https://izhikevich-pinn-results.vercel.app)**. 

The web-based visualization maps the numerical outputs generated by the PyTorch model, including:
* **Training Convergence:** Log-scale loss breakdown (PDE, Initial Condition, and Data) across the training epochs.
* **Accuracy Benchmarks:** Direct MAE and RMSE comparisons against the RK4 numerical baseline.
* **Neuronal Dynamics:** Phase space mapping of membrane voltage vs. recovery state.
* **Spike Timing:** Interspike interval (ISI) analysis and peak potential alignments.
