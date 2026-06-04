# Izhikevich Dynamic Neuron Model: A Multi-Solver Computational Study

> A comprehensive computational neuroscience project implementing and benchmarking five numerical solvers, from classical finite-difference schemes to Physics-Informed Neural Networks. For simulating spiking neuron dynamics using the Izhikevich model, with real-time 3D visualization in Unity.

---

## Table of Contents

- [Overview](#overview)
- [The Izhikevich Neuron Model](#the-izhikevich-neuron-model)
- [Repository Structure](#repository-structure)
- [Solvers at a Glance](#solvers-at-a-glance)
- [Method 1 Explicit Euler](#method-1-explicit-euler)
- [Method 2 Backward (Implicit) Euler](#method-2:-backward-implicit-euler)
- [Method 3 Midpoint / RK2](#method-3--midpoint--rk2)
- [Method 4 Adaptive ExpRESS–Euler](#method-4--adaptive-expresseuler)
- [Method 5 Physics-Informed Neural Network (PINN)](#method-5--physics-informed-neural-network-pinn)
- [Unity 3D Real-Time Visualizer](#unity-3d-real-time-visualizer)
- [Comparative Analysis](#comparative-analysis)
- [Getting Started](#getting-started)
- [Project Report](#project-report)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

This project systematically investigates the numerical simulation of **spiking neuron dynamics** through the lens of the **Izhikevich model**, which is a biologically plausible ODE system that captures the richness of neural firing behavior at low computational cost. Rather than committing to a single numerical strategy, the project implements and rigorously compares five distinct approaches that span the full spectrum from introductory explicit methods to state-of-the-art deep learning:

| Approach | Method | Stability | Accuracy |
|---|---|---|---|
| Classical | Explicit Euler | Conditionally stable | O(h) |
| Classical | Backward (Implicit) Euler | Unconditionally stable | O(h) |
| Classical | Midpoint / RK2 | Conditionally stable | O(h²) |
| Advanced | Adaptive ExpRESS–Euler | Stiff-stable, adaptive | O(h) + error control |
| Deep Learning | PINN (PyTorch) | N/A | Data-free, physics-driven |

Beyond numerical methods, the project includes a **Unity 3D real-time simulator** that visualizes neuron firing as a particle pulse traveling along a 3D neuron mesh, with live parameter control via UI sliders.

---

## The Izhikevich Neuron Model

The **Izhikevich model** (Izhikevich, 2003) is a two-variable ODE system that reproduces the diverse firing patterns of cortical neurons while remaining computationally efficient, a key advantage over the full Hodgkin–Huxley formulation.

### Governing Equations

$$C \frac{dv}{dt} = k(v - v_r)(v - v_t) - w + I$$

$$\frac{dw}{dt} = a\left[b(v - v_r) - w\right]$$

**Spike reset rule:** When $v \geq v_{\text{peak}}$:

$$v \leftarrow c, \qquad w \leftarrow w + d$$

### State Variables

| Variable | Description |
|---|---|
| $v$ | Membrane potential (mV) |
| $w$ | Membrane recovery variable (pA) |

### Model Parameters

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Membrane capacitance | $C$ | 100 pF | Scales the membrane dynamics |
| Resting potential | $v_r$ | −60 mV | Stable equilibrium without input |
| Threshold potential | $v_t$ | −40 mV | Instantaneous spike initiation threshold |
| Spike cutoff | $v_{\text{peak}}$ | 35 mV | Voltage at which spike is declared |
| Reset potential | $c$ | −50 mV | Post-spike reset value of $v$ |
| Recovery increment | $d$ | 100 pA | Post-spike jump in $w$ |
| Voltage scaling | $k$ | 0.7 | Sharpness of spike initiation |
| Recovery speed | $a$ | 0.03 | Time scale of recovery variable |
| Recovery coupling | $b$ | −2 | Sensitivity of $w$ to subthreshold oscillations |
| Input current | $I$ | 0 → 70 pA | Step current applied at $t = 101$ ms |

---

## Repository Structure

```
NeuroSolve/
│
├── Backward_Euler_method/
│   ├── code/
│   │   └── Backward_Euler_Method.py
│   ├── overview/
│   │   ├── Backword Euler final.pdf          
│   │   └── readme.md                
│   └── plots/
│         
├── DL-PINN_model/
│   ├── code/
│   │   ├── Explicit Euler.ipynb      ← Reference solution generation
│   │   ├── PINN_Model.ipynb          ← Guided PINN (main)
│   │   └── Pure_PINN.ipynb           ← Baseline pure PINN
│   ├── overview/
|   |   ├── Results.js
|   |   └── readme.md
│   └── plots/
│
├── Explicit_Euler_method/
│   ├── code/
│   │   └── Euler.py
│   ├── overview/
|   |   ├── Explicit Euler Method.pdf
|   |   └── readme.md   
│   └── plots/
│
├── Midpoint_(RK2)_method/
│   ├── code/
│   │   └── Midpoint(RK2).py
│   ├── overview/
|   |   ├── Midpoint (RK2) Method.pdf
|   |   └── readme.md                       
│   └── plots/
│
├── Presentation/
│
├── Report/
|
├── adaptive_exponential_Rosenbrock–Euler_method/
│   ├── code/
|   |   ├── adaptive_express_euler.py
│   │   └── adaptive_express_euler_results.csv
│   ├── graphs/
|   |                 
│   └── overview/
|       ├── adaptive_exponential_Rosenbrock–Euler_method
|       └── readme.md
|
├── unity/
│   ├── COMPLETE_NEURON_MODEL.blend   ← Blender source model
│   ├── COMPLETE_NEURON_MODEL.fbx     ← Unity-ready 3D model
│   ├── firing_neurons/               ← Unity project root
│   │   ├── Assets/
│   │   ├── Packages/
│   │   └── ProjectSettings/
│   └── README.md
│
└── README.md
```

---

## Method 1: Explicit Euler

**File:** `Explicit_Euler_method/Code/Euler.py`

The **Explicit (Forward) Euler** method is the simplest first-order numerical integration scheme and serves as the project's baseline and reference solver. Its update rule applies derivatives evaluated at the current time step to extrapolate the next state:

$$y_{n+1} = y_n + h \cdot f(t_n, y_n)$$

### Simulation Configuration

| Setting | Value |
|---|---|
| Time range | 0 – 1000 ms |
| Step size $h$ | 1 ms |
| Initial conditions | $v_0 = -60$ mV, $w_0 = 0$ pA |
| Input current | $I = 0$ for $t < 101$ ms, $I = 70$ pA otherwise |

### Reference Output

| Time (ms) | $v$ (mV) | $w$ (pA) |
|---|---|---|
| 0 | −60.0000 | 0.0000 |
| 250 | (see output) | (see output) |
| 500 | (see output) | (see output) |
| 750 | (see output) | (see output) |
| 1000 | (see output) | (see output) |

### Visualizations

![Membrane potential v(t)](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/Explicit_Euler_method/plots/euler_v.png?raw=true)

![Recovery variable w(t](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/Explicit_Euler_method/plots/euler_w.png?raw=true)

![W vs V](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/Explicit_Euler_method/plots/euler_phase_plane.png?raw=true)

### Assessment

**Strengths**
- Trivial to implement; no linear system solves required
- Fast per-step computation: one function evaluation per step
- Ideal baseline for comparing more advanced solvers

**Limitations**
- First-order accurate, global error is $O(h)$
- Conditionally stable, requires small $h$ for correct spike shapes
- Unsuitable for stiff dynamics without step-size reduction

---

## Method 2: Backward (Implicit) Euler

**File:** `Backward_Euler_method/code/Backward_Euler_Method.py`  

The **Backward (Implicit) Euler** method resolves the stability issue of its explicit counterpart by evaluating the derivative at the *next* time step, making it unconditionally stable for linear systems:

$$y_{n+1} = y_n + h \cdot f(t_{n+1}, y_{n+1})$$

Because the Izhikevich ODE is nonlinear, this implicit equation cannot be solved in closed form. The implementation uses **Newton's method** at each time step to iteratively solve the nonlinear system, with convergence diagnostics printed for the first few steps.

### Simulation Configuration

| Setting | Value |
|---|---|
| Time range | 0 – 1000 ms |
| Step size $h$ | 0.25 ms |
| Input current | Constant $I = 100$ pA |
| Newton solver | Per-step iteration with convergence check |

### Visualizations

![Membrane potential v(t)](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/Backward_Euler_method/plots/membrane%20potential.png?raw=true)

![Recovery variable w(t)](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/Backward_Euler_method/plots/Recovery%20Variable.png)

![W vs V](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/Backward_Euler_method/plots/Phase%20plane(W%26V).png?raw=true)

### Assessment

**Strengths**
- Unconditionally stable, can use larger time steps without divergence
- Better suited to stiff regions of the ODE (near-spike dynamics)
- Appropriate for long-duration simulations

**Limitations**
- Requires solving a nonlinear system at each step (Newton iteration overhead)
- Higher implementation complexity than explicit methods
- Not significantly more accurate (still first-order) despite the added cost

---

## Method 3: Midpoint / RK2

**File:** `Midpoint_(RK2)_method/code/Midpoint(RK2).py`  

The **Midpoint method** (a second-order Runge-Kutta scheme) improves accuracy over Euler by evaluating the derivative at the *midpoint* of the time interval rather than only at the endpoints. It makes two function evaluations per step:

**Step 1: Slope at current point:**

$$k_{1v} = \frac{1}{C}\left[k(v - v_r)(v - v_t) - w + I\right], \quad k_{1w} = a\left[b(v - v_r) - w\right]$$

**Step 2: Midpoint estimate:**

$$v_{\text{mid}} = v + \frac{h}{2} k_{1v}, \quad w_{\text{mid}} = w + \frac{h}{2} k_{1w}$$

**Step 3: Slope at midpoint:**

$$k_{2v} = \frac{1}{C}\left[k(v_{\text{mid}} - v_r)(v_{\text{mid}} - v_t) - w_{\text{mid}} + I\right]$$

**Step 4: Full update using midpoint slope:**

$$v_{n+1} = v_n + h \cdot k_{2v}, \quad w_{n+1} = w_n + h \cdot k_{2w}$$

The implementation logs all intermediate values ($k_1$, $k_2$, midpoint estimates) in a Pandas DataFrame for inspection and validation.

### Visualizations

![Midpoint](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/Midpoint_(RK2)_method/plots/All%20plots%20of%20midpoint%20method.png)


### Assessment

**Strengths**
- Second-order accuracy, global error $O(h^2)$; significantly better than Euler for the same step size
- Only two function evaluations per step; modest computational cost
- Good balance between simplicity and accuracy

**Limitations**
- Conditionally stable; sensitive to stiff regions
- Less accurate than RK4 for the same cost
- Not adaptive, fixed step size throughout the simulation

---

## Method 4: Adaptive ExpRESS–Euler

**File:** `adaptive_exponential_Rosenbrock–Euler_method/code/adaptive_express_euler.py`

The **Adaptive Exponential Rosenbrock–Euler (ExpRESS–Euler)** method is the most mathematically sophisticated classical solver in this project. It addresses the stiffness of the Izhikevich ODE directly by incorporating the system Jacobian into the update formula, using a matrix exponential-based integrating factor $\varphi_1(hJ)$:

$$y_{n+1} = y_n + h \cdot \varphi_1(hJ_n) \cdot f(y_n)$$

where $\varphi_1(Z) = (I - Z)^{-1}$ approximates the exponential integrating factor for stiff systems.

**Adaptive step-size control** is achieved via Richardson extrapolation: a full-step solution $y_1$ is compared against a two-half-step solution $y_2$, and the local error estimate drives the step-size update:

$$h_{\text{new}} = h \cdot \min\left(\max\left(\sqrt{\frac{\text{tol}}{\|y_1 - y_2\|}},\ f_{\text{min}}\right),\ f_{\text{max}}\right)$$

### Adaptive Step Control Parameters

| Parameter | Value | Description |
|---|---|---|
| Initial $h$ | 0.25 ms | Starting step size |
| Tolerance | 0.5 | Local error threshold |
| $h_{\text{min}}$ | 0.01 ms | Minimum allowed step |
| $h_{\text{max}}$ | 2.0 ms | Maximum allowed step |
| $f_{\text{min}}$ | 0.1 | Minimum shrink factor |
| $f_{\text{max}}$ | 5.0 | Maximum growth factor |

### Error Analysis vs. Explicit Euler Reference

| Time (ms) | $v_{\text{sim}}$ | $v_{\text{ref}}$ | $\Delta v$ | $\Delta v$ % | $w_{\text{sim}}$ | $w_{\text{ref}}$ | $\Delta w$ |
|---|---|---|---|---|---|---|---|
| 0 | −60.00 | −60.00 | 0.00 | 0.00% | 0.000 | 0.000 | 0.000 |
| 250 | −55.01 | −54.48 | 0.53 | 0.97% | 5.950 | 6.283 | 0.333 |
| 500 | −51.00 | −50.62 | 0.38 | 0.76% | 60.00 | 59.09 | 0.909 |
| 750 | −50.00 | −49.55 | 0.45 | 0.90% | −13.0 | −12.48 | 0.524 |
| 1000 | −54.00 | −53.70 | 0.30 | 0.56% | 2.000 | 1.565 | 0.435 |

### Visualizations

![Membrane potential v(t)](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/adaptive_exponential_Rosenbrock%E2%80%93Euler_method/graphs/v(t)vst.png?raw=true)

![Recovery variable w(t](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/adaptive_exponential_Rosenbrock%E2%80%93Euler_method/graphs/w(t)vst.png?raw=true)

![W vs V](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/adaptive_exponential_Rosenbrock%E2%80%93Euler_method/graphs/w(t)vsv(t).png?raw=true)

### Assessment

**Strengths**
- Numerically stable on stiff systems without requiring tiny fixed step sizes
- Adaptive stepping yields maximum efficiency, large steps between spikes, small steps during them
- Jacobian-informed update reduces integration error in nonlinear regimes

**Limitations**
- Requires Jacobian computation and matrix inversion at every step
- Only first-order accurate in the base scheme (accuracy comes from adaptivity, not order)
- More complex to implement and debug than standard RK methods

---

## Method 5: Physics-Informed Neural Network (PINN)

**Files:** `DL-PINN_model/code/PINN_Model.ipynb`, `Pure_PINN.ipynb`

The **Physics-Informed Neural Network (PINN)** approach replaces the ODE solver entirely with a neural network trained to satisfy the governing differential equations as a soft constraint in the loss function, no time-stepping, and no discretization. The network learns a continuous function $\hat{v}(t)$, $\hat{w}(t)$ by minimizing a composite loss:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_1 \mathcal{L}_{\text{ODE}_v} + \lambda_2 \mathcal{L}_{\text{ODE}_w} + \mathcal{L}_{\text{IC}}$$

where the ODE residuals enforce the Izhikevich equations at a set of collocation points distributed over the simulation interval.

### Network Architecture

| Component | Specification |
|---|---|
| Hidden layers | 4 × 128 neurons |
| Activation | Tanh |
| Total parameters | 50,050 |
| Input normalization | Yes |
| Outputs | $v(t)$, normalized $w(t)$ |

### Training Configuration

| Phase | Optimizer | Iterations/Epochs |
|---|---|---|
| Phase 1 | Adam | 8,000 epochs |
| Phase 2 | L-BFGS | 100 iterations |
| Collocation points | Adaptive, 2,000 total | — |
| Total training time | 1,252.9 seconds | — |
| Time per epoch | 0.157 s | — |

### Pure PINN vs. Guided PINN

A baseline **Pure PINN** was trained first without external guidance, and found to fail at capturing the sharp action potential spikes due to the continuous-function assumption of neural networks. The solution was to introduce a **feedback mechanism** from the Explicit Euler solver:

1. Detect spike times from the Euler reference solution
2. Concentrate adaptive collocation points around high-dynamic spike regions
3. Retrain with these informed collocation points

This guided approach achieves near-perfect alignment in spike timing, action potential amplitude, and recovery dynamics.

![Pure PINN output](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/DL-PINN_model/plots/Pure_PINN_Graph.png?raw=true)

![Guided PINN vs. reference](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/DL-PINN_model/plots/PINN_vs_REF.png?raw=true)

### Performance Metrics

| Metric | Voltage $v$ | Recovery $w$ |
|---|---|---|
| MAE | 0.311 mV | 0.412 |
| RMSE | 0.507 mV | 0.555 |
| Max Error | 4.896 mV | 3.031 |
| Physics Residual | 1.11 | 0.057 |
| Inference speed | 161,264 pts/s | — |

### Neuronal Behavior Validation

| Property | PINN | Reference |
|---|---|---|
| Spikes detected | 6 | 6 |
| First spike (ms) | 202 | 202 |
| Interspike interval (ms) | 148 | 148 |
| Resting potential (mV) | −59.9 | −60.0 |
| Action potential amplitude (mV) | 94.9 | — |

### Visualizations

![PINN training loss curve](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/DL-PINN_model/plots/TrainingLoss.png?raw=true)

![ ODE residuals over time](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/DL-PINN_model/plots/Residuals.png?raw=true)

![Error over time](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/DL-PINN_model/plots/ErrorOverTime.png?raw=true)

![Phase space diagram](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/DL-PINN_model/plots/PhaseGraph.png?raw=true)

### Biomedical Significance

The PINN formulation is particularly relevant to applications where governing equations are known but data is scarce or expensive. In computational neuroscience this includes:

- **Neural prosthetics design:** continuous, differentiable models of spike timing for control interfaces
- **Epilepsy modeling:** data-free simulation of pathological firing patterns
- **Brain-computer interfaces:** real-time inference at 161 K+ points/second
- **Neuropharmacology:** differentiating through the ODE to study drug-induced parameter shifts

---

## Unity 3D Real-Time Visualizer

**Directory:** `unity/firing_neurons/`  
**3D Assets:** `COMPLETE_NEURON_MODEL.blend`, `COMPLETE_NEURON_MODEL.fbx`

The project includes a full **Unity 3D interactive simulator** that renders the Izhikevich model in real time. When the membrane potential crosses $v_{\text{peak}}$, a particle effect (a light pulse or energy orb) propagates along the 3D neuron mesh, providing an intuitive spatial representation of signal propagation.

### Features

- Real-time Izhikevich ODE integration using **Explicit Euler** and **RK4** (toggle-switchable in Play Mode)
- Interactive **UI sliders** for live parameter adjustment:
  - Input current $I$
  - Membrane capacitance $C$
  - Scaling constants $k$, $a$, $b$
- Custom **3D neuron model** (Blender-authored, FBX-exported)
- Simulation resolution: 1 ms per step
- Universal Render Pipeline (URP) compatible

https://github.com/user-attachments/assets/6ce72532-d7c4-4fa1-ae68-a77b5de2a807

### Requirements

| Requirement | Version |
|---|---|
| Unity | 2022 LTS or later |
| Render Pipeline | URP or Built-in |
| Blender (optional) | Any recent version |

### Setup

```bash
# Clone the repository
git clone https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model.git

# Open Unity Hub → Add Project → navigate to unity/firing_neurons/
# Unity will import all assets automatically

# In the Unity Editor:
# 1. Open Scenes/MainScene.unity
# 2. Press Play
# 3. Adjust sliders, neuron fires when I is large enough
# 4. Toggle the "RK4" checkbox to switch integration method
```

---

## Comparative Analysis

### Stability and Accuracy Summary

| Method | Order | Stability | Step Size | Jacobian Required | Relative Cost |
|---|---|---|---|---|---|
| Explicit Euler | 1st | Conditional | Fixed (1 ms) | No | Lowest |
| Backward Euler | 1st | Unconditional | Fixed (0.25 ms) | Implicit (Newton) | Medium |
| Midpoint / RK2 | 2nd | Conditional | Fixed (1 ms) | No | Low |
| Adaptive ExpRESS–Euler | 1st + adaptive | Stiff-stable | Adaptive (0.01–2 ms) | Yes | Medium–High |
| PINN | N/A | N/A | Continuous | No | Highest (training) |

![full comparison table](https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model/blob/main/Report/Compare_All.png?raw=true)


### Key Findings

The Explicit Euler method establishes the performance baseline and is used as the reference solution for error benchmarking. The Backward Euler method trades computational overhead (Newton iterations) for guaranteed stability, making it better suited to longer simulations or larger step sizes. The Midpoint / RK2 method provides a meaningful accuracy improvement over Euler at modest extra cost and is the recommended starting point for practitioners seeking more than a baseline. The Adaptive ExpRESS–Euler method demonstrates the most sophisticated classical strategy: its Jacobian-based formulation handles the stiffness near spikes gracefully, and its adaptive stepping automatically concentrates evaluation effort where the solution changes fastest. The PINN approach is unique in requiring no time-stepping at all, instead learning the entire solution trajectory as a continuous function. Although its training cost is high (1,252 s), its inference speed (161 K points/s) and perfect spike count agreement with classical methods underscore its viability as a mesh-free alternative for parameter-rich biomedical modeling tasks.

---

## Getting Started

### Prerequisites

```bash
# Python environment
python >= 3.8

# Core dependencies (all methods)
pip install numpy matplotlib

# Midpoint method
pip install pandas

# Adaptive ExpRESS–Euler
pip install scipy

# PINN model
pip install torch
pip install jupyter notebook
```

### Running the Classical Solvers

```bash
# Explicit Euler
cd Explicit_Euler_method/Code
python Euler.py

# Backward Euler
cd Backward_Euler_method/code
python Backward_Euler_Method.py

# Midpoint RK2
cd "Midpoint_(RK2)_method/code"
python "Midpoint(RK2).py"

# Adaptive ExpRESS–Euler
cd "adaptive_exponential_Rosenbrock–Euler_method/code"
python adaptive_express_euler.py
```

### Running the PINN

```bash
cd DL-PINN_model/code

# For the baseline (pure PINN, limited performance):
jupyter notebook Pure_PINN.ipynb

# For the guided PINN (best accuracy):
jupyter notebook PINN_Model.ipynb
```

All notebooks are self-contained and can also be run on **Google Colab** without local GPU setup.

---

## Project Report

A comprehensive written report covering the mathematical derivations, implementation decisions, error analysis, and biomedical context for all five methods is available at:

📄 [`Report/DNM_Report.pdf`](Report/DNM_Report.pdf)

---

## License

This project is released under the **MIT License**. You are free to use, modify, and distribute it with proper attribution.

```
MIT License — Copyright (c) 2025 Mohamed Badawy and contributors
```

---

> *"The Izhikevich model reproduces spiking and bursting behavior of known types of cortical neurons. It combines the biologically plausibility of Hodgkin–Huxley-type dynamics and the computational efficiency of integrate-and-fire neurons."*
> — Eugene M. Izhikevich, 2003
