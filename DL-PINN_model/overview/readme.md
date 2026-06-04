# 🧠 Dynamic Neuron Model, PINN Method

[This project](https://izhikevich-pinn-results.vercel.app) is a dynamic visualization tool built in **JavaScript** to showcase the results of a Physics-Informed Neural Network (PINN) trained to simulate **neuron spiking dynamics** using the **Izhikevich model**.

Rather than just presenting static numbers, this tool brings the analysis to life with **interactive plots and animated graphs** — offering an intuitive understanding of model performance, errors, neuronal behavior, and biomedical significance.

---

## 📽️ Live Demo (Video)

[![Watch Demo](https://img.youtube.com/vi/3EBOD50E0xU/0.jpg)](https://www.youtube.com/watch?v=3EBOD50E0xU)

---

## 📊 Key Analysis Visualized

Each of the following sections is visualized dynamically in your browser via the JavaScript app:

### 1. 🔧 Neural Network Architecture
- 4 hidden layers × 128 neurons
- 50,050 total parameters
- Tanh activation, input normalization
- Outputs: `v(t)`, normalized `w(t)`

### 2. 🚀 Training Performance
- Training time: 1252.9s
- Optimizers: Adam (8000 epochs), L-BFGS (100 iterations)
- 2000 adaptive collocation points
- Final loss breakdown displayed visually

### 3. 📉 Error Metrics
| Metric         | Voltage (v) | Recovery (w) |
|----------------|-------------|--------------|
| MAE            | 0.311 mV    | 0.412        |
| RMSE           | 0.507 mV    | 0.555        |
| Max Error      | 4.896 mV    | 3.031        |
| Physics Residual | 1.11       | 0.057        |

### 4. 🧪 Neuronal Behavior
- 6 spikes detected (both PINN and reference)
- Perfect spike timing alignment (first spike at 202 ms)
- Accurate interspike interval: 148 ms
- Resting potential: -59.9 mV
- Action potential amplitude: 94.9 mV

### 5. ⚙️ Computational Performance
- Time per epoch: 0.157 s
- Inference speed: 161,264 points/sec

### 6. 📈 Dynamic Graphs Included
- PINN vs. Reference voltage and recovery
- Error over time
- Training loss curve
- Residuals plot
- Phase space diagram

---

## 🧠 Biomedical Significance

This model offers direct applicability in:

- Neural prosthetics design
- Epilepsy modeling
- Brain-computer interface research
- Computational neuroscience simulations
- Drug effect analysis (neuropharmacology)

---
