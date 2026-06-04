# 🧠 Neuron Dynamics Simulator in Unity

This Unity project simulates the electrical behavior of a spiking neuron using the **Izhikevich Dynamic Neuron Model**. Users can visualize neuron firing in real time with a particle effect traveling along the neuron's 3D model, and adjust model parameters using UI sliders.

---

## 🚀 Features

- Real-time simulation of a spiking neuron using **Euler’s Method**, **RK4 Method**
- Editable parameters:
  - Input Current (I)
  - Capacitance (C)
  - Model constants (k, a, b)
- Firing effect visualized as a light pulse or energy orb traveling along the neuron body
- Simulation time resolution of 1ms per step

---

## 🧬 The Dynamic Neuron Model

The following differential equations define the model:
C * dv/dt = k*(v - vr)(v - vt) - w + I
dw/dt = a(b*(v - vr) - w)


Where:
- `v`: membrane potential
- `w`: recovery variable
- `C`: Membrane capacitance
- `vr`: Resting membrane potential
- `vt`: Instantaneous threshold potential
- `k`: Controls the sharpness of spike initiation
- `a`: Recovery time scale
- `b`: Sensitivity of recovery variable
- `c`: Reset value of membrane potential after spike
- `d`: Recovery variable increment after spike
- `v_peak`: Spike cutoff (maximum voltage)
- `I`: input current

Spikes are detected when `v >= v_peak`, triggering a reset and firing effect.

---

### 🛠️ Installation and Setup

### 1. Requirements
- Unity 2022 or later (URP or Built-in Render Pipeline)
- Git installed on your system
- Blender (optional, for editing the neuron model)

---

### 2. Clone the Repository from GitHub

If you're using **GitHub Desktop**:
1. Click **"Code" > "Open with GitHub Desktop"**
2. Choose a local path to clone the repository
3. Click **"Clone"**

If you're using the **command line**:
git clone https://github.com/Kareem-Taha-05/Izhikevich-Dynamic-Neuron-Model.git

---

### 3. Open the Project in Unity

1. Launch **Unity Hub**.
2. Click on **Open**.
3. Navigate to the folder where you cloned the repository (it should include folders like `Assets/`, `ProjectSettings/`, etc.).
4. Select the main project folder and click **Open**.
5. Unity will begin importing assets and compiling scripts. Wait until the editor is fully ready.

---

### 4. Run the Simulation

1. In the **Project** window, go to the `Scenes/` folder and double-click `MainScene.unity` to open it.
2. Click the **Play** button at the top of the Unity editor to start the simulation.
3. You will see sliders on the canvas for controlling neuron parameters:
   - **I** – Input current
   - **C** – Membrane capacitance
   - **k**, **a**, **b** – Constants controlling the differential behavior
4. Adjust the sliders during simulation. When the input current is high enough, the neuron will spike.
5. In Play Mode, use the toggle labeled “RK4” to switch between:
   - **Unchecked** – Simulation runs using Euler’s method
   - **Checked** – Simulation runs using the Runge-Kutta 4th order (RK4) method
6. A visual effect will move along the neuron's body, simulating a signal.

> Tip: The simulation runs in real-time and responds instantly to slider adjustments.

---

✅ You’re now ready to experiment with neuron dynamics in Unity!

---

#### 5. Demo Video
https://github.com/user-attachments/assets/6ce72532-d7c4-4fa1-ae68-a77b5de2a807
