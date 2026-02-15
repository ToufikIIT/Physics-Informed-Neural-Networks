# Burgers' Equation Solver using Physics-Informed Neural Networks (PINN)

This repository contains a **clean, from-scratch implementation** of Physics-Informed Neural Networks (PINNs) to solve the Burgers' equation in both 1D and 2D domains. 

Unlike standard implementations that rely on PyTorch or TensorFlow's automatic differentiation, this project **manually implements the forward and backward passes** for the neural network, including the computation of higher-order spatial and temporal derivatives ($u_x, u_t, u_{xx}$, etc.) required for the physics-based loss functions.

## 📂 Directory Structure

```
Burgers_equation/
├── 1 Dimensional/
│   ├── Burgers_inviscid/   # Solution for Inviscid Burgers' equation (no viscosity)
│   └── Burgers_viscous/    # Solution for Viscous Burgers' equation (diffusion term included)
├── 2 Dimensional/
│   └── Burgers_viscous/    # Solution for 2D Viscous Burgers' equation
└── shock_wave.py           # Specialized solver for sharp shock waves using viscosity continuation
```

## 🧠 Key Features

- **No Autograd Framework**: The neural network layers (`Linear`, `Tanh`) and the backpropagation logic are implemented using pure `NumPy`.
- **Manual Derivative Calculation**: The network manually computes derivatives like $\frac{\partial u}{\partial x}$ and $\frac{\partial^2 u}{\partial x^2}$ during the forward pass by applying the chain rule to the activation functions.
- **Physics-Informed Loss**: The loss function includes:
  - **PDE Residual**: Enforces the governing equation at collocation points.
  - **Initial Conditions (IC)**: Matches the solution at $t=0$.
  - **Boundary Conditions (BC)**: Enforces conditions at the domain boundaries.
- **Viscosity Continuation (`shock_wave.py`)**: Implements an annealing strategy that starts with high viscosity and gradually reduces it to capture sharp shock fronts without numerical instability.

## 📐 Mathematical Formulations

### 1. Inviscid Burgers' Equation (1D)
Describes fluid motion without viscosity.
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0
$$

### 2. Viscous Burgers' Equation (1D)
Includes a diffusion term with viscosity $\nu$.
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}
$$

### 3. Viscous Burgers' Equation (2D)
Extends the equation to two spatial dimensions ($x, y$).
$$
\begin{cases}
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) \\
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
\end{cases}
$$

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- `numpy`
- `matplotlib`

### Running the Solvers

**1D Viscous Burgers:**
```bash
cd "1 Dimensional/Burgers_viscous"
python main1.py
```

**2D Viscous Burgers:**
```bash
cd "2 Dimensional/Burgers_viscous"
python main.py
```

**Shock Wave Simulation (Viscosity Annealing):**
```bash
python shock_wave.py
```

### Outputs
The scripts typically output:
1. Training progress (loss values).
2. Comparisons between projected PINN solutions and exact analytical solutions.
3. Visualization plots (heatmaps, 3D surface plots, and cross-sections).

## ⚠️ Implementation Notes
Since this is a manual implementation, the backward pass explicitly calculates gradients for the weights ($W$) and biases ($b$) by backpropagating through the PDE residuals. This demonstrates a deep understanding of the computational graph involved in PINN training.
