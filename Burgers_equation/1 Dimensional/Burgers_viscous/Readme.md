# Physics-Informed Neural Network for Burgers' Equation

## Overview

This project implements a **Physics-Informed Neural Network (PINN)** to solve the one-dimensional viscous Burgers' equation with a traveling wave solution. The PINN approach combines deep learning with physics constraints, enabling the neural network to learn solutions that satisfy the underlying partial differential equation (PDE).

---

## Mathematical Formulation

### Burgers' Equation

The one-dimensional viscous Burgers' equation is a nonlinear PDE given by:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

where:
- $u(x,t)$ is the velocity field
- $\nu > 0$ is the kinematic viscosity coefficient (set to $\nu = 0.1$ in this implementation)
- $x \in [-2, 4]$ is the spatial domain
- $t \in [0, 2]$ is the temporal domain

This equation combines nonlinear advection ($u \partial u / \partial x$) with viscous diffusion ($\nu \partial^2 u / \partial x^2$), making it a fundamental model for studying shock waves and turbulence.

### Exact Traveling Wave Solution

The implementation uses a traveling wave solution of the form:

$$u(x,t) = \frac{2}{1 + e^{(x - ct)/\nu}}$$

where $c = 1$ is the wave speed. This solution represents a smooth transition (shock-like) that propagates with constant velocity. The exponential term $(x - ct)/\nu$ controls the sharpness of the transition, with smaller $\nu$ producing steeper gradients.

**Verification**: Substituting this form into Burgers' equation confirms it satisfies the PDE:
- $\frac{\partial u}{\partial t} = \frac{2c e^{(x-ct)/\nu}}{\nu(1 + e^{(x-ct)/\nu})^2}$
- $u \frac{\partial u}{\partial x} = -\frac{4 e^{(x-ct)/\nu}}{\nu(1 + e^{(x-ct)/\nu})^3}$
- $\nu \frac{\partial^2 u}{\partial x^2} = \frac{2 e^{(x-ct)/\nu}(e^{(x-ct)/\nu} - 1)}{\nu(1 + e^{(x-ct)/\nu})^3}$

Combining these terms yields the PDE residual equal to zero.

---

## Physics-Informed Neural Network Methodology

### Network Architecture

The PINN employs a feedforward neural network $u_\theta(x,t)$ parameterized by $\theta = \{W_i, b_i\}$ (weights and biases), with architecture:

- **Input layer**: 2 neurons (spatial coordinate $x$ and time $t$)
- **Hidden layers**: 3 layers with 20 neurons each
- **Activation**: Hyperbolic tangent ($\tanh$) nonlinearity
- **Output layer**: 1 neuron (predicted solution $u(x,t)$)

### Automatic Differentiation for PDE Residuals

A key innovation in this implementation is the use of **automatic differentiation** to compute partial derivatives of the network output with respect to inputs:

- $\frac{\partial u_\theta}{\partial x} = u_x(x,t)$
- $\frac{\partial u_\theta}{\partial t} = u_t(x,t)$  
- $\frac{\partial^2 u_\theta}{\partial x^2} = u_{xx}(x,t)$

These derivatives are computed through the chain rule during forward propagation, enabling exact computation of the PDE residual:

$$R(x,t) = \frac{\partial u_\theta}{\partial t} + u_\theta \frac{\partial u_\theta}{\partial x} - \nu \frac{\partial^2 u_\theta}{\partial x^2}$$

### Loss Function

The training objective combines three loss components:

#### 1. PDE Residual Loss (Collocation Points)

$$L_f = \frac{1}{N_f} \sum_{i=1}^{N_f} \left[ R(x_f^{(i)}, t_f^{(i)}) \right]^2$$

where $(x_f^{(i)}, t_f^{(i)})$ are $N_f = 100$ randomly sampled collocation points within the domain $\Omega = [-2, 4] \times [0, 2]$. This ensures the network satisfies the PDE throughout the domain.

#### 2. Initial Condition Loss

$$L_i = \frac{1}{N_i} \sum_{j=1}^{N_i} \left[ u_\theta(x_i^{(j)}, 0) - u_{\text{exact}}(x_i^{(j)}, 0) \right]^2$$

where $(x_i^{(j)}, 0)$ are $N_i = 50$ points on the initial time slice $t = 0$, enforcing the initial condition.

#### 3. Boundary Condition Loss

$$L_b = \frac{1}{N_b} \sum_{k=1}^{N_b} \left[ u_\theta(x_b^{(k)}, t_b^{(k)}) - u_{\text{exact}}(x_b^{(k)}, t_b^{(k)}) \right]^2$$

where $(x_b^{(k)}, t_b^{(k)})$ are $N_b = 100$ points on the spatial boundaries $x = -2$ and $x = 4$ for various times. This enforces Dirichlet boundary conditions (using exact solution values).

#### Total Loss

The total loss function is:

$$L_{\text{total}} = L_f + L_i + L_b$$

This multi-objective optimization ensures the network satisfies the PDE, initial conditions, and boundary conditions simultaneously.

---

## Implementation Details

### Gradient Computation

The backward pass computes gradients for all loss components:

1. **PDE residual gradients**: 
   - $\frac{\partial L_f}{\partial u} = \frac{2}{N_f} u_x \cdot R$
   - $\frac{\partial L_f}{\partial u_x} = \frac{2}{N_f} u \cdot R$
   - $\frac{\partial L_f}{\partial u_t} = \frac{2}{N_f} R$
   - $\frac{\partial L_f}{\partial u_{xx}} = -\frac{2\nu}{N_f} R$

2. **Initial condition gradients**:
   - $\frac{\partial L_i}{\partial u} = \frac{2}{N_i}(u_\theta - u_{\text{exact}})$

3. **Boundary condition gradients**:
   - $\frac{\partial L_b}{\partial u} = \frac{2}{N_b}(u_\theta - u_{\text{exact}})$

These gradients are backpropagated through the network using the chain rule, with special handling for derivative terms ($u_x$, $u_t$, $u_{xx}$).

### Training Hyperparameters

- **Epochs**: 15,000
- **Batch size**: 100 collocation points per epoch
- **Learning rate**: $\alpha = 10^{-3}$
- **Momentum coefficient**: $\gamma = 0.9$ (for momentum-based optimizer)
- **Viscosity**: $\nu = 0.1$

### Optimization Algorithm

The network uses a **momentum-based gradient descent** optimizer:

$$v_t = \gamma v_{t-1} + \nabla_\theta L$$
$$\theta_{t+1} = \theta_t - \alpha v_t$$

where $v_t$ is the velocity vector, $\gamma$ is the momentum coefficient, and $\alpha$ is the learning rate.

---

## Key Features

1. **Automatic Differentiation**: Computes exact derivatives $u_x$, $u_t$, $u_{xx}$ during forward pass
2. **Physics-Informed Learning**: Incorporates PDE residual directly into loss function
3. **Boundary-Value Problem**: Handles both initial and boundary conditions
4. **From Scratch Implementation**: Custom neural network layers with derivative tracking

---

## Results

After training, the PINN accurately approximates the traveling wave solution across different time instances. The network learns:
- The smooth transition profile of the traveling wave
- The temporal evolution of the solution
- Satisfies the nonlinear PDE throughout the domain

Visualization compares the PINN prediction against the exact solution at multiple time steps ($t = 0.0, 0.5, 0.75$), demonstrating agreement between the learned and analytical solutions.

---


## File Structure

```
PINN_burgers/
├── main.py       # Main implementation with PINN architecture and training loop
└── Readme.md     # This documentation file
```

---

## Usage

Run the training script:

```bash
python main.py
```

The script will:
1. Initialize the PINN with the specified architecture
2. Train for 15,000 epochs using the combined loss function
3. Generate visualization comparing PINN predictions with exact solutions at multiple time instances

---

## Mathematical Notes

### Why Traveling Wave Solutions?

Traveling wave solutions of the form $u(x,t) = f(x - ct)$ are particularly useful because they transform the PDE into an ODE. Substituting into Burgers' equation:

$$-c f'(\xi) + f(\xi) f'(\xi) = \nu f''(\xi), \quad \xi = x - ct$$

This allows analytical solution techniques and provides benchmark solutions for numerical methods.

### Numerical Stability

The implementation clips the exponent $(x - ct)/\nu$ to $[-50, 50]$ to prevent numerical overflow in the exponential function, ensuring stable computation of the exact solution during training.

---

## Extensions

Potential enhancements to this implementation:

1. **Adaptive sampling**: Focus collocation points in regions of high solution gradients
2. **Weighted losses**: Adjust relative importance of PDE, initial, and boundary losses
3. **Curriculum learning**: Gradually increase solution complexity during training
4. **Different architectures**: Experiment with residual connections or attention mechanisms
5. **Inverse problems**: Infer unknown parameters (e.g., $\nu$) from solution data

