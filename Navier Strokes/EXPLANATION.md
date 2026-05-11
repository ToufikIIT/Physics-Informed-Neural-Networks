# Detailed Explanation and Derivations (Navier-Stokes PINN)

This document explains the code in [main.py](main.py) from first principles: the PDE, the stream-function formulation, the PINN architecture, how higher-order derivatives are propagated, and how gradients are derived and backpropagated.

## 1) Problem Setup

We solve the steady 2D incompressible Navier-Stokes equations on $[0,1] \times [0,1]$ with viscosity $\nu$:

$$
\begin{aligned}
&u u_x + v u_y + p_x - \nu (u_{xx} + u_{yy}) = 0,\\
&u v_x + v v_y + p_y - \nu (v_{xx} + v_{yy}) = 0,\\
&u_x + v_y = 0.
\end{aligned}
$$

The code uses the Kovasznay analytical solution (a classic exact solution of steady Navier-Stokes) to provide boundary targets and validation.

### Kovasznay Solution

Define the Reynolds number and parameter:

$$
Re = 100, \quad \nu = \frac{1}{Re}, \quad \lambda = \frac{Re}{2} - \sqrt{\frac{Re^2}{4} + 4\pi^2}.
$$

Exact fields:

$$
\begin{aligned}
&u(x,y) = 1 - e^{\lambda x} \cos(2\pi y),\\
&v(x,y) = \frac{\lambda}{2\pi} e^{\lambda x} \sin(2\pi y),\\
&p(x,y) = \tfrac{1}{2} \left(1 - e^{2\lambda x}\right).
\end{aligned}
$$

These appear in `exact_u`, `exact_v`, `exact_p`.

## 2) Stream Function Formulation

To satisfy incompressibility automatically, the code uses a stream function $\psi(x,y)$:

$$
\begin{aligned}
&u = \psi_y,\\
&v = -\psi_x.
\end{aligned}
$$

Then

$$
\begin{aligned}
&u_x + v_y = \psi_{yx} - \psi_{xy} = 0
\end{aligned}
$$

by equality of mixed partials. This removes the divergence constraint from the loss. The network outputs two scalar fields:

$$
\text{network}(x,y) = [\psi(x,y), p(x,y)].
$$

## 3) Network Inputs, Outputs, and Derivatives

The input is $X = [x, y]^T$ with shape $2 \times N$. The forward pass returns the output and its derivatives up to third order:

$$
\left(\text{out}, \text{out}_x, \text{out}_y, \text{out}_{xx}, \text{out}_{yy}, \text{out}_{xy}, \text{out}_{xxx}, \text{out}_{xxy}, \text{out}_{xyy}, \text{out}_{yyy}\right).
$$

The code computes these derivatives **analytically through the network** by applying chain rules layer by layer (no automatic differentiation library).

## 4) Layer Derivative Propagation (Forward)

Each layer is either:

- Linear: $z = W h + b$
- SiLU: $a = \text{SiLU}(z) = z \sigma(z)$

where $h$ is the layer input and $z$ is the pre-activation output.

### 4.1) Linear Layer

For $z = W h + b$ and each derivative tensor of $h$:

$$
\begin{aligned}
&z_x = W h_x, \quad z_y = W h_y,\\
&z_{xx} = W h_{xx}, \quad z_{yy} = W h_{yy}, \quad z_{xy} = W h_{xy},\\
&z_{xxx} = W h_{xxx}, \quad z_{xxy} = W h_{xxy},\\
&z_{xyy} = W h_{xyy}, \quad z_{yyy} = W h_{yyy}.
\end{aligned}
$$

This is why the `Linear.forward` just multiplies all derivative arrays by $W$.

### 4.2) SiLU Activation

Define $s = \sigma(z) = \frac{1}{1 + e^{-z}}$ and $a = z s$.

The code precomputes the first four derivatives of $a$ with respect to $z$:

$$
\begin{aligned}
&f_1 = \frac{d a}{d z}, \\
&f_2 = \frac{d^2 a}{d z^2}, \\
&f_3 = \frac{d^3 a}{d z^3}, \\
&f_4 = \frac{d^4 a}{d z^4}.
\end{aligned}
$$

These are derived from $a = z s$ and the derivatives of the sigmoid:

$$
\begin{aligned}
&s' = s(1-s),\\
&s'' = s'(1 - 2s),\\
&s''' = s''(1 - 2s) - 2(s')^2,\\
&s'''' = s'''(1 - 2s) - 6 s' s''.
\end{aligned}
$$

Then:

$$
\begin{aligned}
&f_1 = s + z s',\\
&f_2 = 2 s' + z s'',\\
&f_3 = 3 s'' + z s''',\\
&f_4 = 4 s''' + z s''''.
\end{aligned}
$$

The code computes these as `f1..f4` and then uses the multivariate chain rule to propagate spatial derivatives.

### 4.3) Example: First and Second Spatial Derivatives

Using $a(z(x,y))$:

$$
\begin{aligned}
&a_x = f_1 z_x,\\
&a_y = f_1 z_y,\\
&a_{xx} = f_2 z_x^2 + f_1 z_{xx},\\
&a_{yy} = f_2 z_y^2 + f_1 z_{yy},\\
&a_{xy} = f_2 z_x z_y + f_1 z_{xy}.
\end{aligned}
$$

### 4.4) Third Spatial Derivatives

The code applies the higher-order chain rule, for example:

$$
\begin{aligned}
&a_{xxx} = f_3 z_x^3 + 3 f_2 z_x z_{xx} + f_1 z_{xxx},\\
&a_{xxy} = f_3 z_x^2 z_y + f_2 (2 z_x z_{xy} + z_{xx} z_y) + f_1 z_{xxy},\\
&a_{xyy} = f_3 z_x z_y^2 + f_2 (2 z_y z_{xy} + z_{yy} z_x) + f_1 z_{xyy},\\
&a_{yyy} = f_3 z_y^3 + 3 f_2 z_y z_{yy} + f_1 z_{yyy}.
\end{aligned}
$$

These appear in `SiLU.forward`.

## 5) PDE Residuals

Let the network output be:

$$
\text{out} = [\psi, p].
$$

Then:

$$
\begin{aligned}
&u = \psi_y,\\
&v = -\psi_x,\\
&u_x = \psi_{xy}, \quad u_y = \psi_{yy},\\
&v_x = -\psi_{xx}, \quad v_y = -\psi_{xy},\\
&u_{xx} = \psi_{xxy}, \quad u_{yy} = \psi_{yyy},\\
&v_{xx} = -\psi_{xxx}, \quad v_{yy} = -\psi_{xyy},\\
&p_x = p_x, \quad p_y = p_y.
\end{aligned}
$$

The residuals are:

$$
\begin{aligned}
&R_u = u u_x + v u_y + p_x - \nu(u_{xx} + u_{yy}),\\
&R_v = u v_x + v v_y + p_y - \nu(v_{xx} + v_{yy}).
\end{aligned}
$$

## 6) Loss Function

The code uses log-cosh for robustness:

$$
\mathcal{L}_{pde} = \mathbb{E}\left[\log\cosh(R_u)\right] + \mathbb{E}\left[\log\cosh(R_v)\right].
$$

The derivative is:

$$
\frac{d}{dR} \log\cosh(R) = \tanh(R).
$$

So:

$$
\frac{\partial \mathcal{L}_{pde}}{\partial R_u} = \frac{1}{N} \tanh(R_u),
\quad
\frac{\partial \mathcal{L}_{pde}}{\partial R_v} = \frac{1}{N} \tanh(R_v).
$$

The code divides by the number of PDE points explicitly.

## 7) Backpropagation of PDE Residuals

The backpropagation here is **manual**: we distribute the gradient of $R_u, R_v$ into gradients of network outputs and their derivatives.

### 7.1) Notation

Let $dR_u = \frac{\partial \mathcal{L}}{\partial R_u}$ and $dR_v = \frac{\partial \mathcal{L}}{\partial R_v}$. We need to accumulate gradients for:

$$
\frac{\partial \mathcal{L}}{\partial \psi},
\frac{\partial \mathcal{L}}{\partial \psi_x},
\frac{\partial \mathcal{L}}{\partial \psi_y},
\frac{\partial \mathcal{L}}{\partial \psi_{xx}},
\ldots
$$

The code stores these in arrays named `d_out, d_ox, d_oy, d_oxx, ...` where:

- `out` is $[\psi, p]$
- `ox` is derivative of `out` w.r.t. $x$
- `oy` is derivative w.r.t. $y$
- `oxx` is second derivative, etc.

### 7.2) Derivatives of Residuals

Recall:

$$
R_u = u u_x + v u_y + p_x - \nu(u_{xx} + u_{yy}).
$$

and $u = \psi_y$, $v = -\psi_x$.

Partial derivatives (treating all fields as independent for backprop):

$$
\begin{aligned}
\frac{\partial R_u}{\partial u} &= u_x,\\
\frac{\partial R_u}{\partial u_x} &= u,\\
\frac{\partial R_u}{\partial u_y} &= v,\\
\frac{\partial R_u}{\partial v} &= u_y,\\
\frac{\partial R_u}{\partial p_x} &= 1,\\
\frac{\partial R_u}{\partial u_{xx}} &= -\nu,\\
\frac{\partial R_u}{\partial u_{yy}} &= -\nu.
\end{aligned}
$$

Similarly for $R_v$:

$$
\begin{aligned}
\frac{\partial R_v}{\partial v} &= v_y,\\
\frac{\partial R_v}{\partial v_x} &= u,\\
\frac{\partial R_v}{\partial v_y} &= v,\\
\frac{\partial R_v}{\partial u} &= v_x,\\
\frac{\partial R_v}{\partial p_y} &= 1,\\
\frac{\partial R_v}{\partial v_{xx}} &= -\nu,\\
\frac{\partial R_v}{\partial v_{yy}} &= -\nu.
\end{aligned}
$$

Then apply the chain rules $u = \psi_y$, $v = -\psi_x$.

### 7.3) Mapping to Network Derivatives

Because $u$ and $v$ depend on $\psi$, we map gradients into derivatives of $\psi$:

- $u = \psi_y$ so gradients on $u$ go to $\psi_y$.
- $v = -\psi_x$ so gradients on $v$ go to $-\psi_x$.
- $u_x = \psi_{xy}$, $u_y = \psi_{yy}$.
- $v_x = -\psi_{xx}$, $v_y = -\psi_{xy}$.
- $u_{xx} = \psi_{xxy}$, $u_{yy} = \psi_{yyy}$.
- $v_{xx} = -\psi_{xxx}$, $v_{yy} = -\psi_{xyy}$.

This is why the code updates `d_ox`, `d_oy`, `d_oxx`, `d_oxy`, `d_oxxx`, `d_oxxy`, `d_oxyy`, `d_oyyy` with the signs shown.

### 7.4) Example: How `d_ox` and `d_oy` Are Built

From the code:

```python
    d_ox[0:1] = dRu * (-uy) + dRv * (-vy)
    d_oy[0:1] = dRu * (ux) + dRv * (vx)
```

Interpretation:

- `d_ox[0:1]` corresponds to $\frac{\partial \mathcal{L}}{\partial \psi_x}$.
- Since $v = -\psi_x$, any term using $v$ contributes a negative sign.

For example, in $R_u$ the term $v u_y$ gives gradient $\partial R_u / \partial v = u_y$, then $\partial v / \partial \psi_x = -1$ so contribution is $-u_y$. Multiply by $dR_u$ and accumulate.

### 7.5) Pressure Gradients

Pressure appears directly via $p_x$ and $p_y$:

```python
    d_ox[1:2], d_oy[1:2] = dRu, dRv
```

- `d_ox[1:2]` is $\partial \mathcal{L} / \partial p_x$.
- `d_oy[1:2]` is $\partial \mathcal{L} / \partial p_y$.

## 8) Boundary Condition Loss

The boundary conditions enforce $u$ and $v$ on the boundary using the exact solution.

Boundary loss:

$$
\mathcal{L}_{bc} = \mathbb{E}\left[\log\cosh(u - u_{exact})\right] + \mathbb{E}\left[\log\cosh(v - v_{exact})\right].
$$

Then:

$$
\frac{\partial \mathcal{L}_{bc}}{\partial u} = \frac{1}{N_{bc}} \tanh(u - u_{exact}),
\quad
\frac{\partial \mathcal{L}_{bc}}{\partial v} = \frac{1}{N_{bc}} \tanh(v - v_{exact}).
$$

Mapping to $\psi$ derivatives:

- $u = \psi_y \Rightarrow$ gradient goes to $\psi_y$.
- $v = -\psi_x \Rightarrow$ gradient goes to $-\psi_x$.

That yields:

```python
    d_bc_ox[0:1] = -dv_bc
    d_bc_oy[0:1] = du_bc
```

## 9) Backprop Through the Network

Once the gradient arrays for $(\text{out}, \text{out}_x, \text{out}_y, \ldots)$ are computed, the network backpropagation occurs layer by layer.

### 9.1) Linear Layer Backward

Given $z = W h + b$, the gradient for $W$ and $b$ is:

$$
\begin{aligned}
&\frac{\partial \mathcal{L}}{\partial W} = dz \cdot h^T + dz_x \cdot h_x^T + dz_y \cdot h_y^T + \cdots,\\
&\frac{\partial \mathcal{L}}{\partial b} = \sum_i dz_i.
\end{aligned}
$$

All derivative terms contribute to the same $W$ because $W$ multiplies every derivative of $h$.

The gradient w.r.t. $h$ and its derivatives is simply:

$$
\begin{aligned}
&dh = W^T dz, \quad dh_x = W^T dz_x, \quad dh_y = W^T dz_y,\\
&dh_{xx} = W^T dz_{xx}, \quad \ldots
\end{aligned}
$$

### 9.2) SiLU Backward (Multivariate)

Because $a = f(z)$, the backward pass uses the higher-order derivatives $f_1, f_2, f_3, f_4$ stored in the forward pass. The code applies the multivariate chain rule to map gradients on $a, a_x, a_y, a_{xx}, ...$ back to gradients on $z, z_x, z_y, z_{xx}, ...$.

For example, the gradient on $z_x$ includes contributions from:

- $a_x = f_1 z_x$
- $a_{xx} = f_2 z_x^2 + f_1 z_{xx}$
- $a_{xy} = f_2 z_x z_y + f_1 z_{xy}$
- $a_{xxx}, a_{xxy}, a_{xyy}$ terms

This is why the backward formulas in `SiLU.backward` are long and include terms like:

$$
\frac{\partial a_{xx}}{\partial z_x} = 2 f_2 z_x,
\quad
\frac{\partial a_{xxx}}{\partial z_x} = 3 f_3 z_x^2 + 3 f_2 z_{xx},
\quad \ldots
$$

## 10) Optimization (Adam)

Each linear layer uses Adam with parameters:

$$
\beta_1 = 0.9, \; \beta_2 = 0.999, \; \epsilon = 10^{-8}.
$$

For parameter $\theta$ and gradient $g$:

$$
\begin{aligned}
&m_t = \beta_1 m_{t-1} + (1 - \beta_1) g,\\
&v_t = \beta_2 v_{t-1} + (1 - \beta_2) g^2,\\
&\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\\
&\hat{v}_t = \frac{v_t}{1 - \beta_2^t},\\
&\theta \leftarrow \theta - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
\end{aligned}
$$

## 11) Pressure Anchoring

The PDE only determines pressure up to a constant. The code shifts the predicted pressure to match the mean of the exact pressure:

$$
\tilde{p} = p_{pred} - \overline{p_{pred}} + \overline{p_{exact}}.
$$

This avoids comparing two pressure fields that differ by a constant offset.

## 12) Visualization

The code plots:

- 2D contour comparisons for $u$, $v$, and $p$ with absolute error.
- Training loss in log scale.
- 3D surface plots for $u$, $v$, and $p$ plus error.

These are saved as images in the folder (`u.png`, `v.png`, `p.png`, etc.) if you choose to save them after plotting.

## 13) Summary of Data Flow

1. Sample interior points.
2. Forward pass through the network with derivatives.
3. Build PDE residuals.
4. Compute log-cosh loss and its gradient.
5. Map residual gradients back to network derivatives.
6. Sample boundary points and compute BC loss.
7. Combine PDE and BC gradients.
8. Backprop through the network.
9. Update weights with Adam.

If you want, I can also add step-by-step derivations for each term in `SiLU.backward` in a separate appendix.
