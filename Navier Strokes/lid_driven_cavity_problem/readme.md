
# Lid-Driven Cavity Flow (PINN)

Physics-Informed Neural Network (PINN) solvers for the steady 2D incompressible Navier-Stokes lid-driven cavity problem on the unit square. This folder includes two implementations:

- A pure NumPy version with manual higher-order derivatives and a custom optimizer.
- A PyTorch version using autograd with Adam + L-BFGS and a benchmark comparison against Ghia et al. (1982).

## Problem setup

We solve the steady incompressible Navier-Stokes equations on $[0,1] \times [0,1]$:

$$
u u_x + v u_y + p_x - \nu (u_{xx} + u_{yy}) = 0,
$$
$$
u v_x + v v_y + p_y - \nu (v_{xx} + v_{yy}) = 0,
$$
$$
u_x + v_y = 0.
$$

The velocity field is parameterized with a stream function $\psi$:

$$
u = \psi_y, \quad v = -\psi_x.
$$

### Boundary conditions (lid-driven cavity)

- Top wall (moving lid): $u=1$, $v=0$ at $y=1$.
- Bottom, left, right walls (no-slip): $u=0$, $v=0$.
- Stream function anchored on the boundary: $\psi = 0$ (closed streamline).

Default viscosity is $\nu = 1 / Re$ with $Re=100$.

## Files

- [main.py](main.py): NumPy PINN with manual derivatives up to third order and log-cosh loss.
- [pytorch.py](pytorch.py): PyTorch PINN with autograd, Adam + L-BFGS, and benchmark plots.

## Requirements

Python 3.9+ recommended.

NumPy version:

```bash
pip install numpy matplotlib
```

PyTorch version (CPU or CUDA):

```bash
pip install torch numpy matplotlib
```

## Quick start

From this folder:

```bash
python main.py
```

or

```bash
python pytorch.py
```

## What each script does

### NumPy implementation (main.py)

- Network outputs $[\psi, p]$.
- Manual propagation of spatial derivatives up to third order through SiLU layers.
- Log-cosh loss on PDE residuals and boundary losses.
- Plots shown at the end of training (velocity magnitude, streamlines, u/v fields, 3D surfaces, loss curve).

### PyTorch implementation (pytorch.py)

- Autograd-based derivatives with a deeper network and SiLU activations.
- Training schedule:
	- Adam optimizer for the first phase.
	- L-BFGS refinement on a fixed collocation set.
- Chebyshev-clustered sampling in the domain.
- Smooth lid profile to reduce corner singularities.
- Pressure gauge loss at the center to fix the pressure constant.
- Saves plots to `./outputs`:
	- `cavity_2d.png`
	- `loss_curve.png`
	- `centerline_benchmark.png` (compares against Ghia et al. 1982)
	- `3d_u_Velocity.png`, `3d_v_Velocity.png`, `3d_Pressure.png`

## Key parameters to tune

NumPy (main.py):

- `epochs`: number of training epochs.
- `lr`: learning rate.
- `Re`: Reynolds number (sets viscosity `nu`).
- `N_bc`: boundary points per side.
- `bc_w`: boundary loss weight.
- Network architecture: `PINN([2, 64, 64, 64, 64, 2])`.

PyTorch (pytorch.py):

- `EPOCHS`, `LBFGS_MAX`, `LR`.
- `RE`, `NU`.
- `N_PDE`, `N_BC`.
- `BC_WEIGHT`, `GAUGE_WEIGHT`, `EPS`.
- `CHECKPOINT_STEPS`: epochs to snapshot models for the benchmark plot.

## Notes and tips

- Pressure is defined up to a constant. The PyTorch version adds a center-point gauge loss; the NumPy version anchors $\psi$ on the boundary.
- If training is unstable, increase `BC_WEIGHT` and reduce `LR`.
- For reproducibility, set random seeds (NumPy and torch) near the top of each script.
- For faster runs, lower `N_PDE`, `N_BC`, or the network width.

