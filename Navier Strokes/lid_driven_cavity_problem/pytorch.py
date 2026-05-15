"""
Physics-Informed Neural Network (PINN) for Lid-Driven Cavity Flow
Re = 100, using stream-function formulation with PyTorch autograd.

Stream function: psi
  u =  d(psi)/dy
  v = -d(psi)/dx
  (continuity is automatically satisfied)

PDE: Navier-Stokes (steady, incompressible)
  u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
  u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0

Boundary Conditions (unit square [0,1]^2):
  Top    (y=1): u=1, v=0, psi=0
  Bottom (y=0): u=0, v=0, psi=0
  Left   (x=0): u=0, v=0, psi=0
  Right  (x=1): u=0, v=0, psi=0
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS     = 15_000
LR         = 2e-3
RE         = 100.0
NU         = 1.0 / RE
N_PDE      = 1000   # collocation points per epoch
N_BC       = 250    # boundary points per side per epoch
BC_WEIGHT  = 70.0   # weight on boundary loss

# ── Logcosh loss (robust, smooth alternative to MSE) ──────────────────────────
def logcosh_loss(residual: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.log(torch.cosh(residual + 1e-12)))

# ── Network ────────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    """
    Input:  (x, y)  →  2 neurons
    Output: (psi, p) →  2 neurons
    Hidden: 4 × 64  with SiLU activations
    """
    def __init__(self, layers=(2, 64, 64, 64, 64, 2)):
        super().__init__()
        blocks = []
        for i in range(len(layers) - 1):
            blocks.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                blocks.append(nn.SiLU())
        self.net = nn.Sequential(*blocks)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """xy: (N, 2) → (N, 2)  [psi, p]"""
        return self.net(xy)


# ── Derivative helper using autograd ──────────────────────────────────────────
def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """First-order gradient, summed over output dims → shape matches inputs."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True
    )[0]


def get_derivatives(xy: torch.Tensor, net: PINN):
    """
    Returns all velocity/pressure fields and required derivatives.
    xy must have requires_grad=True.
    """
    out   = net(xy)            # (N, 2)
    psi   = out[:, 0:1]        # stream function
    p     = out[:, 1:2]        # pressure

    # ── 1st derivatives of psi ────────────────────────────────────────────────
    psi_grad = grad(psi, xy)          # (N, 2)
    psi_x    = psi_grad[:, 0:1]
    psi_y    = psi_grad[:, 1:2]

    # velocity
    u =  psi_y     # u =  ∂ψ/∂y
    v = -psi_x     # v = -∂ψ/∂x

    # ── 2nd derivatives of psi → 1st of u, v ─────────────────────────────────
    u_grad = grad(u, xy)
    u_x    = u_grad[:, 0:1]
    u_y    = u_grad[:, 1:2]

    v_grad = grad(v, xy)
    v_x    = v_grad[:, 0:1]
    v_y    = v_grad[:, 1:2]

    # ── 3rd derivatives of psi → 2nd of u, v (needed for viscous terms) ───────
    u_xx = grad(u_x, xy)[:, 0:1]
    u_yy = grad(u_y, xy)[:, 1:2]
    v_xx = grad(v_x, xy)[:, 0:1]
    v_yy = grad(v_y, xy)[:, 1:2]

    # ── pressure gradients ────────────────────────────────────────────────────
    p_grad = grad(p, xy)
    p_x    = p_grad[:, 0:1]
    p_y    = p_grad[:, 1:2]

    return psi, p, u, v, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y


# ── Model + Optimiser ──────────────────────────────────────────────────────────
net       = PINN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

loss_history = []

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
for ep in range(EPOCHS):

    optimizer.zero_grad()

    # ── 1. PDE loss (interior collocation points) ─────────────────────────────
    xy_pde = torch.rand(N_PDE, 2, device=device, requires_grad=True)

    _, _, u, v, u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy, p_x, p_y = \
        get_derivatives(xy_pde, net)

    Ru = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)   # x-momentum
    Rv = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)   # y-momentum

    loss_pde = logcosh_loss(Ru) + logcosh_loss(Rv)

    # ── 2. Boundary losses ────────────────────────────────────────────────────
    eps = 1e-3
    x_b = torch.FloatTensor(N_BC).uniform_(eps, 1 - eps).to(device)
    y_b = torch.FloatTensor(N_BC).uniform_(eps, 1 - eps).to(device)
    ones  = torch.ones(N_BC,  device=device)
    zeros = torch.zeros(N_BC, device=device)

    def make_bc_xy(x_coords, y_coords):
        xy = torch.stack([x_coords, y_coords], dim=1)
        xy.requires_grad_(True)
        return xy

    # Top (moving lid): u=1, v=0, psi=0
    xy_top  = make_bc_xy(x_b, ones)
    # Bottom:           u=0, v=0, psi=0
    xy_bot  = make_bc_xy(x_b, zeros)
    # Left:             u=0, v=0, psi=0
    xy_left = make_bc_xy(zeros, y_b)
    # Right:            u=0, v=0, psi=0
    xy_right= make_bc_xy(ones,  y_b)

    def bc_loss(xy, u_target, v_target):
        out    = net(xy)
        psi_bc = out[:, 0:1]

        psi_grad = grad(psi_bc, xy)
        u_bc     =  psi_grad[:, 1:2]   #  ∂ψ/∂y
        v_bc     = -psi_grad[:, 0:1]   # -∂ψ/∂x

        l_u   = logcosh_loss(u_bc.squeeze() - u_target)
        l_v   = logcosh_loss(v_bc.squeeze() - v_target)
        l_psi = logcosh_loss(psi_bc.squeeze())          # psi = 0 on walls
        return l_u + l_v + l_psi

    loss_bc = BC_WEIGHT * (
        bc_loss(xy_top,   ones,  zeros) +   # lid moves at u=1
        bc_loss(xy_bot,   zeros, zeros) +
        bc_loss(xy_left,  zeros, zeros) +
        bc_loss(xy_right, zeros, zeros)
    )

    # ── 3. Total loss + back-prop ─────────────────────────────────────────────
    total_loss = loss_pde + loss_bc
    total_loss.backward()
    optimizer.step()

    loss_history.append(total_loss.item())

    if ep % 1000 == 0:
        print(f"Epoch {ep:5d} | Loss = {total_loss.item():.6f}  "
              f"(PDE = {loss_pde.item():.6f}, BC = {loss_bc.item():.6f})")

print("Training complete. Generating visualizations…")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION ON A UNIFORM GRID
# ══════════════════════════════════════════════════════════════════════════════
N  = 100
x1 = np.linspace(0, 1, N)
y1 = np.linspace(0, 1, N)
X, Y = np.meshgrid(x1, y1)

XY_eval = torch.tensor(
    np.stack([X.flatten(), Y.flatten()], axis=1),
    dtype=torch.float32, device=device
).requires_grad_(True)

with torch.enable_grad():
    out_eval = net(XY_eval)
    psi_eval = out_eval[:, 0:1]
    p_eval   = out_eval[:, 1:2]

    psi_g  = grad(psi_eval, XY_eval)
    u_eval =  psi_g[:, 1:2]
    v_eval = -psi_g[:, 0:1]

psi_pred = psi_eval.detach().cpu().numpy().reshape(N, N)
p_pred   = p_eval  .detach().cpu().numpy().reshape(N, N)
u_pred   = u_eval  .detach().cpu().numpy().reshape(N, N)
v_pred   = v_eval  .detach().cpu().numpy().reshape(N, N)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# 1. 2 × 2 summary
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
speed = np.sqrt(u_pred**2 + v_pred**2)

c0 = axs[0, 0].contourf(X, Y, speed, 100, cmap='jet')
axs[0, 0].set_title("Velocity Magnitude")
fig.colorbar(c0, ax=axs[0, 0])

axs[0, 1].streamplot(X, Y, u_pred, v_pred, color=speed, cmap='jet', density=1.5)
axs[0, 1].set_title("Streamlines")
axs[0, 1].set_xlim([0, 1]); axs[0, 1].set_ylim([0, 1])

c2 = axs[1, 0].contourf(X, Y, u_pred, 100, cmap='jet')
axs[1, 0].set_title("Horizontal Velocity u")
fig.colorbar(c2, ax=axs[1, 0])

c3 = axs[1, 1].contourf(X, Y, v_pred, 100, cmap='jet')
axs[1, 1].set_title("Vertical Velocity v")
fig.colorbar(c3, ax=axs[1, 1])

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/cavity_2d.png", dpi=150)
plt.show()

# 2. Training loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.yscale("log")
plt.title("Cavity Flow Training Loss (LogCosh)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/loss_curve.png", dpi=150)
plt.show()

# 3. 3-D surface plots
def plot_3d(field, name):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, field, cmap='jet', edgecolor='none')
    ax.set_title(f"Predicted {name} (3D)")
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel(name)
    fig.colorbar(ax.plot_surface(X, Y, field, cmap='jet', edgecolor='none'), shrink=0.5)
    plt.tight_layout()
    plt.savefig(f"/mnt/user-data/outputs/3d_{name.replace(' ', '_')}.png", dpi=150)
    plt.show()

plot_3d(u_pred, "u Velocity")
plot_3d(v_pred, "v Velocity")
plot_3d(p_pred, "Pressure")

print("All plots saved to /mnt/user-data/outputs/")