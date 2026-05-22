import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EPOCHS        = 25_000
LBFGS_MAX     = 5_000
LR            = 1e-3
RE            = 100.0
NU            = 1.0 / RE
L             = 20.0           
H             = 1.0            
U_IN          = 1.0            

Q  = U_IN * H 

N_PDE         = 10_000
N_BC          = 500

BC_WEIGHT     = 100.0
GAUGE_WEIGHT  = 10.0

EPS           = 1e-3         
CHECKPOINT_STEPS = [5000, 10000, 15000, 20000, 25000]

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# ANALYTICAL FULLY-DEVELOPED PROFILE  (Poiseuille, plane channel)
# For mean velocity U_IN = 1 and height H = 1:
#   u_fd(y) = 6 * U_IN * (y/H) * (1 - y/H)  →  max = 1.5 at y = 0.5

def poiseuille(y):
    return 6.0 * U_IN * (y / H) * (1.0 - y / H)

def mse_loss(x):
    return torch.mean(x ** 2)

def grad(outputs, inputs):
    return torch.autograd.grad( outputs, inputs, grad_outputs=torch.ones_like(outputs),create_graph=True)[0]

def make_xy(x_coords, y_coords):
    xy = torch.stack([x_coords, y_coords], dim=1)
    xy.requires_grad_(True)
    return xy

# NETWORK
class PINN(nn.Module):

    def __init__(self, layers=(2, 64, 64, 64, 64, 2)):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        # Normalise to [0,1]^2 so the network sees a square input domain
        x_n = xy[:, 0:1] / L
        y_n = xy[:, 1:2] / H
        return self.net(torch.cat([x_n, y_n], dim=1))

# DERIVATIVES
def get_derivatives(xy, net):

    out   = net(xy)
    psi   = out[:, 0:1]
    p     = out[:, 1:2]

    # Stream function:  u = ∂ψ/∂y,  v = -∂ψ/∂x
    psi_g = grad(psi, xy)
    u     = psi_g[:, 1:2]
    v     = -psi_g[:, 0:1]

    u_g   = grad(u, xy);     u_x  = u_g[:, 0:1];  u_y  = u_g[:, 1:2]
    v_g   = grad(v, xy);     v_x  = v_g[:, 0:1];  v_y  = v_g[:, 1:2]

    u_xx  = grad(u_x, xy)[:, 0:1]
    u_yy  = grad(u_y, xy)[:, 1:2]
    v_xx  = grad(v_x, xy)[:, 0:1]
    v_yy  = grad(v_y, xy)[:, 1:2]

    p_g   = grad(p, xy);     p_x  = p_g[:, 0:1];  p_y  = p_g[:, 1:2]

    return (psi, p,u, v,u_x, u_y, v_x, v_y,u_xx, u_yy, v_xx, v_yy,p_x, p_y)

# BOUNDARY LOSSES
def wall_bc_loss(xy, psi_wall, net):

    out      = net(xy)
    psi      = out[:, 0:1]
    psi_g    = grad(psi, xy)
    u        = psi_g[:, 1:2]
    v        = -psi_g[:, 0:1]
    return mse_loss(u) + mse_loss(v) + mse_loss(psi - psi_wall)


def inlet_bc_loss(xy, net):
    """
    Inlet (x = 0): uniform flow u = U_IN, v = 0.
    Since ψ(0, 0) = 0 and u = ∂ψ/∂y = U_IN = 1, integrating gives ψ(0, y) = y.
    Enforcing ψ = y at inlet also pins the flow rate consistently.
    """
    out   = net(xy)
    psi   = out[:, 0:1]
    psi_g = grad(psi, xy)
    u     = psi_g[:, 1:2]
    v     = -psi_g[:, 0:1]
    y_in  = xy[:, 1:2]
    return mse_loss(u - U_IN) + mse_loss(v) + mse_loss(psi - y_in)


def outlet_bc_loss(xy, net):
    """
    Outlet (x = L): zero-gradient (Neumann) condition.
      ∂u/∂x = 0  →  ∂²ψ/∂x∂y = 0
      v = 0       →  ∂ψ/∂x    = 0
    Enforcing ∂ψ/∂x = 0 makes v = 0 and, together with the NS equations
    already driving ∂u/∂x → 0, drives the flow to the developed state.
    """
    out   = net(xy)
    psi   = out[:, 0:1]
    psi_g = grad(psi, xy)
    psi_x = psi_g[:, 0:1]                # = -v; enforce → 0
    psi_y = psi_g[:, 1:2]                # = u
    u_x   = grad(psi_y, xy)[:, 0:1]     # ∂u/∂x; enforce → 0
    return mse_loss(psi_x) + mse_loss(u_x)


def pressure_gauge_loss(net):
    """
    Pin p = 0 at the outlet centre (x = L, y = H/2) to fix the gauge.
    Placing the anchor at the outlet is natural because pressure is normally
    specified there in fully-developed channel flow problems.
    """
    xy_a = torch.tensor([[L, 0.5 * H]], dtype=torch.float32, device=device)
    p    = net(xy_a)[:, 1:2]
    return mse_loss(p)

# MODEL + OPTIMIZER
net       = PINN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

loss_history = []
checkpoints  = {}

# PHASE 1 : ADAM
print("\nStarting Adam training...\n")

for epoch in range(1, EPOCHS + 1):

    optimizer.zero_grad()

    # PDE collocation points
    # Chebyshev clustering in y captures the near-wall gradients;
    # uniform sampling in x covers the full development length.
    xy_u   = torch.rand(N_PDE, 2, device=device)
    x_pde  = L * xy_u[:, 0]
    y_pde  = H * 0.5 * (1.0 - torch.cos(np.pi * xy_u[:, 1]))
    xy_pde = torch.stack([x_pde, y_pde], dim=1)
    xy_pde.requires_grad_(True)

    (psi, p,
     u, v,
     u_x, u_y, v_x, v_y,
     u_xx, u_yy, v_xx, v_yy,
     p_x, p_y) = get_derivatives(xy_pde, net)

    Ru       = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    Rv       = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    loss_pde = mse_loss(Ru) + mse_loss(Rv)

    # Boundary points
    x_b      = torch.FloatTensor(N_BC).uniform_(EPS, L - EPS).to(device)
    y_b      = torch.FloatTensor(N_BC).uniform_(EPS, H - EPS).to(device)

    y_zero   = torch.zeros(N_BC, device=device)
    y_top    = torch.full((N_BC,), H, device=device)
    x_zero   = torch.zeros(N_BC, device=device)
    x_out    = torch.full((N_BC,), L, device=device)

    xy_bot   = make_xy(x_b,    y_zero)
    xy_top   = make_xy(x_b,    y_top)
    xy_in    = make_xy(x_zero, y_b)
    xy_out   = make_xy(x_out,  y_b)

    loss_bc  = BC_WEIGHT * (
        wall_bc_loss(xy_bot, 0.0, net) +    # ψ = 0 on bottom streamline
        wall_bc_loss(xy_top, Q,   net) +    # ψ = Q on top   streamline
        inlet_bc_loss(xy_in,       net) +
        outlet_bc_loss(xy_out,     net)
    )

    loss_gauge = GAUGE_WEIGHT * pressure_gauge_loss(net)
    total_loss = loss_pde + loss_bc + loss_gauge

    total_loss.backward()
    optimizer.step()

    loss_history.append(total_loss.item())

    if epoch in CHECKPOINT_STEPS:
        checkpoints[epoch] = copy.deepcopy(net.state_dict())
        print(f"Checkpoint saved at epoch {epoch}")

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:5d} | Loss={total_loss.item():.4e} | "
            f"PDE={loss_pde.item():.4e} | "
            f"BC={loss_bc.item():.4e} | "
            f"Gauge={loss_gauge.item():.4e}"
        )

# PHASE 2 : L-BFGS  (fixed collocation points)
print("\nStarting L-BFGS optimisation...\n")

xy_u_l   = torch.rand(N_PDE, 2, device=device)
x_pde_l  = L * xy_u_l[:, 0]
y_pde_l  = H * 0.5 * (1.0 - torch.cos(np.pi * xy_u_l[:, 1]))
xy_pde_l = torch.stack([x_pde_l, y_pde_l], dim=1)
xy_pde_l.requires_grad_(True)

x_b_l    = torch.FloatTensor(N_BC).uniform_(EPS, L - EPS).to(device)
y_b_l    = torch.FloatTensor(N_BC).uniform_(EPS, H - EPS).to(device)

y_zero_l = torch.zeros(N_BC, device=device)
y_top_l  = torch.full((N_BC,), H, device=device)
x_zero_l = torch.zeros(N_BC, device=device)
x_out_l  = torch.full((N_BC,), L, device=device)

xy_bot_l = make_xy(x_b_l,   y_zero_l)
xy_top_l = make_xy(x_b_l,   y_top_l)
xy_in_l  = make_xy(x_zero_l, y_b_l)
xy_out_l = make_xy(x_out_l,  y_b_l)

lbfgs_opt = torch.optim.LBFGS(
    net.parameters(),
    lr=1.0,
    max_iter=LBFGS_MAX,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=50,
    line_search_fn='strong_wolfe'
)

lbfgs_step = [0]

def closure():
    lbfgs_opt.zero_grad()
    (psi, p,u, v,u_x, u_y, v_x, v_y,u_xx, u_yy, v_xx, v_yy,p_x, p_y) = get_derivatives(xy_pde_l, net)
    Ru       = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    Rv       = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    loss_pde = mse_loss(Ru) + mse_loss(Rv)

    loss_bc  = BC_WEIGHT * (
        wall_bc_loss(xy_bot_l, 0.0, net) +
        wall_bc_loss(xy_top_l, Q,   net) +
        inlet_bc_loss(xy_in_l,       net) +
        outlet_bc_loss(xy_out_l,     net)
    )

    loss_gauge = GAUGE_WEIGHT * pressure_gauge_loss(net)
    total_loss = loss_pde + loss_bc + loss_gauge
    total_loss.backward()

    lbfgs_step[0] += 1
    if lbfgs_step[0] % 10  == 0: loss_history.append(total_loss.item())
    if lbfgs_step[0] % 100 == 0:
        print(f"L-BFGS iter {lbfgs_step[0]:4d} | Loss={total_loss.item():.6e}")
    return total_loss

lbfgs_opt.step(closure)
checkpoints["L-BFGS Final"] = copy.deepcopy(net.state_dict())
print("\nTraining complete.\n")

# EVALUATION GRID
Nx, Ny = 400, 100
x_eval  = np.linspace(0, L, Nx)
y_eval  = np.linspace(0, H, Ny)
X, Y    = np.meshgrid(x_eval, y_eval)

XY_eval = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1),dtype=torch.float32, device=device).requires_grad_(True)

out_eval  = net(XY_eval)
psi_eval  = out_eval[:, 0:1]
p_eval    = out_eval[:, 1:2]
pg        = grad(psi_eval, XY_eval)
u_eval    = pg[:, 1:2]
v_eval    = -pg[:, 0:1]

psi_pred  = psi_eval.detach().cpu().numpy().reshape(Ny, Nx)
p_pred    = p_eval.detach().cpu().numpy().reshape(Ny, Nx)
u_pred    = u_eval.detach().cpu().numpy().reshape(Ny, Nx)
v_pred    = v_eval.detach().cpu().numpy().reshape(Ny, Nx)

del out_eval, psi_eval, p_eval, pg, u_eval, v_eval, XY_eval

# PLOT 1 : FIELD OVERVIEW (u, v, p)
fig, axs = plt.subplots(3, 1, figsize=(18, 9))

c0 = axs[0].contourf(X, Y, u_pred, 100, cmap='jet')
axs[0].set_title("Horizontal Velocity u")
axs[0].set_ylabel("y"); fig.colorbar(c0, ax=axs[0])

c1 = axs[1].contourf(X, Y, v_pred, 100, cmap='RdBu_r')
axs[1].set_title("Vertical Velocity v  (should be ≈ 0 except near inlet)")
axs[1].set_ylabel("y"); fig.colorbar(c1, ax=axs[1])

c2 = axs[2].contourf(X, Y, p_pred, 100, cmap='coolwarm')
axs[2].set_title("Pressure p  (should decrease linearly along x)")
axs[2].set_xlabel("x"); axs[2].set_ylabel("y"); fig.colorbar(c2, ax=axs[2])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "channel_fields.png"), dpi=150)
plt.show()
print("Saved channel_fields.png")

# PLOT 2 : VELOCITY PROFILE DEVELOPMENT
x_stations = [0, 1, 2, 5, 10, 15, 20]
colors      = plt.cm.plasma(np.linspace(0.1, 0.9, len(x_stations)))
y_fine      = np.linspace(0, H, 300)

fig2, ax2 = plt.subplots(figsize=(9, 7))

for xi, col in zip(x_stations, colors):
    xy_prof = torch.tensor(
        np.stack([np.full_like(y_fine, xi), y_fine], axis=1),
        dtype=torch.float32, device=device
    ).requires_grad_(True)

    out_p  = net(xy_prof)
    psi_p  = out_p[:, 0:1]
    u_p    = grad(psi_p, xy_prof)[:, 1:2]
    u_np   = u_p.detach().cpu().numpy().flatten()

    ax2.plot(u_np, y_fine, color=col, linewidth=1.8, label=f"x = {xi}")

# Analytical Poiseuille
ax2.plot(poiseuille(y_fine), y_fine,'k--', linewidth=2.5, label="Poiseuille (fully developed)")

ax2.set_xlabel("u velocity")
ax2.set_ylabel("y")
ax2.set_title(f"Velocity Profile Development — Re = {int(RE)}\n"
              f"(entry length ≈ 0.05 × Re × 2H = {0.05*RE*2*H:.0f})")
ax2.set_xlim([-0.1, 1.8])
ax2.legend(loc="upper left")
ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "velocity_profiles.png"), dpi=150)
plt.show()
print("Saved velocity_profiles.png")

# PLOT 3 : CHECKPOINT COMPARISON — profile at x = L (outlet)
ckpt_colors = {
    5000:           '#89b4fa',
    10000:          '#a6e3a1',
    15000:          '#f38ba8',
    20000:          '#cba6f7',
    25000:          '#fab387',
    'L-BFGS Final': '#d20f39'
}

tmp_model = PINN().to(device)
fig3, ax3 = plt.subplots(figsize=(7, 8))

for step, state in checkpoints.items():
    tmp_model.load_state_dict(state)
    tmp_model.eval()

    xy_prof = torch.tensor(
        np.stack([np.full_like(y_fine, L), y_fine], axis=1),
        dtype=torch.float32, device=device
    ).requires_grad_(True)

    out_p  = tmp_model(xy_prof)
    psi_p  = out_p[:, 0:1]
    u_p    = grad(psi_p, xy_prof)[:, 1:2]
    u_np   = u_p.detach().cpu().numpy().flatten()

    label  = f"Step {step}" if isinstance(step, int) else step
    lw     = 2.2 if step == "L-BFGS Final" else 1.2
    ax3.plot(u_np, y_fine, color=ckpt_colors.get(step, 'gray'),
             linewidth=lw, label=label)

ax3.plot(poiseuille(y_fine), y_fine, 'k--', linewidth=2.5,label="Poiseuille (analytical)")

ax3.set_xlabel("u velocity")
ax3.set_ylabel("y")
ax3.set_title(f"Outlet Profile vs. Analytical  (Re={int(RE)})")
ax3.set_xlim([-0.1, 1.8])
ax3.legend()
ax3.grid(True)
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, "outlet_profile_comparison.png"), dpi=150)
plt.show()
print("Saved outlet_profile_comparison.png")

# PLOT 4 : CENTERLINE PRESSURE  (should be linear after entry length)
j_mid = Ny // 2
fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.plot(x_eval, p_pred[j_mid, :], linewidth=2)
ax4.set_xlabel("x")
ax4.set_ylabel("p  (centreline)")
ax4.set_title("Centreline Pressure — linear in fully-developed region")
ax4.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pressure_centreline.png"), dpi=150)
plt.show()
print("Saved pressure_centreline.png")

# PLOT 5 : STREAMLINES
speed = np.sqrt(u_pred**2 + v_pred**2)
fig5, ax5 = plt.subplots(figsize=(18, 4))
strm = ax5.streamplot(X, Y, u_pred, v_pred,color=speed, cmap='jet', density=2.0,linewidth=0.8)
fig5.colorbar(strm.lines, ax=ax5, label="|U|")
ax5.set_title("Streamlines")
ax5.set_xlabel("x"); ax5.set_ylabel("y")
ax5.set_xlim([0, L]); ax5.set_ylim([0, H])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "channel_streamlines.png"), dpi=150)
plt.show()
print("Saved channel_streamlines.png")

# PLOT 6 : TRAINING LOSS
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.yscale("log")
plt.title("Training Loss")
plt.xlabel("Iteration"); plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
plt.show()
print("Saved loss_curve.png")

print("\nAll outputs saved in ./outputs/")