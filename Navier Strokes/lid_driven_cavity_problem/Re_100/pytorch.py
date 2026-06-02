import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# HYPERPARAMETERS
EPOCHS           = 25_000
LBFGS_MAX        = 5_000
LR               = 1e-3
RE               = 100.0
NU               = 1.0 / RE
N_PDE            = 10_000
N_BC             = 500
BC_WEIGHT        = 100.0
GAUGE_WEIGHT     = 10.0    
EPS              = 1e-3   

CHECKPOINT_STEPS = [5000, 10000, 15000, 20000, 25000]

# GHIA BENCHMARK DATA  (Re = 100, Ghia et al. 1982)
# u-velocity along the vertical centreline x = 0.5

GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000
])

GHIA_U = np.array([
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434,
    -0.10150, -0.15662, -0.21090, -0.20581, -0.13641,
     0.00332,  0.23151,  0.68717,  0.73722,
     0.78871,  0.84123,  1.00000
])

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# LOSS
def mse_loss(x):
    return torch.mean(x ** 2)

# NETWORK
class PINN(nn.Module):

    def __init__(self, layers=(2, 64, 64, 64, 64, 64, 2)):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)
        self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# AUTOGRAD HELPER
def grad(outputs, inputs):

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )[0]

# DERIVATIVES
def get_derivatives(xy, net):
    out = net(xy)
    psi = out[:, 0:1]
    p   = out[:, 1:2]

    psi_grad = grad(psi, xy)

    psi_x = psi_grad[:, 0:1]
    psi_y = psi_grad[:, 1:2]

    u = psi_y
    v = -psi_x

    u_grad = grad(u, xy)

    u_x = u_grad[:, 0:1]
    u_y = u_grad[:, 1:2]

    v_grad = grad(v, xy)

    v_x = v_grad[:, 0:1]
    v_y = v_grad[:, 1:2]

    u_xx = grad(u_x, xy)[:, 0:1]
    u_yy = grad(u_y, xy)[:, 1:2]

    v_xx = grad(v_x, xy)[:, 0:1]
    v_yy = grad(v_y, xy)[:, 1:2]

    p_grad = grad(p, xy)

    p_x = p_grad[:, 0:1]
    p_y = p_grad[:, 1:2]

    return (
        psi, p,
        u, v,
        u_x, u_y,
        v_x, v_y,
        u_xx, u_yy,
        v_xx, v_yy,
        p_x, p_y
    )

# CENTERLINE EVALUATION
def eval_centerline_u(model, n_pts=200):

    y_cl = np.linspace(0, 1, n_pts)
    x_cl = np.full_like(y_cl, 0.5)

    xy_cl = torch.tensor(
        np.stack([x_cl, y_cl], axis=1),
        dtype=torch.float32,
        device=device
    ).requires_grad_(True)

    out = model(xy_cl)
    psi = out[:, 0:1]
    psi_grad = grad(psi, xy_cl)
    u = psi_grad[:, 1:2]
    return y_cl, u.detach().cpu().numpy().flatten()

# BOUNDARY LOSS
def bc_loss(xy, u_target, v_target, net):
    out = net(xy)
    psi = out[:, 0:1]
    psi_grad = grad(psi, xy)
    u = psi_grad[:, 1:2]
    v = -psi_grad[:, 0:1]
    u_target = u_target.unsqueeze(1)
    v_target = v_target.unsqueeze(1)
    l_u   = mse_loss(u - u_target)
    l_v   = mse_loss(v - v_target)
    l_psi = mse_loss(psi)

    return l_u + l_v + l_psi

_xy_anchor = torch.tensor(
    [[0.5, 0.5]], dtype=torch.float32, device=device
)

def pressure_gauge_loss(net):
    p_center = net(_xy_anchor)[:, 1:2]
    return mse_loss(p_center)

# MODEL + OPTIMIZER
net = PINN().to(device)

optimizer = torch.optim.Adam(
    net.parameters(),
    lr=LR
)

loss_history = []
checkpoints  = {}

def make_bc_xy(x_coords, y_coords):
    xy = torch.stack([x_coords, y_coords], dim=1)
    xy.requires_grad_(True)
    return xy

print("\nStarting Adam training...\n")
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()

    # PDE POINTS (Chebyshev-clustered sampling)
    xy_uniform = torch.rand(N_PDE, 2, device=device)
    xy_pde = 0.5 * (1.0 - torch.cos(np.pi * xy_uniform))
    xy_pde.requires_grad_(True)

    (
        _, _,
        u, v,
        u_x, u_y,
        v_x, v_y,
        u_xx, u_yy,
        v_xx, v_yy,
        p_x, p_y
    ) = get_derivatives(xy_pde, net)

    # NAVIER–STOKES RESIDUALS
    Ru = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    Rv = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    loss_pde = mse_loss(Ru) + mse_loss(Rv)

    # BOUNDARY CONDITIONS
    x_b = torch.FloatTensor(N_BC).uniform_(EPS, 1.0 - EPS).to(device)
    y_b = torch.FloatTensor(N_BC).uniform_(EPS, 1.0 - EPS).to(device)

    ones  = torch.ones(N_BC, device=device)
    zeros = torch.zeros(N_BC, device=device)

    xy_top   = make_bc_xy(x_b, ones)
    xy_bot   = make_bc_xy(x_b, zeros)
    xy_left  = make_bc_xy(zeros, y_b)
    xy_right = make_bc_xy(ones, y_b)

    # Smooth lid: avoids corner velocity singularities
    u_top = 1.0 - (2.0 * x_b - 1.0) ** 16

    loss_bc = BC_WEIGHT * (

        bc_loss(xy_top,   u_top,  zeros, net) +
        bc_loss(xy_bot,   zeros,  zeros, net) +
        bc_loss(xy_left,  zeros,  zeros, net) +
        bc_loss(xy_right, zeros,  zeros, net)
    )

  
    # PRESSURE GAUGE
    loss_gauge = GAUGE_WEIGHT * pressure_gauge_loss(net)

    # TOTAL LOSS
    total_loss = loss_pde + loss_bc + loss_gauge
    total_loss.backward()
    optimizer.step()
    loss_history.append(total_loss.item())

    # checkpoints
    if epoch in CHECKPOINT_STEPS:
        checkpoints[epoch] = copy.deepcopy(net.state_dict())
        print(f"Checkpoint saved at epoch {epoch}")

    # logs
    if epoch % 1000 == 0:

        print(
            f"Epoch {epoch:5d} | "
            f"Loss = {total_loss.item():.8f} | "
            f"PDE = {loss_pde.item():.8f} | "
            f"BC = {loss_bc.item():.8f} | "
            f"Gauge = {loss_gauge.item():.8f}"
        )

print("\nStarting L-BFGS optimization...\n")

# Fixed collocation points for L-BFGS
xy_uniform_lbfgs = torch.rand(N_PDE, 2, device=device)
xy_pde_lbfgs = 0.5 * (
    1.0 - torch.cos(np.pi * xy_uniform_lbfgs)
)
xy_pde_lbfgs.requires_grad_(True)

x_b_lbfgs = torch.FloatTensor(N_BC).uniform_(EPS, 1.0 - EPS).to(device)
y_b_lbfgs = torch.FloatTensor(N_BC).uniform_(EPS, 1.0 - EPS).to(device)

ones_lbfgs  = torch.ones(N_BC, device=device)
zeros_lbfgs = torch.zeros(N_BC, device=device)

xy_top_lbfgs   = make_bc_xy(x_b_lbfgs,   ones_lbfgs)
xy_bot_lbfgs   = make_bc_xy(x_b_lbfgs,   zeros_lbfgs)
xy_left_lbfgs  = make_bc_xy(zeros_lbfgs,  y_b_lbfgs)
xy_right_lbfgs = make_bc_xy(ones_lbfgs,   y_b_lbfgs)

u_top_lbfgs = 1.0 - (2.0 * x_b_lbfgs - 1.0) ** 16

lbfgs_optimizer = torch.optim.LBFGS(
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
    lbfgs_optimizer.zero_grad()
    (
        _, _,
        u, v,
        u_x, u_y,
        v_x, v_y,
        u_xx, u_yy,
        v_xx, v_yy,
        p_x, p_y
    ) = get_derivatives(xy_pde_lbfgs, net)

    Ru = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    Rv = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
    loss_pde = mse_loss(Ru) + mse_loss(Rv)
    loss_bc = BC_WEIGHT * (

        bc_loss(xy_top_lbfgs,   u_top_lbfgs,   zeros_lbfgs, net) +
        bc_loss(xy_bot_lbfgs,   zeros_lbfgs,   zeros_lbfgs, net) +
        bc_loss(xy_left_lbfgs,  zeros_lbfgs,   zeros_lbfgs, net) +
        bc_loss(xy_right_lbfgs, zeros_lbfgs,   zeros_lbfgs, net)
    )

    loss_gauge = GAUGE_WEIGHT * pressure_gauge_loss(net)
    total_loss = loss_pde + loss_bc + loss_gauge
    total_loss.backward()
    lbfgs_step[0] += 1

    if lbfgs_step[0] % 10 == 0:
        loss_history.append(total_loss.item())

    if lbfgs_step[0] % 100 == 0:

        print(
            f"L-BFGS Iter {lbfgs_step[0]:5d} | "
            f"Loss = {total_loss.item():.10f}"
        )

    return total_loss

lbfgs_optimizer.step(closure)
checkpoints["L-BFGS Final"] = copy.deepcopy(net.state_dict())
print("\nTraining complete.\n")

# EVALUATION GRID
N = 200

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

X, Y = np.meshgrid(x, y)

XY_eval = torch.tensor(
    np.stack([X.flatten(), Y.flatten()], axis=1),
    dtype=torch.float32,
    device=device
).requires_grad_(True)

out_eval = net(XY_eval)

psi_eval = out_eval[:, 0:1]
p_eval   = out_eval[:, 1:2]

psi_grad_eval = grad(psi_eval, XY_eval)

u_eval = psi_grad_eval[:, 1:2]
v_eval = -psi_grad_eval[:, 0:1]

psi_pred = psi_eval.detach().cpu().numpy().reshape(N, N)
p_pred   = p_eval.detach().cpu().numpy().reshape(N, N)
u_pred   = u_eval.detach().cpu().numpy().reshape(N, N)
v_pred   = v_eval.detach().cpu().numpy().reshape(N, N)

del out_eval, psi_eval, p_eval, psi_grad_eval, u_eval, v_eval, XY_eval

# PLOTS : 2D
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
speed = np.sqrt(u_pred**2 + v_pred**2)

# velocity magnitude
c0 = axs[0, 0].contourf(X, Y, speed, 100, cmap='jet')
axs[0, 0].set_title("Velocity Magnitude")
fig.colorbar(c0, ax=axs[0, 0])

# streamlines
axs[0, 1].streamplot(X, Y, u_pred, v_pred,
                     color=speed, cmap='jet', density=1.5)
axs[0, 1].set_title("Streamlines")
axs[0, 1].set_xlim([0, 1])
axs[0, 1].set_ylim([0, 1])

# u velocity
c2 = axs[1, 0].contourf(X, Y, u_pred, 100, cmap='jet')
axs[1, 0].set_title("Horizontal Velocity u")
fig.colorbar(c2, ax=axs[1, 0])

# v velocity
c3 = axs[1, 1].contourf(X, Y, v_pred, 100, cmap='jet')
axs[1, 1].set_title("Vertical Velocity v")
fig.colorbar(c3, ax=axs[1, 1])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cavity_2d.png"), dpi=150)
plt.show()
print("Saved cavity_2d.png")

# LOSS CURVE
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.yscale("log")
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
plt.show()
print("Saved loss_curve.png")

# BENCHMARK COMPARISON
ckpt_colors = {
    5000:          '#89b4fa',
    10000:         '#a6e3a1',
    15000:         '#f38ba8',
    20000:         '#cba6f7',
    25000:         '#fab387',
    'L-BFGS Final':'#d20f39'
}

tmp_model = PINN().to(device)

fig_bench, ax_bench = plt.subplots(figsize=(7, 9))

for step, state in checkpoints.items():

    tmp_model.load_state_dict(state)
    tmp_model.eval()

    y_cl, u_cl = eval_centerline_u(tmp_model)

    color = ckpt_colors.get(step, 'gray')

    label = f"Step {step}" if isinstance(step, int) else step

    ax_bench.plot(
        u_cl, y_cl,
        color=color,
        linewidth=2.0 if step == "L-BFGS Final" else 1.2,
        label=label
    )

ax_bench.scatter(
    GHIA_U, GHIA_Y,
    color='black', s=60, zorder=5,
    label='Ghia et al. (1982)'
)

ax_bench.axvline(0,   color='gray', linestyle='--', linewidth=0.8)
ax_bench.axhline(0.5, color='gray', linestyle='--', linewidth=0.8)

ax_bench.set_xlabel("u velocity")
ax_bench.set_ylabel("y / L")
ax_bench.set_title(f"Centerline Velocity Comparison (Re={int(RE)})")

# Narrower x-range suits Re=100 (weaker recirculation than Re=1000)
ax_bench.set_xlim([-0.4, 1.2])
ax_bench.set_ylim([0.0, 1.0])

ax_bench.legend()
ax_bench.grid(True)

fig_bench.tight_layout()
fig_bench.savefig(os.path.join(output_dir, "centerline_benchmark.png"), dpi=150)
plt.show()
print("Saved centerline_benchmark.png")

# 3D PLOTS
def plot_3d(field, name):

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, field, cmap='jet', edgecolor='none')

    ax.set_title(f"{name} (3D)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(name)

    fig.colorbar(surf, ax=ax, shrink=0.5)
    plt.tight_layout()

    filename = os.path.join(output_dir, f"3d_{name.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved {filename}")

plot_3d(u_pred, "u Velocity")
plot_3d(v_pred, "v Velocity")
plot_3d(p_pred, "Pressure")

