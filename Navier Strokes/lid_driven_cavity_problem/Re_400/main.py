"""
PINN  -  Lid-driven cavity flow  (Re = 400)  
═════════════════════════════════════════════════
Improvements 
  1. Fourier feature embedding   – captures steep near-wall gradients
  2. Residual-block MLP          – better gradient flow, avoids vanishing grads
  3. Wall-clustered collocation  – ~25 % of PDE points packed near walls
  4. Cosine-annealing LR         – smooth decay 1e-3 → 5e-5 over Adam phase
  5. Gradient clipping           – stabilises Adam on stiff Re=400 residuals
  6. More training               – 40k Adam  +  10k L-BFGS
  7. Lower BC weight (50)        – keeps PDE residual competitive
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── HYPERPARAMETERS ──────────────────────────────────────────────────────────
EPOCHS       = 40_000
LBFGS_MAX    = 10_000
LR_START     = 1e-3
LR_END       = 5e-5
RE           = 400.0
NU           = 1.0 / RE
N_PDE        = 15_000
WALL_FRAC    = 0.25        # fraction of PDE points in near-wall strips
WALL_DELTA   = 0.05        # strip width
N_BC         = 800
BC_WEIGHT    = 50.0
GAUGE_WEIGHT = 10.0
GRAD_CLIP    = 1.0
EPS          = 1e-3

CHECKPOINT_STEPS = [5000, 10000, 20000, 30000, 40000]

# ── GHIA BENCHMARK DATA  (Re = 400) ──────────────────────────────────────────
# Table I  –  u along vertical centreline  x = 0.5
GHIA_Y = np.array([
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000
])
GHIA_U = np.array([
     0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726,
    -0.17119, -0.11477,  0.02135,  0.16256,  0.29093,  0.55892,  0.61756,
     0.68439,  0.75837,  1.00000
])

# Table II  –  v along horizontal centreline  y = 0.5
GHIA_X_V = np.array([
    1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
    0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
    0.0703, 0.0625, 0.0000
])
GHIA_V = np.array([
     0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993,
    -0.38598,  0.05186,  0.30174,  0.30203,  0.28124,  0.22965,  0.20920,
     0.19713,  0.18360,  0.00000
])

output_dir = "./outputs_re400_v2"
os.makedirs(output_dir, exist_ok=True)

# ── UTILITIES ────────────────────────────────────────────────────────────────
def mse_loss(x):
    return torch.mean(x ** 2)

# ── 1. FOURIER FEATURE EMBEDDING ─────────────────────────────────────────────
class FourierEmbedding(nn.Module):
    """
    (x, y) → [cos(2π B x), sin(2π B x)]  with  B ~ N(0, σ²).
    Output dimension = 2 * n_freqs.
    """
    def __init__(self, in_dim=2, n_freqs=64, sigma=1.0):
        super().__init__()
        B = torch.randn(in_dim, n_freqs) * sigma
        self.register_buffer("B", B)   # fixed (not trained)

    def forward(self, x):
        proj = 2.0 * np.pi * (x @ self.B)          # (N, n_freqs)
        return torch.cat([torch.cos(proj),
                          torch.sin(proj)], dim=-1) # (N, 2*n_freqs)

# ── 2. RESIDUAL BLOCK ─────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))

# ── 3. FULL NETWORK ───────────────────────────────────────────────────────────
class PINN(nn.Module):
    """
    FourierEmbed(64 freqs → 128 dim)
      → Linear(128) → 6 × ResBlock(128) → Linear(2)
    Outputs: (ψ, p)
    """
    def __init__(self, n_freqs=64, hidden=128, n_res=6):
        super().__init__()
        embed_dim   = 2 * n_freqs
        self.embed  = FourierEmbedding(2, n_freqs, sigma=1.0)
        self.input  = nn.Linear(embed_dim, hidden)
        self.res    = nn.ModuleList([ResBlock(hidden) for _ in range(n_res)])
        self.output = nn.Linear(hidden, 2)
        self.act    = nn.SiLU()
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.act(self.input(self.embed(x)))
        for blk in self.res:
            h = blk(h)
        return self.output(h)

# ── AUTOGRAD HELPERS ─────────────────────────────────────────────────────────
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )[0]

def get_derivatives(xy, net):
    out  = net(xy)
    psi, p = out[:, 0:1], out[:, 1:2]
    pg   = grad(psi, xy)
    u, v = pg[:, 1:2], -pg[:, 0:1]
    ug   = grad(u, xy);  u_x, u_y = ug[:, 0:1], ug[:, 1:2]
    vg   = grad(v, xy);  v_x, v_y = vg[:, 0:1], vg[:, 1:2]
    u_xx = grad(u_x, xy)[:, 0:1];  u_yy = grad(u_y, xy)[:, 1:2]
    v_xx = grad(v_x, xy)[:, 0:1];  v_yy = grad(v_y, xy)[:, 1:2]
    ppg  = grad(p, xy);  p_x, p_y = ppg[:, 0:1], ppg[:, 1:2]
    return (psi, p, u, v, u_x, u_y, v_x, v_y,
            u_xx, u_yy, v_xx, v_yy, p_x, p_y)

# ── 4. WALL-CLUSTERED COLLOCATION SAMPLING ───────────────────────────────────
def sample_pde_points(n_total, wall_frac, wall_delta):
    """
    Returns (n_total, 2) tensor with requires_grad=True.
    wall_frac * n_total points are drawn uniformly inside
    near-wall strips of width wall_delta from each of the 4 walls.
    The remainder use Chebyshev-cosine clustering.
    """
    n_wall  = int(n_total * wall_frac)
    n_inner = n_total - n_wall

    # Chebyshev-clustered interior
    uv    = torch.rand(n_inner, 2, device=device)
    inner = 0.5 * (1.0 - torch.cos(np.pi * uv))

    # Near-wall strips (uniform per strip)
    xw    = torch.rand(n_wall, device=device)
    yw    = torch.rand(n_wall, device=device)
    which = torch.randint(0, 4, (n_wall,), device=device)

    # 0 = bottom, 1 = top, 2 = left, 3 = right
    x_fin = torch.where(which == 2, xw * wall_delta,
            torch.where(which == 3, 1.0 - xw * wall_delta, xw))
    y_fin = torch.where(which == 0, yw * wall_delta,
            torch.where(which == 1, 1.0 - yw * wall_delta, yw))

    pts = torch.cat([inner, torch.stack([x_fin, y_fin], dim=1)], dim=0)
    pts = pts.clamp(EPS, 1.0 - EPS)
    pts.requires_grad_(True)
    return pts

# ── BOUNDARY LOSS ─────────────────────────────────────────────────────────────
def bc_loss(xy, u_tgt, v_tgt, net):
    psi  = net(xy)[:, 0:1]
    pg   = grad(psi, xy)
    u    = pg[:, 1:2];  v = -pg[:, 0:1]
    return (mse_loss(u - u_tgt.unsqueeze(1)) +
            mse_loss(v - v_tgt.unsqueeze(1)) +
            mse_loss(psi))

def make_bc_xy(xc, yc):
    xy = torch.stack([xc, yc], dim=1)
    xy.requires_grad_(True)
    return xy

_anchor = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

def gauge_loss(net):
    return mse_loss(net(_anchor)[:, 1:2])

# ── CENTRELINE EVALUATION ─────────────────────────────────────────────────────
def eval_u(model, n=200):
    yc = np.linspace(0, 1, n)
    xy = torch.tensor(np.stack([np.full(n, 0.5), yc], 1),
                      dtype=torch.float32, device=device).requires_grad_(True)
    u  = grad(model(xy)[:, 0:1], xy)[:, 1:2]
    return yc, u.detach().cpu().numpy().flatten()

def eval_v(model, n=200):
    xc = np.linspace(0, 1, n)
    xy = torch.tensor(np.stack([xc, np.full(n, 0.5)], 1),
                      dtype=torch.float32, device=device).requires_grad_(True)
    v  = -grad(model(xy)[:, 0:1], xy)[:, 0:1]
    return xc, v.detach().cpu().numpy().flatten()

# ── MODEL + SCHEDULER ─────────────────────────────────────────────────────────
net       = PINN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR_START)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR_END
)

loss_history = []
checkpoints  = {}

# ── ADAM TRAINING ─────────────────────────────────────────────────────────────
print(f"\nStarting Adam training  (Re = {int(RE)}, v2 — {EPOCHS} epochs)…\n")

for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()

    xy_pde = sample_pde_points(N_PDE, WALL_FRAC, WALL_DELTA)

    (_, _, u, v, u_x, u_y, v_x, v_y,
     u_xx, u_yy, v_xx, v_yy, p_x, p_y) = get_derivatives(xy_pde, net)

    Ru       = u*u_x + v*u_y + p_x - NU*(u_xx + u_yy)
    Rv       = u*v_x + v*v_y + p_y - NU*(v_xx + v_yy)
    loss_pde = mse_loss(Ru) + mse_loss(Rv)

    x_b   = torch.FloatTensor(N_BC).uniform_(EPS, 1-EPS).to(device)
    y_b   = torch.FloatTensor(N_BC).uniform_(EPS, 1-EPS).to(device)
    ones  = torch.ones(N_BC,  device=device)
    zeros = torch.zeros(N_BC, device=device)
    u_top = 1.0 - (2.0*x_b - 1.0)**16

    loss_bc = BC_WEIGHT * (
        bc_loss(make_bc_xy(x_b,  ones),  u_top, zeros, net) +
        bc_loss(make_bc_xy(x_b,  zeros), zeros, zeros, net) +
        bc_loss(make_bc_xy(zeros, y_b),  zeros, zeros, net) +
        bc_loss(make_bc_xy(ones,  y_b),  zeros, zeros, net)
    )

    total = loss_pde + loss_bc + GAUGE_WEIGHT * gauge_loss(net)
    total.backward()

    # gradient clipping
    torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)

    optimizer.step()
    scheduler.step()
    loss_history.append(total.item())

    if epoch in CHECKPOINT_STEPS:
        checkpoints[epoch] = copy.deepcopy(net.state_dict())
        print(f"  Checkpoint @ epoch {epoch} | lr={scheduler.get_last_lr()[0]:.2e}")

    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss={total.item():.6e} | "
              f"PDE={loss_pde.item():.6e} | BC={loss_bc.item():.6e} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

# ── L-BFGS FINE-TUNING ────────────────────────────────────────────────────────
print("\nStarting L-BFGS fine-tuning…\n")

xy_pde_l = sample_pde_points(N_PDE, WALL_FRAC, WALL_DELTA)
x_bl  = torch.FloatTensor(N_BC).uniform_(EPS, 1-EPS).to(device)
y_bl  = torch.FloatTensor(N_BC).uniform_(EPS, 1-EPS).to(device)
ones_l  = torch.ones(N_BC,  device=device)
zeros_l = torch.zeros(N_BC, device=device)
u_top_l = 1.0 - (2.0*x_bl - 1.0)**16

xy_top_l   = make_bc_xy(x_bl,   ones_l)
xy_bot_l   = make_bc_xy(x_bl,   zeros_l)
xy_left_l  = make_bc_xy(zeros_l, y_bl)
xy_right_l = make_bc_xy(ones_l,  y_bl)

lbfgs  = torch.optim.LBFGS(
    net.parameters(), lr=1.0, max_iter=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=100, line_search_fn='strong_wolfe'
)
step_l = [0]

def closure():
    lbfgs.zero_grad()
    (_, _, u, v, u_x, u_y, v_x, v_y,
     u_xx, u_yy, v_xx, v_yy, p_x, p_y) = get_derivatives(xy_pde_l, net)

    Ru = u*u_x + v*u_y + p_x - NU*(u_xx + u_yy)
    Rv = u*v_x + v*v_y + p_y - NU*(v_xx + v_yy)

    total = (mse_loss(Ru) + mse_loss(Rv) +
             BC_WEIGHT * (bc_loss(xy_top_l,   u_top_l, zeros_l, net) +
                          bc_loss(xy_bot_l,   zeros_l, zeros_l, net) +
                          bc_loss(xy_left_l,  zeros_l, zeros_l, net) +
                          bc_loss(xy_right_l, zeros_l, zeros_l, net)) +
             GAUGE_WEIGHT * gauge_loss(net))
    total.backward()

    step_l[0] += 1
    if step_l[0] % 10  == 0: loss_history.append(total.item())
    if step_l[0] % 500 == 0:
        print(f"  L-BFGS {step_l[0]:5d} | Loss={total.item():.8e}")
    return total

lbfgs.step(closure)
checkpoints["L-BFGS Final"] = copy.deepcopy(net.state_dict())
print("\nTraining complete.\n")

# ── EVALUATION GRID ──────────────────────────────────────────────────────────
N = 200
x = np.linspace(0, 1, N);  y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

XY_e = torch.tensor(np.stack([X.flatten(), Y.flatten()], 1),
                    dtype=torch.float32, device=device).requires_grad_(True)
out_e     = net(XY_e)
psi_e     = out_e[:, 0:1];  p_e = out_e[:, 1:2]
pg_e      = grad(psi_e, XY_e)
u_e       = pg_e[:, 1:2];   v_e = -pg_e[:, 0:1]

psi_pred = psi_e.detach().cpu().numpy().reshape(N, N)
p_pred   = p_e.detach().cpu().numpy().reshape(N, N)
u_pred   = u_e.detach().cpu().numpy().reshape(N, N)
v_pred   = v_e.detach().cpu().numpy().reshape(N, N)
del out_e, psi_e, p_e, pg_e, u_e, v_e, XY_e

# ── 2-D FLOW FIELD ───────────────────────────────────────────────────────────
speed = np.sqrt(u_pred**2 + v_pred**2)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

c0 = axs[0,0].contourf(X, Y, speed, 100, cmap='jet')
axs[0,0].set_title(f"Velocity Magnitude  (Re={int(RE)})"); fig.colorbar(c0, ax=axs[0,0])

axs[0,1].streamplot(X, Y, u_pred, v_pred, color=speed, cmap='jet', density=1.5)
axs[0,1].set_title(f"Streamlines  (Re={int(RE)})")
axs[0,1].set_xlim([0,1]); axs[0,1].set_ylim([0,1])

c2 = axs[1,0].contourf(X, Y, u_pred, 100, cmap='jet')
axs[1,0].set_title("Horizontal Velocity u"); fig.colorbar(c2, ax=axs[1,0])

c3 = axs[1,1].contourf(X, Y, v_pred, 100, cmap='jet')
axs[1,1].set_title("Vertical Velocity v"); fig.colorbar(c3, ax=axs[1,1])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cavity_2d.png"), dpi=150)
plt.show(); print("Saved cavity_2d.png")

# ── LOSS CURVE ───────────────────────────────────────────────────────────────
plt.figure(figsize=(8,5))
plt.plot(loss_history); plt.yscale("log")
plt.title(f"Training Loss  (Re={int(RE)}, v2)"); plt.xlabel("Iteration"); plt.ylabel("Loss")
plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
plt.show(); print("Saved loss_curve.png")

# ── BENCHMARK PLOTS ───────────────────────────────────────────────────────────
ckpt_colors = {
     5000: '#89b4fa', 10000: '#a6e3a1', 20000: '#f38ba8',
    30000: '#cba6f7', 40000: '#fab387', 'L-BFGS Final': '#d20f39'
}
tmp = PINN().to(device)

# u centreline
fig_u, ax_u = plt.subplots(figsize=(7, 9))
for step, state in checkpoints.items():
    tmp.load_state_dict(state); tmp.eval()
    yc, uc = eval_u(tmp)
    lw    = 2.2 if step == "L-BFGS Final" else 1.2
    label = f"Step {step}" if isinstance(step, int) else step
    ax_u.plot(uc, yc, color=ckpt_colors.get(step, 'gray'), lw=lw, label=label)

ax_u.scatter(GHIA_U, GHIA_Y, color='black', s=60, zorder=5, label='Ghia et al. (1982)')
ax_u.axvline(0, color='gray', ls='--', lw=0.8)
ax_u.axhline(0.5, color='gray', ls='--', lw=0.8)
ax_u.set_xlabel("u velocity"); ax_u.set_ylabel("y / L")
ax_u.set_title(f"Centreline u-Velocity  (Re={int(RE)})")
ax_u.set_xlim([-0.6, 1.2]); ax_u.set_ylim([0.0, 1.0])
ax_u.legend(); ax_u.grid(True)
fig_u.tight_layout()
fig_u.savefig(os.path.join(output_dir, "centerline_u_benchmark.png"), dpi=150)
plt.show(); print("Saved centerline_u_benchmark.png")

# v centreline
fig_v, ax_v = plt.subplots(figsize=(9, 7))
for step, state in checkpoints.items():
    tmp.load_state_dict(state); tmp.eval()
    xc, vc = eval_v(tmp)
    lw    = 2.2 if step == "L-BFGS Final" else 1.2
    label = f"Step {step}" if isinstance(step, int) else step
    ax_v.plot(xc, vc, color=ckpt_colors.get(step, 'gray'), lw=lw, label=label)

ax_v.scatter(GHIA_X_V, GHIA_V, color='black', s=60, zorder=5, label='Ghia et al. (1982)')
ax_v.axhline(0, color='gray', ls='--', lw=0.8)
ax_v.axvline(0.5, color='gray', ls='--', lw=0.8)
ax_v.set_xlabel("x / L"); ax_v.set_ylabel("v velocity")
ax_v.set_title(f"Centreline v-Velocity  (Re={int(RE)})")
ax_v.set_xlim([0.0, 1.0]); ax_v.set_ylim([-0.6, 0.4])
ax_v.legend(); ax_v.grid(True)
fig_v.tight_layout()
fig_v.savefig(os.path.join(output_dir, "centerline_v_benchmark.png"), dpi=150)
plt.show(); print("Saved centerline_v_benchmark.png")

# ── 3-D SURFACE PLOTS ─────────────────────────────────────────────────────────
def plot_3d(field, name):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, field, cmap='jet', edgecolor='none')
    ax.set_title(f"{name}  (Re={int(RE)}, 3D)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel(name)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    plt.tight_layout()
    fn = os.path.join(output_dir, f"3d_{name.replace(' ','_')}.png")
    plt.savefig(fn, dpi=150); plt.show(); print(f"Saved {fn}")

plot_3d(u_pred, "u Velocity")
plot_3d(v_pred, "v Velocity")
plot_3d(p_pred, "Pressure")