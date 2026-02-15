import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
N_modes = 8
iterations = 8000
learning_rate = 1e-4     
omega_0 = 30             
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

# =========================
# EXACT SOLUTION
# =========================
def exact_solution(x, y, N):
    val = np.zeros_like(x)
    for n in range(1, N+1):
        val += (np.sinh(n*np.pi*(1-x)) / np.sinh(n*np.pi)) * np.sin(n*np.pi*y)
    return val / N


# =========================
# SIREN LAYER
# =========================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1/self.linear.in_features, 1/self.linear.in_features)
            else:
                bound = np.sqrt(6/self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# =========================
# SIREN MODEL
# =========================
class SIREN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            SineLayer(2, 128, omega_0, is_first=True),
            SineLayer(128, 128, omega_0),
            SineLayer(128, 128, omega_0),
            SineLayer(128, 128, omega_0),
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)


# =========================
# FIXED BOUNDARY POINTS
# =========================
n_bc = 400
y_in = torch.rand(n_bc,1,device=device)
x_in = torch.zeros_like(y_in)

target_in = torch.zeros_like(y_in)
for n in range(1, N_modes+1):
    target_in += torch.sin(n*np.pi*y_in)
target_in /= N_modes

y_out = torch.rand(n_bc,1,device=device)
x_out = torch.ones_like(y_out)

x_wall = torch.rand(n_bc,1,device=device)
y_bot = torch.zeros_like(x_wall)
y_top = torch.ones_like(x_wall)


# =========================
# LOSS FUNCTION
# =========================
def compute_loss(model, x_col, y_col):

    phi = model(x_col, y_col)

    dphi_dx = torch.autograd.grad(phi, x_col,
                                  torch.ones_like(phi),
                                  create_graph=True)[0]
    dphi_dy = torch.autograd.grad(phi, y_col,
                                  torch.ones_like(phi),
                                  create_graph=True)[0]

    d2phi_dx2 = torch.autograd.grad(dphi_dx, x_col,
                                    torch.ones_like(dphi_dx),
                                    create_graph=True)[0]
    d2phi_dy2 = torch.autograd.grad(dphi_dy, y_col,
                                    torch.ones_like(dphi_dy),
                                    create_graph=True)[0]

    loss_pde = torch.mean((d2phi_dx2 + d2phi_dy2)**2)

    loss_in = torch.mean((model(x_in,y_in)-target_in)**2)
    loss_out = torch.mean(model(x_out,y_out)**2)
    loss_bot = torch.mean(model(x_wall,y_bot)**2)
    loss_top = torch.mean(model(x_wall,y_top)**2)

    loss_bc = loss_in + loss_out + loss_bot + loss_top

    return loss_pde + 5.0*loss_bc


# =========================
# TRAINING
# =========================
model = SIREN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3000, gamma=0.5
)

print(f"Training SIREN PINN for N={N_modes}")

for i in range(iterations):

    optimizer.zero_grad()

    x_col = torch.rand(5000,1,device=device,requires_grad=True)
    y_col = torch.rand(5000,1,device=device,requires_grad=True)

    loss = compute_loss(model, x_col, y_col)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if i % 1000 == 0:
        print(f"Iter {i}, Loss: {loss.item():.6f}")


# =========================
# VISUALIZATION
# =========================
fig, axs = plt.subplots(2,2,figsize=(14,10))

y_vals = np.linspace(0,1,400)
x_slices = [0.0,0.25,0.5,0.75]

for ax, x_val in zip(axs.flat, x_slices):

    x_tensor = torch.ones(400,1,device=device)*x_val
    y_tensor = torch.tensor(y_vals.reshape(-1,1),
                            dtype=torch.float32,
                            device=device)

    with torch.no_grad():
        u_pred = model(x_tensor,y_tensor).cpu().numpy()

    u_exact = exact_solution(
        x_val*np.ones_like(y_vals),
        y_vals,
        N_modes
    )

    ax.plot(y_vals, u_exact.flatten(), 'k--', linewidth=2, label="Exact")
    ax.plot(y_vals, u_pred.flatten(), 'r-', linewidth=2, label="SIREN PINN")
    ax.set_title(f"x = {x_val}")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle(f"SIREN PINN vs Exact (N={N_modes})", fontsize=14)
plt.tight_layout()
plt.show()
