""" import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_modes = 10     # <--- Change this to test N=1, 3, 5, 10
iterations = 6000
learning_rate = 1e-3

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- THE PINN MODEL ---
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Standard MLP
        self.net = nn.Sequential(
            nn.Linear(2, 60),
            nn.Tanh(),
            nn.Linear(60, 60),
            nn.Tanh(),
            #nn.Linear(50, 50),
            #nn.Tanh(),
            nn.Linear(60, 1)
        )

    def forward(self, x, y):
        # Concatenate x and y
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

# --- PHYSICS LOSS ---
def physics_loss(model, x, y):
    x.requires_grad = True
    y.requires_grad = True
    phi = model(x, y)
    
    # First derivatives
    dphi_dx = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True)[0]
    dphi_dy = torch.autograd.grad(phi, y, torch.ones_like(phi), create_graph=True)[0]
    
    # Second derivatives (Laplacian)
    d2phi_dx2 = torch.autograd.grad(dphi_dx, x, torch.ones_like(dphi_dx), create_graph=True)[0]
    d2phi_dy2 = torch.autograd.grad(dphi_dy, y, torch.ones_like(dphi_dy), create_graph=True)[0]
    
    # Residual: del^2 phi = 0
    residual = d2phi_dx2 + d2phi_dy2
    return torch.mean(residual**2)

# --- BOUNDARY CONDITIONS ---
def boundary_loss(model, N):
    # 1. Inlet (x=0) -> Dirichlet with COSINE series
    y_in = torch.rand(200, 1)
    x_in = torch.zeros(200, 1)
    
    # --- THE FIX IS HERE ---
    # We use COS instead of SIN so the derivative at y=0 and y=1 is zero.
    target_val = torch.zeros_like(y_in)
    for n in range(1, N + 1):
        target_val += torch.cos(n * np.pi * y_in)
        
    phi_pred = model(x_in, y_in)
    loss_inlet = torch.mean((phi_pred - target_val)**2)

    # 2. Bottom Wall (y=0) -> d(phi)/dy = 0
    x_bot = torch.rand(100, 1)
    y_bot = torch.zeros(100, 1)
    y_bot.requires_grad = True
    phi_bot = model(x_bot, y_bot)
    dphi_dy_bot = torch.autograd.grad(phi_bot, y_bot, torch.ones_like(phi_bot), create_graph=True)[0]
    loss_bot = torch.mean(dphi_dy_bot**2)

    # 3. Top Wall (y=1) -> d(phi)/dy = 0
    x_top = torch.rand(100, 1)
    y_top = torch.ones(100, 1)
    y_top.requires_grad = True
    phi_top = model(x_top, y_top)
    dphi_dy_top = torch.autograd.grad(phi_top, y_top, torch.ones_like(phi_top), create_graph=True)[0]
    loss_top = torch.mean(dphi_dy_top**2)

    # 4. Outlet (x=1) -> d(phi)/dy = 0
    x_out = torch.ones(100, 1)
    y_out = torch.rand(100, 1)
    y_out.requires_grad = True
    phi_out = model(x_out, y_out)
    dphi_dy_out = torch.autograd.grad(phi_out, y_out, torch.ones_like(phi_out), create_graph=True)[0]
    loss_out = torch.mean(dphi_dy_out**2)
    
    return loss_inlet + loss_bot + loss_top + loss_out

# --- TRAINING LOOP ---
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training PyTorch PINN with Corrected Inlet (Cosine) for N={N_modes}...")

loss_history = []

for i in range(iterations):
    optimizer.zero_grad()
    
    # Collocation points
    x_col = torch.rand(2000, 1)
    y_col = torch.rand(2000, 1)
    
    loss_pde = physics_loss(model, x_col, y_col)
    loss_bc = boundary_loss(model, N_modes)
    
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if i % 500 == 0:
        print(f"Iter {i}, Loss: {loss.item():.5f}")

# --- VISUALIZATION ---
# 1. Inlet Reconstruction Check
y_plot = torch.linspace(0, 1, 300).view(-1, 1)
x_plot = torch.zeros_like(y_plot)

with torch.no_grad():
    phi_pred = model(x_plot, y_plot)

# Exact Target (Cosine)
phi_exact = torch.zeros_like(y_plot)
for n in range(1, N_modes + 1):
    phi_exact += torch.cos(n * np.pi * y_plot)

plt.figure(figsize=(12, 5))

# Plot 1: Inlet Match
plt.subplot(1, 2, 1)
plt.plot(y_plot.numpy(), phi_exact.numpy(), 'k--', label='Exact Input (Cosine)', linewidth=2)
plt.plot(y_plot.numpy(), phi_pred.numpy(), 'r-', label='PINN Prediction', linewidth=2)
plt.title(f'Inlet Reconstruction (N={N_modes})')
plt.xlabel('y')
plt.ylabel('phi')
plt.legend()
plt.grid(True)

# Plot 2: 2D Heatmap of the Pipe
with torch.no_grad():
    x_grid = torch.linspace(0, 1, 100)
    y_grid = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
    inputs = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    Z = model(inputs[:, 0:1], inputs[:, 1:2]).reshape(100, 100)

plt.subplot(1, 2, 2)
plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), 50, cmap='jet')
plt.colorbar(label='phi')
plt.title('Flow Field inside Pipe')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()  """

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N_modes = 7
iterations = 12000    
learning_rate = 1e-3

#torch.manual_seed(42)
#np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

def compute_loss(model, x_col, y_col, N):
    phi = model(x_col, y_col)
    
    dphi_dx = torch.autograd.grad(phi, x_col, torch.ones_like(phi), create_graph=True)[0]
    dphi_dy = torch.autograd.grad(phi, y_col, torch.ones_like(phi), create_graph=True)[0]
    
    d2phi_dx2 = torch.autograd.grad(dphi_dx, x_col, torch.ones_like(dphi_dx), create_graph=True)[0]
    d2phi_dy2 = torch.autograd.grad(dphi_dy, y_col, torch.ones_like(dphi_dy), create_graph=True)[0]
    
    loss_pde = torch.mean((d2phi_dx2 + d2phi_dy2)**2)

    y_in = torch.rand(200, 1, device=device)
    x_in = torch.zeros(200, 1, device=device)
    
    target_val = torch.zeros_like(y_in)
    for n in range(1, N + 1):
        target_val += torch.cos(n * np.pi * y_in)
        
    phi_in = model(x_in, y_in)
    loss_inlet = torch.mean((phi_in - target_val)**2)

    x_bot = torch.rand(100, 1, device=device)
    y_bot = torch.zeros(100, 1, device=device, requires_grad=True)
    phi_bot = model(x_bot, y_bot)
    dphi_dy_bot = torch.autograd.grad(phi_bot, y_bot, torch.ones_like(phi_bot), create_graph=True)[0]
    
    x_top = torch.rand(100, 1, device=device)
    y_top = torch.ones(100, 1, device=device, requires_grad=True)
    phi_top = model(x_top, y_top)
    dphi_dy_top = torch.autograd.grad(phi_top, y_top, torch.ones_like(phi_top), create_graph=True)[0]
    
    x_out = torch.ones(100, 1, device=device)
    y_out = torch.rand(100, 1, device=device, requires_grad=True)
    phi_out = model(x_out, y_out)
    dphi_dy_out = torch.autograd.grad(phi_out, y_out, torch.ones_like(phi_out), create_graph=True)[0]
    
    loss_bc = loss_inlet + torch.mean(dphi_dy_bot**2) + torch.mean(dphi_dy_top**2) + torch.mean(dphi_dy_out**2)
    
    return loss_pde + 30.0 * loss_bc

model = PINN().to(device)

x_col = torch.rand(2000, 1, device=device, requires_grad=True)
y_col = torch.rand(2000, 1, device=device, requires_grad=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

print(f"Training with Adam Only (N={N_modes})...")

loss_history = []

for i in range(iterations):
    optimizer.zero_grad()
    
    loss = compute_loss(model, x_col, y_col, N_modes)
    loss.backward()
    
    optimizer.step()
    
    #scheduler.step(loss)
    
    loss_history.append(loss.item())
    
    if i % 1000 == 0:
        print(f"Iter {i}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

""" # --- VISUALIZATION  ---

x_vals = torch.linspace(0, 1, 100)
y_vals = torch.linspace(0, 1, 100)
X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')

inputs = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1).to(device)

with torch.no_grad():
    Z_pred = model(inputs[:, 0:1], inputs[:, 1:2])
    Z_pred = Z_pred.reshape(100, 100).cpu().numpy()

Z_exact = torch.zeros_like(X)
for n in range(1, N_modes + 1):
    term = torch.exp(-n * np.pi * X) * torch.cos(n * np.pi * Y)
    Z_exact += term
Z_exact = Z_exact.numpy()

fig = plt.figure(figsize=(16, 12))

# PLOT 1: 2D Inlet Comparison (x=0)
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(y_vals.numpy(), Z_exact[0, :], 'k--', label='Exact Inlet (Analytical)', linewidth=2)
ax1.plot(y_vals.numpy(), Z_pred[0, :], 'r-', label='PINN Prediction', linewidth=2)
ax1.set_title(f'Inlet Reconstruction at x=0 (N={N_modes})', fontsize=14)
ax1.set_xlabel('y')
ax1.set_ylabel('phi')
ax1.legend()
ax1.grid(True, alpha=0.3)

# PLOT 2: 3D Surface - EXACT
ax2 = fig.add_subplot(2, 2, 3, projection='3d')
surf2 = ax2.plot_surface(X.numpy(), Y.numpy(), Z_exact, cmap='viridis', 
                         linewidth=0, antialiased=False, alpha=0.8)
ax2.set_title('Expected (Exact Analytical)', fontsize=14)
ax2.set_xlabel('x (Depth)')
ax2.set_ylabel('y (Width)')
ax2.set_zlabel('phi')
ax2.view_init(elev=30, azim=45)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

# PLOT 3: 3D Surface - PREDICTED
ax3 = fig.add_subplot(2, 2, 4, projection='3d')
surf3 = ax3.plot_surface(X.numpy(), Y.numpy(), Z_pred, cmap='viridis', 
                         linewidth=0, antialiased=False, alpha=0.8)
ax3.set_title('Predicted (PINN)', fontsize=14)
ax3.set_xlabel('x (Depth)')
ax3.set_ylabel('y (Width)')
ax3.set_zlabel('phi')
ax3.view_init(elev=30, azim=45)
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show() """
# --- VISUALIZATION: 2D SLICES ---

x_slices = [0.0, 0.25, 0.5, 0.75]
y_vals = torch.linspace(0, 1, 400).unsqueeze(1).to(device)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

with torch.no_grad():
    for i, x0 in enumerate(x_slices):

        x_vals = torch.full_like(y_vals, x0)

        # PINN prediction
        phi_pred = model(x_vals, y_vals).cpu().numpy().flatten()

        # Exact solution
        phi_exact = np.zeros_like(phi_pred)
        for n in range(1, N_modes + 1):
            phi_exact += np.exp(-n * np.pi * x0) * np.cos(n * np.pi * y_vals.cpu().numpy().flatten())

        ax = axes[i]
        ax.plot(y_vals.cpu().numpy(), phi_exact, 'k--', linewidth=2, label='Exact')
        ax.plot(y_vals.cpu().numpy(), phi_pred, 'r-', linewidth=2, label='PINN')

        ax.set_title(f"x = {x0}")
        ax.set_xlabel("y")
        ax.set_ylabel("φ")
        ax.grid(True, alpha=0.3)
        ax.legend()

plt.suptitle(f"Predicted vs Exact Solution at Different x-slices (N={N_modes})", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
