import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
N_modes = 7          # <--- Set to 10. (If this works, try 15!)
iterations = 15000    # Increased to 15,000 to give it time to learn high freq
learning_rate = 1e-3

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- THE PINN MODEL ---
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        # INCREASED WIDTH: 64 -> 100 neurons per layer
        # This gives the network more "capacity" to memorize the wiggles.
        self.net = nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1)
        )
        
        # Xavier Initialization (Crucial for Tanh)
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

# --- LOSS FUNCTION ---
def compute_loss(model, x_col, y_col, N):
    # 1. Physics Loss
    phi = model(x_col, y_col)
    
    dphi_dx = torch.autograd.grad(phi, x_col, torch.ones_like(phi), create_graph=True)[0]
    dphi_dy = torch.autograd.grad(phi, y_col, torch.ones_like(phi), create_graph=True)[0]
    d2phi_dx2 = torch.autograd.grad(dphi_dx, x_col, torch.ones_like(dphi_dx), create_graph=True)[0]
    d2phi_dy2 = torch.autograd.grad(dphi_dy, y_col, torch.ones_like(dphi_dy), create_graph=True)[0]
    
    loss_pde = torch.mean((d2phi_dx2 + d2phi_dy2)**2)

    # 2. Boundary Loss
    # Inlet (x=0)
    y_in = torch.rand(300, 1, device=device) # More points (300) for better sampling
    x_in = torch.zeros(300, 1, device=device)
    target_val = torch.zeros_like(y_in)
    for n in range(1, N + 1):
        target_val += torch.cos(n * np.pi * y_in)
    
    phi_in = model(x_in, y_in)
    loss_inlet = torch.mean((phi_in - target_val)**2)

    # Walls (Neumann)
    x_bot = torch.rand(100, 1, device=device)
    y_bot = torch.zeros(100, 1, device=device, requires_grad=True)
    dphi_dy_bot = torch.autograd.grad(model(x_bot, y_bot), y_bot, torch.ones((100,1), device=device), create_graph=True)[0]

    x_top = torch.rand(100, 1, device=device)
    y_top = torch.ones(100, 1, device=device, requires_grad=True)
    dphi_dy_top = torch.autograd.grad(model(x_top, y_top), y_top, torch.ones((100,1), device=device), create_graph=True)[0]

    x_out = torch.ones(100, 1, device=device)
    y_out = torch.rand(100, 1, device=device, requires_grad=True)
    dphi_dy_out = torch.autograd.grad(model(x_out, y_out), y_out, torch.ones((100,1), device=device), create_graph=True)[0]
    
    # Weight BCs higher (20.0)
    loss_bc = loss_inlet + torch.mean(dphi_dy_bot**2) + torch.mean(dphi_dy_top**2) + torch.mean(dphi_dy_out**2)
    
    return loss_pde + 20.0 * loss_bc

# --- TRAINING LOOP ---
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Reliable StepLR Scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

print(f"Training Robust PINN for N={N_modes}...")

# Fixed collocation points for stability
x_col = torch.rand(3000, 1, device=device, requires_grad=True)
y_col = torch.rand(3000, 1, device=device, requires_grad=True)

for i in range(iterations):
    optimizer.zero_grad()
    loss = compute_loss(model, x_col, y_col, N_modes)
    loss.backward()
    optimizer.step()
    #scheduler.step()
    
    if i % 1000 == 0:
        print(f"Iter {i}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

# --- VISUALIZATION ---
# Grid for plotting
x_vals = torch.linspace(0, 1, 100)
y_vals = torch.linspace(0, 1, 100)
X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
inputs = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1).to(device)

# 1. Prediction
with torch.no_grad():
    Z_pred = model(inputs[:, 0:1], inputs[:, 1:2]).reshape(100, 100).cpu().numpy()

# 2. Exact Solution
Z_exact = torch.zeros_like(X)
for n in range(1, N_modes + 1):
    term = torch.exp(-n * np.pi * X) * torch.cos(n * np.pi * Y)
    Z_exact += term
Z_exact = Z_exact.numpy()

# 3. Plots
fig = plt.figure(figsize=(16, 8))

# Plot A: 2D Inlet (x=0)
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(y_vals.numpy(), Z_exact[0, :], 'k--', label='Expected (Exact)', linewidth=2)
ax1.plot(y_vals.numpy(), Z_pred[0, :], 'r-', label='Predicted (PINN)', linewidth=2)
ax1.set_title(f'Inlet Reconstruction at x=0 (N={N_modes})')
ax1.set_xlabel('y')
ax1.set_ylabel('phi')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot B: 3D Surface - PREDICTED
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax2.plot_surface(X.numpy(), Y.numpy(), Z_pred, cmap='viridis', antialiased=False, alpha=0.9)
ax2.set_title('Predicted 3D Solution (PINN)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('phi')
ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()