import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_modes = 5
iterations = 12000    
learning_rate = 1e-3

# Set seed for reproducibility (Optional but recommended)
#torch.manual_seed(42)
#np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PINN MODEL ---
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
            nn.Linear(64, 1)
        )
        
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        # Concatenate x and y: [Batch, 2]
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

# --- LOSS FUNCTION ---
def compute_loss(model, x_col, y_col, N):
    # 1. Physics Loss (Interior)
    phi = model(x_col, y_col)
    
    # First derivatives
    dphi_dx = torch.autograd.grad(phi, x_col, torch.ones_like(phi), create_graph=True)[0]
    dphi_dy = torch.autograd.grad(phi, y_col, torch.ones_like(phi), create_graph=True)[0]
    
    # Second derivatives (Laplacian)
    d2phi_dx2 = torch.autograd.grad(dphi_dx, x_col, torch.ones_like(dphi_dx), create_graph=True)[0]
    d2phi_dy2 = torch.autograd.grad(dphi_dy, y_col, torch.ones_like(dphi_dy), create_graph=True)[0]
    
    loss_pde = torch.mean((d2phi_dx2 + d2phi_dy2)**2)

    # 2. Boundary Conditions
    
    # A. Inlet (x=0) -> Dirichlet: sum(cos(n*pi*y))
    y_in = torch.rand(200, 1, device=device)
    x_in = torch.zeros(200, 1, device=device)
    
    target_val = torch.zeros_like(y_in)
    for n in range(1, N + 1):
        target_val += torch.cos(n * np.pi * y_in)
        
    phi_in = model(x_in, y_in)
    loss_inlet = torch.mean((phi_in - target_val)**2)

    # B. Bottom Wall (y=0) -> Neumann: dphi/dy = 0
    x_bot = torch.rand(100, 1, device=device)
    y_bot = torch.zeros(100, 1, device=device, requires_grad=True)
    phi_bot = model(x_bot, y_bot)
    dphi_dy_bot = torch.autograd.grad(phi_bot, y_bot, torch.ones_like(phi_bot), create_graph=True)[0]
    
    # C. Top Wall (y=1) -> Neumann: dphi/dy = 0
    x_top = torch.rand(100, 1, device=device)
    y_top = torch.ones(100, 1, device=device, requires_grad=True)
    phi_top = model(x_top, y_top)
    dphi_dy_top = torch.autograd.grad(phi_top, y_top, torch.ones_like(phi_top), create_graph=True)[0]
    
    # D. Outlet (x=1) -> Neumann: dphi/dy = 0 (As per your request)
    x_out = torch.ones(100, 1, device=device)
    y_out = torch.rand(100, 1, device=device, requires_grad=True)
    phi_out = model(x_out, y_out)
    dphi_dy_out = torch.autograd.grad(phi_out, y_out, torch.ones_like(phi_out), create_graph=True)[0]
    
    # Total Boundary Loss
    loss_bc = loss_inlet + torch.mean(dphi_dy_bot**2) + torch.mean(dphi_dy_top**2) + torch.mean(dphi_dy_out**2)
    
    return loss_pde + 30.0 * loss_bc

# --- TRAINING SETUP ---
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Corrected Scheduler: StepLR step() takes no arguments
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.5)

print(f"Training with Adam (N={N_modes})...")

loss_history = []

for i in range(iterations):
    optimizer.zero_grad()
    
    # FIX: Sample NEW collocation points every iteration to avoid overfitting
    x_col = torch.rand(2000, 1, device=device, requires_grad=True)
    y_col = torch.rand(2000, 1, device=device, requires_grad=True)
    
    loss = compute_loss(model, x_col, y_col, N_modes)
    loss.backward()
    
    optimizer.step()
    #scheduler.step()
    
    loss_history.append(loss.item())
    
    if i % 1000 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Iter {i}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

# --- VISUALIZATION ---
x_slices = [0.0, 0.25, 0.5, 0.75]
y_vals = torch.linspace(0, 1, 400).unsqueeze(1).to(device)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

with torch.no_grad():
    for i, x0 in enumerate(x_slices):
        x_vals = torch.full_like(y_vals, x0)

        # PINN prediction
        phi_pred = model(x_vals, y_vals).cpu().numpy().flatten()

        # Exact solution: sum( exp(-n*pi*x) * cos(n*pi*y) )
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