import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N_modes = 5     
iterations = 10000
learning_rate = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(42)
#np.random.seed(42)

def get_exact_solution(x, y, N):
    val = np.zeros_like(x)
    for n in range(1, N + 1):
        term = (np.sinh(n * np.pi * (1 - x)) / np.sinh(n * np.pi)) * np.sin(n * np.pi * y)
        val += term
    return val

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 74),
            nn.Tanh(),
            nn.Linear(74, 74),
            nn.Tanh(),
            nn.Linear(74, 74),
            nn.Tanh(),
            nn.Linear(74, 74),
            nn.Tanh(),
            nn.Linear(74, 1)
        )

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
    target_in = torch.zeros_like(y_in)
    for n in range(1, N + 1):
        target_in += torch.sin(n * np.pi * y_in)
    loss_inlet = torch.mean((model(x_in, y_in) - target_in)**2)

    y_out = torch.rand(100, 1, device=device)
    x_out = torch.ones(100, 1, device=device)
    loss_outlet = torch.mean(model(x_out, y_out)**2)

    x_wall = torch.rand(100, 1, device=device)
    loss_bot = torch.mean(model(x_wall, torch.zeros_like(x_wall))**2)
    loss_top = torch.mean(model(x_wall, torch.ones_like(x_wall))**2)

    return loss_pde + 10.0 * (loss_inlet + loss_outlet + loss_bot + loss_top)

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training for N={N_modes}...")

for i in range(iterations):
    optimizer.zero_grad()
    
    x_col = torch.rand(2000, 1, device=device, requires_grad=True)
    y_col = torch.rand(2000, 1, device=device, requires_grad=True)
    
    loss = compute_loss(model, x_col, y_col, N_modes)
    loss.backward()
    optimizer.step()
    
    if i % 1000 == 0:
        print(f"Iter {i}, Loss: {loss.item():.5f}")

# ---  VISUALIZATION ---
y_plot = np.linspace(0, 1, 200)
x_plot = np.zeros_like(y_plot) # At x=0

# Exact
val_exact = np.zeros_like(y_plot)
for n in range(1, N_modes + 1):
    val_exact += np.sin(n * np.pi * y_plot)

# Predicted
pt_x = torch.tensor(x_plot.reshape(-1,1), dtype=torch.float32, device=device)
pt_y = torch.tensor(y_plot.reshape(-1,1), dtype=torch.float32, device=device)
with torch.no_grad():
    val_pred = model(pt_x, pt_y).cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(y_plot, val_exact, 'k--', label=f'Exact Input (N={N_modes})', linewidth=2)
plt.plot(y_plot, val_pred, 'r-', label='PINN Prediction', linewidth=2)
plt.title(f"PINN Capability Test: N={N_modes}")
plt.xlabel("y")
plt.ylabel("phi")
plt.legend()
plt.grid(True)
plt.show()