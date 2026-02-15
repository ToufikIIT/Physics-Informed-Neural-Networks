""" import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
np.random.seed(42)

N_modes = 5
epochs = 10000
batch_size = 1800
n_bc_samples = 150      # samples per BC (inlet, walls, outlet)
lr = 0.004
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

w_pde = 1.0
w_bc = 30.0

# --- ACTIVATION FUNCTIONS ---
def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x): t = np.tanh(x); return -2 * t * (1 - t**2)
def ddd_tanh(x): t = np.tanh(x); return (-2 + 6*t**2) * (1 - t**2)

# --- 1. LINEAR LAYER (With ADAM) ---
class Linear:
    def __init__(self, in_f, out_f):
        # Xavier Initialization
        self.W = np.random.randn(out_f, in_f) * np.sqrt(1 / in_f)
        self.b = np.zeros((out_f, 1))
        
        # ADAM Cache (m = momentum, v = velocity)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        
        # Time step for bias correction
        self.t = 0

    def forward(self, h, h_x, h_y, h_xx, h_yy):
        self.h, self.h_x, self.h_y = h, h_x, h_y
        self.h_xx, self.h_yy = h_xx, h_yy
        
        return (self.W @ h + self.b), \
               (self.W @ h_x), (self.W @ h_y), \
               (self.W @ h_xx), (self.W @ h_yy)

    def backward(self, d_z, d_zx, d_zy, d_zxx, d_zyy):
        # Gradients w.r.t Weights
        dw = (d_z @ self.h.T) + \
             (d_zx @ self.h_x.T) + (d_zy @ self.h_y.T) + \
             (d_zxx @ self.h_xx.T) + (d_zyy @ self.h_yy.T)
        db = np.sum(d_z, axis=1, keepdims=True)

        # Gradients w.r.t Inputs
        d_h   = self.W.T @ d_z
        d_hx  = self.W.T @ d_zx
        d_hy  = self.W.T @ d_zy
        d_hxx = self.W.T @ d_zxx
        d_hyy = self.W.T @ d_zyy
        
        return (dw, db), (d_h, d_hx, d_hy, d_hxx, d_hyy)

    def step(self, grads, lr, beta1, beta2, epsilon):
        dw, db = grads
        self.t += 1
        
        # --- ADAM UPDATE (Weights) ---
        # 1. Update Momentum (m) and Velocity (v)
        self.mW = beta1 * self.mW + (1 - beta1) * dw
        self.vW = beta2 * self.vW + (1 - beta2) * (dw**2)
        
        # 2. Bias Correction
        m_hat_W = self.mW / (1 - beta1**self.t)
        v_hat_W = self.vW / (1 - beta2**self.t)
        
        # 3. Update Weights
        self.W -= lr * m_hat_W / (np.sqrt(v_hat_W) + epsilon)

        # --- ADAM UPDATE (Biases) ---
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * (db**2)
        
        m_hat_b = self.mb / (1 - beta1**self.t)
        v_hat_b = self.vb / (1 - beta2**self.t)
        
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

# --- 2. TANH LAYER (Forward Mode Chain Rule) ---
class Tanh:
    def forward(self, z, z_x, z_y, z_xx, z_yy):
        self.z, self.z_x, self.z_y = z, z_x, z_y
        self.z_xx, self.z_yy = z_xx, z_yy
        
        self.s = tanh(z)
        self.ds = d_tanh(z)
        self.dds = dd_tanh(z)
        
        self.a = self.s
        self.a_x = self.ds * z_x
        self.a_y = self.ds * z_y
        self.a_xx = self.dds * (z_x**2) + self.ds * z_xx
        self.a_yy = self.dds * (z_y**2) + self.ds * z_yy
        
        return self.a, self.a_x, self.a_y, self.a_xx, self.a_yy

    def backward(self, d_a, d_ax, d_ay, d_axx, d_ayy):
        ddds = ddd_tanh(self.z)
        
        # dL/dz (Full Chain Rule)
        term1 = d_a * self.ds
        term2 = d_ax * (self.dds * self.z_x)
        term3 = d_ay * (self.dds * self.z_y)
        term4 = d_axx * (ddds * (self.z_x**2) + self.dds * self.z_xx)
        term5 = d_ayy * (ddds * (self.z_y**2) + self.dds * self.z_yy)
        d_z = term1 + term2 + term3 + term4 + term5
        
        # dL/dz_x, dL/dz_y
        d_zx = (d_ax * self.ds) + (d_axx * 2 * self.dds * self.z_x)
        d_zy = (d_ay * self.ds) + (d_ayy * 2 * self.dds * self.z_y)
        
        # dL/dz_xx, dL/dz_yy
        d_zxx = d_axx * self.ds
        d_zyy = d_ayy * self.ds
        
        return d_z, d_zx, d_zy, d_zxx, d_zyy

# --- 3. PINN NETWORK ---
class PINN:
    def __init__(self, layers_cfg):
        self.layers = []
        for i in range(len(layers_cfg) - 1):
            self.layers.append(Linear(layers_cfg[i], layers_cfg[i+1]))
            if i < len(layers_cfg) - 2: 
                self.layers.append(Tanh())

    def forward(self, x_in):
        h = x_in
        # Initialize Derivatives (Broadcasting manually)
        h_x = np.zeros_like(x_in); h_x[0, :] = 1.0 
        h_y = np.zeros_like(x_in); h_y[1, :] = 1.0
        h_xx = np.zeros_like(x_in)
        h_yy = np.zeros_like(x_in)
        
        for layer in self.layers:
            h, h_x, h_y, h_xx, h_yy = layer.forward(h, h_x, h_y, h_xx, h_yy)
        return h, h_x, h_y, h_xx, h_yy

    def backward(self, d_u, d_ux, d_uy, d_uxx, d_uyy):
        grads = (d_u, d_ux, d_uy, d_uxx, d_uyy)
        l_grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, Tanh):
                grads = layer.backward(*grads)
            else:
                w_g, i_g = layer.backward(*grads)
                l_grads.append(w_g)
                grads = i_g
        return l_grads[::-1]

    def update(self, l_grads, lr, beta1, beta2, epsilon):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(l_grads[idx], lr, beta1, beta2, epsilon)
                idx += 1

# --- TRAINING LOOP ---
net = PINN([2, 70, 50, 1])

print(f"Training ADAM PINN (Scratch) for N={N_modes}...")

for epoch in range(epochs + 1):
    
    # 1. PDE LOSS
    x_c = np.random.rand(1, batch_size)
    y_c = np.random.rand(1, batch_size)
    X_c = np.vstack([x_c, y_c])
    
    u, ux, uy, uxx, uyy = net.forward(X_c)
    
    residual = uxx + uyy
    loss_pde = np.mean(residual**2)
    d_res = (2.0/batch_size) * residual
    grads_f = net.backward(np.zeros_like(u), np.zeros_like(ux), np.zeros_like(uy), d_res, d_res)

    # 2. INLET (x=0) — Dirichlet u = Σ cos(nπy)
    y_in = np.random.rand(1, n_bc_samples)
    x_in = np.zeros_like(y_in)
    X_in = np.vstack([x_in, y_in])

    u_in, _, _, _, _ = net.forward(X_in)

    target = np.zeros_like(y_in)
    for n in range(1, N_modes + 1):
        target += np.cos(n * np.pi * y_in)

    diff_in = u_in - target
    loss_in = np.mean(diff_in**2)
    d_in = (2.0 / n_bc_samples) * diff_in
    grads_in = net.backward(d_in, np.zeros_like(u_in), np.zeros_like(u_in), np.zeros_like(u_in), np.zeros_like(u_in))

    # 3. WALLS (y=0, y=1) — Neumann ∂u/∂y = 0
    x_b = np.random.rand(1, n_bc_samples)
    X_neu = np.hstack([np.vstack([x_b, np.zeros_like(x_b)]), np.vstack([x_b, np.ones_like(x_b)])])
    n_wall = 2 * n_bc_samples

    _, _, uy_neu, _, _ = net.forward(X_neu)
    loss_neu = np.mean(uy_neu**2)
    d_neu = (2.0 / n_wall) * uy_neu
    grads_neu = net.backward(np.zeros_like(uy_neu), np.zeros_like(uy_neu), d_neu, np.zeros_like(uy_neu), np.zeros_like(uy_neu))

    # 4. OUTLET (x=1) — Neumann ∂u/∂n = 0 ⇒ u_x = 0
    y_out = np.random.rand(1, n_bc_samples)
    x_out = np.ones_like(y_out)
    X_out = np.vstack([x_out, y_out])

    _, ux_out, _, _, _ = net.forward(X_out)
    loss_out = np.mean(ux_out**2)
    d_out = (2.0 / n_bc_samples) * ux_out
    grads_out = net.backward(np.zeros_like(ux_out), d_out, np.zeros_like(ux_out), np.zeros_like(ux_out), np.zeros_like(ux_out))

    # UPDATE
    final_grads = []
    for g1, g2, g3, g4 in zip(grads_f, grads_in, grads_neu, grads_out):
        dw = w_pde*g1[0] + w_bc*g2[0] + w_bc*g3[0] + w_bc*g4[0]
        db = w_pde*g1[1] + w_bc*g2[1] + w_bc*g3[1] + w_bc*g4[1]
        final_grads.append((dw, db))
        
    net.update(final_grads, lr, beta1, beta2, epsilon)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | PDE: {loss_pde:.5f} | Inlet: {loss_in:.5f} | Walls: {loss_neu:.5f} | Outlet: {loss_out:.5f}")

# --- VISUALIZATION ---
grid_size = 60
x_vals = np.linspace(0, 1, grid_size)
y_vals = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x_vals, y_vals)
inputs = np.vstack([X.ravel(), Y.ravel()])

u_pred, _, _, _, _ = net.forward(inputs)
Z_pred = u_pred.reshape(grid_size, grid_size)

# Exact solution for Laplace with inlet u(0,y)=Σ cos(nπy) and walls u_y=0 (semi-infinite strip)
Z_exact = np.zeros_like(X)
for n in range(1, N_modes + 1):
    Z_exact += np.exp(-n * np.pi * X) * np.cos(n * np.pi * Y)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(y_vals, Z_exact[:, 0], 'k--', label='Exact', linewidth=2)
ax1.plot(y_vals, Z_pred[:, 0], 'r-', label='Adam PINN', linewidth=2)
ax1.set_title(f'Inlet at x=0 (N={N_modes})')
ax1.legend()
ax1.grid(True)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z_pred, cmap='viridis')
ax2.set_title('Predicted Solution (Adam)')
plt.show() """

import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_modes = 3           
epochs = 5000        # More epochs to refine the interior
lr = 0.001            
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8        

# --- WEIGHTS (Crucial for balancing) ---
w_pde = 2.0           # Increased weight for Physics
w_bc = 10.0           # Boundary weight

# --- ACTIVATION FUNCTIONS ---
def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x): t = np.tanh(x); return -2 * t * (1 - t**2)

# --- LINEAR LAYER (Adam Optimizer built-in) ---
class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(out_f, in_f) * np.sqrt(1 / in_f)
        self.b = np.zeros((out_f, 1))
        # Adam Cache
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0

    def forward(self, h, h_x, h_y, h_xx, h_yy):
        self.h, self.h_x, self.h_y = h, h_x, h_y
        self.h_xx, self.h_yy = h_xx, h_yy
        return (self.W@h + self.b), (self.W@h_x), (self.W@h_y), (self.W@h_xx), (self.W@h_yy)

    def backward(self, d_z, d_zx, d_zy, d_zxx, d_zyy):
        dw = (d_z @ self.h.T) + (d_zx @ self.h_x.T) + (d_zy @ self.h_y.T) + (d_zxx @ self.h_xx.T) + (d_zyy @ self.h_yy.T)
        db = np.sum(d_z, axis=1, keepdims=True)
        d_h = self.W.T @ d_z
        d_hx, d_hy = self.W.T @ d_zx, self.W.T @ d_zy
        d_hxx, d_hyy = self.W.T @ d_zxx, self.W.T @ d_zyy
        return (dw, db), (d_h, d_hx, d_hy, d_hxx, d_hyy)

    def step(self, grads, lr):
        dw, db = grads
        self.t += 1
        # Adam Update for W
        self.mW = beta1 * self.mW + (1 - beta1) * dw
        self.vW = beta2 * self.vW + (1 - beta2) * (dw**2)
        m_hat = self.mW / (1 - beta1**self.t)
        v_hat = self.vW / (1 - beta2**self.t)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        # Adam Update for b
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * (db**2)
        m_hat_b = self.mb / (1 - beta1**self.t)
        v_hat_b = self.vb / (1 - beta2**self.t)
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

# --- TANH LAYER ---
class Tanh:
    def forward(self, z, z_x, z_y, z_xx, z_yy):
        self.z, self.z_x, self.z_y, self.z_xx, self.z_yy = z, z_x, z_y, z_xx, z_yy
        self.s = tanh(z); self.ds = d_tanh(z); self.dds = dd_tanh(z)
        self.a = self.s
        self.a_x, self.a_y = self.ds * z_x, self.ds * z_y
        self.a_xx = self.dds * (z_x**2) + self.ds * z_xx
        self.a_yy = self.dds * (z_y**2) + self.ds * z_yy
        return self.a, self.a_x, self.a_y, self.a_xx, self.a_yy

    def backward(self, d_a, d_ax, d_ay, d_axx, d_ayy):
        # Full Chain Rule (Recovered from previous step)
        term1 = d_a * self.ds
        term2 = d_ax * self.dds * self.z_x + d_ay * self.dds * self.z_y
        term3 = d_axx * (dd_tanh(self.z)*(self.z_x**2)*-2*self.s + self.dds*self.z_xx) # Approx for speed
        # Simplified robust backward for 2nd order:
        d_z = d_a * self.ds + d_ax * self.dds * self.z_x + d_ay * self.dds * self.z_y + \
              d_axx * (self.dds * self.z_xx) + d_ayy * (self.dds * self.z_yy) 
        d_zx = d_ax * self.ds + d_axx * 2 * self.dds * self.z_x
        d_zy = d_ay * self.ds + d_ayy * 2 * self.dds * self.z_y
        d_zxx = d_axx * self.ds
        d_zyy = d_ayy * self.ds
        return d_z, d_zx, d_zy, d_zxx, d_zyy

class PINN:
    def __init__(self, layers_cfg):
        self.layers = [Linear(layers_cfg[i], layers_cfg[i+1]) for i in range(len(layers_cfg)-1)]
        self.activations = [Tanh() for _ in range(len(layers_cfg)-2)]
    
    def forward(self, x):
        h = x; h_x = np.zeros_like(x); h_x[0,:]=1; h_y = np.zeros_like(x); h_y[1,:]=1
        h_xx = np.zeros_like(x); h_yy = np.zeros_like(x)
        for i, layer in enumerate(self.layers):
            h, h_x, h_y, h_xx, h_yy = layer.forward(h, h_x, h_y, h_xx, h_yy)
            if i < len(self.activations):
                h, h_x, h_y, h_xx, h_yy = self.activations[i].forward(h, h_x, h_y, h_xx, h_yy)
        return h, h_x, h_y, h_xx, h_yy

    def backward(self, d_u, d_ux, d_uy, d_uxx, d_uyy):
        grads = (d_u, d_ux, d_uy, d_uxx, d_uyy)
        l_grads = []
        for i in reversed(range(len(self.layers))):
            if i < len(self.activations): grads = self.activations[i].backward(*grads)
            gw, gi = self.layers[i].backward(*grads)
            l_grads.append(gw); grads = gi
        return l_grads[::-1]

    def update(self, l_grads, lr):
        for i, layer in enumerate(self.layers): layer.step(l_grads[i], lr)

# --- MAIN EXECUTION ---
net = PINN([2, 50, 50, 50, 1])

for epoch in range(epochs + 1):
    # 1. PDE LOSS (Sample 1000 points! High volume sampling fixes the "Droop")
    X_c = np.random.rand(2, 1000) 
    u, ux, uy, uxx, uyy = net.forward(X_c)
    res = uxx + uyy
    d_res = (2.0/1000) * res
    gf = net.backward(np.zeros_like(u), np.zeros_like(ux), np.zeros_like(uy), d_res, d_res)

    # 2. INLET (x=0)
    y_in = np.random.rand(1, 100); X_in = np.vstack([np.zeros((1,100)), y_in])
    u_in, _, _, _, _ = net.forward(X_in)
    target = sum(np.cos(n * np.pi * y_in) for n in range(1, N_modes + 1))
    d_in = (2.0/100) * (u_in - target)
    gin = net.backward(d_in, np.zeros_like(u_in), np.zeros_like(u_in), np.zeros_like(u_in), np.zeros_like(u_in))

    # 3. WALLS (Neumann)
    x_b = np.random.rand(1, 100)
    X_neu = np.hstack([np.vstack([x_b, np.zeros_like(x_b)]), np.vstack([x_b, np.ones_like(x_b)])])
    _, _, uy_neu, _, _ = net.forward(X_neu)
    d_neu = (2.0/200) * uy_neu
    gneu = net.backward(np.zeros_like(uy_neu), np.zeros_like(uy_neu), d_neu, np.zeros_like(uy_neu), np.zeros_like(uy_neu))

    # Update
    final_grads = []
    for g1, g2, g3 in zip(gf, gin, gneu):
        final_grads.append((w_pde*g1[0] + w_bc*g2[0] + w_bc*g3[0], w_pde*g1[1] + w_bc*g2[1] + w_bc*g3[1]))
    net.update(final_grads, lr)
    
    if epoch % 1000 == 0: print(f"Iter {epoch} | PDE Loss: {np.mean(res**2):.5f}")

# --- REPLICATE YOUR PLOT ---
# Exact Solution Function
def get_exact(x, y):
    sol = np.zeros_like(x)
    for n in range(1, N_modes+1): sol += np.exp(-n*np.pi*x) * np.cos(n*np.pi*y)
    return sol

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(15, 8))
y_plot = np.linspace(0, 1, 100).reshape(1, -1)
x_locs = [0.0, 0.25, 0.5, 0.75]

for ax, x_val in zip(axs.flat, x_locs):
    X_plot = np.vstack([np.ones_like(y_plot)*x_val, y_plot])
    u_pred, _, _, _, _ = net.forward(X_plot)
    u_exact = get_exact(X_plot[0], X_plot[1])
    
    ax.plot(y_plot.flatten(), u_exact.flatten(), 'k--', linewidth=2, label='Exact')
    ax.plot(y_plot.flatten(), u_pred.flatten(), 'r-', linewidth=2, label='PINN')
    ax.set_title(f'x = {x_val}')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()