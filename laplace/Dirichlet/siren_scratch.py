import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_modes = 5
epochs = 5000         # SIREN converges much faster!
batch_size = 1000
lr = 0.0005           # SIREN needs a smaller learning rate

w_pde = 1.0
w_bc = 15.0           # Slightly higher BC weight helps SIREN lock frequencies

# SIREN Hyperparameter
OMEGA_0 = 30.0        # The frequency factor (Critical for SIREN)

# --- ACTIVATION FUNCTIONS (SINE) ---
def sin_act(x): return np.sin(x)
def d_sin(x): return np.cos(x)
def dd_sin(x): return -np.sin(x)
def d3_sin(x): return -np.cos(x)

class Linear:
    def __init__(self, in_f, out_f, is_first=False):
        self.in_f = in_f
        self.out_f = out_f
        
        # --- SIREN INITIALIZATION (Crucial) ---
        # 1. First Layer: Uniform(-1/fan_in, 1/fan_in)
        # 2. Hidden Layers: Uniform(-sqrt(6/fan_in)/omega, sqrt(6/fan_in)/omega)
        
        if is_first:
            limit = 1.0 / in_f
            self.W = np.random.uniform(-limit, limit, (out_f, in_f))
        else:
            limit = np.sqrt(6 / in_f) / OMEGA_0
            self.W = np.random.uniform(-limit, limit, (out_f, in_f))
            
        self.b = np.zeros((out_f, 1))
        
        # Adam Optimizer
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t = 0

    def forward(self, h, h_x, h_y, h_xx, h_yy):
        self.h, self.h_x, self.h_y = h, h_x, h_y
        self.h_xx, self.h_yy = h_xx, h_yy
        return (self.W@h + self.b), (self.W@h_x), (self.W@h_y), (self.W@h_xx), (self.W@h_yy)

    def backward(self, d_z, d_zx, d_zy, d_zxx, d_zyy):
        dw = (d_z @ self.h.T) + (d_zx @ self.h_x.T) + (d_zy @ self.h_y.T) + \
             (d_zxx @ self.h_xx.T) + (d_zyy @ self.h_yy.T)
        db = np.sum(d_z, axis=1, keepdims=True)
        
        d_h = self.W.T @ d_z
        d_hx, d_hy = self.W.T @ d_zx, self.W.T @ d_zy
        d_hxx, d_hyy = self.W.T @ d_zxx, self.W.T @ d_zyy
        return (dw, db), (d_h, d_hx, d_hy, d_hxx, d_hyy)

    def step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        dw, db = grads
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * dw
        self.vW = beta2 * self.vW + (1 - beta2) * (dw**2)
        m_hat = self.mW / (1 - beta1**self.t)
        v_hat = self.vW / (1 - beta2**self.t)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * (db**2)
        m_hat_b = self.mb / (1 - beta1**self.t)
        v_hat_b = self.vb / (1 - beta2**self.t)
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

class Sine:
    def forward(self, z, z_x, z_y, z_xx, z_yy):
        # Apply OMEGA_0 scaling implicitly via the weights (standard SIREN practice)
        # We just compute sin(z) here.
        self.z, self.z_x, self.z_y, self.z_xx, self.z_yy = z, z_x, z_y, z_xx, z_yy
        
        self.s = sin_act(z)
        self.ds = d_sin(z)
        self.dds = dd_sin(z)
        self.ddds = d3_sin(z)
        
        self.a = self.s
        self.a_x = self.ds * z_x
        self.a_y = self.ds * z_y
        self.a_xx = self.dds * (z_x**2) + self.ds * z_xx
        self.a_yy = self.dds * (z_y**2) + self.ds * z_yy
        return self.a, self.a_x, self.a_y, self.a_xx, self.a_yy

    def backward(self, d_a, d_ax, d_ay, d_axx, d_ayy):
        # Chain rule logic is identical to Tanh, just different derivatives
        d_z = d_a * self.ds + \
              d_ax * (self.dds * self.z_x) + \
              d_ay * (self.dds * self.z_y) + \
              d_axx * (self.ddds * (self.z_x**2) + self.dds * self.z_xx) + \
              d_ayy * (self.ddds * (self.z_y**2) + self.dds * self.z_yy)
              
        d_zx = d_ax * self.ds + d_axx * (2 * self.dds * self.z_x)
        d_zy = d_ay * self.ds + d_ayy * (2 * self.dds * self.z_y)
        d_zxx = d_axx * self.ds
        d_zyy = d_ayy * self.ds
        return d_z, d_zx, d_zy, d_zxx, d_zyy

class PINN:
    def __init__(self, layers_cfg):
        self.layers = []
        # Flag the first layer for special initialization
        self.layers.append(Linear(layers_cfg[0], layers_cfg[1], is_first=True))
        for i in range(1, len(layers_cfg)-1):
            self.layers.append(Linear(layers_cfg[i], layers_cfg[i+1], is_first=False))
            
        self.activations = [Sine() for _ in range(len(layers_cfg)-2)]
     
    def forward(self, x):
        h = x
        h_x = np.zeros_like(x); h_x[0,:]=1; h_y = np.zeros_like(x); h_y[1,:]=1
        h_xx = np.zeros_like(x); h_yy = np.zeros_like(x)
        
        # SIREN Trick: Scale input by Omega_0 immediately
        h = h * OMEGA_0
        h_x = h_x * OMEGA_0
        h_y = h_y * OMEGA_0
        h_xx = h_xx * (OMEGA_0**2) # Chain rule applies to 2nd deriv scaling too!
        h_yy = h_yy * (OMEGA_0**2)

        for i, layer in enumerate(self.layers):
            h, h_x, h_y, h_xx, h_yy = layer.forward(h, h_x, h_y, h_xx, h_yy)
            if i < len(self.activations):
                h, h_x, h_y, h_xx, h_yy = self.activations[i].forward(h, h_x, h_y, h_xx, h_yy)
        return h, h_x, h_y, h_xx, h_yy

    def backward(self, d_u, d_ux, d_uy, d_uxx, d_uyy):
        grads = (d_u, d_ux, d_uy, d_uxx, d_uyy)
        l_grads = []
        for i in reversed(range(len(self.layers))):
            gw, gi = self.layers[i].backward(*grads)
            l_grads.append(gw)
            grads = gi
            if i > 0: 
                grads = self.activations[i-1].backward(*grads)
        return l_grads[::-1]

    def update(self, l_grads, lr):
        for i, layer in enumerate(self.layers): layer.step(l_grads[i], lr)

# --- TRAINING ---
# SIREN typically needs fewer neurons to fit high frequency
net = PINN([2, 60, 60, 60, 60, 1]) 

print(f"Training SIREN PINN (Omega={OMEGA_0}) for N={N_modes}...")

for epoch in range(epochs + 1):

    X_c = np.random.rand(2, 1000)
    u, ux, uy, uxx, uyy = net.forward(X_c)
    res = uxx + uyy
    d_res = (2.0/1000) * res
    gf = net.backward(np.zeros_like(u), np.zeros_like(ux), np.zeros_like(uy), d_res, d_res)

    # 2. BC Inlet
    y_in = np.random.rand(1, 100)
    X_in = np.vstack([np.zeros((1,100)), y_in])
    u_in, _, _, _, _ = net.forward(X_in)
    target_in = sum(np.sin(n * np.pi * y_in) for n in range(1, N_modes + 1))
    d_in = (2.0/100) * w_bc * (u_in - target_in)
    gin = net.backward(d_in, np.zeros_like(u_in), np.zeros_like(u_in), np.zeros_like(u_in), np.zeros_like(u_in))

    # 3. BC Walls
    r = np.random.rand(1, 100); o = np.ones((1, 100)); z = np.zeros((1, 100))
    X_walls = np.hstack([np.vstack([o, r]), np.vstack([r, z]), np.vstack([r, o])])
    u_walls, _, _, _, _ = net.forward(X_walls)
    d_walls = (2.0/300) * w_bc * u_walls
    gwalls = net.backward(d_walls, np.zeros_like(u_walls), np.zeros_like(u_walls), np.zeros_like(u_walls), np.zeros_like(u_walls))

    # Update
    final_grads = []
    for g1, g2, g3 in zip(gf, gin, gwalls):
        final_grads.append((w_pde*g1[0] + g2[0] + g3[0], w_pde*g1[1] + g2[1] + g3[1]))
    net.update(final_grads, lr)
    
    if epoch % 1000 == 0:
        loss = np.mean(res**2) + np.mean((u_in-target_in)**2) + np.mean(u_walls**2)
        print(f"Iter {epoch} | Loss: {loss:.5f}")

# --- VISUALIZATION ---
# (Same visualization code as previous response)
def get_exact(x, y, N):
    val = np.zeros_like(x)
    for n in range(1, N + 1):
        numerator = np.sinh(n * np.pi * (1 - x))
        denominator = np.sinh(n * np.pi)
        term = (numerator / denominator) * np.sin(n * np.pi * y)
        val += term
    return val

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
y_vals = np.linspace(0, 1, 100)
x_slices = [0.1, 0.25, 0.5, 0.75]

for ax, x_val in zip(axs.flat, x_slices):
    X_plot = np.vstack([np.ones((1,100))*x_val, y_vals.reshape(1,100)])
    u_pred, _, _, _, _ = net.forward(X_plot)
    u_exact = get_exact(X_plot[0], X_plot[1], N_modes)
    
    ax.plot(y_vals, u_exact.flatten(), 'k--', linewidth=2, label='Exact')
    ax.plot(y_vals, u_pred.flatten(), 'r-', linewidth=2, label='SIREN')
    ax.set_title(f"x = {x_val}")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()