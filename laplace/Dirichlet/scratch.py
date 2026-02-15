import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
N_modes = 5          
epochs = 15000       # Increased slightly for better convergence
batch_size = 500     
lr = 0.001           # Lowered slightly for stability

w_pde = 1.0
w_bc = 10.0            

# --- ACTIVATION FUNCTIONS & DERIVATIVES ---
def tanh(x): 
    return np.tanh(x)

def d_tanh(x): 
    return 1 - np.tanh(x)**2

def dd_tanh(x): 
    t = np.tanh(x)
    return -2 * t * (1 - t**2)

def d3_tanh(x): 
    # derivative of -2t(1-t^2) -> -2(1-3t^2)(1-t^2)
    t = np.tanh(x)
    return -2 * (1 - t**2) * (1 - 3*t**2)

# --- LINEAR LAYER (Forward Mode AD capable) ---
class Linear:
    def __init__(self, in_f, out_f):
        # Xavier Initialization
        self.W = np.random.randn(out_f, in_f) * np.sqrt(2 / (in_f + out_f))
        self.b = np.zeros((out_f, 1))
        
        # Adam Optimizer States
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self.t = 0

    def forward(self, h, h_x, h_y, h_xx, h_yy):
        self.h, self.h_x, self.h_y = h, h_x, h_y
        self.h_xx, self.h_yy = h_xx, h_yy
        
        # Chain rule for linear transformation z = Wh + b
        # z_x = W * h_x, z_xx = W * h_xx, etc.
        return (self.W@h + self.b), (self.W@h_x), (self.W@h_y), (self.W@h_xx), (self.W@h_yy)

    def backward(self, d_z, d_zx, d_zy, d_zxx, d_zyy):
        # Gradients of Weights sum contributions from value, 1st deriv, and 2nd deriv paths
        dw = (d_z @ self.h.T) + (d_zx @ self.h_x.T) + (d_zy @ self.h_y.T) + \
             (d_zxx @ self.h_xx.T) + (d_zyy @ self.h_yy.T)
        
        db = np.sum(d_z, axis=1, keepdims=True) # Only d_z contributes to bias (derivatives of bias are 0)
        
        # Gradients wrt input (h)
        d_h = self.W.T @ d_z
        d_hx, d_hy = self.W.T @ d_zx, self.W.T @ d_zy
        d_hxx, d_hyy = self.W.T @ d_zxx, self.W.T @ d_zyy
        
        return (dw, db), (d_h, d_hx, d_hy, d_hxx, d_hyy)

    def step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        dw, db = grads
        self.t += 1
        
        # Update W
        self.mW = beta1 * self.mW + (1 - beta1) * dw
        self.vW = beta2 * self.vW + (1 - beta2) * (dw**2)
        m_hat = self.mW / (1 - beta1**self.t)
        v_hat = self.vW / (1 - beta2**self.t)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Update b
        self.mb = beta1 * self.mb + (1 - beta1) * db
        self.vb = beta2 * self.vb + (1 - beta2) * (db**2)
        m_hat_b = self.mb / (1 - beta1**self.t)
        v_hat_b = self.vb / (1 - beta2**self.t)
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

# --- TANH LAYER (With 2nd Order Chain Rule) ---
class Tanh:
    def forward(self, z, z_x, z_y, z_xx, z_yy):
        self.z, self.z_x, self.z_y, self.z_xx, self.z_yy = z, z_x, z_y, z_xx, z_yy
        
        # Precompute derivatives of activation function
        self.s = tanh(z)
        self.ds = d_tanh(z)
        self.dds = dd_tanh(z)
        self.ddds = d3_tanh(z) # Needed for backward pass of 2nd derivatives
        
        self.a = self.s
        self.a_x = self.ds * z_x
        self.a_y = self.ds * z_y
        
        # Product rule: (f(g(x)))'' = f''(g) * (g')^2 + f'(g) * g''
        self.a_xx = self.dds * (z_x**2) + self.ds * z_xx
        self.a_yy = self.dds * (z_y**2) + self.ds * z_yy
        
        return self.a, self.a_x, self.a_y, self.a_xx, self.a_yy

    def backward(self, d_a, d_ax, d_ay, d_axx, d_ayy):
        # Backprop through the computation graph defined in forward
        
        # 1. Gradient wrt z (Accumulate from all paths)
        # Path from a: d_a * ds
        # Path from a_x: d_ax * dds * z_x
        # Path from a_xx: d_axx * (ddds * z_x^2 + dds * z_xx)  <-- FIX WAS HERE
        d_z = d_a * self.ds + \
              d_ax * (self.dds * self.z_x) + \
              d_ay * (self.dds * self.z_y) + \
              d_axx * (self.ddds * (self.z_x**2) + self.dds * self.z_xx) + \
              d_ayy * (self.ddds * (self.z_y**2) + self.dds * self.z_yy)
              
        # 2. Gradient wrt z_x
        d_zx = d_ax * self.ds + d_axx * (2 * self.dds * self.z_x)
        
        # 3. Gradient wrt z_y
        d_zy = d_ay * self.ds + d_ayy * (2 * self.dds * self.z_y)
        
        # 4. Gradient wrt z_xx and z_yy
        d_zxx = d_axx * self.ds
        d_zyy = d_ayy * self.ds
        
        return d_z, d_zx, d_zy, d_zxx, d_zyy

# --- PINN NETWORK ---
class PINN:
    def __init__(self, layers_cfg):
        self.layers = [Linear(layers_cfg[i], layers_cfg[i+1]) for i in range(len(layers_cfg)-1)]
        self.activations = [Tanh() for _ in range(len(layers_cfg)-2)]
     
    def forward(self, x):
        # Initial derivatives of input (Identity wrt itself)
        h = x
        h_x = np.zeros_like(x); h_x[0,:]=1 
        h_y = np.zeros_like(x); h_y[1,:]=1
        h_xx = np.zeros_like(x)
        h_yy = np.zeros_like(x)
        
        for i, layer in enumerate(self.layers):
            h, h_x, h_y, h_xx, h_yy = layer.forward(h, h_x, h_y, h_xx, h_yy)
            if i < len(self.activations):
                h, h_x, h_y, h_xx, h_yy = self.activations[i].forward(h, h_x, h_y, h_xx, h_yy)
        return h, h_x, h_y, h_xx, h_yy

    def backward(self, d_u, d_ux, d_uy, d_uxx, d_uyy):
        grads = (d_u, d_ux, d_uy, d_uxx, d_uyy)
        l_grads = []
        
        # Iterate backwards
        for i in reversed(range(len(self.layers))):
            gw, gi = self.layers[i].backward(*grads)
            l_grads.append(gw)
            grads = gi
            if i > 0: # Backprop through activation if not at input
                grads = self.activations[i-1].backward(*grads)
                
        return l_grads[::-1]

    def update(self, l_grads, lr):
        for i, layer in enumerate(self.layers): layer.step(l_grads[i], lr)

# --- TRAINING SETUP ---
net = PINN([2, 50, 50, 50, 1]) # Reduced size slightly for speed/stability

print(f"Training Scratch PINN (Exact Sinh Solution) for N={N_modes}...")

loss_history = []

for epoch in range(epochs + 1):
    
    # 1. PDE Residuals (Collocation Points)
    X_c = np.random.rand(2, 1000)
    u, ux, uy, uxx, uyy = net.forward(X_c)
    
    res = uxx + uyy # Laplace Eq: u_xx + u_yy = 0
    d_res = (2.0/1000) * res # Derivative of MSE wrt res
    
    # Backward PDE: We act as if 'res' is the output we want to minimize
    # We pass d_res into the slots for d_uxx and d_uyy
    gf = net.backward(np.zeros_like(u), np.zeros_like(ux), np.zeros_like(uy), d_res, d_res)

    # 2. Inlet BC (x=0, varying y)
    y_in = np.random.rand(1, 100)
    X_in = np.vstack([np.zeros((1,100)), y_in])
    u_in, _, _, _, _ = net.forward(X_in)
    
    target_in = sum(np.sin(n * np.pi * y_in) for n in range(1, N_modes + 1))
    d_in = (2.0/100) * w_bc * (u_in - target_in)
    gin = net.backward(d_in, np.zeros_like(u_in), np.zeros_like(u_in), np.zeros_like(u_in), np.zeros_like(u_in))

    # 3. Walls/Outlet BC (u=0)
    r = np.random.rand(1, 100); o = np.ones((1, 100)); z = np.zeros((1, 100))
    X_outlet = np.vstack([o, r]) # x=1
    X_bot    = np.vstack([r, z]) # y=0
    X_top    = np.vstack([r, o]) # y=1
    X_walls  = np.hstack([X_outlet, X_bot, X_top])
    
    u_walls, _, _, _, _ = net.forward(X_walls)
    target_walls = np.zeros_like(u_walls)
    d_walls = (2.0/300) * w_bc * (u_walls - target_walls)
    gwalls = net.backward(d_walls, np.zeros_like(u_walls), np.zeros_like(u_walls), np.zeros_like(u_walls), np.zeros_like(u_walls))

    # 4. Accumulate Gradients
    final_grads = []
    for g1, g2, g3 in zip(gf, gin, gwalls):
        dw = w_pde*g1[0] + g2[0] + g3[0] # w_bc is already multiplied in d_in/d_walls
        db = w_pde*g1[1] + g2[1] + g3[1]
        final_grads.append((dw, db))
    
    net.update(final_grads, lr)
    
    loss_val = np.mean(res**2) + np.mean((u_in-target_in)**2) + np.mean((u_walls-target_walls)**2)
    loss_history.append(loss_val)

    if epoch % 1000 == 0: 
        print(f"Iter {epoch} | Total Loss: {loss_val:.5f} | PDE Res: {np.mean(res**2):.5f}")

# --- VISUALIZATION ---
def get_exact(x, y, N):
    val = np.zeros_like(x)
    for n in range(1, N + 1):
        if n % 2 == 0: continue # Optional optimization: depending on BCs, even modes might be 0, but safe to keep
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
    ax.plot(y_vals, u_pred.flatten(), 'r-', linewidth=2, label='PINN')
    ax.set_title(f"Cross section at x = {x_val}")
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle(f"Laplace Eq Solution (Forward Mode AD from Scratch)\nN_modes={N_modes}, Epochs={epochs}")
plt.tight_layout()
plt.show()