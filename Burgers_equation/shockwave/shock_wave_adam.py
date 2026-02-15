""" import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
nu = 0.0001           
epochs = 5000       
batch_size = 3000    
lr = 1e-3             # Adam generally works well with 1e-3
beta1 = 0.9           # Adam parameter
beta2 = 0.999         # Adam parameter
epsilon = 1e-8        # Adam parameter (stability)

w_pde = 1.0
w_ic = 10.0
w_bc = 10.0

# --- Exact Solution & Helpers ---
def get_exact_solution(x, y, t):
    arg = (x + y - t) / (4 * nu)
    wave = np.tanh(arg)
    u = 0.5 - 0.5 * wave
    v = 0.5 - 0.5 * wave
    return u, v

def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x): t = np.tanh(x); return -2 * t * (1 - t**2)
def ddd_tanh(x): t = np.tanh(x); return (-2 + 6*t**2) * (1 - t**2)

class Linear:
    def __init__(self, in_f, out_f):
        # Xavier Initialization
        self.W = np.random.randn(out_f, in_f) * np.sqrt(1/in_f)
        self.b = np.zeros((out_f, 1))
        
        # Adam Parameters: First moment (m) and Second moment (v)
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

    def forward(self, h, h_x, h_y, h_t, h_xx, h_yy):
        self.h, self.h_x, self.h_y, self.h_t = h, h_x, h_y, h_t
        self.h_xx, self.h_yy = h_xx, h_yy
        
        z = self.W @ h + self.b
        return (z, 
                self.W @ h_x, self.W @ h_y, self.W @ h_t, 
                self.W @ h_xx, self.W @ h_yy)

    def backward(self, d_z, d_zx, d_zy, d_zt, d_zxx, d_zyy):
        dw = (d_z @ self.h.T) + \
             (d_zx @ self.h_x.T) + (d_zy @ self.h_y.T) + (d_zt @ self.h_t.T) + \
             (d_zxx @ self.h_xx.T) + (d_zyy @ self.h_yy.T)
             
        db = np.sum(d_z, axis=1, keepdims=True)
        d_prev = (self.W.T@d_z, 
                  self.W.T@d_zx, self.W.T@d_zy, self.W.T@d_zt, 
                  self.W.T@d_zxx, self.W.T@d_zyy)
        return (dw, db), d_prev

    def step(self, grads, lr, beta1, beta2, epsilon, t_step):
        # Unpack gradients
        g_W = grads[0]
        g_b = grads[1]

        # --- Adam Update for Weights (W) ---
        self.m_W = beta1 * self.m_W + (1 - beta1) * g_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * (g_W ** 2)
        
        # Bias correction
        m_W_hat = self.m_W / (1 - beta1 ** t_step)
        v_W_hat = self.v_W / (1 - beta2 ** t_step)
        
        # Update
        self.W -= lr * m_W_hat / (np.sqrt(v_W_hat) + epsilon)

        # --- Adam Update for Biases (b) ---
        self.m_b = beta1 * self.m_b + (1 - beta1) * g_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (g_b ** 2)
        
        # Bias correction
        m_b_hat = self.m_b / (1 - beta1 ** t_step)
        v_b_hat = self.v_b / (1 - beta2 ** t_step)
        
        # Update
        self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

class Tanh:
    def forward(self, z, z_x, z_y, z_t, z_xx, z_yy):
        self.z, self.z_x, self.z_y, self.z_t = z, z_x, z_y, z_t
        self.z_xx, self.z_yy = z_xx, z_yy
        
        self.s = tanh(z)
        self.ds = d_tanh(z)
        self.dds = dd_tanh(z)
        
        a = self.s
        a_x = self.ds * z_x
        a_y = self.ds * z_y
        a_t = self.ds * z_t
        a_xx = self.dds * (z_x**2) + self.ds * z_xx
        a_yy = self.dds * (z_y**2) + self.ds * z_yy
        
        return a, a_x, a_y, a_t, a_xx, a_yy

    def backward(self, d_a, d_ax, d_ay, d_at, d_axx, d_ayy):
        ddds = ddd_tanh(self.z)
        d_z = (d_a * self.ds) + \
              (d_ax * self.dds * self.z_x) + \
              (d_ay * self.dds * self.z_y) + \
              (d_at * self.dds * self.z_t) + \
              (d_axx * (ddds * self.z_x**2 + self.dds * self.z_xx)) + \
              (d_ayy * (ddds * self.z_y**2 + self.dds * self.z_yy))
              
        d_zx = (d_ax * self.ds) + (d_axx * 2 * self.dds * self.z_x)
        d_zy = (d_ay * self.ds) + (d_ayy * 2 * self.dds * self.z_y)
        d_zt = d_at * self.ds
        d_zxx = d_axx * self.ds
        d_zyy = d_ayy * self.ds
        
        return d_z, d_zx, d_zy, d_zt, d_zxx, d_zyy

class PINN:
    def __init__(self, layers_cfg):
        self.layers = []
        for i in range(len(layers_cfg) - 1):
            self.layers.append(Linear(layers_cfg[i], layers_cfg[i+1]))
            if i < len(layers_cfg) - 2: self.layers.append(Tanh())

    def forward(self, x_in):
        h = x_in
        h_x = np.zeros_like(x_in); h_x[0,:] = 1.0 
        h_y = np.zeros_like(x_in); h_y[1,:] = 1.0
        h_t = np.zeros_like(x_in); h_t[2,:] = 1.0
        h_xx = np.zeros_like(x_in)
        h_yy = np.zeros_like(x_in)
        
        for layer in self.layers:
            h, h_x, h_y, h_t, h_xx, h_yy = layer.forward(h, h_x, h_y, h_t, h_xx, h_yy)
        return h, h_x, h_y, h_t, h_xx, h_yy

    def backward(self, d_u, d_ux, d_uy, d_ut, d_uxx, d_uyy):
        grads = (d_u, d_ux, d_uy, d_ut, d_uxx, d_uyy)
        l_grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, Tanh): 
                grads = layer.backward(*grads)
            else: 
                w_g, i_g = layer.backward(*grads)
                l_grads.append(w_g)
                grads = i_g
        return l_grads[::-1]

    def update(self, l_grads, lr, beta1, beta2, epsilon, t_step):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(l_grads[idx], lr, beta1, beta2, epsilon, t_step)
                idx+=1

net = PINN([3, 40, 50, 40, 2])

print(f"Starting 2D PINN training with Adam (nu={nu})...")

global_step = 0 # Step counter for Adam bias correction

for epoch in range(epochs + 1):
    global_step += 1
    
    # 1. Physics Loss (PDE Residuals)
    x_c = np.random.uniform(-1, 1, (1, batch_size))
    y_c = np.random.uniform(-1, 1, (1, batch_size))
    t_c = np.random.uniform(0, 0.5, (1, batch_size))
    X_c = np.vstack([x_c, y_c, t_c])
    
    uv, uv_x, uv_y, uv_t, uv_xx, uv_yy = net.forward(X_c)
    u, v = uv[0:1], uv[1:2]
    ux, vx = uv_x[0:1], uv_x[1:2]
    uy, vy = uv_y[0:1], uv_y[1:2]
    ut, vt = uv_t[0:1], uv_t[1:2]
    uxx, vxx = uv_xx[0:1], uv_xx[1:2]
    uyy, vyy = uv_yy[0:1], uv_yy[1:2]
    
    f_u = ut + u*ux + v*uy - nu*(uxx + uyy)
    f_v = vt + u*vx + v*vy - nu*(vxx + vyy)
    
    loss_f = np.mean(f_u**2 + f_v**2)
    
    dfu = (2/batch_size)*f_u
    dfv = (2/batch_size)*f_v
    
    grad_u = dfu * ux + dfv * vx
    grad_v = dfv * vy + dfu * uy
    
    grads_f = net.backward(
        np.vstack([grad_u, grad_v]),         
        np.vstack([dfu*u, dfv*u]),            
        np.vstack([dfu*v, dfv*v]),           
        np.vstack([dfu, dfv]),             
        np.vstack([-nu*dfu, -nu*dfv]),        
        np.vstack([-nu*dfu, -nu*dfv])        
    )
    
    # 2. Initial Condition Loss
    x_i = np.random.uniform(-1, 1, (1, 800))
    y_i = np.random.uniform(-1, 1, (1, 800))
    t_i = np.zeros_like(x_i)
    X_i = np.vstack([x_i, y_i, t_i])
    
    uv_i, _, _, _, _, _ = net.forward(X_i)
    u_true_i, v_true_i = get_exact_solution(x_i, y_i, t_i)
    
    diff_u_i = uv_i[0] - u_true_i
    diff_v_i = uv_i[1] - v_true_i
    loss_i = np.mean(diff_u_i**2 + diff_v_i**2)
    
    grads_i = net.backward(
        (2/800)*np.vstack([diff_u_i, diff_v_i]),
        np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i),
        np.zeros_like(uv_i), np.zeros_like(uv_i)
    )
    
    # 3. Boundary Condition Loss
    nb = 700 
    yb_v = np.random.uniform(-1, 1, (1, nb))
    xb_h = np.random.uniform(-1, 1, (1, nb))
    
    xb = np.hstack([-np.ones((1,nb)), np.ones((1,nb)), xb_h, xb_h])
    yb = np.hstack([yb_v, yb_v, -np.ones((1,nb)), np.ones((1,nb))])
    tb = np.random.uniform(0, 0.5, (1, nb*4))
    X_b = np.vstack([xb, yb, tb])
    
    uv_b, _, _, _, _, _ = net.forward(X_b)
    u_true_b, v_true_b = get_exact_solution(xb, yb, tb)
    
    diff_u_b = uv_b[0] - u_true_b
    diff_v_b = uv_b[1] - v_true_b
    loss_b = np.mean(diff_u_b**2 + diff_v_b**2)
    
    grads_b = net.backward(
        (2/(nb*4))*np.vstack([diff_u_b, diff_v_b]),
        np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b),
        np.zeros_like(uv_b), np.zeros_like(uv_b)
    )
    
    # 4. Aggregate Gradients and Update
    final_grads = []
    for g1, g2, g3 in zip(grads_f, grads_i, grads_b):
        dw = w_pde*g1[0] + w_ic*g2[0] + w_bc*g3[0]
        db = w_pde*g1[1] + w_ic*g2[1] + w_bc*g3[1]
        final_grads.append((dw, db))
        
    net.update(final_grads, lr, beta1, beta2, epsilon, global_step)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Total Loss: {loss_f+loss_i+loss_b:.5f} | PDE: {loss_f:.5f}")
        
        
# --- Graphs ---

print(f"\nTraining completed. Generating results...")
N = 60
x_v = np.linspace(-1, 1, N)
y_v = np.linspace(-1, 1, N)
Xg, Yg = np.meshgrid(x_v, y_v)
t_val = 0.25 

X_flat = np.vstack([
    Xg.flatten(),
    Yg.flatten(),
    np.full_like(Xg.flatten(), t_val)
])

uv_pred, _, _, _, _, _ = net.forward(X_flat)
u_pred = uv_pred[0].reshape(N, N)

u_true, _ = get_exact_solution(Xg, Yg, t_val)

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(Xg, Yg, u_pred, cmap='viridis')
ax1.set_title(f"Predicted u(x,y,t={t_val})")

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(Xg, Yg, u_true, cmap='viridis')
ax2.set_title(f"Exact u(x,y,t={t_val})")

ax3 = fig.add_subplot(2, 2, 3)
cf3 = ax3.contourf(Xg, Yg, u_pred, 20, cmap='viridis')
ax3.set_title("Predicted u (2D)")
plt.colorbar(cf3, ax=ax3)

ax4 = fig.add_subplot(2, 2, 4)
cf4 = ax4.contourf(Xg, Yg, u_true, 20, cmap='viridis')
ax4.set_title("Exact u (2D)")
plt.colorbar(cf4, ax=ax4)

plt.suptitle(f"2D Burgers Equation (Adam Optimizer, t={t_val})", fontsize=16)
N_slice = 100
x_slice = np.linspace(-1, 1, N_slice)
y_slice = np.zeros_like(x_slice)

times = [0.0, 0.5]

plt.figure(figsize=(12, 8))

for i, t_val in enumerate(times):
    t_slice = np.full_like(x_slice, t_val)
    X_slice = np.vstack([x_slice, y_slice, t_slice])

    uv_pred_slice, _, _, _, _, _ = net.forward(X_slice)
    u_pred_line = uv_pred_slice[0].flatten()

    u_true_line, _ = get_exact_solution(x_slice, y_slice, t_slice)

    plt.subplot(2, 1, i + 1)
    plt.plot(x_slice, u_true_line, 'k-', linewidth=2.5, label='Exact')
    plt.plot(x_slice, u_pred_line, 'r--', linewidth=2.5, label='PINN')
    plt.title(f"u(x, y=0, t={t_val})")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True, alpha=0.5)
    plt.legend()

plt.suptitle("1D Cross-Sections (Adam Optimizer)", fontsize=16)
plt.tight_layout()
plt.show() """

# RAR
""" 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Configuration & Hyperparameters ---
nu = 0.0001           # Viscosity (Low nu = steep shock)
epochs = 8000         # Increased epochs for RAR to have time to work
batch_size = 3000     
lr = 0.001            # Adam default learning rate
beta1 = 0.9           # Adam beta1
beta2 = 0.999         # Adam beta2
epsilon = 1e-8        # Adam epsilon

# Loss Weights
w_pde = 1.0
w_ic = 10.0
w_bc = 10.0

# --- 2. Exact Solution & Activation Functions ---
def get_exact_solution(x, y, t):
    # Exact solution for 2D Burgers (moving wave)
    arg = (x + y - t) / (4 * nu)
    wave = np.tanh(arg)
    u = 0.5 - 0.5 * wave
    v = 0.5 - 0.5 * wave
    return u, v

def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x): t = np.tanh(x); return -2 * t * (1 - t**2)
def ddd_tanh(x): t = np.tanh(x); return (-2 + 6*t**2) * (1 - t**2)

# --- 3. Neural Network Layers (with Adam) ---
class Linear:
    def __init__(self, in_f, out_f):
        # Xavier Initialization
        self.W = np.random.randn(out_f, in_f) * np.sqrt(1/in_f)
        self.b = np.zeros((out_f, 1))
        
        # Adam Moments Initialization
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

    def forward(self, h, h_x, h_y, h_t, h_xx, h_yy):
        self.h, self.h_x, self.h_y, self.h_t = h, h_x, h_y, h_t
        self.h_xx, self.h_yy = h_xx, h_yy
        
        z = self.W @ h + self.b
        return (z, 
                self.W @ h_x, self.W @ h_y, self.W @ h_t, 
                self.W @ h_xx, self.W @ h_yy)

    def backward(self, d_z, d_zx, d_zy, d_zt, d_zxx, d_zyy):
        dw = (d_z @ self.h.T) + \
             (d_zx @ self.h_x.T) + (d_zy @ self.h_y.T) + (d_zt @ self.h_t.T) + \
             (d_zxx @ self.h_xx.T) + (d_zyy @ self.h_yy.T)
             
        db = np.sum(d_z, axis=1, keepdims=True)
        d_prev = (self.W.T@d_z, 
                  self.W.T@d_zx, self.W.T@d_zy, self.W.T@d_zt, 
                  self.W.T@d_zxx, self.W.T@d_zyy)
        return (dw, db), d_prev

    def step(self, grads, lr, beta1, beta2, epsilon, t_step):
        # Unpack gradients
        g_W = grads[0]
        g_b = grads[1]

        # --- Adam Update for Weights (W) ---
        self.m_W = beta1 * self.m_W + (1 - beta1) * g_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * (g_W ** 2)
        
        # Bias correction
        m_W_hat = self.m_W / (1 - beta1 ** t_step)
        v_W_hat = self.v_W / (1 - beta2 ** t_step)
        
        # Update
        self.W -= lr * m_W_hat / (np.sqrt(v_W_hat) + epsilon)

        # --- Adam Update for Biases (b) ---
        self.m_b = beta1 * self.m_b + (1 - beta1) * g_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (g_b ** 2)
        
        # Bias correction
        m_b_hat = self.m_b / (1 - beta1 ** t_step)
        v_b_hat = self.v_b / (1 - beta2 ** t_step)
        
        # Update
        self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

class Tanh:
    def forward(self, z, z_x, z_y, z_t, z_xx, z_yy):
        self.z, self.z_x, self.z_y, self.z_t = z, z_x, z_y, z_t
        self.z_xx, self.z_yy = z_xx, z_yy
        
        self.s = tanh(z)
        self.ds = d_tanh(z)
        self.dds = dd_tanh(z)
        
        a = self.s
        a_x = self.ds * z_x
        a_y = self.ds * z_y
        a_t = self.ds * z_t
        a_xx = self.dds * (z_x**2) + self.ds * z_xx
        a_yy = self.dds * (z_y**2) + self.ds * z_yy
        
        return a, a_x, a_y, a_t, a_xx, a_yy

    def backward(self, d_a, d_ax, d_ay, d_at, d_axx, d_ayy):
        ddds = ddd_tanh(self.z)
        d_z = (d_a * self.ds) + \
              (d_ax * self.dds * self.z_x) + \
              (d_ay * self.dds * self.z_y) + \
              (d_at * self.dds * self.z_t) + \
              (d_axx * (ddds * self.z_x**2 + self.dds * self.z_xx)) + \
              (d_ayy * (ddds * self.z_y**2 + self.dds * self.z_yy))
              
        d_zx = (d_ax * self.ds) + (d_axx * 2 * self.dds * self.z_x)
        d_zy = (d_ay * self.ds) + (d_ayy * 2 * self.dds * self.z_y)
        d_zt = d_at * self.ds
        d_zxx = d_axx * self.ds
        d_zyy = d_ayy * self.ds
        
        return d_z, d_zx, d_zy, d_zt, d_zxx, d_zyy

class PINN:
    def __init__(self, layers_cfg):
        self.layers = []
        for i in range(len(layers_cfg) - 1):
            self.layers.append(Linear(layers_cfg[i], layers_cfg[i+1]))
            if i < len(layers_cfg) - 2: self.layers.append(Tanh())

    def forward(self, x_in):
        h = x_in
        h_x = np.zeros_like(x_in); h_x[0,:] = 1.0 
        h_y = np.zeros_like(x_in); h_y[1,:] = 1.0
        h_t = np.zeros_like(x_in); h_t[2,:] = 1.0
        h_xx = np.zeros_like(x_in)
        h_yy = np.zeros_like(x_in)
        
        for layer in self.layers:
            h, h_x, h_y, h_t, h_xx, h_yy = layer.forward(h, h_x, h_y, h_t, h_xx, h_yy)
        return h, h_x, h_y, h_t, h_xx, h_yy

    def backward(self, d_u, d_ux, d_uy, d_ut, d_uxx, d_uyy):
        grads = (d_u, d_ux, d_uy, d_ut, d_uxx, d_uyy)
        l_grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, Tanh): 
                grads = layer.backward(*grads)
            else: 
                w_g, i_g = layer.backward(*grads)
                l_grads.append(w_g)
                grads = i_g
        return l_grads[::-1]

    def update(self, l_grads, lr, beta1, beta2, epsilon, t_step):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(l_grads[idx], lr, beta1, beta2, epsilon, t_step)
                idx+=1

# --- 4. Main Training Loop with RAR ---

# Deeper network for high-frequency shock capture
net = PINN([3, 50, 60, 60, 60, 2]) 

print(f"Starting 2D PINN training (Adam + RAR, nu={nu})...")

global_step = 0
X_adaptive = np.empty((3, 0)) # Buffer to store "hard" points

for epoch in range(epochs + 1):
    global_step += 1
    
    # --- Step 1: Dynamic Batch Creation ---
    # A. Standard random points
    x_c = np.random.uniform(-1, 1, (1, batch_size))
    y_c = np.random.uniform(-1, 1, (1, batch_size))
    t_c = np.random.uniform(0, 0.5, (1, batch_size))
    X_random = np.vstack([x_c, y_c, t_c])
    
    # B. Combine with Adaptive points (if any exist)
    if X_adaptive.shape[1] > 0:
        # If we have too many adaptive points, pick a random subset (e.g., 2000)
        n_adapt = X_adaptive.shape[1]
        n_sample = min(n_adapt, 2000)
        idx_mix = np.random.choice(n_adapt, n_sample, replace=False)
        X_c = np.hstack([X_random, X_adaptive[:, idx_mix]])
    else:
        X_c = X_random

    # --- Step 2: PDE Residual Loss ---
    uv, uv_x, uv_y, uv_t, uv_xx, uv_yy = net.forward(X_c)
    u, v = uv[0:1], uv[1:2]
    ux, vx = uv_x[0:1], uv_x[1:2]
    uy, vy = uv_y[0:1], uv_y[1:2]
    ut, vt = uv_t[0:1], uv_t[1:2]
    uxx, vxx = uv_xx[0:1], uv_xx[1:2]
    uyy, vyy = uv_yy[0:1], uv_yy[1:2]
    
    # Burgers residuals
    f_u = ut + u*ux + v*uy - nu*(uxx + uyy)
    f_v = vt + u*vx + v*vy - nu*(vxx + vyy)
    
    loss_f = np.mean(f_u**2 + f_v**2)
    
    # PDE Gradients
    curr_batch_size = X_c.shape[1]
    dfu = (2/curr_batch_size)*f_u
    dfv = (2/curr_batch_size)*f_v
    
    grad_u = dfu * ux + dfv * vx
    grad_v = dfv * vy + dfu * uy
    
    grads_f = net.backward(
        np.vstack([grad_u, grad_v]),         
        np.vstack([dfu*u, dfv*u]),            
        np.vstack([dfu*v, dfv*v]),           
        np.vstack([dfu, dfv]),             
        np.vstack([-nu*dfu, -nu*dfv]),        
        np.vstack([-nu*dfu, -nu*dfv])        
    )

    # --- Step 3: Initial Condition Loss ---
    x_i = np.random.uniform(-1, 1, (1, 800))
    y_i = np.random.uniform(-1, 1, (1, 800))
    t_i = np.zeros_like(x_i)
    X_i = np.vstack([x_i, y_i, t_i])
    
    uv_i, _, _, _, _, _ = net.forward(X_i)
    u_true_i, v_true_i = get_exact_solution(x_i, y_i, t_i)
    
    diff_u_i = uv_i[0] - u_true_i
    diff_v_i = uv_i[1] - v_true_i
    loss_i = np.mean(diff_u_i**2 + diff_v_i**2)
    
    grads_i = net.backward(
        (2/800)*np.vstack([diff_u_i, diff_v_i]),
        np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i),
        np.zeros_like(uv_i), np.zeros_like(uv_i)
    )

    # --- Step 4: Boundary Condition Loss ---
    nb = 700 
    yb_v = np.random.uniform(-1, 1, (1, nb))
    xb_h = np.random.uniform(-1, 1, (1, nb))
    
    xb = np.hstack([-np.ones((1,nb)), np.ones((1,nb)), xb_h, xb_h])
    yb = np.hstack([yb_v, yb_v, -np.ones((1,nb)), np.ones((1,nb))])
    tb = np.random.uniform(0, 0.5, (1, nb*4))
    X_b = np.vstack([xb, yb, tb])
    
    uv_b, _, _, _, _, _ = net.forward(X_b)
    u_true_b, v_true_b = get_exact_solution(xb, yb, tb)
    
    diff_u_b = uv_b[0] - u_true_b
    diff_v_b = uv_b[1] - v_true_b
    loss_b = np.mean(diff_u_b**2 + diff_v_b**2)
    
    grads_b = net.backward(
        (2/(nb*4))*np.vstack([diff_u_b, diff_v_b]),
        np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b),
        np.zeros_like(uv_b), np.zeros_like(uv_b)
    )

    # --- Step 5: Update Weights (Adam) ---
    final_grads = []
    for g1, g2, g3 in zip(grads_f, grads_i, grads_b):
        dw = w_pde*g1[0] + w_ic*g2[0] + w_bc*g3[0]
        db = w_pde*g1[1] + w_ic*g2[1] + w_bc*g3[1]
        final_grads.append((dw, db))
        
    net.update(final_grads, lr, beta1, beta2, epsilon, global_step)

    # --- Step 6: RAR (Residual-based Adaptive Refinement) ---
    # Every 200 epochs, scan for points where the PDE is failing (the shock)
    if epoch % 200 == 0 and epoch > 0 and epoch < (epochs - 200):
        # 1. Create a dense grid to scan
        N_dense = 5000
        x_d = np.random.uniform(-1, 1, (1, N_dense))
        y_d = np.random.uniform(-1, 1, (1, N_dense))
        t_d = np.random.uniform(0, 0.5, (1, N_dense))
        X_d = np.vstack([x_d, y_d, t_d])
        
        # 2. Check PDE residual
        uv_d, uv_xd, uv_yd, uv_td, uv_xxd, uv_yyd = net.forward(X_d)
        u_d, v_d = uv_d[0:1], uv_d[1:2]
        
        # Calculate residual magnitude
        f_u_d = uv_td[0:1] + u_d*uv_xd[0:1] + v_d*uv_yd[0:1] - nu*(uv_xxd[0:1] + uv_yyd[0:1])
        f_v_d = uv_td[1:2] + u_d*uv_xd[1:2] + v_d*uv_yd[1:2] - nu*(uv_xxd[1:2] + uv_yyd[1:2])
        err_d = f_u_d**2 + f_v_d**2
        
        # 3. Pick the worst points
        # Sort indices by error (ascending), take the last 500 (highest error)
        indices = np.argsort(err_d.flatten())[-500:] 
        X_new_hard = X_d[:, indices]
        
        # 4. Add to adaptive buffer
        X_adaptive = np.hstack([X_adaptive, X_new_hard])
        print(f"  [RAR] Epoch {epoch}: Added {X_new_hard.shape[1]} shock points. Buffer size: {X_adaptive.shape[1]}")

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss_f+loss_i+loss_b:.5f} | PDE: {loss_f:.5f} | IC: {loss_i:.5f}")

# --- 5. Visualization ---
print(f"\nTraining completed. Generating results...")
N = 60
x_v = np.linspace(-1, 1, N)
y_v = np.linspace(-1, 1, N)
Xg, Yg = np.meshgrid(x_v, y_v)
t_val = 0.25 

X_flat = np.vstack([
    Xg.flatten(),
    Yg.flatten(),
    np.full_like(Xg.flatten(), t_val)
])

uv_pred, _, _, _, _, _ = net.forward(X_flat)
u_pred = uv_pred[0].reshape(N, N)
u_true, _ = get_exact_solution(Xg, Yg, t_val)

fig = plt.figure(figsize=(14, 10))

# 3D Surface Plots
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(Xg, Yg, u_pred, cmap='viridis')
ax1.set_title(f"Predicted u(x,y,t={t_val})")

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(Xg, Yg, u_true, cmap='viridis')
ax2.set_title(f"Exact u(x,y,t={t_val})")

# 2D Contour Plots
ax3 = fig.add_subplot(2, 2, 3)
cf3 = ax3.contourf(Xg, Yg, u_pred, 20, cmap='viridis')
ax3.set_title("Predicted u (2D)")
plt.colorbar(cf3, ax=ax3)

ax4 = fig.add_subplot(2, 2, 4)
cf4 = ax4.contourf(Xg, Yg, u_true, 20, cmap='viridis')
ax4.set_title("Exact u (2D)")
plt.colorbar(cf4, ax=ax4)

plt.suptitle(f"2D Burgers (Adam + RAR, nu={nu})", fontsize=16)

# 1D Slices (Cross-sections)
N_slice = 100
x_slice = np.linspace(-1, 1, N_slice)
y_slice = np.zeros_like(x_slice)
times = [0.0, 0.5]

plt.figure(figsize=(12, 8))
for i, t_val in enumerate(times):
    t_slice = np.full_like(x_slice, t_val)
    X_slice = np.vstack([x_slice, y_slice, t_slice])

    uv_pred_slice, _, _, _, _, _ = net.forward(X_slice)
    u_pred_line = uv_pred_slice[0].flatten()

    u_true_line, _ = get_exact_solution(x_slice, y_slice, t_slice)

    plt.subplot(2, 1, i + 1)
    plt.plot(x_slice, u_true_line, 'k-', linewidth=2.5, label='Exact')
    plt.plot(x_slice, u_pred_line, 'r--', linewidth=2.5, label='PINN (Adam+RAR)')
    plt.title(f"u(x, y=0, t={t_val})")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True, alpha=0.5)
    plt.legend()

plt.suptitle("1D Cross-Sections: Shock Capturing Performance", fontsize=16)
plt.tight_layout()
plt.show() """

# Curricullam + RAR
""" 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# We will NOT use a single fixed nu. We use a schedule.
# We start easy (0.05) and get harder (to 0.0001)
nu_schedule = [0.05, 0.01, 0.005, 0.001, 0.0001]
epochs_schedule = [2000, 2000, 2000, 3000, 5000] # More epochs for harder stages

batch_size = 3000     
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

w_pde = 1.0
w_ic = 10.0
w_bc = 10.0

# --- Helper Functions ---
def get_exact_solution(x, y, t, nu_val):
    # Note: Exact solution depends on current nu
    arg = (x + y - t) / (4 * nu_val)
    wave = np.tanh(arg)
    u = 0.5 - 0.5 * wave
    v = 0.5 - 0.5 * wave
    return u, v

def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x): t = np.tanh(x); return -2 * t * (1 - t**2)
def ddd_tanh(x): t = np.tanh(x); return (-2 + 6*t**2) * (1 - t**2)

# --- Neural Network Classes (Adam Implementation) ---
class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(out_f, in_f) * np.sqrt(1/in_f)
        self.b = np.zeros((out_f, 1))
        self.reset_adam()

    def reset_adam(self):
        # We reset optimizer momentum when we change physics (nu)
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

    def forward(self, h, h_x, h_y, h_t, h_xx, h_yy):
        self.h, self.h_x, self.h_y, self.h_t = h, h_x, h_y, h_t
        self.h_xx, self.h_yy = h_xx, h_yy
        z = self.W @ h + self.b
        return (z, self.W @ h_x, self.W @ h_y, self.W @ h_t, self.W @ h_xx, self.W @ h_yy)

    def backward(self, d_z, d_zx, d_zy, d_zt, d_zxx, d_zyy):
        dw = (d_z @ self.h.T) + (d_zx @ self.h_x.T) + (d_zy @ self.h_y.T) + \
             (d_zt @ self.h_t.T) + (d_zxx @ self.h_xx.T) + (d_zyy @ self.h_yy.T)
        db = np.sum(d_z, axis=1, keepdims=True)
        d_prev = (self.W.T@d_z, self.W.T@d_zx, self.W.T@d_zy, self.W.T@d_zt, 
                  self.W.T@d_zxx, self.W.T@d_zyy)
        return (dw, db), d_prev

    def step(self, grads, lr, beta1, beta2, epsilon, t_step):
        g_W, g_b = grads[0], grads[1]
        
        self.m_W = beta1 * self.m_W + (1 - beta1) * g_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * (g_W ** 2)
        m_W_hat = self.m_W / (1 - beta1 ** t_step)
        v_W_hat = self.v_W / (1 - beta2 ** t_step)
        self.W -= lr * m_W_hat / (np.sqrt(v_W_hat) + epsilon)

        self.m_b = beta1 * self.m_b + (1 - beta1) * g_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (g_b ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** t_step)
        v_b_hat = self.v_b / (1 - beta2 ** t_step)
        self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

class Tanh:
    def forward(self, z, z_x, z_y, z_t, z_xx, z_yy):
        self.z, self.z_x, self.z_y, self.z_t = z, z_x, z_y, z_t
        self.z_xx, self.z_yy = z_xx, z_yy
        self.s = tanh(z); self.ds = d_tanh(z); self.dds = dd_tanh(z)
        
        a = self.s
        a_x = self.ds * z_x
        a_y = self.ds * z_y
        a_t = self.ds * z_t
        a_xx = self.dds * (z_x**2) + self.ds * z_xx
        a_yy = self.dds * (z_y**2) + self.ds * z_yy
        return a, a_x, a_y, a_t, a_xx, a_yy

    def backward(self, d_a, d_ax, d_ay, d_at, d_axx, d_ayy):
        ddds = ddd_tanh(self.z)
        d_z = (d_a * self.ds) + (d_ax * self.dds * self.z_x) + (d_ay * self.dds * self.z_y) + \
              (d_at * self.dds * self.z_t) + (d_axx * (ddds * self.z_x**2 + self.dds * self.z_xx)) + \
              (d_ayy * (ddds * self.z_y**2 + self.dds * self.z_yy))
              
        d_zx = (d_ax * self.ds) + (d_axx * 2 * self.dds * self.z_x)
        d_zy = (d_ay * self.ds) + (d_ayy * 2 * self.dds * self.z_y)
        d_zt = d_at * self.ds
        d_zxx = d_axx * self.ds
        d_zyy = d_ayy * self.ds
        return d_z, d_zx, d_zy, d_zt, d_zxx, d_zyy

class PINN:
    def __init__(self, layers_cfg):
        self.layers = []
        for i in range(len(layers_cfg) - 1):
            self.layers.append(Linear(layers_cfg[i], layers_cfg[i+1]))
            if i < len(layers_cfg) - 2: self.layers.append(Tanh())

    def forward(self, x_in):
        h = x_in
        h_x = np.zeros_like(x_in); h_x[0,:] = 1.0 
        h_y = np.zeros_like(x_in); h_y[1,:] = 1.0
        h_t = np.zeros_like(x_in); h_t[2,:] = 1.0
        h_xx = np.zeros_like(x_in)
        h_yy = np.zeros_like(x_in)
        for layer in self.layers:
            h, h_x, h_y, h_t, h_xx, h_yy = layer.forward(h, h_x, h_y, h_t, h_xx, h_yy)
        return h, h_x, h_y, h_t, h_xx, h_yy

    def backward(self, d_u, d_ux, d_uy, d_ut, d_uxx, d_uyy):
        grads = (d_u, d_ux, d_uy, d_ut, d_uxx, d_uyy)
        l_grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, Tanh): 
                grads = layer.backward(*grads)
            else: 
                w_g, i_g = layer.backward(*grads)
                l_grads.append(w_g)
                grads = i_g
        return l_grads[::-1]

    def update(self, l_grads, lr, beta1, beta2, epsilon, t_step):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(l_grads[idx], lr, beta1, beta2, epsilon, t_step)
                idx+=1
    
    def reset_optimizer(self):
        for layer in self.layers:
            if isinstance(layer, Linear): layer.reset_adam()

# --- Training Loop with Curriculum Learning ---

net = PINN([3, 50, 60, 60, 60, 2]) # Deep network

print(f"Starting Curriculum Learning Training...")
print(f"Schedule: {nu_schedule}")

global_step = 0
X_adaptive = np.empty((3, 0)) # Adaptive sampling buffer

# OUTER LOOP: Iterate through viscosities (Easy -> Hard)
for stage, current_nu in enumerate(nu_schedule):
    
    # Reset optimizer momentum for the new stage (optional but often helps stability)
    net.reset_optimizer()
    stage_epochs = epochs_schedule[stage]
    print(f"\n--- STAGE {stage+1}/{len(nu_schedule)}: nu = {current_nu} ({stage_epochs} epochs) ---")
    
    for epoch in range(stage_epochs + 1):
        global_step += 1
        
        # 1. Dynamic Batching (Mix Random + Adaptive)
        x_c = np.random.uniform(-1, 1, (1, batch_size))
        y_c = np.random.uniform(-1, 1, (1, batch_size))
        t_c = np.random.uniform(0, 0.5, (1, batch_size))
        X_random = np.vstack([x_c, y_c, t_c])
        
        if X_adaptive.shape[1] > 0:
            idx_mix = np.random.choice(X_adaptive.shape[1], min(X_adaptive.shape[1], 2000), replace=False)
            X_c = np.hstack([X_random, X_adaptive[:, idx_mix]])
        else:
            X_c = X_random

        # 2. PDE Loss (Using current_nu!)
        uv, uv_x, uv_y, uv_t, uv_xx, uv_yy = net.forward(X_c)
        u, v = uv[0:1], uv[1:2]
        ux, vx = uv_x[0:1], uv_x[1:2]
        uy, vy = uv_y[0:1], uv_y[1:2]
        ut, vt = uv_t[0:1], uv_t[1:2]
        uxx, vxx = uv_xx[0:1], uv_xx[1:2]
        uyy, vyy = uv_yy[0:1], uv_yy[1:2]
        
        f_u = ut + u*ux + v*uy - current_nu*(uxx + uyy)
        f_v = vt + u*vx + v*vy - current_nu*(vxx + vyy)
        
        loss_f = np.mean(f_u**2 + f_v**2)
        
        # Gradients
        bs = X_c.shape[1]
        dfu, dfv = (2/bs)*f_u, (2/bs)*f_v
        grad_u = dfu * ux + dfv * vx
        grad_v = dfv * vy + dfu * uy
        
        grads_f = net.backward(
            np.vstack([grad_u, grad_v]), np.vstack([dfu*u, dfv*u]), np.vstack([dfu*v, dfv*v]),
            np.vstack([dfu, dfv]), np.vstack([-current_nu*dfu, -current_nu*dfv]), np.vstack([-current_nu*dfu, -current_nu*dfv])
        )

        # 3. Initial Condition Loss
        x_i = np.random.uniform(-1, 1, (1, 1000))
        y_i = np.random.uniform(-1, 1, (1, 1000))
        t_i = np.zeros_like(x_i)
        X_i = np.vstack([x_i, y_i, t_i])
        
        uv_i, _, _, _, _, _ = net.forward(X_i)
        u_true_i, v_true_i = get_exact_solution(x_i, y_i, t_i, current_nu) # Use current_nu for IC too if it varies
        
        diff_u_i, diff_v_i = uv_i[0] - u_true_i, uv_i[1] - v_true_i
        loss_i = np.mean(diff_u_i**2 + diff_v_i**2)
        grads_i = net.backward((2/1000)*np.vstack([diff_u_i, diff_v_i]), np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i))

        # 4. Boundary Condition Loss
        nb = 800 
        yb_v, xb_h = np.random.uniform(-1, 1, (1, nb)), np.random.uniform(-1, 1, (1, nb))
        xb = np.hstack([-np.ones((1,nb)), np.ones((1,nb)), xb_h, xb_h])
        yb = np.hstack([yb_v, yb_v, -np.ones((1,nb)), np.ones((1,nb))])
        tb = np.random.uniform(0, 0.5, (1, nb*4))
        X_b = np.vstack([xb, yb, tb])
        
        uv_b, _, _, _, _, _ = net.forward(X_b)
        u_true_b, v_true_b = get_exact_solution(xb, yb, tb, current_nu)
        
        diff_u_b, diff_v_b = uv_b[0] - u_true_b, uv_b[1] - v_true_b
        loss_b = np.mean(diff_u_b**2 + diff_v_b**2)
        grads_b = net.backward((2/(nb*4))*np.vstack([diff_u_b, diff_v_b]), np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b))

        # 5. Update
        final_grads = []
        for g1, g2, g3 in zip(grads_f, grads_i, grads_b):
            dw = w_pde*g1[0] + w_ic*g2[0] + w_bc*g3[0]
            db = w_pde*g1[1] + w_ic*g2[1] + w_bc*g3[1]
            final_grads.append((dw, db))
            
        net.update(final_grads, lr, beta1, beta2, epsilon, global_step)

        # 6. RAR (Adaptive Sampling) - Crucial for sharpening
        if epoch % 500 == 0 and epoch < (stage_epochs - 100):
            N_dense = 5000
            x_d = np.random.uniform(-1, 1, (1, N_dense))
            y_d = np.random.uniform(-1, 1, (1, N_dense))
            t_d = np.random.uniform(0, 0.5, (1, N_dense))
            X_d = np.vstack([x_d, y_d, t_d])
            
            uv_d, uv_xd, uv_yd, uv_td, uv_xxd, uv_yyd = net.forward(X_d)
            u_d, v_d = uv_d[0:1], uv_d[1:2]
            f_u_d = uv_td[0:1] + u_d*uv_xd[0:1] + v_d*uv_yd[0:1] - current_nu*(uv_xxd[0:1] + uv_yyd[0:1])
            f_v_d = uv_td[1:2] + u_d*uv_xd[1:2] + v_d*uv_yd[1:2] - current_nu*(uv_xxd[1:2] + uv_yyd[1:2])
            err_d = f_u_d**2 + f_v_d**2
            
            indices = np.argsort(err_d.flatten())[-500:] 
            X_new_hard = X_d[:, indices]
            X_adaptive = np.hstack([X_adaptive, X_new_hard])
            print(f"  [RAR] Added {X_new_hard.shape[1]} hard points. Total Buffer: {X_adaptive.shape[1]}")

        if epoch % 500 == 0:
            print(f"  Epoch {epoch} | Loss: {loss_f+loss_i+loss_b:.5f} | PDE: {loss_f:.5f}")

# --- Plotting Results ---
print(f"\nTraining completed. Visualizing result for FINAL nu={nu_schedule[-1]}")
N = 80
x_v = np.linspace(-1, 1, N)
y_v = np.linspace(-1, 1, N)
Xg, Yg = np.meshgrid(x_v, y_v)
t_val = 0.25 

X_flat = np.vstack([Xg.flatten(), Yg.flatten(), np.full_like(Xg.flatten(), t_val)])
uv_pred, _, _, _, _, _ = net.forward(X_flat)
u_pred = uv_pred[0].reshape(N, N)
u_true, _ = get_exact_solution(Xg, Yg, t_val, 0.0001) # Compare against the hardest target

fig = plt.figure(figsize=(12, 5))

# Cross Section
ax1 = fig.add_subplot(1, 2, 1)
x_slice = np.linspace(-1, 1, 200)
X_slice = np.vstack([x_slice, np.zeros_like(x_slice), np.full_like(x_slice, t_val)])
uv_slice, _, _, _, _, _ = net.forward(X_slice)
u_slice_true, _ = get_exact_solution(x_slice, np.zeros_like(x_slice), t_val, 0.0001)

ax1.plot(x_slice, u_slice_true, 'k-', linewidth=2, label='Exact (Sharp)')
ax1.plot(x_slice, uv_slice[0], 'r--', linewidth=2, label='PINN Prediction')
ax1.set_title(f"Cross Section at y=0, t={t_val}")
ax1.legend()
ax1.grid(True)

# 3D View
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(Xg, Yg, u_pred, cmap='viridis')
ax2.set_title(f"Predicted Surface (nu=0.0001)")

plt.tight_layout()
plt.show() """

# Curricullam

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Curriculum Schedule: Start easy (thick shock), end hard (thin shock)
nu_schedule = [0.05, 0.01, 0.005, 0.001, 0.0001]
epochs_schedule = [2000, 2000, 2000, 3000, 5000] 

batch_size = 3000     
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

w_pde = 1.0
w_ic = 10.0
w_bc = 10.0

# --- Helper Functions ---
def get_exact_solution(x, y, t, nu_val):
    arg = (x + y - t) / (4 * nu_val)
    wave = np.tanh(arg)
    u = 0.5 - 0.5 * wave
    v = 0.5 - 0.5 * wave
    return u, v

def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x): t = np.tanh(x); return -2 * t * (1 - t**2)
def ddd_tanh(x): t = np.tanh(x); return (-2 + 6*t**2) * (1 - t**2)

# --- Neural Network Classes (Adam Implementation) ---
class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(out_f, in_f) * np.sqrt(1/in_f)
        self.b = np.zeros((out_f, 1))
        self.reset_adam()

    def reset_adam(self):
        # Reset momentum when physics changes
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)

    def forward(self, h, h_x, h_y, h_t, h_xx, h_yy):
        self.h, self.h_x, self.h_y, self.h_t = h, h_x, h_y, h_t
        self.h_xx, self.h_yy = h_xx, h_yy
        z = self.W @ h + self.b
        return (z, self.W @ h_x, self.W @ h_y, self.W @ h_t, self.W @ h_xx, self.W @ h_yy)

    def backward(self, d_z, d_zx, d_zy, d_zt, d_zxx, d_zyy):
        dw = (d_z @ self.h.T) + (d_zx @ self.h_x.T) + (d_zy @ self.h_y.T) + \
             (d_zt @ self.h_t.T) + (d_zxx @ self.h_xx.T) + (d_zyy @ self.h_yy.T)
        db = np.sum(d_z, axis=1, keepdims=True)
        d_prev = (self.W.T@d_z, self.W.T@d_zx, self.W.T@d_zy, self.W.T@d_zt, 
                  self.W.T@d_zxx, self.W.T@d_zyy)
        return (dw, db), d_prev

    def step(self, grads, lr, beta1, beta2, epsilon, t_step):
        g_W, g_b = grads[0], grads[1]
        
        self.m_W = beta1 * self.m_W + (1 - beta1) * g_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * (g_W ** 2)
        m_W_hat = self.m_W / (1 - beta1 ** t_step)
        v_W_hat = self.v_W / (1 - beta2 ** t_step)
        self.W -= lr * m_W_hat / (np.sqrt(v_W_hat) + epsilon)

        self.m_b = beta1 * self.m_b + (1 - beta1) * g_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (g_b ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** t_step)
        v_b_hat = self.v_b / (1 - beta2 ** t_step)
        self.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

class Tanh:
    def forward(self, z, z_x, z_y, z_t, z_xx, z_yy):
        self.z, self.z_x, self.z_y, self.z_t = z, z_x, z_y, z_t
        self.z_xx, self.z_yy = z_xx, z_yy
        self.s = tanh(z); self.ds = d_tanh(z); self.dds = dd_tanh(z)
        
        a = self.s
        a_x = self.ds * z_x
        a_y = self.ds * z_y
        a_t = self.ds * z_t
        a_xx = self.dds * (z_x**2) + self.ds * z_xx
        a_yy = self.dds * (z_y**2) + self.ds * z_yy
        return a, a_x, a_y, a_t, a_xx, a_yy

    def backward(self, d_a, d_ax, d_ay, d_at, d_axx, d_ayy):
        ddds = ddd_tanh(self.z)
        d_z = (d_a * self.ds) + (d_ax * self.dds * self.z_x) + (d_ay * self.dds * self.z_y) + \
              (d_at * self.dds * self.z_t) + (d_axx * (ddds * self.z_x**2 + self.dds * self.z_xx)) + \
              (d_ayy * (ddds * self.z_y**2 + self.dds * self.z_yy))
              
        d_zx = (d_ax * self.ds) + (d_axx * 2 * self.dds * self.z_x)
        d_zy = (d_ay * self.ds) + (d_ayy * 2 * self.dds * self.z_y)
        d_zt = d_at * self.ds
        d_zxx = d_axx * self.ds
        d_zyy = d_ayy * self.ds
        return d_z, d_zx, d_zy, d_zt, d_zxx, d_zyy

class PINN:
    def __init__(self, layers_cfg):
        self.layers = []
        for i in range(len(layers_cfg) - 1):
            self.layers.append(Linear(layers_cfg[i], layers_cfg[i+1]))
            if i < len(layers_cfg) - 2: self.layers.append(Tanh())

    def forward(self, x_in):
        h = x_in
        h_x = np.zeros_like(x_in); h_x[0,:] = 1.0 
        h_y = np.zeros_like(x_in); h_y[1,:] = 1.0
        h_t = np.zeros_like(x_in); h_t[2,:] = 1.0
        h_xx = np.zeros_like(x_in)
        h_yy = np.zeros_like(x_in)
        for layer in self.layers:
            h, h_x, h_y, h_t, h_xx, h_yy = layer.forward(h, h_x, h_y, h_t, h_xx, h_yy)
        return h, h_x, h_y, h_t, h_xx, h_yy

    def backward(self, d_u, d_ux, d_uy, d_ut, d_uxx, d_uyy):
        grads = (d_u, d_ux, d_uy, d_ut, d_uxx, d_uyy)
        l_grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, Tanh): 
                grads = layer.backward(*grads)
            else: 
                w_g, i_g = layer.backward(*grads)
                l_grads.append(w_g)
                grads = i_g
        return l_grads[::-1]

    def update(self, l_grads, lr, beta1, beta2, epsilon, t_step):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(l_grads[idx], lr, beta1, beta2, epsilon, t_step)
                idx+=1
    
    def reset_optimizer(self):
        for layer in self.layers:
            if isinstance(layer, Linear): layer.reset_adam()

# --- Main Training Loop (Curriculum ONLY) ---

net = PINN([3, 60, 60, 60, 2]) # Deep network

print(f"Starting Curriculum Learning Training (No RAR)...")
print(f"Schedule: {nu_schedule}")

global_step = 0

# Iterate through viscosities (Easy -> Hard)
for stage, current_nu in enumerate(nu_schedule):
    
    # Reset optimizer momentum for the new stage
    net.reset_optimizer()
    stage_epochs = epochs_schedule[stage]
    print(f"\n--- STAGE {stage+1}/{len(nu_schedule)}: nu = {current_nu} ({stage_epochs} epochs) ---")
    
    for epoch in range(stage_epochs + 1):
        global_step += 1
        
        # 1. Standard Batching (No Adaptive Sampling)
        x_c = np.random.uniform(-1, 1, (1, batch_size))
        y_c = np.random.uniform(-1, 1, (1, batch_size))
        t_c = np.random.uniform(0, 0.5, (1, batch_size))
        X_c = np.vstack([x_c, y_c, t_c])

        # 2. PDE Loss (Using current_nu)
        uv, uv_x, uv_y, uv_t, uv_xx, uv_yy = net.forward(X_c)
        u, v = uv[0:1], uv[1:2]
        ux, vx = uv_x[0:1], uv_x[1:2]
        uy, vy = uv_y[0:1], uv_y[1:2]
        ut, vt = uv_t[0:1], uv_t[1:2]
        uxx, vxx = uv_xx[0:1], uv_xx[1:2]
        uyy, vyy = uv_yy[0:1], uv_yy[1:2]
        
        f_u = ut + u*ux + v*uy - current_nu*(uxx + uyy)
        f_v = vt + u*vx + v*vy - current_nu*(vxx + vyy)
        
        loss_f = np.mean(f_u**2 + f_v**2)
        
        # Gradients
        bs = X_c.shape[1]
        dfu, dfv = (2/bs)*f_u, (2/bs)*f_v
        grad_u = dfu * ux + dfv * vx
        grad_v = dfv * vy + dfu * uy
        
        grads_f = net.backward(
            np.vstack([grad_u, grad_v]), np.vstack([dfu*u, dfv*u]), np.vstack([dfu*v, dfv*v]),
            np.vstack([dfu, dfv]), np.vstack([-current_nu*dfu, -current_nu*dfv]), np.vstack([-current_nu*dfu, -current_nu*dfv])
        )

        # 3. Initial Condition Loss
        x_i = np.random.uniform(-1, 1, (1, 1000))
        y_i = np.random.uniform(-1, 1, (1, 1000))
        t_i = np.zeros_like(x_i)
        X_i = np.vstack([x_i, y_i, t_i])
        
        uv_i, _, _, _, _, _ = net.forward(X_i)
        u_true_i, v_true_i = get_exact_solution(x_i, y_i, t_i, current_nu)
        
        diff_u_i, diff_v_i = uv_i[0] - u_true_i, uv_i[1] - v_true_i
        loss_i = np.mean(diff_u_i**2 + diff_v_i**2)
        grads_i = net.backward((2/1000)*np.vstack([diff_u_i, diff_v_i]), np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i))

        # 4. Boundary Condition Loss
        nb = 800 
        yb_v, xb_h = np.random.uniform(-1, 1, (1, nb)), np.random.uniform(-1, 1, (1, nb))
        xb = np.hstack([-np.ones((1,nb)), np.ones((1,nb)), xb_h, xb_h])
        yb = np.hstack([yb_v, yb_v, -np.ones((1,nb)), np.ones((1,nb))])
        tb = np.random.uniform(0, 0.5, (1, nb*4))
        X_b = np.vstack([xb, yb, tb])
        
        uv_b, _, _, _, _, _ = net.forward(X_b)
        u_true_b, v_true_b = get_exact_solution(xb, yb, tb, current_nu)
        
        diff_u_b, diff_v_b = uv_b[0] - u_true_b, uv_b[1] - v_true_b
        loss_b = np.mean(diff_u_b**2 + diff_v_b**2)
        grads_b = net.backward((2/(nb*4))*np.vstack([diff_u_b, diff_v_b]), np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b))

        # 5. Update
        final_grads = []
        for g1, g2, g3 in zip(grads_f, grads_i, grads_b):
            dw = w_pde*g1[0] + w_ic*g2[0] + w_bc*g3[0]
            db = w_pde*g1[1] + w_ic*g2[1] + w_bc*g3[1]
            final_grads.append((dw, db))
            
        net.update(final_grads, lr, beta1, beta2, epsilon, global_step)

        if epoch % 500 == 0:
            print(f"  Epoch {epoch} | Loss: {loss_f+loss_i+loss_b:.5f} | PDE: {loss_f:.5f}")

# --- Plotting Results ---
print(f"\nTraining completed. Visualizing result for FINAL nu={nu_schedule[-1]}")
N = 80
x_v = np.linspace(-1, 1, N)
y_v = np.linspace(-1, 1, N)
Xg, Yg = np.meshgrid(x_v, y_v)
t_val = 0.25 

X_flat = np.vstack([Xg.flatten(), Yg.flatten(), np.full_like(Xg.flatten(), t_val)])
uv_pred, _, _, _, _, _ = net.forward(X_flat)
u_pred = uv_pred[0].reshape(N, N)

fig = plt.figure(figsize=(12, 5))

# Cross Section
ax1 = fig.add_subplot(1, 2, 1)
x_slice = np.linspace(-1, 1, 200)
X_slice = np.vstack([x_slice, np.zeros_like(x_slice), np.full_like(x_slice, t_val)])
uv_slice, _, _, _, _, _ = net.forward(X_slice)
u_slice_true, _ = get_exact_solution(x_slice, np.zeros_like(x_slice), t_val, 0.0001)

ax1.plot(x_slice, u_slice_true, 'k-', linewidth=2, label='Exact (Sharp)')
ax1.plot(x_slice, uv_slice[0], 'r--', linewidth=2, label='PINN Prediction')
ax1.set_title(f"Cross Section at y=0, t={t_val}")
ax1.legend()
ax1.grid(True)

# 3D View
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(Xg, Yg, u_pred, cmap='viridis')
ax2.set_title(f"Predicted Surface (nu=0.0001)")

plt.tight_layout()
plt.show()