import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Target viscosity (The sharp shock you want)
target_nu = 0.0001   
# Starting viscosity (The easy smooth wave)
start_nu = 0.2       

epochs = 35000       # Increased epochs to give time for annealing
batch_size = 200    
lr = 1e-3
gamma = 0.9

w_pde = 1.0
w_ic = 10.0
w_bc = 10.0

# --- Helper Functions ---

# UPDATED: Now accepts 'nu_val' so the solution gets sharper over time
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

# --- Neural Network Classes ---

class Linear:
    def __init__(self, in_f, out_f):
        # FIX: Changed to Xavier Initialization (better for Tanh/PINNs)
        self.W = np.random.randn(out_f, in_f) * np.sqrt(1/in_f)
        self.b = np.zeros((out_f, 1))
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

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

    def step(self, grads, lr, gamma):
        self.vW = gamma * self.vW + grads[0]
        self.vb = gamma * self.vb + grads[1]
        self.W -= lr * self.vW
        self.b -= lr * self.vb

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

    def update(self, l_grads, lr, gamma):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(l_grads[idx], lr, gamma)
                idx+=1

# --- Main Training Loop ---

net = PINN([3, 40, 40, 40, 2])

print(f"Starting Viscosity Continuation Training...")
print(f"Annealing Nu: {start_nu} -> {target_nu}")

for epoch in range(epochs + 1):
    
    # --- ANNEALING STEP ---
    # Calculate current viscosity for this epoch (Logarithmic decay is best)
    # This slowly lowers nu from 0.2 down to 0.0001
    decay_rate = np.log(target_nu / start_nu) / epochs
    current_nu = start_nu * np.exp(decay_rate * epoch)
    
    # 1. PDE LOSS (Physics)
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
    
    # Use current_nu for the physics equation
    f_u = ut + u*ux + v*uy - current_nu*(uxx + uyy)
    f_v = vt + u*vx + v*vy - current_nu*(vxx + vyy)
    
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
        np.vstack([-current_nu*dfu, -current_nu*dfv]),        
        np.vstack([-current_nu*dfu, -current_nu*dfv])        
    )

    # 2. INITIAL CONDITION LOSS
    x_i = np.random.uniform(-1, 1, (1, 100))
    y_i = np.random.uniform(-1, 1, (1, 100))
    t_i = np.zeros_like(x_i)
    X_i = np.vstack([x_i, y_i, t_i])
    
    uv_i, _, _, _, _, _ = net.forward(X_i)
    # Important: The "True" solution we match against must also use current_nu
    u_true_i, v_true_i = get_exact_solution(x_i, y_i, t_i, current_nu)
    
    diff_u_i = uv_i[0] - u_true_i
    diff_v_i = uv_i[1] - v_true_i
    loss_i = np.mean(diff_u_i**2 + diff_v_i**2)
    
    grads_i = net.backward(
        (2/100)*np.vstack([diff_u_i, diff_v_i]),
        np.zeros_like(uv_i), np.zeros_like(uv_i), np.zeros_like(uv_i),
        np.zeros_like(uv_i), np.zeros_like(uv_i)
    )

    # 3. BOUNDARY CONDITION LOSS
    nb = 50 
    yb_v = np.random.uniform(-1, 1, (1, nb))
    xb_h = np.random.uniform(-1, 1, (1, nb))
    
    xb = np.hstack([-np.ones((1,nb)), np.ones((1,nb)), xb_h, xb_h])
    yb = np.hstack([yb_v, yb_v, -np.ones((1,nb)), np.ones((1,nb))])
    tb = np.random.uniform(0, 0.5, (1, nb*4))
    X_b = np.vstack([xb, yb, tb])
    
    uv_b, _, _, _, _, _ = net.forward(X_b)
    # Boundary target must also adapt to current_nu
    u_true_b, v_true_b = get_exact_solution(xb, yb, tb, current_nu)
    
    diff_u_b = uv_b[0] - u_true_b
    diff_v_b = uv_b[1] - v_true_b
    loss_b = np.mean(diff_u_b**2 + diff_v_b**2)
    
    grads_b = net.backward(
        (2/(nb*4))*np.vstack([diff_u_b, diff_v_b]),
        np.zeros_like(uv_b), np.zeros_like(uv_b), np.zeros_like(uv_b),
        np.zeros_like(uv_b), np.zeros_like(uv_b)
    )

    # 4. UPDATE
    final_grads = []
    for g1, g2, g3 in zip(grads_f, grads_i, grads_b):
        dw = w_pde*g1[0] + w_ic*g2[0] + w_bc*g3[0]
        db = w_pde*g1[1] + w_ic*g2[1] + w_bc*g3[1]
        final_grads.append((dw, db))
        
    net.update(final_grads, lr, gamma)

    if epoch % 5000 == 0:
        print(f"Epoch {epoch} | Nu: {current_nu:.6f} | Total Loss: {loss_f+loss_i+loss_b:.5f}")


# --- Graphs ---

print(f"\nTraining completed. Generating results for Nu={target_nu}...")
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

# Compare against the final target viscosity
u_true, _ = get_exact_solution(Xg, Yg, t_val, target_nu)

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(Xg, Yg, u_pred, cmap='viridis')
ax1.set_title(f"Predicted u (Sharp, nu={target_nu})")

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.plot_surface(Xg, Yg, u_true, cmap='viridis')
ax2.set_title(f"Exact u (Sharp, nu={target_nu})")

ax3 = fig.add_subplot(2, 2, 3)
cf3 = ax3.contourf(Xg, Yg, u_pred, 20, cmap='viridis')
ax3.set_title("Predicted u (2D)")
plt.colorbar(cf3, ax=ax3)

ax4 = fig.add_subplot(2, 2, 4)
cf4 = ax4.contourf(Xg, Yg, u_true, 20, cmap='viridis')
ax4.set_title("Exact u (2D)")
plt.colorbar(cf4, ax=ax4)

plt.suptitle(f"2D Burgers - Sharp Shock Wave Simulation (Nu={target_nu})", fontsize=16)

# 1D Cuts to see the steepness
N_slice = 200 # More points to see the sharp line
x_slice = np.linspace(-1, 1, N_slice)
y_slice = np.zeros_like(x_slice)
times = [0.0, 0.4]

plt.figure(figsize=(12, 8))

for i, t_val in enumerate(times):
    t_slice = np.full_like(x_slice, t_val)
    X_slice = np.vstack([x_slice, y_slice, t_slice])

    uv_pred_slice, _, _, _, _, _ = net.forward(X_slice)
    u_pred_line = uv_pred_slice[0].flatten()

    u_true_line, _ = get_exact_solution(x_slice, y_slice, t_slice, target_nu)

    plt.subplot(2, 1, i + 1)
    plt.plot(x_slice, u_true_line, 'k-', linewidth=2.0, label='Exact (Shock)')
    plt.plot(x_slice, u_pred_line, 'r--', linewidth=2.0, label='PINN Prediction')
    plt.title(f"u(x, y=0, t={t_val})")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True, alpha=0.5)
    plt.legend()

plt.suptitle("1D Cross-Sections: Attempting to Capture the Step", fontsize=16)
plt.tight_layout()
plt.show()