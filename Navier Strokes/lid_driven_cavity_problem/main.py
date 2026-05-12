import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(42)

epochs = 15000  
lr = 2 * 1e-3
Re = 100.0
nu = 1.0 / Re

# LOGCOSH LOSS HELPER
def logcosh(x):
    return np.log(np.cosh(x + 1e-12))

# SiLU ACTIVATION & DERIVATIVES
class SiLU:
    def forward(self, z, zx, zy, zxx, zyy, zxy, zxxx, zxxy, zxyy, zyyy):
        self.z = z
        self.zx = zx
        self.zy = zy
        self.zxx = zxx
        self.zyy = zyy
        self.zxy = zxy
        self.zxxx = zxxx
        self.zxxy = zxxy
        self.zxyy = zxyy
        self.zyyy = zyyy

        # Safe sigmoid
        s = np.where(z > 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
        s2 = s * (1 - s)
        s3 = s2 * (1 - 2*s)
        s4 = s3 * (1 - 2*s) - 2 * (s2**2)
        s5 = s4 * (1 - 2*s) - 6 * s2 * s3

        self.f1 = s + z * s2
        self.f2 = 2*s2 + z * s3
        self.f3 = 3*s3 + z * s4
        self.f4 = 4*s4 + z * s5

        a = z * s
        ax = self.f1 * zx
        ay = self.f1 * zy
        axx = self.f2 * (zx**2) + self.f1 * zxx
        ayy = self.f2 * (zy**2) + self.f1 * zyy
        axy = self.f2 * (zx*zy) + self.f1 * zxy

        axxx = self.f3 * (zx**3) + 3 * self.f2 * zx * zxx + self.f1 * zxxx
        axxy = self.f3 * (zx**2) * zy + self.f2 * (2 * zx * zxy + zxx * zy) + self.f1 * zxxy
        axyy = self.f3 * zx * (zy**2) + self.f2 * (2 * zy * zxy + zyy * zx) + self.f1 * zxyy
        ayyy = self.f3 * (zy**3) + 3 * self.f2 * zy * zyy + self.f1 * zyyy

        return a, ax, ay, axx, ayy, axy, axxx, axxy, axyy, ayyy

    def backward(self, da, dax, day, daxx, dayy, daxy, daxxx, daxxy, daxyy, dayyy):
        f1, f2, f3, f4 = self.f1, self.f2, self.f3, self.f4
        zx, zy = self.zx, self.zy
        zxx, zyy, zxy = self.zxx, self.zyy, self.zxy
        zxxx, zxxy, zxyy, zyyy = self.zxxx, self.zxxy, self.zxyy, self.zyyy

        dzxxx = daxxx * f1
        dzxxy = daxxy * f1
        dzxyy = daxyy * f1
        dzyyy = dayyy * f1

        dzxx = daxx * f1 + daxxx * (3 * f2 * zx) + daxxy * (f2 * zy)
        dzxy = daxy * f1 + daxxy * (2 * f2 * zx) + daxyy * (2 * f2 * zy)
        dzyy = dayy * f1 + dayyy * (3 * f2 * zy) + daxyy * (f2 * zx)

        dzx = (dax * f1 + daxx * (2 * f2 * zx) + daxy * (f2 * zy)
             + daxxx * (3 * f3 * zx**2 + 3 * f2 * zxx)
             + daxxy * (2 * f3 * zx * zy + 2 * f2 * zxy)
             + daxyy * (f3 * zy**2 + f2 * zyy))

        dzy = (day * f1 + dayy * (2 * f2 * zy) + daxy * (f2 * zx)
             + dayyy * (3 * f3 * zy**2 + 3 * f2 * zyy)
             + daxyy * (2 * f3 * zx * zy + 2 * f2 * zxy)
             + daxxy * (f3 * zx**2 + f2 * zxx))

        dz = (da * f1 + dax * (f2 * zx) + day * (f2 * zy)
            + daxx * (f3 * zx**2 + f2 * zxx)
            + dayy * (f3 * zy**2 + f2 * zyy)
            + daxy * (f3 * zx * zy + f2 * zxy)
            + daxxx * (f4 * zx**3 + 3 * f3 * zx * zxx + f2 * zxxx)
            + daxxy * (f4 * (zx**2) * zy + f3 * (2 * zx * zxy + zxx * zy) + f2 * zxxy)
            + daxyy * (f4 * zx * (zy**2) + f3 * (2 * zy * zxy + zyy * zx) + f2 * zxyy)
            + dayyy * (f4 * zy**3 + 3 * f3 * zy * zyy + f2 * zyyy))

        return dz, dzx, dzy, dzxx, dzyy, dzxy, dzxxx, dzxxy, dzxyy, dzyyy

# LINEAR LAYER w/ ADAM OPTIMIZER
class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(out_f, in_f) * np.sqrt(2 / (in_f + out_f))
        self.b = np.zeros((out_f, 1))

        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0

    def forward(self, h, hx, hy, hxx, hyy, hxy, hxxx, hxxy, hxyy, hyyy):
        self.h, self.hx, self.hy = h, hx, hy
        self.hxx, self.hyy, self.hxy = hxx, hyy, hxy
        self.hxxx, self.hxxy, self.hxyy, self.hyyy = hxxx, hxxy, hxyy, hyyy

        z = self.W @ h + self.b
        zx, zy = self.W @ hx, self.W @ hy
        zxx, zyy, zxy = self.W @ hxx, self.W @ hyy, self.W @ hxy
        zxxx, zxxy, zxyy, zyyy = self.W @ hxxx, self.W @ hxxy, self.W @ hxyy, self.W @ hyyy

        return z, zx, zy, zxx, zyy, zxy, zxxx, zxxy, zxyy, zyyy

    def backward(self, dz, dzx, dzy, dzxx, dzyy, dzxy, dzxxx, dzxxy, dzxyy, dzyyy):
        dW = (dz @ self.h.T + dzx @ self.hx.T + dzy @ self.hy.T
            + dzxx @ self.hxx.T + dzyy @ self.hyy.T + dzxy @ self.hxy.T
            + dzxxx @ self.hxxx.T + dzxxy @ self.hxxy.T
            + dzxyy @ self.hxyy.T + dzyyy @ self.hyyy.T)

        db = np.sum(dz, axis=1, keepdims=True)

        dh, dhx, dhy = self.W.T @ dz, self.W.T @ dzx, self.W.T @ dzy
        dhxx, dhyy, dhxy = self.W.T @ dzxx, self.W.T @ dzyy, self.W.T @ dzxy
        dhxxx, dhxxy = self.W.T @ dzxxx, self.W.T @ dzxxy
        dhxyy, dhyyy = self.W.T @ dzxyy, self.W.T @ dzyyy

        return (dW, db), (dh, dhx, dhy, dhxx, dhyy, dhxy, dhxxx, dhxxy, dhxyy, dhyyy)

    def step(self, grads, lr):
        dW, db = grads
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8

        self.mW = b1 * self.mW + (1 - b1) * dW
        self.vW = b2 * self.vW + (1 - b2) * (dW**2)
        m_hat_W = self.mW / (1 - b1**self.t)
        v_hat_W = self.vW / (1 - b2**self.t)
        self.W -= lr * m_hat_W / (np.sqrt(v_hat_W) + eps)

        self.mb = b1 * self.mb + (1 - b1) * db
        self.vb = b2 * self.vb + (1 - b2) * (db**2)
        m_hat_b = self.mb / (1 - b1**self.t)
        v_hat_b = self.vb / (1 - b2**self.t)
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

# PINN
class PINN:
    def __init__(self, layers):
        self.layers = [Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.acts = [SiLU() for _ in range(len(layers)-2)]

    def forward(self, X):
        h = X
        hx, hy = np.zeros_like(X), np.zeros_like(X)
        hx[0,:], hy[1,:] = 1, 1
        hxx, hyy, hxy = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
        hxxx, hxxy, hxyy, hyyy = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)

        for i, layer in enumerate(self.layers):
            h, hx, hy, hxx, hyy, hxy, hxxx, hxxy, hxyy, hyyy = layer.forward(
                h, hx, hy, hxx, hyy, hxy, hxxx, hxxy, hxyy, hyyy)
            if i < len(self.acts):
                h, hx, hy, hxx, hyy, hxy, hxxx, hxxy, hxyy, hyyy = self.acts[i].forward(
                    h, hx, hy, hxx, hyy, hxy, hxxx, hxxy, hxyy, hyyy)
        return h, hx, hy, hxx, hyy, hxy, hxxx, hxxy, hxyy, hyyy

    def backward(self, *du_args):
        g = du_args
        grads = []
        for i in reversed(range(len(self.layers))):
            gw, g = self.layers[i].backward(*g)
            grads.append(gw)
            if i > 0:
                g = self.acts[i-1].backward(*g)
        return grads[::-1]

    def step(self, grads, lr):
        for i, layer in enumerate(self.layers):
            layer.step(grads[i], lr)

# Architecture (Outputs: psi, p)
net = PINN([2, 64, 64, 64, 64, 2])
loss_history = []

# TRAINING LOOP
for ep in range(epochs):
    # --- 1. PDE POINTS ---
    X = np.random.rand(2, 1000)
    out, ox, oy, oxx, oyy, oxy, oxxx, oxxy, oxyy, oyyy = net.forward(X)

    # u = psi_y, v = -psi_x
    u, v = oy[0:1,:], -ox[0:1,:]
    ux, uy = oxy[0:1,:], oyy[0:1,:]
    vx, vy = -oxx[0:1,:], -oxy[0:1,:]
    uxx, uyy = oxxy[0:1,:], oyyy[0:1,:]
    vxx, vyy = -oxxx[0:1,:], -oxyy[0:1,:]
    px, py = ox[1:2,:], oy[1:2,:]

    # Momentum Equations Residuals
    Ru = u*ux + v*uy + px - nu*(uxx + uyy)
    Rv = u*vx + v*vy + py - nu*(vxx + vyy)

    loss_pde = np.mean(logcosh(Ru)) + np.mean(logcosh(Rv))

    # PDE Backprop
    dRu = np.tanh(Ru) / 1000.0
    dRv = np.tanh(Rv) / 1000.0

    d_out, d_ox, d_oy, d_oxx, d_oyy, d_oxy = [np.zeros_like(out) for _ in range(6)]
    d_oxxx, d_oxxy, d_oxyy, d_oyyy = [np.zeros_like(out) for _ in range(4)]

    # Distribute chain rule for Momentum Equations
    d_ox[0:1] = dRu * (-uy) + dRv * (-vy)
    d_oy[0:1] = dRu * (ux) + dRv * (vx)
    d_ox[1:2], d_oy[1:2] = dRu, dRv
    d_oxx[0:1], d_oyy[0:1] = dRv * (-v), dRu * (-u)
    d_oxy[0:1] = dRu * (u) + dRv * (v)
    d_oxxx[0:1], d_oxyy[0:1] = dRv * nu, dRv * nu
    d_oxxy[0:1], d_oyyy[0:1] = dRu * (-nu), dRu * (-nu)

    gpde = net.backward(d_out, d_ox, d_oy, d_oxx, d_oyy, d_oxy, d_oxxx, d_oxxy, d_oxyy, d_oyyy)

    # --- 2. LID-DRIVEN CAVITY BOUNDARY CONDITIONS ---
    N_bc = 100 # Points per boundary side
    x_b = np.random.rand(N_bc)
    y_b = np.random.rand(N_bc)

    # Top boundary (Moving wall: u=1, v=0)
    X_top = np.vstack([x_b, np.ones_like(x_b)])
    u_top_tgt = np.ones(N_bc)

    # Bottom boundary (Stationary: u=0, v=0)
    X_bot = np.vstack([x_b, np.zeros_like(x_b)])
    u_bot_tgt = np.zeros(N_bc)

    # Left boundary (Stationary: u=0, v=0)
    X_left = np.vstack([np.zeros_like(y_b), y_b])
    u_left_tgt = np.zeros(N_bc)

    # Right boundary (Stationary: u=0, v=0)
    X_right = np.vstack([np.ones_like(y_b), y_b])
    u_right_tgt = np.zeros(N_bc)

    # Combine boundaries
    Xbc = np.hstack([X_top, X_bot, X_left, X_right])

    # Targets
    u_tgt = np.hstack([u_top_tgt, u_bot_tgt, u_left_tgt, u_right_tgt]).reshape(1, -1)
    v_tgt = np.zeros_like(u_tgt) # v is zero on all walls
    psi_tgt = np.zeros_like(u_tgt) # The cavity boundary is a closed streamline (psi = 0)

    out_bc, ox_bc, oy_bc, oxx_bc, oyy_bc, oxy_bc, oxxx_bc, oxxy_bc, oxyy_bc, oyyy_bc = net.forward(Xbc)

    # Network predictions on boundary
    psi_bc = out_bc[0:1,:]
    u_bc, v_bc = oy_bc[0:1,:], -ox_bc[0:1,:]

    # Calculate boundary loss (Velocity + Stream function anchoring)
    loss_bc = np.mean(logcosh(u_bc - u_tgt)) + np.mean(logcosh(v_bc - v_tgt)) + np.mean(logcosh(psi_bc - psi_tgt))

    # BC Backprop
    du_bc = np.tanh(u_bc - u_tgt) / 400.0
    dv_bc = np.tanh(v_bc - v_tgt) / 400.0
    dpsi_bc = np.tanh(psi_bc - psi_tgt) / 400.0

    d_bc_out, d_bc_ox, d_bc_oy, d_bc_oxx, d_bc_oyy, d_bc_oxy = [np.zeros_like(out_bc) for _ in range(6)]
    d_bc_oxxx, d_bc_oxxy, d_bc_oxyy, d_bc_oyyy = [np.zeros_like(out_bc) for _ in range(4)]

    # Route gradients
    d_bc_out[0:1] = dpsi_bc # Anchor stream function
    d_bc_ox[0:1] = -dv_bc
    d_bc_oy[0:1] = du_bc

    gbc = net.backward(d_bc_out, d_bc_ox, d_bc_oy, d_bc_oxx, d_bc_oyy, d_bc_oxy,
                       d_bc_oxxx, d_bc_oxxy, d_bc_oxyy, d_bc_oyyy)

    # === FIXED: Explicit explicit loop to guarantee safe arrays ===
    grads = []
    for g1, g2 in zip(gpde, gbc):
        dW_combined = g1[0] + g2[0] # Corrected line
        db_combined = g1[1] + g2[1]
        grads.append((dW_combined, db_combined))

    net.step(grads, lr)

    total_loss = loss_pde + loss_bc
    loss_history.append(total_loss)

    if ep % 500 == 0:
        print(f"Epoch {ep} | Loss = {total_loss:.6f}")

# VISUALIZATION
print("Training Complete. Generating Visualizations...")

N = 100
x, y = np.linspace(0,1,N), np.linspace(0,1,N)
X, Y = np.meshgrid(x, y)
XY = np.vstack([X.flatten(), Y.flatten()])

out, ox, oy, _, _, _, _, _, _, _ = net.forward(XY)

psi_pred = out[0:1,:].reshape(N,N)
p_pred = out[1:2,:].reshape(N,N)
u_pred = oy[0:1,:].reshape(N,N)
v_pred = -ox[0:1,:].reshape(N,N)

# Plotting the predicted Lid-Driven Cavity Flow (2D)
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 1. Velocity Magnitude Contour
speed = np.sqrt(u_pred**2 + v_pred**2)
contour1 = axs[0, 0].contourf(X, Y, speed, 100, cmap='jet') # Fixed index
axs[0, 0].set_title("Velocity Magnitude Predicted") # Fixed index
fig.colorbar(contour1, ax=axs[0, 0]) # Fixed index

# 2. Streamlines (Primary Vortex)
axs[0, 1].streamplot(X, Y, u_pred, v_pred, color=speed, cmap='jet', density=1.5) # Fixed index
axs[0, 1].set_title("Predicted Flow Streamlines") # Fixed index
axs[0, 1].set_xlim([0, 1]) # Fixed xlim to match problem domain
axs[0, 1].set_ylim([0, 1]) # Fixed ylim to match problem domain

# 3. Horizontal Velocity (u)
contour3 = axs[1, 0].contourf(X, Y, u_pred, 100, cmap='jet') # Fixed index
axs[1, 0].set_title("Horizontal Velocity (u)") # Fixed index
fig.colorbar(contour3, ax=axs[1, 0]) # Fixed index

# 4. Vertical Velocity (v)
contour4 = axs[1, 1].contourf(X, Y, v_pred, 100, cmap='jet')
axs[1, 1].set_title("Vertical Velocity (v)")
fig.colorbar(contour4, ax=axs[1, 1])

plt.tight_layout()
plt.show()

# Training Loss Curve
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.yscale("log")
plt.title("Cavity Flow Training Loss (LogCosh)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# 3D VISUALIZATION
def plot_3d_field(field_data, name):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, field_data, cmap='jet', edgecolor='none')
    ax.set_title(f'Predicted {name} (3D)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(name)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    plt.tight_layout()
    plt.show()

plot_3d_field(u_pred, "u Velocity")
plot_3d_field(v_pred, "v Velocity")
plot_3d_field(p_pred, "Pressure")
