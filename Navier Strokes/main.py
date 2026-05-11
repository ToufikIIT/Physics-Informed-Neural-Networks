import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

epochs = 10000
lr = 3*1e-3
Re = 100.0
nu = 1.0 / Re

lam = Re/2 - np.sqrt((Re**2)/4 + 4*np.pi**2)

def exact_u(x, y):
    return 1 - np.exp(lam*x) * np.cos(2*np.pi*y)

def exact_v(x, y):
    return (lam/(2*np.pi)) * np.exp(lam*x) * np.sin(2*np.pi*y)

def exact_p(x, y):
    return 0.5 * (1 - np.exp(2*lam*x))

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

# Architecture from PDF (Outputs: psi, p)
net = PINN([2, 32, 32, 32, 32, 2])
loss_history = []

# TRAINING LOOP
for ep in range(epochs):
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

    # PDE Backprop (Derivative of logcosh is tanh)
    dRu = np.tanh(Ru) / 1000.0
    dRv = np.tanh(Rv) / 1000.0

    d_out, d_ox, d_oy, d_oxx, d_oyy, d_oxy = [np.zeros_like(out) for _ in range(6)]
    d_oxxx, d_oxxy, d_oxyy, d_oyyy = [np.zeros_like(out) for _ in range(4)]

    # Distribute chain rule for Momentum Equations to network derivatives
    d_ox[0:1] = dRu * (-uy) + dRv * (-vy)
    d_oy[0:1] = dRu * (ux) + dRv * (vx)
    d_ox[1:2], d_oy[1:2] = dRu, dRv
    d_oxx[0:1], d_oyy[0:1] = dRv * (-v), dRu * (-u)
    d_oxy[0:1] = dRu * (u) + dRv * (v)
    d_oxxx[0:1], d_oxyy[0:1] = dRv * nu, dRv * nu
    d_oxxy[0:1], d_oyyy[0:1] = dRu * (-nu), dRu * (-nu)

    gpde = net.backward(d_out, d_ox, d_oy, d_oxx, d_oyy, d_oxy, d_oxxx, d_oxxy, d_oxyy, d_oyyy)

    # --- 2. BOUNDARY CONDITIONS ---
    Xbc = np.random.rand(2, 400)
    sides = np.random.randint(0, 4, 400)
    Xbc[0, sides==0] = 0; Xbc[0, sides==1] = 1
    Xbc[1, sides==2] = 0; Xbc[1, sides==3] = 1

    out_bc, ox_bc, oy_bc, oxx_bc, oyy_bc, oxy_bc, oxxx_bc, oxxy_bc, oxyy_bc, oyyy_bc = net.forward(Xbc)

    u_bc, v_bc = oy_bc[0:1,:], -ox_bc[0:1,:]
    u_ex = exact_u(Xbc[0, :], Xbc[1, :]).reshape(1,-1)
    v_ex = exact_v(Xbc[0, :], Xbc[1, :]).reshape(1,-1)
    loss_bc = np.mean(logcosh(u_bc - u_ex)) + np.mean(logcosh(v_bc - v_ex))

    du_bc = np.tanh(u_bc - u_ex) / 400.0
    dv_bc = np.tanh(v_bc - v_ex) / 400.0

    d_bc_out, d_bc_ox, d_bc_oy, d_bc_oxx, d_bc_oyy, d_bc_oxy = [np.zeros_like(out_bc) for _ in range(6)]
    d_bc_oxxx, d_bc_oxxy, d_bc_oxyy, d_bc_oyyy = [np.zeros_like(out_bc) for _ in range(4)]

    d_bc_ox[0:1] = -dv_bc
    d_bc_oy[0:1] = du_bc

    gbc = net.backward(d_bc_out, d_bc_ox, d_bc_oy, d_bc_oxx, d_bc_oyy, d_bc_oxy,d_bc_oxxx, d_bc_oxxy, d_bc_oxyy, d_bc_oyyy)

    # Combine Gradients & Step Optimizer
    grads = [(g1[0] + g2[0], g1[1] + g2[1]) for g1, g2 in zip(gpde, gbc)]
    net.step(grads, lr)

    total_loss = loss_pde + loss_bc
    loss_history.append(total_loss)

    if ep % 500 == 0:
        print(f"Epoch {ep} | Loss = {total_loss:.6f}")

# VISUALIZATION
N = 100
x, y = np.linspace(0,1,N), np.linspace(0,1,N)
X, Y = np.meshgrid(x, y)
XY = np.vstack([X.flatten(), Y.flatten()])

out, ox, oy, _, _, _, _, _, _, _ = net.forward(XY)

u_pred, v_pred = oy[0:1,:].reshape(N,N), -ox[0:1,:].reshape(N,N)
p_pred = out[1:2,:].reshape(N,N)

u_exact, v_exact, p_exact = exact_u(X,Y), exact_v(X,Y), exact_p(X,Y)

# Anchor pressure to analytical mean (since PINN pressure is relative without BC anchor)
p_pred = p_pred - np.mean(p_pred) + np.mean(p_exact)

def plot_field(exact, pred, name):
    plt.figure(figsize=(15,4))
    plt.subplot(131)
    plt.contourf(X, Y, exact, 100, cmap='jet'); plt.title(f"Exact {name}"); plt.colorbar()
    plt.subplot(132)
    plt.contourf(X, Y, pred, 100, cmap='jet'); plt.title(f"Predicted {name}"); plt.colorbar()
    plt.subplot(133)
    plt.contourf(X, Y, np.abs(exact - pred), 100, cmap='hot'); plt.title(f"{name} Error"); plt.colorbar()
    plt.tight_layout(); plt.show()

plot_field(u_exact, u_pred, "u")
plot_field(v_exact, v_pred, "v")
plot_field(p_exact, p_pred, "Pressure (p)")

plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.yscale("log")
plt.title("Training Loss (LogCosh)")
plt.grid()
plt.show()

# 3D VISUALIZATION

def plot_3d_field(exact, pred, name):

    error = np.abs(exact - pred)

    fig = plt.figure(figsize=(18,5))

    # Exact
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(
        X, Y, exact,
        cmap='jet',
        edgecolor='none'
    )
    ax1.set_title(f'Exact {name}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel(name)
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Predicted
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(
        X, Y, pred,
        cmap='jet',
        edgecolor='none'
    )
    ax2.set_title(f'Predicted {name}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel(name)
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(
        X, Y, error,
        cmap='hot',
        edgecolor='none'
    )
    ax3.set_title(f'{name} Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Error')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.show()

# CALL 3D PLOTS

plot_3d_field(u_exact, u_pred, "u")
plot_3d_field(v_exact, v_pred, "v")
plot_3d_field(p_exact, p_pred, "Pressure")