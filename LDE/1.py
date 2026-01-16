import numpy as np
import matplotlib.pyplot as plt

n_hidden = 30
epochs = 30000
n_collocation = 100

lr = 1e-3
gamma = 0.9
bc_weight = 30.0

np.random.seed(0)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2

def dd_tanh(x):
    return -2 * np.tanh(x) * (1 - np.tanh(x)**2)

def init(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])

w = init((n_hidden, 1))
b = np.zeros((n_hidden, 1))
v = init((1, n_hidden))

class SGD_Momentum:
    def __init__(self, params, lr, gamma):
        self.params = params
        self.lr = lr
        self.gamma = gamma
        self.vel = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for i in range(len(self.params)):
            self.vel[i] = self.gamma * self.vel[i] + grads[i]
            self.params[i] -= self.lr * self.vel[i]

def forward(x, params):
    w, b, v = params
    z = w @ x + b
    h = tanh(z)
    sp = d_tanh(z)

    y = v @ h
    y_x = (v * w.T) @ sp

    return y, y_x, z, h, sp

optimizer = SGD_Momentum([w, b, v], lr, gamma)

for epoch in range(epochs):

    w, b, v = optimizer.params

    x_bc = np.zeros((1, 1))
    y_bc, _, z_bc, h_bc, sp_bc = forward(x_bc, [w, b, v])

    diff_bc = y_bc - 1.0
    loss_bc = diff_bc**2

    dv_bc = 2 * diff_bc @ h_bc.T
    dw_bc = 2 * diff_bc * v.T * sp_bc @ x_bc.T
    db_bc = 2 * diff_bc * v.T * sp_bc

    x_col = np.random.uniform(0, 2, (1, n_collocation))
    y, y_x, z, h, sp = forward(x_col, [w, b, v])
    spp = dd_tanh(z)

    R = y_x + 3*y - 2*x_col
    loss_pde = np.mean(R**2)
    g = (2 / n_collocation) * R

    dR_dv = (w * sp) + 3*h
    dv_pde = g @ dR_dv.T

    dy_dw = v.T * sp * x_col
    dyx_dw = v.T * (sp + w * spp * x_col)
    dR_dw = dyx_dw + 3 * dy_dw
    dw_pde = np.sum(g * dR_dw, axis=1, keepdims=True)

    dy_db = v.T * sp
    dyx_db = v.T * w * spp
    dR_db = dyx_db + 3 * dy_db
    db_pde = np.sum(g * dR_db, axis=1, keepdims=True)

    dv = dv_pde + bc_weight * dv_bc
    dw = dw_pde + bc_weight * dw_bc
    db = db_pde + bc_weight * db_bc

    optimizer.step([dw, db, dv])

    if epoch % 5000 == 0:
        total_loss = loss_pde + bc_weight * loss_bc
        print(f"Epoch {epoch} | Loss = {total_loss.item():.6f}")

x_test = np.linspace(0, 2, 200).reshape(1, -1)
y_pred, _, _, _, _ = forward(x_test, optimizer.params)

y_exact = (11/9)*np.exp(-3*x_test) + (2/3)*x_test - 2/9

plt.plot(x_test.flatten(), y_exact.flatten(), 'k-', label="Exact")
plt.plot(x_test.flatten(), y_pred.flatten(), 'r--', label="PINN")
plt.legend()
plt.grid()
plt.show()
