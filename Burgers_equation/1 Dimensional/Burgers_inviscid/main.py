import numpy as np
import matplotlib.pyplot as plt

a = 0.7
b = -0.2

w_pde = 1.0
w_ic = 10.0
w_bc = 10.0

def exact_solution(x, t):
    return (a * x + b) / (a * t + 1)

epochs = 12000
batch_size = 100
lr = 1e-3
gamma = 0.9

def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x):
    t = np.tanh(x)
    return -2 * t * (1 - t**2)
def ddd_tanh(x):
    t = np.tanh(x)
    return (-2 + 6*t**2) * (1 - t**2)

class Tanh:
    def forward(self, z, z_x, z_t, z_xx):
        self.z, self.z_x, self.z_t, self.z_xx = z, z_x, z_t, z_xx
        self.ds = d_tanh(z)
        self.dds = dd_tanh(z)
        a = tanh(z)
        a_x = self.ds * z_x
        a_t = self.ds * z_t
        a_xx = self.dds * (z_x**2) + self.ds * z_xx
        return a, a_x, a_t, a_xx

    def backward(self, d_a, d_ax, d_at, d_axx):
        ddds = ddd_tanh(self.z)

        d_z = (
            d_a * self.ds
            + d_ax * self.dds * self.z_x
            + d_at * self.dds * self.z_t
            + d_axx * (ddds * self.z_x**2 + self.dds * self.z_xx)
        )

        d_zx = d_ax * self.ds + 2 * d_axx * self.dds * self.z_x
        d_zt = d_at * self.ds
        d_zxx = d_axx * self.ds

        return d_z, d_zx, d_zt, d_zxx


class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(out_f, in_f) * np.sqrt(2/in_f)
        self.b = np.zeros((out_f, 1))
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, h, h_x, h_t, h_xx):
        self.h, self.h_x, self.h_t, self.h_xx = h, h_x, h_t, h_xx
        return (
            self.W @ h + self.b,
            self.W @ h_x,
            self.W @ h_t,
            self.W @ h_xx
        )

    def backward(self, d_z, d_zx, d_zt, d_zxx):
        dW = (
            d_z @ self.h.T
            + d_zx @ self.h_x.T
            + d_zt @ self.h_t.T
            + d_zxx @ self.h_xx.T
        )
        db = np.sum(d_z, axis=1, keepdims=True)

        return (dW, db), (
            self.W.T @ d_z,
            self.W.T @ d_zx,
            self.W.T @ d_zt,
            self.W.T @ d_zxx
        )

    def step(self, grads):
        self.vW = gamma * self.vW + grads[0]
        self.vb = gamma * self.vb + grads[1]
        self.W -= lr * self.vW
        self.b -= lr * self.vb


class PINN:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(Tanh())

    def forward(self, X):
        h = X
        h_x = np.zeros_like(X); h_x[0] = 1
        h_t = np.zeros_like(X); h_t[1] = 1
        h_xx = np.zeros_like(X)

        for layer in self.layers:
            h, h_x, h_t, h_xx = layer.forward(h, h_x, h_t, h_xx)
        return h, h_x, h_t, h_xx

    def backward(self, grads):
        g = grads
        layer_grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, Tanh):
                g = layer.backward(*g)
            else:
                w_g, g = layer.backward(*g)
                layer_grads.append(w_g)
        return layer_grads[::-1]

    def update(self, grads):
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(grads[idx])
                idx += 1


net = PINN([2, 30, 30, 30, 1])

for epoch in range(epochs + 1):
    x = np.random.uniform(-2, 2, (1, batch_size))
    t = np.random.uniform(0, 2, (1, batch_size))
    X = np.vstack([x, t])

    u, ux, ut, _ = net.forward(X)

    f = ut + u * ux
    loss_pde = np.mean(f**2)

    df = (2/batch_size) * f
    grads_pde = net.backward((df * ux, df * u, df, np.zeros_like(df)))

    xi = np.random.uniform(-2, 2, (1, 50))
    ti = np.zeros_like(xi)
    Xi = np.vstack([xi, ti])

    ui, _, _, _ = net.forward(Xi)
    u_true = exact_solution(xi, 0)

    diff = ui - u_true
    loss_ic = np.mean(diff**2)

    grads_ic = net.backward((2/50 * diff,np.zeros_like(diff),np.zeros_like(diff),np.zeros_like(diff)))

    final_grads = []
    for g1, g2 in zip(grads_pde, grads_ic):
        final_grads.append((w_pde*g1[0] + w_ic*g2[0], w_pde*g1[1] + w_ic*g2[1]))

    net.update(final_grads)
    if epoch % 2000 == 0:
        print(f"Epoch {epoch} | Loss = {loss_pde + loss_ic:.6e}")

x_test = np.linspace(-2, 2, 400)
test_times = [0.0, 0.5, 1.0]
plt.figure(figsize=(9, 5))

for t_test in test_times:
    X_test = np.vstack([x_test.reshape(1, -1),t_test * np.ones((1, x_test.size))])
    u_pred, _, _, _ = net.forward(X_test)
    u_true = exact_solution(x_test, t_test)
    plt.plot(x_test, u_true,linewidth=2,label=f"Exact, t={t_test}")
    plt.plot(x_test, u_pred.flatten(),"--",linewidth=2,label=f"PINN, t={t_test}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Inviscid Burgers Equation – Complete Integral\n(All times on same plot)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
