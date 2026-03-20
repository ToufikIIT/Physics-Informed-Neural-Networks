import numpy as np
import matplotlib.pyplot as plt

NU = 0.0               
N_FOURIER = 450        

EPOCHS = 15000         
BATCH_SIZE = 200
LR = 1e-3
GAMMA = 0.9            

W_PDE = 1.0
W_IC = 5.0
W_BC = 10.0


def get_fourier_sgn(x, n_terms):
    """
    Fourier Sine Series approximation of sgn(x) on [-1, 1].
    Formula: 4/pi * sum_{odd n} (sin(n*pi*x) / n)
    """
    u = np.zeros_like(x)
    coeff = 4.0 / np.pi
    
    for k in range(n_terms):
        n = 2 * k + 1  
        term = np.sin(n * np.pi * x) / n
        u += coeff * term
        
    return u

def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x): t = np.tanh(x); return -2 * t * (1 - t**2)
def ddd_tanh(x): t = np.tanh(x); return (-2 + 6*t**2) * (1 - t**2)

class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(out_f, in_f) * np.sqrt(2/in_f)
        self.b = np.zeros((out_f, 1))
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, h, h_x, h_t, h_xx):
        self.h, self.h_x, self.h_t, self.h_xx = h, h_x, h_t, h_xx
        return (self.W @ h + self.b), (self.W @ h_x), (self.W @ h_t), (self.W @ h_xx)

    def backward(self, d_z, d_zx, d_zt, d_zxx):
        dw = (d_z @ self.h.T) + (d_zx @ self.h_x.T) + \
             (d_zt @ self.h_t.T) + (d_zxx @ self.h_xx.T)
        db = np.sum(d_z + d_zx + d_zt + d_zxx, axis=1, keepdims=True)
        
        return (dw, db), (self.W.T @ d_z,
                          self.W.T @ d_zx,
                          self.W.T @ d_zt,
                          self.W.T @ d_zxx)

    def step(self, grads, lr, gamma):
        self.vW = gamma * self.vW + grads[0]
        self.vb = gamma * self.vb + grads[1]
        self.W -= lr * self.vW
        self.b -= lr * self.vb


class Tanh:
    def forward(self, z, z_x, z_t, z_xx):
        self.z, self.z_x, self.z_t, self.z_xx = z, z_x, z_t, z_xx
        self.s = tanh(z)
        self.ds = d_tanh(z)
        self.dds = dd_tanh(z)

        a = self.s
        a_x = self.ds * z_x
        a_t = self.ds * z_t
        a_xx = self.dds * (z_x**2) + self.ds * z_xx
        return a, a_x, a_t, a_xx

    def backward(self, d_a, d_ax, d_at, d_axx):
        ddds = ddd_tanh(self.z)

        d_z = (d_a * self.ds) + \
              (d_ax * self.dds * self.z_x) + \
              (d_at * self.dds * self.z_t) + \
              (d_axx * (ddds * self.z_x**2 + self.dds * self.z_xx))

        d_zx = (d_ax * self.ds) + (d_axx * 2 * self.dds * self.z_x)
        d_zt = d_at * self.ds
        d_zxx = d_axx * self.ds

        return d_z, d_zx, d_zt, d_zxx


class PINN:
    def __init__(self, layers_cfg):
        self.layers = []
        for i in range(len(layers_cfg) - 1):
            self.layers.append(Linear(layers_cfg[i], layers_cfg[i+1]))
            if i < len(layers_cfg) - 2:
                self.layers.append(Tanh())

    def forward(self, x_in):
        h = x_in
        h_x = np.zeros_like(x_in); h_x[0,:] = 1.0 
        h_t = np.zeros_like(x_in); h_t[1,:] = 1.0
        h_xx = np.zeros_like(x_in)

        for layer in self.layers:
            h, h_x, h_t, h_xx = layer.forward(h, h_x, h_t, h_xx)
        return h, h_x, h_t, h_xx

    def backward(self, d_u, d_ux, d_ut, d_uxx):
        grads = (d_u, d_ux, d_ut, d_uxx)
        l_grads = []
        for layer in reversed(self.layers):
            if isinstance(layer, Tanh):
                grads = layer.backward(*grads)
            else:
                w_g, i_g = layer.backward(*grads)
                l_grads.append(w_g)
                grads = i_g
        return l_grads[::-1]

    def update(self, l_grads_list, lr, gamma):
        total_grads = []
        num_linear = len(self.layers)//2 + 1
        
        for i in range(num_linear):
             total_grads.append([np.zeros_like(self.layers[i*2].W), 
                                 np.zeros_like(self.layers[i*2].b)])

        weights = [W_PDE, W_IC, W_BC]
        for g_list, w in zip(l_grads_list, weights):
             for layer_idx, (dw, db) in enumerate(g_list):
                 total_grads[layer_idx][0] += w * dw
                 total_grads[layer_idx][1] += w * db
        
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.step(total_grads[idx], lr, gamma)
                idx += 1

net = PINN([2, 50, 50, 50, 50, 1])

print(f"Training INVISCID PINN (nu=0) on Fourier Signum (N={N_FOURIER})...")

loss_history = []

for epoch in range(EPOCHS + 1):

    x_c = np.random.uniform(-1, 1, (1, BATCH_SIZE))
    t_c = np.random.uniform(0, 1, (1, BATCH_SIZE))
    X_c = np.vstack([x_c, t_c])

    u, ux, ut, uxx = net.forward(X_c)
    
    f = ut + u*ux 
    loss_f = np.mean(f**2)

    df = (2/BATCH_SIZE) * f
    grads_f = net.backward(df*ux, df*u, df, np.zeros_like(df))


    x_i = np.random.uniform(-1, 1, (1, 200))
    t_i = np.zeros_like(x_i)
    X_i = np.vstack([x_i, t_i])

    u_pred_i, _, _, _ = net.forward(X_i)
    
    u_true_i = get_fourier_sgn(x_i, N_FOURIER)

    diff_i = u_pred_i - u_true_i
    loss_i = np.mean(diff_i**2)

    d_ic = (2/200) * diff_i
    grads_i = net.backward(d_ic, np.zeros_like(d_ic), 
                           np.zeros_like(d_ic), np.zeros_like(d_ic))


    t_b = np.random.uniform(0, 1, (1, 100))
    x_left = -1 * np.ones_like(t_b)
    x_right = 1 * np.ones_like(t_b)
    
    x_b = np.hstack([x_left, x_right])
    t_b = np.hstack([t_b, t_b])
    X_b = np.vstack([x_b, t_b])

    u_pred_b, _, _, _ = net.forward(X_b)
    u_true_b = np.sign(x_b) 

    diff_b = u_pred_b - u_true_b
    loss_b = np.mean(diff_b**2)

    d_bc = (2/200) * diff_b
    grads_b = net.backward(d_bc, np.zeros_like(d_bc), 
                           np.zeros_like(d_bc), np.zeros_like(d_bc))


    net.update([grads_f, grads_i, grads_b], LR, GAMMA)

    loss_history.append(loss_f + loss_i + loss_b)

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Total Loss: {loss_history[-1]:.6f} (IC: {loss_i:.6f})")

# 5. VISUALIZATION
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x_test = np.linspace(-1, 1, 300)
t_zero = np.zeros_like(x_test)
X_test = np.vstack([x_test.reshape(1,-1), t_zero.reshape(1,-1)])

u_pred_0, _, _, _ = net.forward(X_test)
u_fourier_0 = get_fourier_sgn(x_test.reshape(1,-1), N_FOURIER).flatten()
u_exact_0 = np.sign(x_test)


plt.plot(x_test, u_exact_0, 'k:', linewidth=2, label='Exact sgn(x)')
plt.plot(x_test, u_fourier_0, 'g-', alpha=0.6, linewidth=1.5, label=f'Fourier (N={N_FOURIER})')
plt.plot(x_test, u_pred_0.flatten(), 'r--', linewidth=2, label='PINN Prediction')
plt.title(f"Initial Condition (t=0)\nInviscid Mode (nu=0)")
plt.xlabel("x")
plt.ylabel("u")
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
times = [0.0, 0.25, 0.5, 0.75]
for t_val in times:
    t_in = np.ones_like(x_test) * t_val
    X_in = np.vstack([x_test.reshape(1,-1), t_in.reshape(1,-1)])
    u_out, _, _, _ = net.forward(X_in)
    plt.plot(x_test, u_out.flatten(), label=f't={t_val}')

plt.title("Inviscid Evolution: Rarefaction Fan")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()