import numpy as np
import matplotlib.pyplot as plt

nu = 0.1          
epochs = 15000    
batch_size = 100
lr = 1e-3
gamma = 0.9      

w_pde = 1.0
w_ic = 10.0
w_bc = 10.0

def exact_solution(x, t):
    exponent = (x - t) / nu
    exponent = np.clip(exponent, -50, 50) 
    return 2.0 / (1.0 + np.exp(exponent))

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
        return (self.W@h + self.b), (self.W@h_x), (self.W@h_t), (self.W@h_xx)

    def backward(self, d_z, d_zx, d_zt, d_zxx):
        dw = (d_z @ self.h.T) + (d_zx @ self.h_x.T) + (d_zt @ self.h_t.T) + (d_zxx @ self.h_xx.T)
        db = np.sum(d_z, axis=1, keepdims=True)
        return (dw, db), (self.W.T@d_z, self.W.T@d_zx, self.W.T@d_zt, self.W.T@d_zxx)

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
        
        self.a = self.s
        self.a_x = self.ds * z_x
        self.a_t = self.ds * z_t
        self.a_xx = self.dds * (z_x**2) + self.ds * z_xx
        return self.a, self.a_x, self.a_t, self.a_xx

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
            if i < len(layers_cfg) - 2: self.layers.append(Tanh())

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
            if isinstance(layer, Tanh): grads = layer.backward(*grads)
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

net = PINN([2, 20, 20, 20, 1])

print(f"Starting PINN training for Traveling Wave (nu={nu})...")

for epoch in range(epochs + 1):
    
    x_c = np.random.uniform(-2, 4, (1, batch_size))
    t_c = np.random.uniform(0, 2, (1, batch_size))
    X_c = np.vstack([x_c, t_c])
    
    u, ux, ut, uxx = net.forward(X_c)
    
    f = ut + u*ux - nu*uxx
    loss_f = np.mean(f**2)
    
    df = (2/batch_size)*f
    grads_f = net.backward(df*ux, df*u, df, -nu*df)

    x_i = np.random.uniform(-2, 4, (1, 50))
    t_i = np.zeros_like(x_i)
    X_i = np.vstack([x_i, t_i])
    
    u_pred_i, _, _, _ = net.forward(X_i)
    u_true_i = exact_solution(x_i, 0) 
    
    diff_i = u_pred_i - u_true_i
    loss_i = np.mean(diff_i**2)
    
    grads_i = net.backward((2/50)*diff_i, np.zeros_like(u_pred_i), np.zeros_like(u_pred_i), np.zeros_like(u_pred_i))

    t_b = np.random.uniform(0, 2, (1, 50))
    x_b_left = -2 * np.ones_like(t_b)
    x_b_right = 4 * np.ones_like(t_b)
    
    x_b = np.hstack([x_b_left, x_b_right])
    t_b = np.hstack([t_b, t_b])
    X_b = np.vstack([x_b, t_b])
    
    u_pred_b, _, _, _ = net.forward(X_b)
    u_true_b = exact_solution(x_b, t_b)
    
    diff_b = u_pred_b - u_true_b
    loss_b = np.mean(diff_b**2)
    
    grads_b = net.backward((2/100)*diff_b, np.zeros_like(u_pred_b), np.zeros_like(u_pred_b), np.zeros_like(u_pred_b))

    final_grads = []
    for g1, g2, g3 in zip(grads_f, grads_i, grads_b):
        dw = w_pde*g1[0] + w_ic*g2[0] + w_bc*g3[0]
        db = w_pde*g1[1] + w_ic*g2[1] + w_bc*g3[1]
        final_grads.append((dw, db))
        
    net.update(final_grads, lr, gamma)

    if epoch % 2000 == 0:
        print(f"Epoch {epoch} | Total Loss: {loss_f + loss_i + loss_b:.6f} | PDE: {loss_f:.5f}")

plt.figure(figsize=(15, 5))
test_times = [0.0, 0.5, 0.75]
x_test = np.linspace(-2, 4, 100)

for i, t_val in enumerate(test_times):
    t_in = np.ones_like(x_test) * t_val
    X_in = np.vstack([x_test.reshape(1,-1), t_in.reshape(1,-1)])
    u_pred, _, _, _ = net.forward(X_in)
    u_true = exact_solution(x_test, t_val)
    
    plt.subplot(1, 3, i+1)
    plt.plot(x_test, u_true, 'k-', linewidth=2, label='Exact')
    plt.plot(x_test, u_pred.flatten(), 'r--', linewidth=2, label='PINN')
    
    plt.title(f"Time t = {t_val}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True)
    if i == 0: plt.legend()

plt.suptitle(f"Burgers Eq: Traveling Wave (c=1, nu={nu})", fontsize=14)
plt.tight_layout()
plt.show()