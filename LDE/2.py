import numpy as np
import matplotlib.pyplot as plt

n_hidden = 30
epochs = 30000           
n_collocation = 100

lr = 0.001           
gamma = 0.99         
ic_weight = 50.0     

def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2
def dd_tanh(x): return -2 * np.tanh(x) * (1 - np.tanh(x)**2)

def init(size):
    return np.random.randn(*size) 

w = init((n_hidden, 1))
b = np.zeros((n_hidden, 1))
v = init((1, n_hidden))

class SGD_Momentum:
    def __init__(self, params, lr=0.001, gamma=0.9):
        self.params = params
        self.lr = lr
        self.gamma = gamma
        self.velocities = [np.zeros_like(p) for p in params]
        
    def step(self, grads):
        updated_params = []
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.velocities[i] = (self.gamma * self.velocities[i]) + (self.lr * grad)
            param_new = param - self.velocities[i]
            updated_params.append(param_new)
        self.params = updated_params
        return updated_params

optimizer = SGD_Momentum([w, b, v], lr=lr, gamma=gamma)

def forward(x_in, params):
    w_curr, b_curr, v_curr = params
    z = w_curr @ x_in + b_curr
    h = tanh(z)
    y = v_curr @ h
    sigma_p = d_tanh(z)
    y_x = (v_curr * w_curr.T) @ sigma_p
    return y, y_x, z, h, sigma_p

for epoch in range(epochs):
    current_w, current_b, current_v = optimizer.params
    
    x_bc = np.zeros((1, 1))
    y_bc, _, _, h_bc, sp_bc = forward(x_bc, [current_w, current_b, current_v])
    
    diff_bc = y_bc - 0.2
    loss_bc = np.sum(diff_bc**2)
    grad_factor_bc = 2.0 * diff_bc
    
    dv_bc = grad_factor_bc @ h_bc.T
    dw_bc = (grad_factor_bc * current_v.T * sp_bc) @ x_bc.T
    db_bc = np.sum(grad_factor_bc * current_v.T * sp_bc, axis=1, keepdims=True)

    x_col = np.random.uniform(0, 2, (1, n_collocation))
    y_pde, y_x_pde, z_pde, h_pde, sp_pde = forward(x_col, [current_w, current_b, current_v])
    
    residual = y_x_pde - 2*y_pde - np.sin(x_col)
    loss_pde = np.mean(residual**2)
    grad_factor_pde = (2.0 / n_collocation) * residual
    spp_pde = dd_tanh(z_pde)
    
    dR_dv = (current_w * sp_pde) - 2 * h_pde
    dv_pde = np.sum(grad_factor_pde * dR_dv, axis=1, keepdims=True).T
    
    dy_dw = current_v.T * sp_pde * x_col
    dyx_dw = (current_v.T * sp_pde) + (current_v.T * current_w * spp_pde * x_col)
    dR_dw = dyx_dw - 2 * dy_dw
    dw_pde = np.sum(grad_factor_pde * dR_dw, axis=1, keepdims=True)
    
    dy_db = current_v.T * sp_pde
    dyx_db = current_v.T * current_w * spp_pde
    dR_db = dyx_db - 2 * dy_db
    db_pde = np.sum(grad_factor_pde * dR_db, axis=1, keepdims=True)
    
    dv_total = dv_pde + (ic_weight * dv_bc)
    dw_total = dw_pde + (ic_weight * dw_bc)
    db_total = db_pde + (ic_weight * db_bc)
    
    optimizer.step([dw_total, db_total, dv_total])
    
    total_loss = loss_pde + (ic_weight * loss_bc)

    if epoch % 5000 == 0:
        print(f"Epoch {epoch}: Loss {total_loss:.5f}")


final_w, final_b, final_v = optimizer.params
x_test = np.linspace(0, 2, 100).reshape(1, 100)
y_pred, _, _, _, _ = forward(x_test, [final_w, final_b, final_v])

y_exact = 0.4 * np.exp(2*x_test) - 0.4*np.sin(x_test) - 0.2*np.cos(x_test)

plt.plot(x_test.flatten(), y_exact.flatten(), 'k-', label="Exact")
plt.plot(x_test.flatten(), y_pred.flatten(), 'r--', label="Predicted")
plt.legend()
plt.grid(True)
plt.show()