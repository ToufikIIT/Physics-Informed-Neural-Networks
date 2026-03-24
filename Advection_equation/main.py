import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ===================== CONFIG =====================
np.random.seed(0)

epochs = 12000
lr = 1e-3

alpha = 0.01
u_vel = 1.0
v_vel = 1.0

k1, k2 = 2*np.pi, 2*np.pi
k = np.sqrt(k1**2 + k2**2)
omega = k * np.sqrt(u_vel**2 + v_vel**2)

w_pde = 1.0
w_ic  = 10.0
w_bc  = 10.0

# ===================== ACTIVATIONS =====================
def tanh(x): return np.tanh(x)
def d_tanh(x): return 1 - np.tanh(x)**2

def dd_tanh(x):
    t = np.tanh(x)
    return -2*t*(1 - t**2)

def d3_tanh(x):
    t = np.tanh(x)
    return -2*(1 - t**2)*(1 - 3*t**2)

# ===================== LINEAR =====================
class Linear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(out_f, in_f) * np.sqrt(2/(in_f+out_f))
        self.b = np.zeros((out_f,1))

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t = 0

    def forward(self, h, hx, hy, ht, hxx, hyy, htt):
        self.h, self.hx, self.hy, self.ht = h, hx, hy, ht
        self.hxx, self.hyy, self.htt = hxx, hyy, htt

        z = self.W @ h + self.b
        zx = self.W @ hx
        zy = self.W @ hy
        zt = self.W @ ht

        zxx = self.W @ hxx
        zyy = self.W @ hyy
        ztt = self.W @ htt

        return z, zx, zy, zt, zxx, zyy, ztt

    def backward(self, dz, dzx, dzy, dzt, dzxx, dzyy, dztt):

        dw = (dz @ self.h.T +
              dzx @ self.hx.T +
              dzy @ self.hy.T +
              dzt @ self.ht.T +
              dzxx @ self.hxx.T +
              dzyy @ self.hyy.T +
              dztt @ self.htt.T)

        db = np.sum(dz, axis=1, keepdims=True)

        dh   = self.W.T @ dz
        dhx  = self.W.T @ dzx
        dhy  = self.W.T @ dzy
        dht  = self.W.T @ dzt

        dhxx = self.W.T @ dzxx
        dhyy = self.W.T @ dzyy
        dhtt = self.W.T @ dztt

        return (dw, db), (dh, dhx, dhy, dht, dhxx, dhyy, dhtt)

    def step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        dw, db = grads
        self.t += 1

        self.mW = b1*self.mW + (1-b1)*dw
        self.vW = b2*self.vW + (1-b2)*(dw**2)

        mW = self.mW/(1-b1**self.t)
        vW = self.vW/(1-b2**self.t)

        self.W -= lr*mW/(np.sqrt(vW)+eps)

        self.mb = b1*self.mb + (1-b1)*db
        self.vb = b2*self.vb + (1-b2)*(db**2)

        mb = self.mb/(1-b1**self.t)
        vb = self.vb/(1-b2**self.t)

        self.b -= lr*mb/(np.sqrt(vb)+eps)

# ===================== TANH =====================
class Tanh:
    def forward(self, z, zx, zy, zt, zxx, zyy, ztt):

        self.z = z
        self.zx, self.zy, self.zt = zx, zy, zt
        self.zxx, self.zyy, self.ztt = zxx, zyy, ztt

        self.s = tanh(z)
        self.ds = d_tanh(z)
        self.dds = dd_tanh(z)
        self.ddds = d3_tanh(z)

        a = self.s

        ax = self.ds*zx
        ay = self.ds*zy
        at = self.ds*zt

        axx = self.dds*(zx**2) + self.ds*zxx
        ayy = self.dds*(zy**2) + self.ds*zyy
        att = self.dds*(zt**2) + self.ds*ztt

        return a, ax, ay, at, axx, ayy, att

    def backward(self, da, dax, day, dat, daxx, dayy, datt):

        dz = (da*self.ds
              + dax*(self.dds*self.zx)
              + day*(self.dds*self.zy)
              + dat*(self.dds*self.zt)
              + daxx*(self.ddds*self.zx**2 + self.dds*self.zxx)
              + dayy*(self.ddds*self.zy**2 + self.dds*self.zyy)
              + datt*(self.ddds*self.zt**2 + self.dds*self.ztt))

        dzx = dax*self.ds + daxx*(2*self.dds*self.zx)
        dzy = day*self.ds + dayy*(2*self.dds*self.zy)
        dzt = dat*self.ds + datt*(2*self.dds*self.zt)

        dzxx = daxx*self.ds
        dzyy = dayy*self.ds
        dztt = datt*self.ds

        return dz, dzx, dzy, dzt, dzxx, dzyy, dztt

# ===================== NETWORK =====================
class PINN:
    def __init__(self, layers):
        self.layers = [Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.acts = [Tanh() for _ in range(len(layers)-2)]

    def forward(self, x):
        h = x
        hx = np.zeros_like(x); hx[0,:]=1
        hy = np.zeros_like(x); hy[1,:]=1
        ht = np.zeros_like(x); ht[2,:]=1

        hxx = np.zeros_like(x)
        hyy = np.zeros_like(x)
        htt = np.zeros_like(x)

        for i,layer in enumerate(self.layers):
            h,hx,hy,ht,hxx,hyy,htt = layer.forward(h,hx,hy,ht,hxx,hyy,htt)
            if i < len(self.acts):
                h,hx,hy,ht,hxx,hyy,htt = self.acts[i].forward(h,hx,hy,ht,hxx,hyy,htt)

        return h,hx,hy,ht,hxx,hyy,htt

    def backward(self, *grads):
        g = grads
        Lg = []

        for i in reversed(range(len(self.layers))):
            gw, g = self.layers[i].backward(*g)
            Lg.append(gw)
            if i>0:
                g = self.acts[i-1].backward(*g)

        return Lg[::-1]

    def step(self, grads, lr):
        for i,layer in enumerate(self.layers):
            layer.step(grads[i], lr)

# ===================== EXACT SOLUTION =====================
def exact(x,y,t):
    return np.exp(-alpha*k**2*t) * np.sin(k1*x + k2*y - omega*t)

# ===================== TRAIN =====================
net = PINN([3,50,50,50,1])

loss_hist = []

for ep in range(epochs):

    # ---- PDE points ----
    X = np.random.rand(3,800)
    u,ux,uy,ut,uxx,uyy,utt = net.forward(X)

    res = ut + u_vel*ux + v_vel*uy - alpha*(uxx+uyy)
    d_res = (2.0/800)*res

    gpde = net.backward(
        np.zeros_like(u),
        u_vel*d_res,
        v_vel*d_res,
        d_res,
        -alpha*d_res,
        -alpha*d_res,
        np.zeros_like(u)
    )

    # ---- IC (t=0) ----
    X_ic = np.random.rand(3,200)
    X_ic[2,:]=0
    u_ic,*_ = net.forward(X_ic)

    target_ic = np.sin(k1*X_ic[0] + k2*X_ic[1])
    d_ic = (2.0/200)*w_ic*(u_ic-target_ic)

    gic = net.backward(d_ic, *( [np.zeros_like(u_ic)]*6 ))

    # ---- BC ----
    X_bc = np.random.rand(3,200)
    X_bc[0,:]=0  # x=0 boundary
    u_bc,*_ = net.forward(X_bc)

    target_bc = exact(X_bc[0],X_bc[1],X_bc[2])
    d_bc = (2.0/200)*w_bc*(u_bc-target_bc)

    gbc = net.backward(d_bc, *( [np.zeros_like(u_bc)]*6 ))

    # ---- Combine ----
    grads = []
    for g1,g2,g3 in zip(gpde,gic,gbc):
        dw = w_pde*g1[0] + g2[0] + g3[0]
        db = w_pde*g1[1] + g2[1] + g3[1]
        grads.append((dw,db))

    net.step(grads,lr)

    loss = np.mean(res**2) + np.mean((u_ic-target_ic)**2)
    loss_hist.append(loss)

    if ep%1000==0:
        print(f"Epoch {ep} Loss {loss:.5e}")

# ===================== PLOT =====================
y = np.linspace(0,1,100)
x = 0.5*np.ones_like(y)

times = [0.0,0.25,0.5,0.75]

plt.figure(figsize=(12,8))

for i,t in enumerate(times):
    X = np.vstack([x,y,np.ones_like(y)*t])
    pred,*_ = net.forward(X)
    exact_val = exact(x,y,t)

    plt.subplot(2,2,i+1)
    plt.plot(y,exact_val.flatten(),'k--',label="Exact")
    plt.plot(y,pred.flatten(),'r-',label="PINN")
    plt.title(f"t={t}")
    plt.legend()
    plt.grid()

plt.suptitle("Advection-Diffusion PINN (Exact vs Predicted)")
plt.tight_layout()
plt.show()

def plot_3d(net, t):

    x = np.linspace(0,1,50)
    y = np.linspace(0,1,50)
    X, Y = np.meshgrid(x,y)

    XYT = np.vstack([
        X.flatten(),
        Y.flatten(),
        np.ones_like(X.flatten())*t
    ])

    pred, *_ = net.forward(XYT)
    pred = pred.reshape(X.shape)

    exact_val = exact(X.flatten(), Y.flatten(), t).reshape(X.shape)

    fig = plt.figure(figsize=(12,5))

    # --- PINN ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, pred, cmap='viridis')
    ax1.set_title(f"PINN (t={t})")

    # --- Exact ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, exact_val, cmap='viridis')
    ax2.set_title(f"Exact (t={t})")

    plt.show()


plot_3d(net, t=0.2)
plot_3d(net, t=0.5)
plot_3d(net, t=0.9)


def animate_solution(net):
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    X,Y = np.meshgrid(x,y)

    fig,ax = plt.subplots()
    img = ax.imshow(np.zeros_like(X),extent=[0,1,0,1],
                    origin='lower',cmap='jet',vmin=-1,vmax=1)

    plt.colorbar(img)

    def update(frame):
        XYT = np.vstack([X.flatten(),Y.flatten(),
                         np.ones_like(X.flatten())*frame])
        pred,*_ = net.forward(XYT)
        Z = pred.reshape(X.shape)

        img.set_array(Z)
        ax.set_title(f"t={frame:.2f}")
        return [img]

    ani = animation.FuncAnimation(fig,update,
                                  frames=np.linspace(0,1,50),
                                  interval=100)

    plt.show()
    return ani

ani = animate_solution(net)