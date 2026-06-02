## Physics-Informed Neural Network — Lid-Driven Cavity Flow (Re = 400)

---

## What is this code solving?

We are solving the **2-D incompressible Navier–Stokes equations** inside a unit square
cavity where the top wall ("lid") slides to the right at velocity u = 1.
The fluid is driven into a clockwise recirculating vortex.

The governing equations are:

```
u·∂u/∂x + v·∂u/∂y = −∂p/∂x + ν·(∂²u/∂x² + ∂²u/∂y²)   ← x-momentum (Ru = 0)
u·∂v/∂x + v·∂v/∂y = −∂p/∂y + ν·(∂²v/∂x² + ∂²v/∂y²)   ← y-momentum (Rv = 0)
∂u/∂x + ∂v/∂y = 0                                         ← continuity (enforced via ψ)
```

Where ν = 1/Re is the kinematic viscosity.  At Re = 400 the flow is significantly
more inertia-dominated than Re = 100, creating steeper velocity gradients near the walls
and a more prominent secondary corner vortex — making it harder to learn.

---

## Section 1 — Imports and Device Setup (Lines 1–22)

```python
"""
PINN  –  Lid-driven cavity flow  (Re = 400)  v2
...
"""
```
**Lines 1–12 — Docstring.**
The triple-quoted string at the top is a module-level docstring. It lists all seven
improvements over v1, acting as a changelog so anyone reading the file immediately
knows what changed and why.

---

```python
import torch
import torch.nn as nn
```
**Lines 14–15 — PyTorch core.**
`torch` provides tensors, automatic differentiation, and GPU support.
`torch.nn` provides the building blocks for neural networks (Linear layers, activations,
Module base class).

---

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```
**Lines 16–18 — NumPy and Matplotlib.**
NumPy handles CPU-side array maths (meshgrids, linspace, benchmark data).
Matplotlib draws all plots. `Axes3D` is imported explicitly to register the 3-D
projection even though it is not called directly — without this import `projection='3d'`
would fail in some environments.

---

```python
import copy, os
```
**Line 19 — Standard library.**
`copy.deepcopy` snapshots the network weights at each checkpoint.
`os.makedirs` creates the output directory.

---

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
**Lines 21–22 — Hardware selection.**
Queries whether a CUDA-capable GPU is present. If yes, all tensors and the model will
live on the GPU (drastically faster for large collocation sets). Otherwise falls back
to CPU. Every tensor created later passes `device=device` to land in the right place.

**Why important:** Re = 400 with 15k collocation points and 6 residual blocks per
forward pass makes CPU training extremely slow (hours). GPU drops this to minutes.

---

## Section 2 — Hyperparameters (Lines 24–40)

```python
EPOCHS    = 40_000
```
**Line 25 — Adam training epochs.**
v1 used 25,000. Increased to 40,000 because Re = 400 has sharper gradients that need
more passes to drive down the PDE residual. The cosine LR schedule means the extra
15k epochs are not wasted — they run at progressively lower learning rates doing
fine-grained adjustment.

---

```python
LBFGS_MAX = 10_000
```
**Line 26 — Maximum L-BFGS iterations.**
v1 had 5,000. Doubled because L-BFGS is the second-order polishing phase, and Re = 400
needs more curvature-guided steps to reach a tight minimum than Re = 100.

---

```python
LR_START  = 1e-3
LR_END    = 5e-5
```
**Lines 27–28 — Learning rate bounds (NEW in v2).**
v1 had a single fixed LR of 1e-3 throughout. v2 introduces a cosine schedule decaying
from 1e-3 down to 5e-5. The start value is aggressive for fast early progress; the
end value is gentle enough for precise convergence without oscillating around the minimum.

---

```python
RE   = 400.0
NU   = 1.0 / RE
```
**Lines 29–30 — Reynolds number and viscosity.**
ν (nu) = 1/Re appears directly in the Navier–Stokes residuals. Re = 400 means
ν = 0.0025, which is 4× smaller than Re = 100's ν = 0.01. Smaller viscosity means
diffusion is weaker — convective terms dominate — and the solution has thinner boundary
layers that the network must resolve.

---

```python
N_PDE     = 15_000
```
**Line 31 — Interior collocation point count.**
v1 used 10,000. Increased to 15,000 to give the PDE residual loss more spatial
coverage, especially now that 25% of points are redirected near walls (see WALL_FRAC).
More points = better average representation of the PDE across the domain.

---

```python
WALL_FRAC  = 0.25
WALL_DELTA = 0.05
```
**Lines 32–33 — Wall clustering parameters (NEW in v2).**
`WALL_FRAC = 0.25` means 3,750 of the 15,000 PDE points will be inside near-wall
strips. `WALL_DELTA = 0.05` defines those strips as the innermost 5% of the domain
adjacent to each wall. This is explained in detail in the `sample_pde_points` section.

---

```python
N_BC      = 800
```
**Line 34 — Boundary collocation point count.**
v1 had 500. Increased to 800 because the lid velocity profile is more strongly
nonlinear at Re = 400 (larger inertial forces → stronger shear at the lid) and more
points better sample the smooth-lid function.

---

```python
BC_WEIGHT  = 100.0  →  50.0   (CHANGED)
```
**Line 35 — Boundary condition loss weight.**
v1 used 100. Reduced to 50 in v2. This is one of the most important tuning choices.

**Why it was too high at 100:**
The total loss is:  `Loss = loss_pde + BC_WEIGHT * loss_bc + GAUGE_WEIGHT * loss_gauge`
With BC_WEIGHT = 100, even a tiny BC error contributes 100× as much as the same-size
PDE error. The optimiser focused almost entirely on matching the boundary velocities
while neglecting the interior physics. The result was a network that satisfied BCs
well but had large Navier–Stokes residuals inside — producing velocities that looked
wrong in the bulk flow. BC_WEIGHT = 50 keeps both terms competitive.

---

```python
GAUGE_WEIGHT = 10.0
```
**Line 36 — Pressure gauge loss weight (unchanged).**
Pressure in incompressible flow is only defined up to an arbitrary constant (only
pressure gradients appear in the equations). Without pinning it, the network can
learn any pressure field shifted by a constant and still satisfy the PDE. The gauge
term fixes p(0.5, 0.5) = 0 (centre of the domain). Weight 10 is mild — just enough
to remove the degree of freedom without over-constraining the pressure field.

---

```python
GRAD_CLIP = 1.0
```
**Line 37 — Gradient clipping threshold (NEW in v2).**
Before each Adam update, if the global L2 norm of all gradients exceeds 1.0, all
gradients are scaled down proportionally so the norm equals exactly 1.0. At Re = 400
the stiff PDE residuals can produce very large gradient spikes early in training
(especially in the near-wall strips where ∂²u/∂y² is large). These spikes would cause
Adam to make a destructive parameter update. Clipping prevents this without needing
to reduce LR.

---

```python
EPS = 1e-3
```
**Line 38 — Boundary exclusion buffer.**
Boundary collocation points and PDE points are sampled from (EPS, 1-EPS) rather than
(0, 1). This avoids placing a PDE point exactly on the boundary where the stream-
function boundary condition (ψ = 0) is separately enforced — having both at the same
location can create conflicting gradient signals.

---

```python
CHECKPOINT_STEPS = [5000, 10000, 20000, 30000, 40000]
```
**Line 40 — Snapshot schedule (CHANGED from v1).**
v1: `[5000, 10000, 15000, 20000, 25000]` — evenly every 5k up to 25k.
v2: `[5000, 10000, 20000, 30000, 40000]` — matches the new 40k epoch total, with
a gap at 15k replaced by 20k to better track the longer training curve.

---

## Section 3 — Benchmark Data (Lines 42–65)

```python
GHIA_Y = np.array([0.0000, 0.0547, ... 1.0000])
GHIA_U = np.array([0.00000, -0.08186, ... 1.00000])
```
**Lines 44–53 — Ghia Table I: u-velocity on the vertical centreline (x = 0.5).**
These are the gold-standard reference values from Ghia, Ghia & Shin (1982), a
benchmark paper that solved the cavity problem with a multigrid finite-difference
method on a very fine grid. 17 y-positions along the vertical mid-line, with the
corresponding u-velocity. Used at the end to verify the PINN's solution quality.

**v1 error that was fixed:** v1 accidentally used Re = 100 Ghia data for the
Re = 400 run. These are the correct Re = 400 values.

---

```python
GHIA_X_V = np.array([1.0000, 0.9688, ... 0.0000])
GHIA_V   = np.array([0.00000, -0.12146, ... 0.00000])
```
**Lines 56–65 — Ghia Table II: v-velocity on the horizontal centreline (y = 0.5) (NEW in v2).**
v1 only benchmarked the u centreline. v2 adds the v centreline from Table II of
the same paper. This is a stricter test because v peaks near the right and left walls
where the recirculation brings fluid up/down, and is more sensitive to errors in
the pressure gradient. Adding this plot helps diagnose whether the solution is
physically correct in both velocity components, not just u.

---

```python
output_dir = "./outputs_re400_v2"
os.makedirs(output_dir, exist_ok=True)
```
**Lines 67–68 — Output directory.**
Creates `./outputs_re400_v2/` (separate from v1's `./outputs_re400/`) so old results
are never overwritten. `exist_ok=True` means no error if it already exists.

---

## Section 4 — Loss Utility (Lines 70–72)

```python
def mse_loss(x):
    return torch.mean(x ** 2)
```
**Lines 71–72 — Mean squared error.**
Squares every element of the residual tensor, then averages. Used for both PDE
residuals (Ru, Rv) and BC errors. Squaring ensures positivity and differentiability.
The mean (not sum) keeps the loss magnitude independent of how many collocation
points are used — important because N_PDE changes between v1 and v2.

---

## Section 5 — Fourier Feature Embedding (Lines 74–88) ★ NEW in v2

```python
class FourierEmbedding(nn.Module):
```
**Line 75 — Module declaration.**
Inherits from `nn.Module` so PyTorch handles device placement (`.to(device)`) and
state saving/loading automatically.

---

```python
def __init__(self, in_dim=2, n_freqs=64, sigma=1.0):
    super().__init__()
    B = torch.randn(in_dim, n_freqs) * sigma
    self.register_buffer("B", B)
```
**Lines 80–83 — Random frequency matrix construction.**

- `in_dim=2`: input is (x, y), so 2 dimensions.
- `n_freqs=64`: 64 random frequency vectors are drawn.
- `B = torch.randn(2, 64) * sigma`: a 2×64 matrix where each column is a random
  2-D frequency vector drawn from N(0, σ²). σ = 1.0 suits a unit-square domain
  (coordinates in [0,1]).
- `register_buffer("B", B)`: B is stored as part of the model state (so it is saved
  and loaded with `state_dict`) but is **not** a trainable parameter — it is fixed
  for the lifetime of the model. This is crucial: B must not change during training
  or the embedding loses meaning.

**Why is this needed?**
Plain MLPs suffer from "spectral bias" (Rahaman et al., 2019) — they learn low-frequency
functions much faster than high-frequency ones. For the cavity at Re = 400, the velocity
boundary layer near each wall changes from 0 to ~1 within a thin strip — a high-
frequency feature in space. Without help, the MLP will smear this transition and
predict smooth, incorrect profiles. The Fourier embedding explicitly provides high-
frequency sinusoidal features as inputs, bypassing the spectral bias.

---

```python
def forward(self, x):
    proj = 2.0 * np.pi * (x @ self.B)
    return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
```
**Lines 85–88 — Forward pass of the embedding.**

- `x` has shape (N, 2) — N collocation points, each with coordinates (xi, yi).
- `x @ self.B` performs matrix multiplication: (N,2) × (2,64) = (N,64).
  Each result element is  xᵢ·B₁ⱼ + yᵢ·B₂ⱼ  — a linear projection of (x,y) onto
  the j-th random frequency vector.
- Multiplying by 2π gives the angular frequency argument.
- `torch.cos(proj)` and `torch.sin(proj)` each produce (N, 64) tensors.
- `torch.cat([cos, sin], dim=-1)` stacks them → (N, 128).

So each (x, y) input point is mapped to a 128-dimensional vector of oscillating
features. The network then learns to combine these sinusoids to build the solution.
Both cos and sin are needed to represent phase shifts (a cos + b sin = amplitude×sin(ω+φ)).

---

## Section 6 — Residual Block (Lines 90–99) ★ NEW in v2

```python
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
```
**Lines 91–96 — Two linear layers + SiLU activation.**

- Both layers map dim → dim (preserving dimension for the skip connection).
- `nn.SiLU()` is the Sigmoid Linear Unit: f(x) = x·σ(x). It is smooth, non-zero
  for negative inputs (unlike ReLU), and has been shown to work better than tanh
  for PINNs because it has non-zero higher-order derivatives — important since
  we compute up to 3rd-order derivatives of the network output.

---

```python
def forward(self, x):
    return x + self.fc2(self.act(self.fc1(x)))
```
**Line 99 — The residual (skip) connection.**
The output is `input + transformation(input)`. The transformation is:
  `fc1 → SiLU → fc2`
And the input is added directly.

**Why this matters for PINNs:**
v1 used a plain sequential MLP: layer1 → SiLU → layer2 → SiLU → ... → output.
Each layer sees only the output of the previous layer. If gradients are small, they
shrink further with each layer (vanishing gradient problem). With a residual connection,
the gradient of the loss with respect to early layers includes a direct path through
the skip connection:
  `∂Loss/∂input = ∂Loss/∂output · (1 + ∂transformation/∂input)`
The "+1" term ensures the gradient never fully vanishes, even with 6 stacked blocks.
This is why ResNets (He et al., 2016) revolutionised deep learning — the same
principle helps here.

---

## Section 7 — Full Network Architecture (Lines 101–128)

```python
class PINN(nn.Module):
    def __init__(self, n_freqs=64, hidden=128, n_res=6):
```
**Lines 102–108 — Parameters.**
- `n_freqs=64`: 64 Fourier frequencies → embedding dim = 128.
- `hidden=128`: all hidden layers have 128 neurons.
- `n_res=6`: six residual blocks.

v1 used a flat `(2, 64, 64, 64, 64, 64, 2)` MLP — 5 hidden layers of 64 neurons.
v2 uses 128 neurons per layer with skip connections — twice the width and a qualitatively
different connectivity pattern.

---

```python
embed_dim  = 2 * n_freqs          # = 128
self.embed = FourierEmbedding(2, n_freqs, sigma=1.0)
self.input = nn.Linear(embed_dim, hidden)
self.res   = nn.ModuleList([ResBlock(hidden) for _ in range(n_res)])
self.output = nn.Linear(hidden, 2)
self.act   = nn.SiLU()
```
**Lines 110–116 — Submodule construction.**

- `self.embed`: maps (N,2) → (N,128) via random Fourier features.
- `self.input`: projects the 128-dim embedding to the 128-dim hidden space.
  (They happen to be the same size here but conceptually different roles.)
- `nn.ModuleList(...)`: creates 6 ResBlock objects and registers them as
  submodules. `ModuleList` (not a plain Python list) ensures PyTorch tracks
  their parameters for optimisation and saving.
- `self.output`: final linear layer maps 128 → 2, outputting (ψ, p).

---

```python
def _init(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
```
**Lines 118–122 — Xavier weight initialisation.**
`self.modules()` iterates over every submodule recursively (including those inside
ResBlocks). For each Linear layer:
- Weights are drawn from N(0, 2/(fan_in + fan_out)) — Xavier normal. This keeps
  the variance of activations roughly constant across layers at initialisation,
  avoiding both vanishing (too small) and exploding (too large) signals.
- Biases start at zero — no initial preference for positive or negative outputs.

v1 used the same Xavier init. Unchanged because it works well.

---

```python
def forward(self, x):
    h = self.act(self.input(self.embed(x)))
    for blk in self.res:
        h = blk(h)
    return self.output(h)
```
**Lines 124–128 — Forward pass.**

1. `self.embed(x)` → (N, 128): Fourier features.
2. `self.input(...)` → (N, 128): linear projection into hidden space.
3. `self.act(...)` → (N, 128): SiLU activation.
4. Loop over 6 ResBlocks: each refines `h` in-place with a skip connection.
5. `self.output(h)` → (N, 2): raw (ψ, p) values for each collocation point.

The data flow is:  `(x,y) → Fourier → Linear → SiLU → ResBlock×6 → (ψ,p)`

---

## Section 8 — Autograd Helpers (Lines 130–149)

```python
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )[0]
```
**Lines 131–136 — Scalar-field gradient via automatic differentiation.**

- `torch.autograd.grad` computes ∂outputs/∂inputs using the chain rule recorded
  in the computational graph.
- `grad_outputs=torch.ones_like(outputs)`: since `outputs` is a vector (N,1),
  this is equivalent to summing over all N points before differentiating —
  giving the total gradient with respect to each input coordinate.
  Result shape: same as `inputs`, i.e. (N, 2) where column 0 is ∂/∂x, column 1 is ∂/∂y.
- `create_graph=True`: **critical**. Tells PyTorch to build a second-level
  computational graph tracking how the gradients themselves depend on the network
  weights. Without this, we could not differentiate the result again to get second
  derivatives (u_xx, u_yy etc.) or backpropagate through the PDE residual.

---

```python
def get_derivatives(xy, net):
    out      = net(xy)
    psi, p   = out[:, 0:1], out[:, 1:2]
```
**Lines 138–140 — Network evaluation.**
The network outputs a (N,2) tensor. Slicing with `0:1` (not just `0`) preserves
the trailing dimension (N,1) instead of collapsing to (N,). This is important for
the subsequent autograd calls which expect a consistent tensor shape.

---

```python
    pg   = grad(psi, xy)
    u, v = pg[:, 1:2], -pg[:, 0:1]
```
**Lines 141–142 — Stream function to velocity.**
The key physics here: by defining u = ∂ψ/∂y and v = −∂ψ/∂x, continuity
(∂u/∂x + ∂v/∂y = 0) is **automatically satisfied** for any ψ, because:
  ∂u/∂x + ∂v/∂y = ∂²ψ/∂x∂y − ∂²ψ/∂y∂x = 0  (mixed partials are equal).

This eliminates the continuity equation as a separate constraint and reduces
the loss to two momentum residuals. This is a well-established PINN trick for
incompressible flows.

`pg[:, 1:2]` is the y-derivative of ψ → u.
`-pg[:, 0:1]` is the negative x-derivative of ψ → v.

---

```python
    ug      = grad(u, xy);  u_x, u_y = ug[:, 0:1], ug[:, 1:2]
    vg      = grad(v, xy);  v_x, v_y = vg[:, 0:1], vg[:, 1:2]
    u_xx    = grad(u_x, xy)[:, 0:1]
    u_yy    = grad(u_y, xy)[:, 1:2]
    v_xx    = grad(v_x, xy)[:, 0:1]
    v_yy    = grad(v_y, xy)[:, 1:2]
    ppg     = grad(p, xy);  p_x, p_y = ppg[:, 0:1], ppg[:, 1:2]
```
**Lines 143–147 — Second-order derivatives.**
Each `grad()` call traverses one more level of the computational graph:
- First call on u gives (∂u/∂x, ∂u/∂y).
- Second call on u_x gives ∂²u/∂x².
- Second call on u_y gives ∂²u/∂y².
These are the viscous diffusion terms in Navier–Stokes: ν·(∂²u/∂x² + ∂²u/∂y²).

Note the indexing: `grad(u_x, xy)[:, 0:1]` takes only the x-component of the gradient
of u_x (which is ∂/∂x of ∂u/∂x = ∂²u/∂x²). Similarly `[:, 1:2]` takes the y-component.

---

## Section 9 — Wall-Clustered Collocation (Lines 151–180) ★ NEW in v2

```python
def sample_pde_points(n_total, wall_frac, wall_delta):
```
**Line 152 — Purpose.**
Generates the set of (x,y) interior points where the PDE residuals (Ru, Rv) are
evaluated each training step. The distribution of these points is one of the biggest
levers in PINN accuracy.

---

```python
    n_wall  = int(n_total * wall_frac)   # = 3,750
    n_inner = n_total - n_wall           # = 11,250
```
**Lines 159–160 — Split budget.**
With n_total = 15,000 and wall_frac = 0.25:
- 3,750 points go near walls (750 per wall on average, though random).
- 11,250 points fill the interior with Chebyshev clustering.

**Why do we need wall points?**
At Re = 400 the viscous boundary layer — where velocity transitions sharply from 0
at the stationary walls to the interior value — is thin. With purely random or even
Chebyshev-clustered sampling, the probability of landing in a strip of width 0.05
near a wall is only 4 × 0.05 = 20%. Without deliberate over-sampling, the PDE
residual loss is almost entirely determined by the interior, and the network never
"sees" enough signal to learn the sharp gradients correctly.

**What was wrong in v1?**
v1 used only Chebyshev-cosine sampling:
  `xy_pde = 0.5 * (1 − cos(π * uniform))`
The cosine transform does cluster points toward 0 and 1 — but only mildly. The density
at the walls is about 1.57× the uniform density, not enough for Re = 400.

---

```python
    uv    = torch.rand(n_inner, 2, device=device)
    inner = 0.5 * (1.0 - torch.cos(np.pi * uv))
```
**Lines 163–164 — Chebyshev-clustered interior points.**
`torch.rand` gives uniform samples in [0,1]. The cosine transform warps them so more
points land near 0 and 1 in each dimension. This is the standard Chebyshev quadrature
node distribution, known to minimise interpolation error for smooth functions.

---

```python
    xw    = torch.rand(n_wall, device=device)
    yw    = torch.rand(n_wall, device=device)
    which = torch.randint(0, 4, (n_wall,), device=device)
```
**Lines 167–169 — Raw random coordinates for wall strip assignment.**
For each wall point, we draw a random (xw, yw) ∈ [0,1]² and an integer `which`
telling us which of the 4 walls (0=bottom, 1=top, 2=left, 3=right) this point will
be moved to. Using `randint` over 4 values gives an approximately equal number per wall.

---

```python
    x_fin = torch.where(which == 2, xw * wall_delta,
            torch.where(which == 3, 1.0 - xw * wall_delta, xw))
    y_fin = torch.where(which == 0, yw * wall_delta,
            torch.where(which == 1, 1.0 - yw * wall_delta, yw))
```
**Lines 172–175 — Vectorised strip assignment.**
`torch.where(condition, value_if_true, value_if_false)` works element-wise.

Reading x_fin logic:
- If the point is assigned to the left wall (which == 2): x = xw × 0.05,
  so x ∈ [0, 0.05] — left strip.
- If assigned to the right wall (which == 3): x = 1 − xw × 0.05,
  so x ∈ [0.95, 1] — right strip.
- Otherwise (bottom/top walls): x stays as xw — free in [0,1].

Reading y_fin logic:
- Bottom wall (which == 0): y = yw × 0.05, so y ∈ [0, 0.05] — bottom strip.
- Top wall (which == 1): y = 1 − yw × 0.05, so y ∈ [0.95, 1] — top strip.
- Otherwise (left/right walls): y stays as yw — free in [0,1].

This is fully vectorised — no Python loop over points — so it runs fast on GPU.

---

```python
    pts = torch.cat([inner, torch.stack([x_fin, y_fin], dim=1)], dim=0)
    pts = pts.clamp(EPS, 1.0 - EPS)
    pts.requires_grad_(True)
    return pts
```
**Lines 177–180 — Assemble and prepare for autograd.**
- `torch.cat([inner, wall_pts], dim=0)`: combines 11,250 interior + 3,750 wall points
  into one (15,000, 2) tensor.
- `.clamp(EPS, 1-EPS)`: ensures no point is exactly on the boundary (avoids conflict
  with the explicit BC loss).
- `.requires_grad_(True)`: tells PyTorch to track derivatives with respect to the
  coordinates x and y. This is what allows `grad(psi, xy)` to compute ∂ψ/∂x.

---

## Section 10 — Boundary Loss (Lines 182–199)

```python
def bc_loss(xy, u_tgt, v_tgt, net):
    psi  = net(xy)[:, 0:1]
    pg   = grad(psi, xy)
    u    = pg[:, 1:2];  v = -pg[:, 0:1]
    return (mse_loss(u - u_tgt.unsqueeze(1)) +
            mse_loss(v - v_tgt.unsqueeze(1)) +
            mse_loss(psi))
```
**Lines 183–189 — Three-term boundary loss.**

The boundary loss has three components:
1. `mse_loss(u − u_tgt)`: penalises horizontal velocity error on this wall.
2. `mse_loss(v − v_tgt)`: penalises vertical velocity error on this wall.
3. `mse_loss(psi)`: enforces ψ = 0 on all walls.

**Why ψ = 0 on walls?**
The stream function ψ is constant on any streamline. The walls are streamlines
(no fluid passes through them). Setting ψ = 0 on all walls is the standard Dirichlet
condition for the stream function in a closed cavity. It also uniquely fixes the
undetermined additive constant in ψ.

`.unsqueeze(1)` reshapes u_tgt from (N,) to (N,1) to match the (N,1) shape of u.

---

```python
def make_bc_xy(xc, yc):
    xy = torch.stack([xc, yc], dim=1)
    xy.requires_grad_(True)
    return xy
```
**Lines 191–194 — Boundary point assembly.**
Takes two 1-D tensors (x-coords and y-coords of N boundary points) and stacks them
into an (N,2) tensor with `requires_grad=True`. The gradient flag is needed because
the stream-function formulation requires differentiating ψ with respect to position
even at boundary points.

---

```python
_anchor = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

def gauge_loss(net):
    return mse_loss(net(_anchor)[:, 1:2])
```
**Lines 196–199 — Pressure gauge fixing.**
Without this, the pressure is only determined up to a constant. The network could
learn p everywhere shifted by +1000 and still satisfy all PDE and BC conditions.
This loss term forces p(0.5, 0.5) = 0 (centre of the domain).

The anchor tensor is created **once** outside the training loop — creating it each
epoch would waste memory and slow things down. `_anchor` is a module-level constant.

---

## Section 11 — Centreline Evaluation (Lines 201–214)

```python
def eval_u(model, n=200):
    yc = np.linspace(0, 1, n)
    xy = torch.tensor(np.stack([np.full(n, 0.5), yc], 1),
                      dtype=torch.float32, device=device).requires_grad_(True)
    u  = grad(model(xy)[:, 0:1], xy)[:, 1:2]
    return yc, u.detach().cpu().numpy().flatten()
```
**Lines 202–207 — u along vertical centreline x = 0.5.**
Creates 200 evenly-spaced points along the vertical line x = 0.5, evaluates the
network, and computes u = ∂ψ/∂y via autograd. `.detach()` removes the tensor from
the computational graph (we don't need gradients for plotting). `.cpu().numpy()` moves
data from GPU memory to a NumPy array for Matplotlib.

---

```python
def eval_v(model, n=200):
    xc = np.linspace(0, 1, n)
    xy = torch.tensor(np.stack([xc, np.full(n, 0.5)], 1), ...)
    v  = -grad(model(xy)[:, 0:1], xy)[:, 0:1]
    return xc, v.detach().cpu().numpy().flatten()
```
**Lines 209–214 — v along horizontal centreline y = 0.5 (NEW in v2).**
Same idea, but evaluates along y = 0.5 and computes v = −∂ψ/∂x (column 0 of the
gradient, negated). This is the second Ghia benchmark comparison that was missing in v1.

---

## Section 12 — Model and Scheduler Setup (Lines 216–224)

```python
net       = PINN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=LR_START)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=LR_END
)
```
**Lines 216–221 — Instantiation.**

- `PINN().to(device)`: creates the network and moves all parameters (weights,
  biases, Fourier buffer B) to GPU/CPU.
- `torch.optim.Adam`: the standard adaptive gradient optimiser. Adam maintains
  per-parameter running estimates of the first moment (mean gradient) and second
  moment (uncentred variance) and normalises updates accordingly. Good for
  noisy losses (new random PDE points each epoch).
- `CosineAnnealingLR`: schedules the learning rate as:
  `lr(t) = LR_END + 0.5*(LR_START − LR_END)*(1 + cos(π*t/T_max))`
  This produces a smooth half-cosine decay from 1e-3 to 5e-5 over 40,000 epochs.

**Why cosine over fixed LR (v1)?**
Fixed LR oscillates around the minimum: too large to settle, too small to converge
fast. Cosine annealing starts aggressive (fast progress), then slows down to let
the optimiser refine the solution precisely. It is a standard technique in deep
learning that typically improves final accuracy by 10–30%.

---

```python
loss_history = []
checkpoints  = {}
```
**Lines 223–224 — Logging containers.**
`loss_history` accumulates one scalar per Adam epoch and one per 10 L-BFGS iterations.
`checkpoints` maps epoch number → deep-copied parameter dict for later plotting.

---

## Section 13 — Adam Training Loop (Lines 226–271)

```python
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
```
**Lines 229–230 — Epoch start.**
`zero_grad()` clears accumulated gradients from the previous step. PyTorch accumulates
gradients by default (useful for gradient accumulation tricks) so this must be called
explicitly each step.

---

```python
    xy_pde = sample_pde_points(N_PDE, WALL_FRAC, WALL_DELTA)
```
**Line 232 — Fresh collocation points each epoch.**
Unlike the L-BFGS phase, Adam resamples PDE points every epoch. This is equivalent
to stochastic mini-batch training — different random samples each step provide
regularisation, prevent the network from overfitting to a fixed point set, and give
a stochastic estimate of the true (infinite-point) PDE residual.

---

```python
    Ru = u*u_x + v*u_y + p_x - NU*(u_xx + u_yy)
    Rv = u*v_x + v*v_y + p_y - NU*(v_xx + v_yy)
    loss_pde = mse_loss(Ru) + mse_loss(Rv)
```
**Lines 237–239 — Navier–Stokes residuals.**

`Ru` is the x-momentum residual. Reading term by term:
- `u*u_x`: convective acceleration, u·∂u/∂x
- `v*u_y`: convective acceleration, v·∂u/∂y
- `p_x`: pressure gradient, ∂p/∂x
- `NU*(u_xx + u_yy)`: viscous diffusion, ν·∇²u

A perfect solution gives Ru = 0 everywhere. The MSE of Ru measures how strongly the
network violates the physics. The network is trained to minimise this.

---

```python
    u_top = 1.0 - (2.0*x_b - 1.0)**16
```
**Line 245 — Smooth lid velocity profile.**
The true lid BC is u = 1 everywhere on y = 1. But at the two top corners (x=0 and
x=1), the lid velocity jumps discontinuously from 1 to 0 (the stationary side walls).
These corner singularities are non-physical and cause the network to struggle
(it tries to learn an infinite gradient).

The smooth profile `1 − (2x−1)^16` equals 1 at x = 0.5, smoothly falls to 0
at x = 0 and x = 1. The exponent 16 makes the transition very sharp, closely
approximating the true lid while eliminating the singularity. This trick was already
in v1 and is kept unchanged.

---

```python
    total = loss_pde + loss_bc + GAUGE_WEIGHT * gauge_loss(net)
    total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()
```
**Lines 254–261 — Backward pass and update.**

1. `total.backward()`: PyTorch traverses the entire computational graph backwards,
   computing ∂total/∂w for every network weight w. This uses the chain rule through
   the Navier–Stokes residuals, through the stream-function derivatives, through the
   residual blocks, through the Fourier embedding — typically 6–8 levels of chain rule.

2. `clip_grad_norm_(..., GRAD_CLIP)` **(NEW in v2)**: computes the global gradient norm
   across all parameters. If it exceeds 1.0, scales all gradients down by the same
   factor. This prevents large early-training updates that could destabilise the
   residual blocks. Added because the wall-clustered points produce much sharper
   gradient signals than uniform sampling.

3. `optimizer.step()`: applies the Adam update to all parameters.

4. `scheduler.step()`: advances the cosine LR schedule by one step.

---

```python
    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss={total.item():.6e} | "
              f"PDE={loss_pde.item():.6e} | BC={loss_bc.item():.6e} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
```
**Lines 268–271 — Progress logging (CHANGED from v1).**
v1 logged every 1,000 epochs. v2 logs every 2,000 to reduce output volume (40k
epochs × 1k-logging = 40 prints vs 20 prints at 2k). Also added `lr=` to confirm
the cosine schedule is decreasing as expected.

---

## Section 14 — L-BFGS Fine-Tuning (Lines 273–318)

```python
xy_pde_l = sample_pde_points(N_PDE, WALL_FRAC, WALL_DELTA)
```
**Line 276 — Fixed collocation set for L-BFGS.**
Unlike Adam (which resamples), L-BFGS uses a **fixed** point set throughout its
entire run. L-BFGS is a quasi-Newton method: it builds a low-rank approximation of
the inverse Hessian from the history of gradient changes. Changing the point set each
step would change the loss function being minimised — the Hessian approximation would
be inconsistent and L-BFGS would fail to converge. Fixed points are mandatory.

---

```python
lbfgs = torch.optim.LBFGS(
    net.parameters(), lr=1.0, max_iter=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=100, line_search_fn='strong_wolfe'
)
```
**Lines 288–292 — L-BFGS configuration (CHANGED from v1).**

- `max_iter=10_000` (was 5,000): doubled to allow more Newton steps.
- `tolerance_grad=1e-8` (was 1e-7): tighter gradient norm stopping criterion.
- `tolerance_change=1e-10` (was 1e-9): tighter loss-change stopping criterion.
- `history_size=100` (was 50): L-BFGS remembers the last 100 gradient pairs
  to approximate the Hessian. More history = better curvature estimate = faster
  convergence. Memory cost: 100 × 2 × N_params gradient vectors.
- `line_search_fn='strong_wolfe'`: at each L-BFGS step, does a 1-D line search
  along the Newton direction, stopping when the Wolfe conditions are satisfied
  (sufficient decrease + curvature condition). This guarantees each step actually
  reduces the loss. Essential for stability of second-order methods.

---

```python
def closure():
    lbfgs.zero_grad()
    ...
    total.backward()
    step_l[0] += 1
    ...
    return total
```
**Lines 295–315 — L-BFGS closure.**
L-BFGS may call the closure multiple times per `.step()` call (for the line search).
The closure must: (1) zero gradients, (2) compute the loss, (3) call backward,
(4) return the loss value. The `step_l` counter is wrapped in a list `[0]` rather than
a plain integer so it can be mutated inside the nested function (Python closure scoping).

---

## Section 15 — Evaluation Grid (Lines 321–337)

```python
N = 200
X, Y = np.meshgrid(np.linspace(0,1,N), np.linspace(0,1,N))
XY_e = torch.tensor(...).requires_grad_(True)
```
**Lines 322–327 — 200×200 evaluation grid.**
After training, we evaluate the solution on a dense 200×200 uniform grid (40,000
points) to create smooth plots. `requires_grad=True` is needed here too because we
must call `grad(psi, XY_e)` to recover u and v.

---

```python
del out_e, psi_e, p_e, pg_e, u_e, v_e, XY_e
```
**Line 337 — Memory cleanup.**
The evaluation tensors with `create_graph=True` attached hold large intermediate
buffers. Explicitly deleting them frees GPU/CPU memory before the plotting phase.

---

## Section 16 — Plots (Lines 339–429)

**2-D flow field (Lines 339–358):**
Four subplots in one figure:
- `contourf(speed)`: filled colour contour of |u| — shows where fast/slow flow is.
- `streamplot(u_pred, v_pred)`: flow lines coloured by speed — visualises the primary
  clockwise vortex and secondary corner vortices.
- `contourf(u_pred)`: horizontal velocity component — should show blue (negative, 
  recirculating) near bottom and red (positive, lid-driven) near top.
- `contourf(v_pred)`: vertical velocity — peaks near the side walls.

---

**Loss curve (Lines 360–366):**
`plt.yscale("log")` is essential — the loss spans several orders of magnitude and
a linear scale would compress the interesting convergence behaviour near the end.

---

**Benchmark comparison plots (Lines 368–413):**

The loop iterates over all saved checkpoints **plus** the L-BFGS Final state,
loading each into a fresh `tmp = PINN()` model, evaluating the centreline,
and plotting. This shows how accuracy improves throughout training.

- Earlier checkpoints (Step 5000, 10000) will likely be far from Ghia's dots.
- The L-BFGS Final line should lie closest to the benchmark.
- If they all miss badly, it means the physics is not being learned — indicating
  a bug or a hyperparameter that needs adjustment.

The scatter plot `ax_u.scatter(GHIA_U, GHIA_Y, ...)` overlays the 17 Ghia reference
points as black dots. Good agreement means the PINN is correctly solving the Navier–
Stokes equations.

---

**3-D surface plots (Lines 415–429):**
`ax.plot_surface(X, Y, field)` renders the scalar field as a height map. Gives
intuition about the 3-D shape of the pressure and velocity fields — e.g. the pressure
peak at the top-right corner and the trough at the top-left should be clearly visible.

---

## Summary: What Changed and Why (v1 → v2)

| What | v1 | v2 | Why |
|---|---|---|---|
| Architecture | Plain 5-layer MLP, width 64 | FourierEmbed + 6 ResBlocks, width 128 | Spectral bias; vanishing gradients |
| Collocation | Chebyshev-cosine uniform | 25% near-wall clustered + 75% Chebyshev | Thin boundary layers at Re=400 need resolution |
| Training epochs | 25k Adam + 5k L-BFGS | 40k Adam + 10k L-BFGS | More passes needed for harder Re |
| Learning rate | Fixed 1e-3 | Cosine 1e-3 → 5e-5 | Avoids oscillating around minimum |
| Gradient clipping | None | Norm clip at 1.0 | Wall points create large gradient spikes |
| BC weight | 100 | 50 | PDE was being ignored; balance restored |
| Collocation count | 10k | 15k | Compensates for the wall-cluster reallocation |
| BC count | 500 | 800 | Better sampling of the smooth lid profile |
| Benchmarks | u centreline only | u + v centrelines | Stricter two-component validation |
| L-BFGS history | 50 | 100 | Better Hessian approximation |
| LR in logs | Not shown | Shown | Diagnose schedule issues |
