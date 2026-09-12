# DeepONet architectures used in this project

Three architectures appear across the three stages. They share the branch–trunk idea and differ in how
the branch consumes the input function and how the output is assembled.

## 0. The operator-learning premise

A DeepONet approximates an operator `G : u ↦ G(u)` by factorising it into a **branch** network that
encodes the input function `u` (sampled at fixed sensor locations) and a **trunk** network that encodes
the query coordinate `y`, combined so that

```
G(u)(y) ≈ Σ_k  b_k(u) · t_k(y)  +  bias
```

The practical consequence for this project is that one trained model serves *any* heat-source field and
*any* boundary configuration in the sampled family, without retraining — which is what makes parametric
design sweeps affordable.

---

## 1. Minimal DeepONet — 1D stage

```
T̂(x | Q) = ⟨ branch(Q), trunk(x) ⟩ + b
```

| Component | Configuration |
|---|---|
| Branch | 2 hidden layers × 128, ReLU |
| Trunk | 2 hidden layers × 128, ReLU |
| Head | inner product + learned bias |

Training: 500 pairs, batch 512, 200 epochs, Adam @ 1e-3, MSE.

Documented from the thesis; the implementing code is not in the archive. See
the *Stage 1* section of the root [`README.md`](../../README.md#stage-1--1d-pointwise-deeponet-feasibility).

---

## 2. Four-network pointwise DeepONet — 2D stage 2a

The 2D problem depends on more than one input function, so the branch and trunk are each split in two:

| Sub-network | Encodes | Hidden layers | Output |
|---|---|---|---|
| Q-branch | flattened heat-source field | (256, 256, 256) | `D = 256` |
| BC-branch | boundary feature vector | (256, 256, 256) | `D = 256` |
| XY-trunk | `(x, y)` normalised to `[0, 1]` | (256, 256, 256, 256) | `D = 256` |
| T-trunk | `t` scaled to `[0, 1]` | (128, 128) | `D = 256` |

Each output passes through a LayerNorm, then two fusion paths are combined:

```
y_concat = Head([e_Q, e_BC, e_XY, e_t])            Head: MLP(4D → 256 → 128 → 1), GELU
y_prod   = Σ (e_Q ⊙ e_BC ⊙ e_XY ⊙ e_t) / √D

ŷ = α_add · y_concat + α_prod · y_prod + b
```

`α_add` and `α_prod` are **learned** scalars (initialised 1.0 and 0.2). The multiplicative path is the
classical DeepONet inner product generalised to four factors; the additive path is an expressive MLP
head over the concatenation. Keeping both, with learned weights, lets the model choose how much
structure to impose.

Dropout 0.3 in the MLP blocks; activation GELU throughout.

**Cost.** Every `(x, y, t)` is an independent sample: 40 × 40 × 9 = 14,400 per simulation, ≈ 2.9 M for
200 simulations. Batches of 16,384 pooled points, 64 steps per epoch, 60 epochs, Adam @ 1e-3.

---

## 3. U-DeepONet — 2D stage 2b (and the basis of stage 3)

Following Diab and Al-Kobaisi (2024). The branch becomes a U-Net acting on the input **grid**, so one
simulation is one training sample:

```
X_static [B, C_in, H, W]
   │  1×1 Conv  (lift C_in → f)
   ├─ UNetBlock2D(f) ── LeakyReLU
   ├─ UNetBlock2D(f) ── LeakyReLU
   └─ UNetBlock2D(f)                       →  x [B, f, H, W]

times [T] ── TrunkTimeMLP (10 layers, width 128, sin) →  t_feat [T, f]

xt = x[:, None, :, :, :] * t_feat[None, :, :, None, None]      # [B, T, f, H, W]
ŷ  = Conv3d_{1×1×1}(f → 1) ( ReLU( Conv3d_{1×1×1}(f → f)(xt) ) )  # [B, T, H, W]
```

### The U-Net block (constant width `f`, LeakyReLU(0.2), BatchNorm)

| Step | Operation | Stride | Skip |
|---|---|---|---|
| c1 | Conv 3×3, f→f | 2 | → u2 |
| c2 | Conv 3×3, f→f | 2 | |
| c3 | Conv 3×3, f→f | 1 | → u1 |
| c4 | Conv 3×3, f→f (bottleneck) | 2 | |
| c5 | Conv 3×3, f→f (bottleneck) | 1 | |
| u1 | ConvTranspose 4×4, f→f, concat c3 | 2 | |
| u2 | ConvTranspose 4×4, 2f→f, concat c1 | 2 | |
| u3 | ConvTranspose 4×4, 2f→f, concat input | 2 | |
| out | Conv 3×3, 2f→f | 1 | |

Three downsamplings means the input height and width must be multiples of 8; the generator pads
accordingly (no padding is needed at `N = 120`).

### Input channels (2D, `C_in = 13`)

```
Q, battery_mask, x, y,
left_val, left_mask, right_val, right_mask,
bottom_val, bottom_mask, top_val, top_mask,
ones
```

Encoding each wall as a *value* map plus a *mask* map lets one network handle constant, Gaussian and
adiabatic walls without changing its input signature.

### Why the outer product

The branch is independent of `t` and the trunk is independent of `(x, y)`. The separable outer product
is what lets a single forward pass emit **all 24 time snapshots at once** — the property responsible
for the ~0.07 s inference time. The cost is that space–time couplings which do not factorise cannot be
represented; for diffusion-driven melting this has not been limiting.

Hyper-parameters: `f = 64`, trunk width 128, trunk depth 10, `sin` activation, dropout 0.0,
Adam @ 1e-4, weight decay 1e-4, batch 16 cases, 60 epochs, MSE on normalised fields.

Normalisation uses **training-split statistics only**: inputs standardised per channel over
(case, H, W); targets standardised with a single scalar mean/std; times scaled by `t / max(t)`.

---

## 4. Unified slice U-Net — 3D stage

Same U-Net encoder, three changes that matter.

```
X_window [B, 13, 18, H, W]        13 neighbouring slices, 18 channels each
   for each slice j:  encode_one_slice(X_j) → feat_j [B, 64, H, W]     (shared weights)
   slice fusion: concat(feat_j) [B, 13·64, H, W] ─1×1 Conv + BN + act─▶ feat [B, 64, H, W]

times [B, T] ── Fourier features (1 + 2·8 dims) ── MLP(→64→64) ──▶ t_emb [B, T, 64]

fuse: concat(feat broadcast over T, t_emb broadcast over H,W) → [B·T, 128, H, W]
      Conv3×3+BN+act → Conv3×3+BN+act → Conv1×1 → ŷ [B, T, H, W]
```

**(a) A 13-slice window instead of one plane.** The encoder is applied with shared weights to each
slice in a window of radius 6 around the target slice, and the features are fused by a 1×1 convolution.
This is how out-of-plane structure re-enters a model that only ever performs 2D convolutions. Window
indices are clamped or reflected at the stack boundaries.

**(b) Fourier time features.** `t ↦ [t, sin(2^k π t), cos(2^k π t)]` for `k = 0…7`, then a small MLP.
Higher-frequency components give the network a sharper handle on the rapid early transient than a plain
coordinate trunk.

**(c) Convolutional space–time fusion instead of an outer product.** Time embeddings are broadcast
spatially and concatenated with the spatial features, then fused by two 3×3 convolutions. This is
strictly more expressive than the separable 2D form, at the cost of a forward pass per time step.

**(d) 18 input channels per slice**: `Q`, three coordinate maps, six boundary-face value maps, six
boundary-**distance** maps, the battery mask and a ones channel. The distance channels are essential —
they are what tells an otherwise identical-looking 2D image whether it is a mid-plane or a near-wall
slice.

**(e) A structured loss.** Beyond MSE:

| Term | Weight | Purpose |
|---|---|---|
| Temporal smoothness | 1e-4 | penalise jitter between consecutive predicted snapshots |
| Gradient matching | 0.15 | match spatial gradients, not just values — keeps fronts sharp |
| Interface weighting (`f` only) | 0.2 | Gaussian emphasis around `f = 0.5` (σ = 0.15, α = 2.0) |

The interface term targets the dominant failure mode identified in 2D: error concentrated on the
solid–liquid front. Up-weighting cells near `f = 0.5` spends optimisation capacity where it matters
rather than on the large, easy, fully-solid or fully-liquid regions.

Hyper-parameters: base width 64, time dim 64, 8 frequencies, window 13, Adam @ 1e-3, weight decay 1e-4,
batch 128 slice samples, 60 epochs configured.

---

## 5. Summary of the progression

| | 1D | 2D pointwise | 2D U-DeepONet | 3D slice |
|---|---|---|---|---|
| Branch input | `Q(x)` vector | flattened `Q`, BC vector | 13-channel grid | 13 × 18-channel slice window |
| Trunk input | `x` | `(x, y)` and `t` | `t` | `t` (Fourier) |
| Sample granularity | one function | one space–time point | one simulation | one slice |
| Output per forward pass | one profile | one scalar | all 24 snapshots | all 24 snapshots for one slice |
| Combination | inner product | additive + multiplicative, learned weights | outer product + 1×1×1 Conv3D | convolutional fusion |
| Parameters that matter | 128-wide MLPs | `D = 256` | `f = 64`, trunk depth 10 | `f = 64`, window 13 |
