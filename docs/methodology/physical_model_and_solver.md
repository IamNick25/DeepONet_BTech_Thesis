# Physical model and reference solver

Every dataset in this repository is produced by the same family of solvers: an **enthalpy-porosity**
formulation of melting in a PCM–metal-foam composite, integrated explicitly in time. This note collects
the formulation in one place; the code is in
[`common/physics/enthalpy_solver_2d.py`](../../common/physics/enthalpy_solver_2d.py) (2D, extracted
verbatim from the final dataset generator) and inside the 3D generation notebook
([`3D/code/data_generation/`](../../3D/code/data_generation/)) for the three-dimensional case.

## 1. The composite

PCM infiltrated into an open-cell metal foam is modelled with **volume-averaged effective properties**
rather than a resolved pore geometry. The foam raises the effective conductivity and lowers the
effective latent-heat capacity per unit volume; its geometry is not meshed.

Composite enthalpy:

```
H = [ C_p,pcm (1 − φ) + C_p,m φ ] T  +  (1 − φ) f_l L
```

with `φ` the foam porosity, `C_p,pcm` and `C_p,m` the specific heats of the PCM and the metal, `f_l` the
liquid fraction (0 solid, 1 liquid) and `L` the latent heat of fusion.

Liquid-fraction update:

```
f_l,new = ( H − [ C_p,pcm (1 − φ) + C_p,m φ ] T_m ) / ( (1 − φ) L )
```

Energy equation with a volumetric source:

```
∂H/∂t = k_avg ∇²T + Q
```

## 2. Time integration

Explicit FTCS. The time step is set from the diffusive stability limit,

```
α  = k / (ρ c_p)
dt = CFL · min(dx², dy²) / (4 α)          (2D;  the 3D solver uses the analogous 3D bound)
```

with CFL = 0.45 in 2D and 0.30 in 3D.

Each step:

1. apply boundary conditions to `T` and re-synchronise `H` on Dirichlet walls;
2. compute the interior Laplacian of `T`;
3. advance interior enthalpy, `H += dt · ( (k/ρ)∇²T + Q/ρ )`;
4. recover `T` and `f` from `H` through the three-branch enthalpy update;
5. re-apply boundary conditions;
6. store a snapshot if the current time has reached the next requested save time.

## 3. The enthalpy update

Given a trial fraction `f* = (H − c_p T_m)/L`:

| Regime | Condition | Temperature | Liquid fraction |
|---|---|---|---|
| Solid | `f* ≤ 0` | `T = H / c_p` | `f = 0` |
| Mushy | `0 < f* < 1` | `T = T_m` | `f = f*` |
| Liquid | `f* ≥ 1` | `T = (H − L) / c_p` | `f = 1` |

This is what pins the temperature to `T_m` across the mushy zone and produces the sharp interface that
the surrogates then have to reproduce — and it is the reason the dominant surrogate error is a
*front-position* error rather than a magnitude error.

## 4. Battery cells

Where a `battery_mask` is supplied, those cells are **excluded from melting**: their phase fraction is
pinned to `f = 0` while their temperature continues to evolve from the enthalpy, so the battery behaves
as a solid heat-generating inclusion embedded in a melting medium.

The 3D solver treats the battery explicitly, with three selectable modes:

- **`lumped`** (used for the 3D dataset) — the battery temperature follows a lumped energy balance. The
  net conduction leaving the battery surface is computed from the *same* temperature field used in the
  Laplacian, so the energy the battery loses is exactly the energy the PCM gains.
- **`dirichlet`** — the battery is held at a fixed temperature `T_BAT_FIXED`.
- **`solid_2d`** — the phase fraction is pinned and the temperature evolves from enthalpy, without a
  lumped balance.

The battery must sit strictly inside the interior (`BAT_MARGIN_M` clearance); otherwise a Dirichlet face
would overwrite battery cells and break the energy balance, and the solver raises an error.

## 5. Boundary conditions

Each wall (2D) or face (3D) is independently either:

- **on, constant** — Dirichlet at a fixed temperature;
- **on, Gaussian** — Dirichlet with a 1D profile `T(s) = base + amp · exp(−½((s − μL)/(σL))²)`
  (2D), or a constant / 1D-Gaussian-along-either-axis / 2D-Gaussian surface (3D);
- **off** — adiabatic, implemented by copying the adjacent interior value.

Sampling ranges used for the datasets: `μ ∈ [0.25, 0.8]`, `σ ∈ [0.12, 0.7]`, `amp ∈ [12, 80]` K,
deterministically seeded per case index so that any case can be regenerated exactly.

## 6. Heat-source generation

Volumetric sources come from three generators
([`common/physics/heat_source_gp.py`](../../common/physics/heat_source_gp.py) for the first):

**Gaussian process.** A squared-exponential (RBF) kernel on the grid,

```
k(x, x') = σ² exp( −‖x − x'‖² / (2 l²) )
```

conditioned on boundary values (zero by default), sampled through a Cholesky factorisation with
adaptive jitter. An optional softplus forces the field positive, and the mean can be retargeted to a
prescribed `q_scale`. Length scale 0.18 and σ = 1 were used throughout, with `q_scale = 1e5` W/m³.

**Battery footprints.** 2–3 axis-aligned or rotated rectangles, circles or ellipses (equal
probability), rendered by 6× supersampling with anti-aliasing and a 3×3 average blur so the edges are
not a single-pixel step, at `1.0–1.2 × 10⁶` W/m³ on a 4.5 W/m³ background. Placement is rejection-
sampled to respect a 3 mm wall clearance and 2 mm inter-cell separation.

**Hybrid.** A weighted sum of a GP field and a battery field, weights drawn from `[0.4, 0.7]` and
`[0.3, 0.6]` respectively.

## 7. Baseline parameters

| Quantity | 2D | 3D |
|---|---|---|
| Grid | 120 × 120 (40 × 40 in the pointwise stage) | 24 × 24 × 24 |
| Domain | 0.05 m × 0.05 m | 0.05 m cube |
| `ρ` | 800 kg/m³ | 800 kg/m³ |
| `c_p` | 2000 J/(kg·K) | 2000 J/(kg·K) |
| `k` | 0.2 W/(m·K) | 0.2 W/(m·K) |
| `L` | 2 × 10⁵ J/kg | 2 × 10⁵ J/kg |
| `T_m` | 330 K | 330 K |
| `T_init` | 300 K | 300 K |
| `T_bound` | 330 K | 330 K |
| `t_end` | 4000 s | 4000 s |
| CFL | 0.45 | 0.30 |
| Snapshots | 24 (9 in the pointwise stage) | 24 |

## 8. A note on the thesis notation

The thesis writes the governing equations with the porosity `φ` and the separate PCM and metal specific
heats explicit (equations 1–3 of Chapter 3). The implemented solver uses single effective `ρ`, `c_p`,
`k` and `L` values, which is the same model with the volume averaging already folded into the
constants. The reported values are the effective ones. This is noted in
[`../repository_audit.md`](../repository_audit.md) under *Potential issues* as something worth checking
if the effective properties ever need to be traced back to a specific porosity and foam material.
