"""
Boundary-condition profile helpers and the 2-D enthalpy-porosity PCM solver.

Extracted VERBATIM from the final 2-D dataset-generation notebook
(`2D/02_u_deeponet/code/data_generation/udeeponet_dataset_generation_2d.ipynb`)
so that the shared physics used by every stage of this project can be read in one
place. Nothing in the numerical scheme, the equations or the parameters has been
changed - only the surrounding notebook scaffolding was removed and the imports
were made explicit.

The notebooks in this repository remain self-contained and do NOT import this
module; it exists as a readable reference copy.
"""

import numpy as np
import matplotlib.pyplot as plt

def _const_profile(val):
    """returns a function f(s) -> constant temperature array"""
    return lambda s: np.full_like(s, float(val), dtype=float)

def _gauss_profile(base, amp, mu, sigma, axis_len):
    """
    1D Gaussian along coordinate s in [0, axis_len]:
    T(s) = base + amp * exp(-0.5 * ((s - mu*axis_len)/(sigma*axis_len))**2)
    mu and sigma are given in 0..1 (relative position / width).
    """
    mu_abs = mu * axis_len
    sig_abs = max(1e-12, sigma * axis_len)
    return lambda s: base + amp * np.exp(-0.5 * ((s - mu_abs)/sig_abs)**2)

def _make_profile(side_cfg, axis_array, axis_len):
    """
    side_cfg: dict like {"on": True/False, "type": "const"/"gauss"/"custom", **params}
    axis_array: x (for top/bottom) or y (for left/right)
    axis_len: Lx (for top/bottom) or Ly (for left/right)
    returns: (on_bool, array_of_temperatures) or (False, None) if OFF
    """
    on = bool(side_cfg.get("on", False))
    if not on:
        return False, None

    typ = side_cfg.get("type", "const").lower()
    if typ == "const":
        prof = _const_profile(side_cfg.get("T", 300.0))
    elif typ == "gauss":
        prof = _gauss_profile(
            base=float(side_cfg.get("base", 330.0)),
            amp=float(side_cfg.get("amp", 20.0)),
            mu=float(side_cfg.get("mu", 0.5)),
            sigma=float(side_cfg.get("sigma", 0.2)),
            axis_len=axis_len,
        )
    elif typ == "custom":
        func = side_cfg.get("func", None)
        if not callable(func):
            raise ValueError("bc 'custom' requires a callable 'func(s_array) -> T_array'")
        prof = func
    else:
        raise ValueError(f"Unknown bc type: {typ}")

    return True, prof(axis_array)

def simulate_pcm_2d_with_source(
    Q_Wm3,
    nx=128, ny=128, Lx=0.05, Ly=0.05,
    rho=800.0, cp=2000.0, k=0.2, L_lat=2e5, Tm=330.0,
    T_init=300.0, Tb=300.0,
    t_end=2500.0, cfl=0.45,
    save_times=(10.0, 60.0, 300.0, 600.0, 1200.0, 2000.0, 2500.0),
    bc=None,
    battery_mask=None,
):
    """
    Q_Wm3: [nx,ny] in (x,y) indexing; x is axis-0, y is axis-1.
    battery_mask: [nx,ny] with 1 in battery region.
      - We enforce: f=0 in battery region (no melting there).
      - Temperature still evolves by conduction/enthalpy there (solid-like region).
    """

    # grid
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")  # [nx,ny]

    # fields
    T = np.full((nx, ny), float(T_init), dtype=np.float64)
    f = np.zeros_like(T)
    H = cp*T + f*L_lat

    # default BCs (same behavior as your notebook if bc None)
    if bc is None:
        bc = {
            "left":   {"on": True, "type": "const", "T": Tb},
            "right":  {"on": True, "type": "const", "T": Tb},
            "bottom": {"on": True, "type": "const", "T": Tb},
            "top":    {"on": True, "type": "const", "T": Tb},
        }

    # battery mask
    if battery_mask is not None:
        battery_mask = (np.asarray(battery_mask) > 0.5)
    else:
        battery_mask = None

    # axis arrays for boundary profiles
    xs = X[:, 0]   # along bottom/top edges (x coordinate)
    ys = Y[0, :]   # along left/right edges (y coordinate)

    # timestep (2D FTCS stability-ish)
    alpha = k/(rho*cp)
    dt = cfl * min(dx*dx, dy*dy) / (4.0*alpha)

    save_times = np.array(sorted(save_times), dtype=float)
    Ts, Fs, ts = [], [], []
    next_k = 0
    t = 0.0

    def apply_BCs(Tarr, Harr):
        # LEFT
        on, arr = _make_profile(bc.get("left", {}), ys, Ly)
        if on:
            Tarr[0, :] = arr
            Harr[0, :] = cp*Tarr[0, :] + f[0, :]*L_lat
        else:
            Tarr[0, :] = Tarr[1, :]  # adiabatic

        # RIGHT
        on, arr = _make_profile(bc.get("right", {}), ys, Ly)
        if on:
            Tarr[-1, :] = arr
            Harr[-1, :] = cp*Tarr[-1, :] + f[-1, :]*L_lat
        else:
            Tarr[-1, :] = Tarr[-2, :]

        # BOTTOM
        on, arr = _make_profile(bc.get("bottom", {}), xs, Lx)
        if on:
            Tarr[:, 0] = arr
            Harr[:, 0] = cp*Tarr[:, 0] + f[:, 0]*L_lat
        else:
            Tarr[:, 0] = Tarr[:, 1]

        # TOP
        on, arr = _make_profile(bc.get("top", {}), xs, Lx)
        if on:
            Tarr[:, -1] = arr
            Harr[:, -1] = cp*Tarr[:, -1] + f[:, -1]*L_lat
        else:
            Tarr[:, -1] = Tarr[:, -2]

    # initial BC
    apply_BCs(T, H)

    while t < t_end + 1e-12:
        apply_BCs(T, H)

        # Laplacian (interior)
        Lap = np.zeros_like(T)
        Lap[1:-1, 1:-1] = (
            (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
            (T[1:-1, 2:]   - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
        )

        # Enthalpy update (interior only)
        H[1:-1, 1:-1] += dt * ((k/rho) * Lap[1:-1, 1:-1] + Q_Wm3[1:-1, 1:-1] / rho)

        # Phase update:
        # - normal PCM everywhere, BUT force f=0 in battery cells (no melt inside batteries)
        if battery_mask is None:
            f_new = (H - cp*Tm) / L_lat
            mush  = (f_new > 0.0) & (f_new < 1.0)
            solid = (f_new <= 0.0)
            liq   = (f_new >= 1.0)

            T[mush]  = Tm
            f[mush]  = f_new[mush]
            T[solid] = H[solid]/cp
            f[solid] = 0.0
            T[liq]   = (H[liq]-L_lat)/cp
            f[liq]   = 1.0
        else:
            pcm = ~battery_mask

            # update only PCM region as phase change material
            f_new_pcm = (H[pcm] - cp*Tm) / L_lat
            mush  = (f_new_pcm > 0.0) & (f_new_pcm < 1.0)
            solid = (f_new_pcm <= 0.0)
            liq   = (f_new_pcm >= 1.0)

            # defaults in PCM: solid relation
            T[pcm] = H[pcm]/cp
            f[pcm] = 0.0

            pcm_idx = np.flatnonzero(pcm.ravel())

            mush_idx = pcm_idx[mush]
            liq_idx  = pcm_idx[liq]

            # mush
            T.ravel()[mush_idx] = Tm
            f.ravel()[mush_idx] = f_new_pcm[mush]

            # liquid
            T.ravel()[liq_idx] = (H.ravel()[liq_idx] - L_lat)/cp
            f.ravel()[liq_idx] = 1.0

            # battery region (no melt)
            T[battery_mask] = H[battery_mask]/cp
            f[battery_mask] = 0.0

        apply_BCs(T, H)

        # Save snapshots at requested times
        if next_k < len(save_times) and t >= save_times[next_k] - 0.5*dt:
            Ts.append(T.copy())
            Fs.append(f.copy())
            ts.append(t)
            next_k += 1

        t += dt

    return np.array(Ts), np.array(Fs), np.array(ts), X, Y, {"dt": dt}
