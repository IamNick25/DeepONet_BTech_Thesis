"""
Gaussian-process (RBF-kernel) volumetric heat-source generator, 2-D.

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

class HeatSource2DRBFPosterior:
    """
    Generate 2D heat-source fields Q(x,y) via a Gaussian-process posterior
    with an RBF (squared-exponential) kernel on a square grid.
    """

    def __init__(self, grid_size=64, length=1.0,
                 length_scale=0.20, sigma=1.0,
                 jitter=1e-6, seed=None):
        self.N = int(grid_size)
        self.L = float(length)
        self.l = float(length_scale)
        self.sigma = float(sigma)
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(seed)

        x = np.linspace(0.0, self.L, self.N, dtype=np.float64)
        y = np.linspace(0.0, self.L, self.N, dtype=np.float64)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.points = np.column_stack((self.X.ravel(), self.Y.ravel()))

        # Boolean mask of boundary grid points
        self.boundary_mask = (
            np.isclose(self.X, 0.0) | np.isclose(self.X, self.L) |
            np.isclose(self.Y, 0.0) | np.isclose(self.Y, self.L)
        ).ravel()

    # RBF (Gaussian) kernel
    def rbf_kernel(self, X1, X2):
        d2 = np.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
        return (self.sigma**2) * np.exp(-0.5 * d2 / (self.l**2))

    # Robust Cholesky with adaptive jitter
    def safe_cholesky(self, K, max_tries=8):
        jitter = self.jitter
        I = np.eye(K.shape[0], dtype=K.dtype)
        for _ in range(max_tries):
            try:
                return np.linalg.cholesky(K + jitter * I)
            except np.linalg.LinAlgError:
                jitter *= 10.0
        raise np.linalg.LinAlgError(f"Cholesky failed; final jitter tried={jitter:g}")

    @staticmethod
    def _softplus(x, beta=6.0):
        # smooth nonnegative map; larger beta -> closer to max(0,x)
        return (1.0/beta) * np.log1p(np.exp(beta*x))

    def sample_posterior(
        self,
        Qb_func=None,
        enforce_zero_mean=True,
        *,
        force_sign=None,            # None | 'positive' | 'negative'
        pos_beta=6.0,               # sharpness for softplus
        target_mean=None,           # set desired mean after transform (e.g., 5e4)
        keep_boundary_zero=True     # keep Q=0 on the boundary after transform
    ):
        """
        Draw a sample Q from the GP posterior conditioned on boundary values Q_b.

        Parameters
        ----------
        Qb_func : callable or None
            Function returning boundary values Q_b at boundary coordinates;
            if None, boundary is conditioned to Q=0 in distribution.
        enforce_zero_mean : bool
            If True and force_sign is None, recenter to zero mean.
        force_sign : None | 'positive' | 'negative'
            Enforce all-positive or all-negative field using a smooth transform.
        pos_beta : float
            Softplus sharpness for positivity; 5–8 is typical.
        target_mean : float or None
            If given, shift the final field so mean(Q) == target_mean.
        keep_boundary_zero : bool
            If True, zero out the boundary *after* any sign/mean transforms.

        Returns
        -------
        Q : (N,N) ndarray
        """
        X_all = self.points
        X_b = X_all[self.boundary_mask]

        # Boundary values for Q
        if Qb_func is None:
            Q_b = np.zeros(len(X_b), dtype=np.float64)
        else:
            Q_b = np.asarray(Qb_func(X_b), dtype=np.float64)

        # Posterior mean and covariance
        K_xx = self.rbf_kernel(X_all, X_all).astype(np.float64)
        K_xb = self.rbf_kernel(X_all, X_b).astype(np.float64)
        K_bb = self.rbf_kernel(X_b, X_b).astype(np.float64) + self.jitter * np.eye(len(X_b))

        mu = K_xb @ np.linalg.solve(K_bb, Q_b)
        K_post = K_xx - K_xb @ np.linalg.solve(K_bb, K_xb.T)

        # Sample from posterior (Gaussian)
        Lp = self.safe_cholesky(K_post)
        z = self.rng.standard_normal(self.N * self.N)
        Q = (mu + Lp @ z).reshape(self.N, self.N)

        # Enforce sign if requested
        if force_sign is None:
            if enforce_zero_mean:
                Q -= Q.mean()
        elif force_sign == 'positive':
            Q = self._softplus(Q, beta=pos_beta)  # strictly >= 0
        elif force_sign == 'negative':
            Q = -self._softplus(Q, beta=pos_beta) # strictly <= 0
        else:
            raise ValueError("force_sign must be None, 'positive', or 'negative'")

        # Optional mean targeting *after* transform
        if target_mean is not None:
            Q += (target_mean - Q.mean())

        # Optionally keep boundary exactly zero (useful for coupling to PDE BCs)
        if keep_boundary_zero:
            Q.ravel()[self.boundary_mask] = 0.0

        return Q

    @staticmethod
    def show(Q, title="Heat source Q(x,y)", cmap="RdBu_r"):
        plt.figure(figsize=(5.0, 4.5), dpi=120)
        im = plt.imshow(Q.T, origin="lower", cmap=cmap, aspect="equal")
        plt.colorbar(im, shrink=0.9, label="Q")
        plt.title(title)
        plt.xlabel("x index")
        plt.ylabel("y index")
        plt.tight_layout()
        plt.show()

