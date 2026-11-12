"""Task-parameterized Gaussian mixture model (TP-GMM), implemented in numpy.

Follows Calinon (2016), *A tutorial on task-parameterized movement learning and
retrieval*. Each demonstration is observed from P task frames {A_j, b_j}; the model
holds, for every component k, one Gaussian per frame N(mu_kj, S_kj) and a shared prior.

* **EM:** the responsibility of component k for a datapoint is
  pi_k prod_j N(xi^(j) | mu_kj, S_kj); the M-step updates each frame's
  Gaussian with those shared responsibilities, plus a per-dimension diagonal
  regularization (units differ: time, metres, radians, gripper 0-100).
* **Initialization:** k-means along time (K equal-mass time bins).
* **Model selection:** BIC = -2 LL + n_params log N.
* **Reproduction:** for new frames, each component's local Gaussians are mapped to
  the global frame (A mu + b, A S A^T) and multiplied (product of Gaussians); then
  Gaussian mixture regression on time gives the trajectory and its covariance.

Angular variables: the frame transform of an angle is a pure offset (A = 1, b = frame
yaw). Before the product, angular means of all frames are aligned to the first one
modulo 2*pi so a 2*pi-shifted copy cannot spoil the product.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from so101_tpgmm.tpgmm.gmm import GMM, gaussian_logpdf, gaussian_product, gmr, logsumexp


@dataclass
class TaskFrame:
    """Affine task frame in the augmented variable space: global = A @ local + b."""

    A: np.ndarray  # (D, D)
    b: np.ndarray  # (D,)

    def to_local(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.A, (np.atleast_2d(X) - self.b).T).T


@dataclass
class TPGMM:
    """TP-GMM with ``n_frames`` frames and ``dim`` variables (dimension 0 = time input)."""

    n_components: int
    n_frames: int
    dim: int
    reg: np.ndarray  # (D,) diagonal regularization (variances)
    angular_dims: tuple[int, ...] = ()
    priors: np.ndarray | None = None  # (K,)
    means: np.ndarray | None = None  # (K, P, D)
    covs: np.ndarray | None = None  # (K, P, D, D)
    log_likelihood: float = float("nan")
    n_iter: int = 0
    n_data: int = 0
    var_names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ EM
    def _init_time_kmeans(self, X: np.ndarray) -> None:
        """K equal-mass bins along time (dimension 0, identical in all frames)."""
        t = X[0][:, 0]
        edges = np.quantile(t, np.linspace(0, 1, self.n_components + 1))
        lab = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, self.n_components - 1)
        K, P, D = self.n_components, self.n_frames, self.dim
        self.priors = np.full(K, 1.0 / K)
        self.means = np.zeros((K, P, D))
        self.covs = np.zeros((K, P, D, D))
        for k in range(K):
            m = lab == k
            for j in range(P):
                Xk = X[j][m]
                self.means[k, j] = Xk.mean(axis=0)
                self.covs[k, j] = np.cov(Xk.T) + np.diag(self.reg)

    def _log_resp(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[1]
        L = np.tile(np.log(self.priors + 1e-300), (N, 1))
        for k in range(self.n_components):
            for j in range(self.n_frames):
                L[:, k] += gaussian_logpdf(X[j], self.means[k, j], self.covs[k, j])
        return L

    def fit(self, X: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> "TPGMM":
        """Fit on local data ``X`` of shape (P, N, D) (all demonstrations concatenated)."""
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] != self.n_frames or X.shape[2] != self.dim:
            raise ValueError(f"Expected (P={self.n_frames}, N, D={self.dim}), got {X.shape}")
        self.n_data = X.shape[1]
        self._init_time_kmeans(X)
        prev = -np.inf
        for it in range(max_iter):
            L = self._log_resp(X)
            norm = logsumexp(L, axis=1)
            ll = float(norm.sum())
            G = np.exp(L - norm[:, None])  # (N, K)
            Nk = G.sum(axis=0) + 1e-12
            self.priors = Nk / Nk.sum()
            for k in range(self.n_components):
                w = G[:, k] / Nk[k]
                for j in range(self.n_frames):
                    mu = w @ X[j]
                    d = X[j] - mu
                    self.means[k, j] = mu
                    self.covs[k, j] = (d * w[:, None]).T @ d + np.diag(self.reg)
            self.n_iter = it + 1
            if abs(ll - prev) <= tol * abs(ll):
                break
            prev = ll
        self.log_likelihood = float(logsumexp(self._log_resp(X), axis=1).sum())
        return self

    @property
    def n_params(self) -> int:
        K, P, D = self.n_components, self.n_frames, self.dim
        return (K - 1) + K * P * (D + D * (D + 1) // 2)

    def bic(self) -> float:
        return -2.0 * self.log_likelihood + self.n_params * np.log(self.n_data)

    # ------------------------------------------------------------------ reproduction
    def product(self, frames: list[TaskFrame]) -> GMM:
        """Global GMM for new task frames (product of the frame-wise Gaussians)."""
        if len(frames) != self.n_frames:
            raise ValueError("One TaskFrame per model frame is required.")
        K, D = self.n_components, self.dim
        means, covs = np.zeros((K, D)), np.zeros((K, D, D))
        for k in range(K):
            mus, Ss = [], []
            for j, F in enumerate(frames):
                mus.append(F.A @ self.means[k, j] + F.b)
                Ss.append(F.A @ self.covs[k, j] @ F.A.T)
            for a in self.angular_dims:  # align angles modulo 2*pi before multiplying
                for j in range(1, len(mus)):
                    mus[j][a] += 2 * np.pi * np.round((mus[0][a] - mus[j][a]) / (2 * np.pi))
            means[k], covs[k] = gaussian_product(mus, Ss)
        return GMM(self.priors.copy(), means, covs)

    def reproduce(self, frames: list[TaskFrame], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, D-1) mean and (N, D-1, D-1) covariance of the outputs at times ``t``."""
        g = self.product(frames)
        return gmr(g, np.asarray(t, dtype=np.float64)[:, None], [0], list(range(1, self.dim)))

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "n_components": self.n_components, "n_frames": self.n_frames, "dim": self.dim,
            "reg": self.reg.tolist(), "angular_dims": list(self.angular_dims),
            "priors": self.priors.tolist(), "means": self.means.tolist(), "covs": self.covs.tolist(),
            "log_likelihood": self.log_likelihood, "n_iter": self.n_iter, "n_data": self.n_data,
            "var_names": self.var_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TPGMM":
        return cls(d["n_components"], d["n_frames"], d["dim"], np.array(d["reg"]), tuple(d["angular_dims"]),
                   np.array(d["priors"]), np.array(d["means"]), np.array(d["covs"]), d["log_likelihood"],
                   d["n_iter"], d["n_data"], d.get("var_names", []))


def select_by_bic(X: np.ndarray, k_range: list[int], reg: np.ndarray, angular_dims: tuple[int, ...] = (),
                  max_iter: int = 200) -> tuple[TPGMM, dict[int, float]]:
    """Fit one TP-GMM per K and return the BIC-optimal model and all BIC values."""
    best, scores = None, {}
    for K in k_range:
        m = TPGMM(K, X.shape[0], X.shape[2], reg, angular_dims).fit(X, max_iter=max_iter)
        scores[K] = m.bic()
        if best is None or scores[K] < best.bic():
            best = m
    return best, scores


def save_models(path: str | Path, models: dict[str, TPGMM], meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "models": {k: m.to_dict() for k, m in models.items()}}, f)


def load_models(path: str | Path) -> tuple[dict[str, TPGMM], dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {k: TPGMM.from_dict(v) for k, v in d["models"].items()}, d["meta"]
