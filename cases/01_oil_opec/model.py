import numpy as np


class OilOPECModel:

    def __init__(self, params: dict):
        self.kappa   = params["kappa"]
        self.theta   = params["theta"]
        self.sigma0  = params["sigma0"]
        self.alpha   = params["alpha"]
        self.lam     = params["lam"]
        self.mu_j    = params["mu_j"]
        self.sigma_j = params["sigma_j"]

    def drift(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.kappa * (self.theta - X)

    def diffusion(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.sigma0 * (1.0 + self.alpha * np.abs(X - self.theta))

    def jump(self, X: np.ndarray, t: float, rng: np.random.Generator, dt: float) -> np.ndarray:
        arrivals = rng.poisson(self.lam * dt, size=X.shape[0])
        sizes    = rng.normal(self.mu_j, self.sigma_j, size=X.shape[0])
        return arrivals * sizes