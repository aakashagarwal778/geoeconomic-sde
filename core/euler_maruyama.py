import numpy as np


def solve(model, X0: float, T: float, dt: float, n_paths: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    paths = np.zeros((n_steps + 1, n_paths))
    paths[0] = X0
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        X = paths[i]
        t = i * dt
        dW  = rng.standard_normal(n_paths) * sqrt_dt
        d   = model.drift(X, t)
        sig = model.diffusion(X, t)
        j   = model.jump(X, t, rng, dt)
        paths[i + 1] = X + d * dt + sig * dW + j

    return paths