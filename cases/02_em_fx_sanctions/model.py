"""
cases/02_em_fx_sanctions/model.py

SDE for USD/RUB log-rate under sanctions shock.

dX = μdt  +  σdW  +  J·dq(λ(t))

where λ(t) = λ₀ + (λ₁ - λ₀)·s(t)
      s(t) ∈ {0,1}  geopolitical tension signal
      J    ~ N(μⱼ, σⱼ²),  μⱼ > 0  (depreciation bias)

Drift    : constant GBM drift μ — captures structural depreciation trend.
           OU mean-reversion was rejected: TRY has no fixed long-run mean,
           it follows a structural one-way depreciation trend. Forcing
           reversion to a historical average distorts the simulation.
Diffusion: constant — continuous FX trading noise
Jump     : compound Poisson with TIME-VARYING intensity λ(t)
           New technique vs Case 1 — λ re-evaluated at every
           time step from the current signal state s(t).

Analytical tractability: FAILS at Q3.
λ(t) is time-varying. Merton's formula requires constant λ.
With λ = λ(t), the jump process is non-stationary — the
characteristic function integral has no closed form when
s(t) switches state. Euler-Maruyama required.
Drift change does not affect tractability — constant drift
is trivially integrable; Q3 remains the binding constraint.
"""

import numpy as np


class SignalGenerator:
    """
    Geopolitical tension signal s(t) in {0,1}.

    mode='simulate'  : stochastic switching via transition probabilities
    mode='historical': fixed tension windows (calibration use only)

    p_escalate   : daily probability of calm -> tension
    p_deescalate : daily probability of tension -> calm
    """

    def __init__(self, mode="simulate", p_escalate=0.02,
                 p_deescalate=0.10, initial_state=0):
        assert mode in ("simulate", "historical")
        self.mode         = mode
        self.p_escalate   = p_escalate
        self.p_deescalate = p_deescalate
        self.state        = initial_state

    def step(self, rng):
        if self.mode == "simulate":
            if self.state == 0:
                if rng.random() < self.p_escalate:
                    self.state = 1
            else:
                if rng.random() < self.p_deescalate:
                    self.state = 0
        return self.state

    def reset(self, state=0):
        self.state = state


class USDRUBModel:

    def __init__(self, params, signal):
        """
        params keys
        -----------
        mu      : constant drift — average daily log-return (annualised)
        sigma   : constant diffusion vol (annualised)
        lam0    : baseline jump intensity — calm period (per year)
        lam1    : elevated jump intensity — tension period (per year)
        mu_j    : mean jump size in log-rate (positive = depreciation bias)
        sigma_j : std of jump size
        """
        self.mu      = params["mu"]     # constant drift — structural depreciation trend
        self.sigma   = params["sigma"]
        self.lam0    = params["lam0"]
        self.lam1    = params["lam1"]
        self.mu_j    = params["mu_j"]
        self.sigma_j = params["sigma_j"]
        self.signal  = signal

    def drift(self, X, t):
        return np.full_like(X, self.mu)

    def diffusion(self, X, t):
        return np.full_like(X, self.sigma)

    def jump(self, X, t, rng, dt):
        """
        Key difference from Case 1:
        λ is evaluated fresh at every time step from signal state.
        Signal is market-wide — same state applies to all paths at time t.
        """
        s     = self.signal.step(rng)
        lam_t = self.lam0 + (self.lam1 - self.lam0) * s

        arrivals = rng.poisson(lam_t * dt, size=X.shape[0])
        sizes    = rng.normal(self.mu_j, self.sigma_j, size=X.shape[0])
        return arrivals * sizes