"""
calibration/estimate.py

MLE calibration for Case 1 — Brent crude SDE parameters.

Pulls live daily Brent crude data from Yahoo Finance.
Estimates κ, θ, σ₀, α via Euler-Maruyama approximate MLE.
Estimates λ, μⱼ, σⱼ from empirical large-move detection.
Writes fitted parameters to calibration/params.json.

Run from repo root:
    python calibration/estimate.py
"""

import numpy as np
import json
import os
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
from datetime import datetime

# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_brent(start="2000-01-01") -> np.ndarray:
    """Pull daily Brent crude log-prices from Yahoo Finance."""
    df = yf.download("BZ=F", start=start, auto_adjust=True, progress=False)
    if df.empty:
        # fallback ticker
        df = yf.download("CL=F", start=start, auto_adjust=True, progress=False)
    prices = df["Close"].dropna().values.flatten()
    print(f"Downloaded {len(prices)} daily prices "
          f"({start} to {datetime.today().strftime('%Y-%m-%d')})")
    return np.log(prices)

# ── Jump detection ────────────────────────────────────────────────────────────

def estimate_jumps(log_prices: np.ndarray, threshold_sigma: float = 3.0):
    """
    Identify jump days as moves exceeding threshold_sigma rolling std.
    Returns λ (jumps/year), μⱼ, σⱼ.
    """
    returns   = np.diff(log_prices)
    roll_std  = np.std(returns)
    threshold = threshold_sigma * roll_std

    jump_mask  = np.abs(returns) > threshold
    jump_moves = returns[jump_mask]
    n_years    = len(returns) / 252

    lam    = len(jump_moves) / n_years
    mu_j   = float(np.mean(jump_moves))   if len(jump_moves) > 0 else 0.0
    sigma_j= float(np.std(jump_moves))    if len(jump_moves) > 0 else 0.05

    print(f"Jump detection: {len(jump_moves)} events over {n_years:.1f} years "
          f"→ λ = {lam:.2f} yr⁻¹")

    # remove jump days before OU calibration
    clean_returns = returns[~jump_mask]
    clean_prices  = np.concatenate([[log_prices[0]],
                                     log_prices[1:][~jump_mask]])
    return lam, mu_j, sigma_j, clean_prices

# ── Euler-Maruyama approximate MLE ───────────────────────────────────────────

def neg_log_likelihood(params: np.ndarray, X: np.ndarray, dt: float) -> float:
    """
    Approximate MLE via Euler-Maruyama transition density.

    At each step t, X(t+dt) | X(t) is approximately:
        N( X(t) + κ(θ-X(t))·dt,  [σ₀(1+α|X(t)-θ|)]²·dt )

    We maximise the sum of log-densities over the observed path.
    """
    kappa, theta, sigma0, alpha = params

    # enforce positivity constraints
    if kappa <= 0 or sigma0 <= 0 or alpha < 0:
        return 1e10

    Xt  = X[:-1]
    Xt1 = X[1:]

    mu_step  = Xt + kappa * (theta - Xt) * dt
    sig_step = sigma0 * (1.0 + alpha * np.abs(Xt - theta)) * np.sqrt(dt)

    # guard against zero or negative std
    sig_step = np.maximum(sig_step, 1e-8)

    ll = norm.logpdf(Xt1, loc=mu_step, scale=sig_step)
    return -np.sum(ll)


def calibrate_ou(X: np.ndarray, dt: float = 1/252) -> dict:
    """
    Fit κ, θ, σ₀, α to clean (jump-removed) log-price series.
    Uses moment estimates as starting values for robustness.
    """
    # moment-based starting values
    theta0  = float(np.mean(X))
    sigma0  = float(np.std(np.diff(X)) / np.sqrt(dt))
    kappa0  = 1.0
    alpha0  = 0.2

    x0     = [kappa0, theta0, sigma0, alpha0]
    bounds = [(0.01, 20), (theta0 - 1, theta0 + 1),
              (0.01, 2.0), (0.0, 5.0)]

    result = minimize(
        neg_log_likelihood,
        x0,
        args=(X, dt),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12}
    )

    if not result.success:
        print(f"Warning: optimiser did not fully converge — {result.message}")

    kappa, theta, sigma0, alpha = result.x
    return {
        "kappa" : round(float(kappa),  4),
        "theta" : round(float(theta),  6),
        "sigma0": round(float(sigma0), 4),
        "alpha" : round(float(alpha),  4),
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dt = 1 / 252

    print("── Fetching Brent crude data ────────────────────────────────")
    log_prices = fetch_brent(start="2000-01-01")

    print("\n── Estimating jump parameters ──────────────────────────────")
    lam, mu_j, sigma_j, clean_prices = estimate_jumps(log_prices)

    print("\n── Calibrating OU + state-dependent σ via MLE ──────────────")
    ou_params = calibrate_ou(clean_prices, dt)

    params = {
        **ou_params,
        "theta_price": round(float(np.exp(ou_params["theta"])), 2),
        "lam"        : round(lam,     4),
        "mu_j"       : round(mu_j,    6),
        "sigma_j"    : round(sigma_j, 4),
        "calibrated_on": datetime.today().strftime("%Y-%m-%d"),
        "data_start"   : "2000-01-01",
        "n_obs"        : len(log_prices),
        "note"         : "Euler-Maruyama approximate MLE. Jump days removed before OU fit."
    }

    print("\n── Results ─────────────────────────────────────────────────")
    for k, v in params.items():
        print(f"  {k:20s}: {v}")

    out_path = os.path.join(os.path.dirname(__file__), "params.json")
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()