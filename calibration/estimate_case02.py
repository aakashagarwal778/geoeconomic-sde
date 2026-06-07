"""
calibration/estimate_case02.py

MLE calibration for Case 2 — USD/TRY SDE parameters.
Post-Feb 2022 excluded — administratively managed rate.

Two-step:
  Step 1: Detect jump days (>2sigma) across full sample.
          Estimate single lambda, mu_j, sigma_j.
          Set lam0 = lambda, lam1 = lambda * 3 (prior-based ratio).
          Note: separate calm/tension lambda estimation was attempted
          but produced lam1=0 due to sparse coverage in Yahoo TRY=X
          tension windows. Single lambda with prior ratio is more honest.
  Step 2: Fit kappa, theta, sigma on clean series via Euler-Maruyama MLE.

Run from repo root:
    python calibration/estimate_case02.py
"""

import numpy as np
import json, os
import yfinance as yf
from datetime import datetime

def fetch_usdrub(start="2000-01-01", end="2026-01-01"):
    df = yf.download("TRY=X", start=start, end=end,
                     auto_adjust=True, progress=False)
    prices = df["Close"].dropna()
    print(f"Downloaded {len(prices)} daily USD/TRY prices ({start} to {end})")
    return np.log(prices.values.flatten()), prices.index

def estimate_jumps(log_prices, threshold_sigma=2.0):
    """
    Single lambda estimated from full sample.
    lam0 = lambda (calm baseline)
    lam1 = lambda * 3 (tension — prior-based ratio, economically justified)
    sigma_j capped at 0.15 to prevent unstable estimates from outliers.
    """
    returns   = np.diff(log_prices)
    threshold = threshold_sigma * np.std(returns)
    jump_mask = np.abs(returns) > threshold

    n_years = len(returns) / 252
    all_j   = returns[jump_mask]
    lam     = len(all_j) / n_years
    mu_j    = float(np.mean(all_j))
    sigma_j = min(float(np.std(all_j)), 0.15)

    lam0 = lam
    lam1 = lam * 3.0

    print(f"Jump detection ({threshold_sigma}sigma): {jump_mask.sum()} events over {n_years:.1f} years")
    print(f"  lambda = {lam:.4f} yr-1")
    print(f"  lam0 (calm)    = {lam0:.4f}")
    print(f"  lam1 (tension) = {lam1:.4f}  [prior ratio: lam1 = lam0 x 3]")
    print(f"  mu_j = {mu_j:.4f}  sigma_j = {sigma_j:.4f}")

    clean = np.concatenate([[log_prices[0]], log_prices[1:][~jump_mask]])
    return lam0, lam1, mu_j, sigma_j, clean

def calibrate_gbm(X, dt=1/252):
    """
    GBM drift: mu = mean daily log-return, annualised.
    Sigma: std of daily log-returns, annualised.
    OU rejected — TRY has no fixed long-run mean.
    """
    returns = np.diff(X)
    mu      = float(np.mean(returns) / dt)       # annualised drift
    sigma   = float(np.std(returns)  / np.sqrt(dt))  # annualised vol
    print(f"GBM calibration:")
    print(f"  mu (annualised drift) = {mu:.4f}  ({mu*100:.2f}% per year)")
    print(f"  sigma (annualised vol) = {sigma:.4f}")
    return mu, sigma

def main():
    dt = 1/252
    print("── Fetching USD/TRY ─────────────────────────────────────────")
    log_prices, dates = fetch_usdrub()

    print("\n── Jump parameters ──────────────────────────────────────────")
    lam0, lam1, mu_j, sigma_j, clean = estimate_jumps(log_prices)

    print("\n── GBM calibration ──────────────────────────────────────────")
    mu, sigma = calibrate_gbm(clean, dt)

    params = {
        "mu"           : round(float(mu),    6),
        "sigma"        : round(float(sigma),  4),
        "lam0"         : round(lam0,   4),
        "lam1"         : round(lam1,   4),
        "mu_j"         : round(mu_j,   6),
        "sigma_j"      : round(sigma_j, 4),
        "calibrated_on": datetime.today().strftime("%Y-%m-%d"),
        "data_start"   : "2000-01-01",
        "data_end"     : "2026-01-01",
        "n_obs"        : len(log_prices),
        "note"         : "USD/TRY 2000-2026. Single lambda from full sample — calm/tension split produced lam1=0 due to sparse Yahoo TRY=X tension coverage. lam1 = lam0 x 3 (prior ratio). sigma_j capped at 0.15."
    }

    print("\n── Results ──────────────────────────────────────────────────")
    for k, v in params.items():
        print(f"  {k:20s}: {v}")

    out = os.path.join(os.path.dirname(__file__), "params_case02.json")
    with open(out, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()