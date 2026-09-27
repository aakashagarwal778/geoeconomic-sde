# Case 2 — SDE Derivation: USD/TRY

## Step 1 — State Variable
X(t) = log(USD/TRY). Increase in X = lira depreciation.
Starting value X₀ = log(38).

## Step 2 — Building the Expression Term by Term

**Drift.**
TRY has no fixed long-run mean. OU mean-reversion was considered and rejected on
economic grounds: the lira follows a structural one-way depreciation trend, and
forcing reversion toward a historical average would distort every simulated path.
The correct specification is a constant drift encoding that trend — the rate of
change of X reflects a persistent, historically stable depreciation independent of
the current level.
→ μ  (constant)

This is the key structural difference from Case 1.

**Diffusion.**
Continuous FX noise — portfolio flows, bid-ask dynamics, routine trading.
Constant σ is defensible here because extreme volatility is captured by the jump
term. No state-dependent scaling needed.
→ σdW

**Jump.**
Sanctions and political shocks arrive as a Poisson process with time-varying
intensity λ(t). Intensity switches between baseline λ₀ (calm) and elevated λ₁
(tension), driven by geopolitical signal s(t) ∈ {0,1}.
Jump sizes are asymmetric — depreciation bias encoded in μⱼ > 0.
→ J·dq(λ(t)),   J ~ N(μⱼ, σⱼ²),   μⱼ > 0

## Step 3 — Full SDE

```
dX = μdt  +  σdW  +  J·dq(λ(t))

where:
  λ(t) = λ₀ + (λ₁ − λ₀)·s(t)
  s(t) ∈ {0,1}   geopolitical tension signal
  J    ~ N(μⱼ, σⱼ²),   μⱼ > 0
```

| Term | Type | Economic meaning |
|------|------|-----------------|
| μdt | Drift | Structural depreciation trend |
| σdW | Diffusion | Continuous FX trading noise |
| J·dq(λ(t)) | Jump | Sanctions/political shock — intensity state-dependent |

## Step 4 — Analytical Tractability Check

**Q1: Is diffusion constant?** Yes. σ is constant — passes.

**Q2: Is drift linear in X?** Yes. μ is constant, so trivially linear — passes.
Note that the change from OU to constant drift does not affect tractability. A
constant drift is trivially integrable; Q3 remains the binding constraint either way.

**Q3: Are jumps tractable?**
No. Merton's formula requires constant λ. Here λ(t) = λ₀ + (λ₁ − λ₀)·s(t)
is time-varying. The moment λ becomes a function of time the jump process is no
longer stationary. The characteristic function of the log-price process requires
evaluating ∫₀ᵀ λ(t)dt, which has no closed form when s(t) switches state
unpredictably — the number of jumps over [0, T] is no longer Poisson with a fixed
parameter, but depends on the entire path of s(t).

**→ FAILS at Q3. Analytical solution does not exist.**

**Decision: Euler-Maruyama numerical solver required.**

## Step 5 — Euler-Maruyama Discretisation

At each step, λ(t) is evaluated from the current signal state before
drawing the Poisson increment:

```
λ(t)         = λ₀ + (λ₁ − λ₀)·s(t)
X(t + Δt)    = X(t) + μ·Δt + σ·√Δt·Z + J·1[Poisson(λ(t)·Δt) ≥ 1]
Z ~ N(0,1),  J ~ N(μⱼ, σⱼ²),  Δt = 1/252
```

New technique vs Case 1: λ is re-evaluated at every time step from s(t).
The solver loop now passes the current signal state into the jump function, and the
signal generator lives inside the model object and advances its state on each call.
The signal is shared across all paths at each step — it represents a market-wide
geopolitical condition, not a path-specific state.

## Step 6 — Parameters

Calibrated on daily USD/TRY, January 2000 to January 2026 (5,459 observations).
Jump days identified at a 2σ threshold and removed before fitting μ and σ.

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| μ | 0.1659 | 16.6% annualised depreciation trend, mean log-return of the jump-removed series |
| σ | 0.1601 | 16.0% baseline annualised vol |
| λ₀ | 8.957 | ≈9 jump events per year in calm periods, from full-sample detection |
| λ₁ | 26.871 | ≈27 events per year in tension periods — **prior, not estimated**: λ₁ = 3 × λ₀ |
| μⱼ | 0.0094 | Small positive bias — shocks slightly depreciation-skewed |
| σⱼ | 0.0408 | Typical jump ≈4% log-rate move per event |

Parameters estimated in `calibration/estimate_case02.py`, logged to
`calibration/params_case02.json`.

**On λ₁.** Separate calm/tension estimation produced λ₁ = 0 because Yahoo Finance
coverage inside the tension windows is too sparse to identify a distinct intensity.
The 3× ratio is a stated prior. Since time-varying intensity is the defining feature
of this case, the magnitude of the regime difference should be read as assumed rather
than measured; the qualitative direction is supported by the event history in
`hypothesis_case.md`.

**On the jump character.** λ₀ of roughly 9 events per year at 4% each is a different
profile from Case 1's Brent calibration (4.12 events per year at 10%). TRY's history
is one of chronic political-driven volatility — frequent moderate shocks — which is
also why the detection threshold is 2σ here against 3σ for Brent.
