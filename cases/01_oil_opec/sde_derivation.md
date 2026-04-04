# Case 1 — SDE Derivation

## Expression Build

**Drift:** Rate of change pulled toward θ proportional to distance.
→ κ(θ − X)

**Diffusion:** Vol rises as price deviates from band center.
→ σ₀(1 + α|X − θ|)

**Jump:** Poisson arrivals at rate λ, each shifting log-price by J ~ N(μⱼ, σⱼ²).
→ J·dq

## Full SDE
```
dX = κ(θ − X)dt  +  σ₀(1 + α|X − θ|)dW  +  J·dq
```

## Analytical Tractability
Fails at diffusion check. σ(X) = σ₀(1 + α|X − θ|) is nonlinear in X.
The |X − θ| term has derivative sign(X − θ) — discontinuous at X = θ.
Itô's formula requires computing ½f''(X)·σ(X)²dt, which produces an integral
with no closed form. → Numerical solver required.

## Discretisation (Euler-Maruyama, Δt = 1/252)
```
X(t+Δt) = X(t) + κ(θ−X)·Δt + σ₀(1+α|X−θ|)·√Δt·Z + J·Bernoulli(λΔt)
Z ~ N(0,1),  J ~ N(μⱼ, σⱼ²)
```

## Parameters
| Parameter | Value   | Reasoning                                 |
|-----------|---------|-------------------------------------------|
| κ         | 2.1     | ~4 month half-life, matches OPEC cycle    |
| θ         | log(78) | Midpoint of OPEC fiscal breakeven band    |
| σ₀        | 0.18    | Baseline annualised vol, historical Brent |
| α         | 0.40    | Vol scaling near band boundaries          |
| λ         | 3.0     | ~3 material geopolitical shocks/year      |
| μⱼ        | 0.0     | Shocks symmetric on average               |
| σⱼ        | 0.06    | Typical ±6% log-price move per event      |

Formal MLE calibration to be done in `calibration/estimate.py`.