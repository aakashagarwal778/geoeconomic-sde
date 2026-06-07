# Case 2 — SDE Derivation: USD/RUB

## Step 1 — State Variable
X(t) = log(USD/RUB). Increase in X = rouble depreciation.
Starting value X₀ = log(current USD/RUB spot).

## Step 2 — Building the Expression Term by Term

**Drift.**
The rouble reverts toward macro-fundamental equilibrium θ at speed κ.
Same OU argument as Case 1 — restoring force proportional to deviation.
→ κ(θ − X)

**Diffusion.**
Continuous FX noise — portfolio flows, bid-ask dynamics, routine trading.
Constant σ is defensible here because extreme vol is captured by the jump
term. No state-dependent scaling needed.
→ σdW

**Jump.**
Sanctions shocks arrive as a Poisson process with time-varying intensity λ(t).
Intensity switches between baseline λ₀ (calm) and elevated λ₁ (tension)
driven by geopolitical signal s(t) ∈ {0,1}.
Jump sizes are asymmetric — depreciation bias encoded in μⱼ > 0.
→ J·dq(λ(t)),   J ~ N(μⱼ, σⱼ²),   μⱼ > 0

## Step 3 — Full SDE

```
dX = κ(θ − X)dt  +  σdW  +  J·dq(λ(t))

where:
  λ(t) = λ₀ + (λ₁ − λ₀)·s(t)
  s(t) ∈ {0,1}   geopolitical tension signal
  J    ~ N(μⱼ, σⱼ²),   μⱼ > 0
```

| Term | Type | Economic meaning |
|------|------|-----------------|
| κ(θ − X)dt | Drift | Macro-fundamental mean reversion |
| σdW | Diffusion | Continuous FX trading noise |
| J·dq(λ(t)) | Jump | Sanctions shock — intensity state-dependent |

## Step 4 — Analytical Tractability Check

**Q1: Is diffusion constant?** Yes. σ is constant — passes.

**Q2: Is drift linear in X?** Yes. κ(θ − X) is linear — passes.

**Q3: Are jumps tractable?**
No. Merton's formula requires constant λ. Here λ(t) = λ₀ + (λ₁ − λ₀)·s(t)
is time-varying. The moment λ becomes a function of time the jump process
is no longer stationary. The characteristic function of the log-price process
loses its closed form — the integral ∫λ(t)dt has no analytic expression
when s(t) switches state unpredictably.

**→ FAILS at Q3. Analytical solution does not exist.**

**Decision: Euler-Maruyama numerical solver required.**

## Step 5 — Euler-Maruyama Discretisation

At each step, λ(t) is evaluated from the current signal state before
drawing the Poisson increment:

```
λ(t)         = λ₀ + (λ₁ − λ₀)·s(t)
X(t + Δt)    = X(t) + κ(θ − X(t))·Δt + σ·√Δt·Z + J·Bernoulli(λ(t)·Δt)
Z ~ N(0,1),  J ~ N(μⱼ, σⱼ²)
```

New technique vs Case 1: λ is re-evaluated at every time step from s(t).
The solver loop now passes the current signal state into the jump function.

## Step 6 — Parameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| κ | — | MLE from 2000–2022 USD/RUB data |
| θ | — | Long-run log-rate, MLE estimated |
| σ | — | Baseline annualised vol, MLE estimated |
| λ₀ | — | Calm-period jump intensity, MLE estimated |
| λ₁ | — | Tension-period jump intensity, MLE estimated |
| μⱼ | — | Mean jump size, positive (depreciation bias) |
| σⱼ | — | Jump size std dev |

All parameters estimated via MLE in calibration/estimate_case02.py.
