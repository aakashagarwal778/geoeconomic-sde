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
