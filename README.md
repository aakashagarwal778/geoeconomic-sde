# Geoeconomic SDE Framework

Translating geopolitical and macroeconomic structure into quantitative probability
distributions using custom Stochastic Differential Equations built from first principles.

---

## Purpose

This project has two equal objectives.

**Geoeconomic applicability.** Every model here is grounded in a real, observable
relationship between a geopolitical mechanism and a financial variable. The outputs
are not academic — they are probability distributions over price levels at tradeable
horizons, designed to identify where standard market models misprice risk.

**SDE construction and numerical methods mastery.** Each case is an exercise in
deriving a custom SDE from scratch — not fitting a known form to data, but building
each term from an economic argument, checking analytically why closed form fails,
and implementing the numerical solver by hand. By Case 5, the full solver handles
state-dependent diffusion, compound Poisson jumps, correlated multi-asset noise,
and Markov regime switching — all built incrementally from a single base loop.

---

## Methodology

Every case follows this sequence without exception:

```
Observe a real relationship
        ↓
Construct the economic argument for why it persists
        ↓
Identify the state variable and its natural units
        ↓
Express restoring forces and shocks in plain language
        ↓
Translate each argument into a mathematical term
        ↓
Assemble the full SDE: dX = drift·dt + diffusion·dW + jump·dq
        ↓
Analytical tractability check — identify exactly where closed form fails
        ↓
Discretise via Euler-Maruyama and implement the numerical solver
        ↓
Monte Carlo simulation → probability distribution → tradeable output
```

The analytical tractability check is not optional. Understanding *why* a specific
equation cannot be solved analytically — and identifying the exact term that breaks
it — is as important as the simulation itself.

Calibration is treated as a test of the economic argument, not a formality. Where the
fitted parameters contradict the hypothesis that motivated the model, the contradiction
is reported rather than smoothed over — see the note under Case 1.

---

## Architecture

```
geoeconomic-sde/
├── core/
│   ├── euler_maruyama.py       # Base solver loop — shared across all cases
│   ├── noise.py                # Brownian increments, Poisson draws, Cholesky
│   ├── diagnostics.py          # Analytical tractability checklist
│   └── output.py               # Path plots, probability tables, scenario scores
│
├── cases/
│   ├── 01_oil_opec/            # Brent crude under OPEC+ production discipline
│   ├── 02_em_fx_sanctions/     # USD/TRY under sanctions and political jump shocks
│   ├── 03_yield_equity/        # Yield curve and equity coupled SDE system
│   ├── 04_commodity_regime/    # Commodity price with Markov regime switching
│   └── 05_scenario_engine/     # Full geoeconomic scenario engine (Cases 1–4 unified)
│
├── calibration/
│   ├── estimate.py             # Case 1 — MLE parameter estimation
│   ├── estimate_case02.py      # Case 2 — MLE parameter estimation
│   ├── params.json             # Case 1 calibration log — committed to git
│   ├── params_case02.json      # Case 2 calibration log — committed to git
│   └── data/                   # Raw market data — not tracked by git
│
└── docs/                       # LaTeX-compiled framework PDF per case
```

Each case directory contains:
- `hypothesis_case.md` — economic argument and mechanism, written before any code
- `sde_derivation.md` — term-by-term expression build and tractability log
- `model.py` — drift, diffusion, and jump functions specific to that case
- `run.py` — entry point, calls core solver, saves results
- `results/` — path plots and probability tables, generated on run

The `core/` solver is extended incrementally. Each case adds exactly one new
numerical technique to the base loop — nothing is reimplemented, only extended.

---

## Cases

### Case 1 — Brent Crude Mean-Reversion Under OPEC+ Band Enforcement `[complete]`

OPEC+ fiscal breakevens (~$70–75) and US shale breakevens (~$85–90) create a
structural band enforced by proportional production responses. The proportionality
of both responses is what makes price mean-reverting, not merely bounded.

**SDE**
```
dX = κ(θ − X)dt  +  σ₀(1 + α|X − θ|)dW  +  J·dq
```

| Term | Role |
|------|------|
| κ(θ − X)dt | OU drift — OPEC/shale restoring force |
| σ₀(1 + α\|X − θ\|)dW | State-dependent diffusion — vol rises near band boundaries |
| J·dq | Compound Poisson jump — discrete geopolitical shocks |

**Why analytical solution fails:** σ(X) = σ₀(1 + α|X − θ|) is nonlinear in X.
The |X − θ| term has derivative sign(X − θ) — discontinuous at X = θ.
The resulting Fokker-Planck PDE has a non-smooth diffusion coefficient and
admits no closed-form transition density.

**New technique introduced:** Euler-Maruyama base solver, state-dependent diffusion,
compound Poisson jump sampling.

**Calibration** — daily Brent, January 2000 to April 2026 (4,648 observations).
Jump days detected at 3σ and removed before fitting the continuous parameters.

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| κ | 0.35 | Half-life ≈2 years — reversion is slow |
| θ | log(87.11) | Historical equilibrium at $87 |
| σ₀ | 0.258 | 25.8% baseline annualised vol |
| α | 1.39 | Strong state-dependent vol scaling near band boundaries |
| λ | 4.12 | ≈4 jump events per year |
| μⱼ | −0.013 | Marginal negative bias |
| σⱼ | 0.105 | Jump sizes ≈10% log-price move per event |

**Where the calibration disagrees with the hypothesis.** The economic argument implies
rapid correction — a prior of κ = 2.1 would give a four-month half-life. The MLE
returned κ = 0.35, roughly a two-year half-life, and θ = $87 rather than the $78 band
midpoint. Both are consistent with the wider literature on crude, where mean reversion
is well documented but slow.

The honest reading is that the band mechanism exists but operates over a far longer
horizon than the fiscal-breakeven story suggests, and that a 2000–2026 estimation window
contains the 2011–2014 supercycle, during which the band did not hold at all. The model
is retained with the fitted parameters; the hypothesis is treated as partially rejected
rather than confirmed. Re-estimating on a post-2015 sample is the obvious next test.

Full derivation: `docs/case01_framework.pdf`

---

### Case 2 — USD/TRY Under Sanctions Shock `[complete]`

The Turkish lira has no fixed long-run mean. Chronic inflation differentials, unorthodox
monetary policy and persistent current account deficits produce a structural one-way
depreciation trend, punctuated by discrete shocks that cluster around political and
sanctions events. OU drift was considered and rejected on economic grounds — fitting
mean reversion to a structurally trending currency produces a distorted θ that pulls
simulations toward a level the rate has permanently left.

**SDE**
```
dX = μdt  +  σdW  +  J·dq(λ(t))

where  λ(t) = λ₀ + (λ₁ − λ₀)·s(t),   s(t) ∈ {0,1}
```

| Term | Role |
|------|------|
| μdt | Constant drift — structural depreciation trend |
| σdW | Constant diffusion — continuous FX trading noise |
| J·dq(λ(t)) | Compound Poisson jump with time-varying intensity |

**Why analytical solution fails:** Merton's formula requires constant λ. With
λ(t) = λ₀ + (λ₁ − λ₀)·s(t), the jump process is non-stationary and the characteristic
function requires ∫₀ᵀ λ(t)dt, which has no closed form when s(t) switches
unpredictably — the jump count over [0, T] depends on the entire path of s(t).

**New technique introduced:** state-dependent Poisson intensity λ(t), re-evaluated at
every solver step from a signal generator held inside the model object. The signal is
market-wide, shared across all paths at each step.

**Calibration** — daily USD/TRY, January 2000 to January 2026 (5,459 observations).
Jump days detected at 2σ and removed before fitting μ and σ. The threshold is lower
than Case 1's 3σ because TRY's shocks are frequent and moderate rather than rare and
large; a 3σ filter classifies almost nothing as a jump in this series.

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| μ | 0.1659 | 16.6% annualised depreciation trend |
| σ | 0.1601 | 16.0% baseline annualised vol |
| λ₀ | 8.957 | ≈9 jump events per year, calm periods |
| λ₁ | 26.871 | ≈27 events per year, tension periods — **prior, not estimated** |
| μⱼ | 0.0094 | Small positive bias — shocks depreciation-skewed |
| σⱼ | 0.0408 | Typical jump ≈4% log-rate move per event |

**Known limitation.** Separate estimation of λ₀ and λ₁ produced λ₁ = 0: Yahoo Finance
coverage inside the tension windows is too sparse to identify a distinct intensity.
λ₁ = 3 × λ₀ is therefore a stated prior. Since time-varying intensity is what this case
exists to demonstrate, the magnitude of the regime difference should be read as assumed
rather than measured. The direction is supported by the event history in
`hypothesis_case.md` — the 2018 Brunson crisis, the 2021 central bank dismissal, and
the 2023 post-election period.

**Output** (X₀ = 38, 10,000 paths per regime, 1-year horizon)

| Regime | P(dep > 5%) | P(app > 5%) | Median | 95th pct |
|--------|-------------|-------------|--------|----------|
| Calm | 83.9% | 7.0% | 48.8 | 69.4 |
| Tension | 85.5% | 6.5% | 49.9 | 71.6 |

The regime signal lives in the tail, not the centre. Calm and tension differ by only
1.6 percentage points on P(dep > 5%), but the 95th percentile moves from 69.4 to 71.6.
That is the characteristic signature of increased jump intensity — more frequent
moderate shocks accumulate into a heavier extreme tail while the median barely moves.
A single constant implied vol cannot capture that regime-conditional tail shift.

Full derivation: `docs/case02_framework.pdf`

---

### Case 3 — Yield Curve and Equity Coupled System `[upcoming]`

Two SDEs sharing correlated Brownian motions with cross-asset drift terms.
New technique: Cholesky decomposition for correlated noise generation.

### Case 4 — Commodity Regime Switching `[upcoming]`

SDE parameters governed by an unobserved Markov chain switching between
geopolitical regimes. New technique: hidden-state forward simulation.

### Case 5 — Geoeconomic Scenario Engine `[upcoming]`

Cases 1–4 unified into a four-dimensional coupled system. Oil feeds FX drift,
FX feeds equity volatility, rates influence oil discounting. Full joint Monte Carlo
with scenario probability scoring. New technique: 4D correlated simulation.

---

## Running a Case

Calibrate first, then run:

```bash
python calibration/estimate.py            # Case 1
python cases/01_oil_opec/run.py

python calibration/estimate_case02.py     # Case 2
python cases/02_em_fx_sanctions/run.py
```

Each entry point falls back to prior parameters if the calibration log is absent, and
prints which set it is using. Output is saved to the relevant `cases/*/results/`.

## Dependencies

```bash
pip install numpy pandas matplotlib scipy yfinance
```
