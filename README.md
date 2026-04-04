# Geoeconomic SDE Framework

Translating geopolitical and macroeconomic structure into quantitative probability
distributions using custom Stochastic Differential Equations built from first principles.

---

## Purpose

This project has two equal objectives.

**Geoeconomic applicability.** Every model here is grounded in a real, observable
relationship between a geopolitical mechanism and a financial variable. The outputs
are not academic — they are probability distributions over financial relevant at tradeable
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

---

## Architecture

```
geoeconomic-sde/
├── core/
│   ├── euler_maruyama.py     # Base solver loop — shared across all cases
│   ├── noise.py              # Brownian increments, Poisson draws, Cholesky
│   ├── diagnostics.py        # Analytical tractability checklist
│   └── output.py             # Path plots, probability tables, scenario scores
│
├── cases/
│   ├── 01_oil_opec/          # Brent crude under OPEC+ production discipline
│   ├── 02_em_fx_sanctions/   # EM FX under geopolitical jump shocks
│   ├── 03_yield_equity/      # Yield curve and equity coupled SDE system
│   ├── 04_commodity_regime/  # Commodity price with Markov regime switching
│   └── 05_scenario_engine/   # Full geoeconomic scenario engine (Cases 1–4 unified)
│
├── calibration/
│   ├── estimate.py           # MLE / method of moments parameter estimation
│   ├── params.json           # Versioned calibration log — committed to git
│   └── data/                 # Raw market data — not tracked by git
│
└── docs/                     # LaTeX-compiled PDF per case
```

Each case directory contains:
- `hypothesis.md` — economic argument and mechanism, written before any code
- `sde_derivation.md` — term-by-term expression build and tractability log
- `model.py` — drift, diffusion, and jump functions specific to that case
- `run.py` — entry point, calls core solver, saves results
- `results/` — path plots and probability tables, generated on run

The `core/` solver is extended incrementally. Each case adds exactly one new
numerical technique to the base loop — nothing is reimplemented, only extended.

---

## Cases

### Case 1 — Brent Crude Under OPEC+ Production Discipline `[complete]`

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

**Output** (X₀ = $82, θ = $78, 10,000 paths, 6-month horizon)

| Horizon | P(< $70) | P(> $90) | P(in band) | Median |
|---------|----------|----------|------------|--------|
| 30d | 1.3% | 5.4% | 93.4% | $81.0 |
| 60d | 4.7% | 8.7% | 86.7% | $80.3 |
| 90d | 7.0% | 9.3% | 83.7% | $79.7 |

Starting at $82, the ceiling breach is ~4x more likely than the floor breach at
30 days due to geometric proximity in log-price space. Standard Black-Scholes prices
equidistant strikes symmetrically — this model does not. The asymmetry is the edge.

Full derivation: `docs/case01_oil_opec.pdf`

---

### Case 2 — EM FX Under Sanctions Shock `[upcoming]`

Jump intensity as a time-varying function of a geopolitical news signal.
New technique: state-dependent Poisson intensity λ(t).

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

```bash
python cases/01_oil_opec/run.py
```

Output is saved to `cases/01_oil_opec/results/`.

## Dependencies

```bash
pip install numpy pandas matplotlib scipy
```