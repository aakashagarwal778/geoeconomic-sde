# Case 2 — Hypothesis: USD/TRY Under Sanctions Shock

## Observation
USD/TRY does not drift smoothly. Over calm periods it exhibits slow, persistent
depreciation driven by inflation differentials and current account dynamics. This
continuous trend is punctuated by discrete, large, directional moves that cluster
around identifiable geopolitical and political events. The clustering is the key
observation — jumps do not arrive uniformly through time. They arrive in bursts
when geopolitical tension is elevated.

## Economic Argument

**The structural depreciation trend.** The Turkish lira has been on a one-way
depreciation path since 2018, driven by chronic inflation differentials (Turkish CPI
persistently 40–85% above US CPI), repeated episodes of unorthodox monetary policy,
and sustained current account deficits. There is no fixed long-run mean toward which
the rate reverts.

This is the structural difference from Case 1. Brent has a defensible equilibrium
anchored by production economics; TRY does not. Fitting an Ornstein-Uhlenbeck drift
here would produce a historically distorted θ that pulls simulations back toward a
level the currency has permanently left. A constant GBM drift μ is the economically
honest specification: TRY depreciates at a historically stable average rate, with
shocks adding discrete moves on top.

The jumps come from a separate channel:

**Sanctions and designations.** OFAC designations and secondary-sanctions threats
reprice lira exposure immediately and discontinuously rather than gradually.

**Monetary policy shocks.** Abrupt central bank leadership changes remove the
expectation of orthodox rate defence, repricing the currency in a single session.

**Expectation cascades.** Market participants front-run anticipated policy or
sanctions moves, creating self-reinforcing sell pressure that amplifies the initial
move.

## Why Jump Intensity Must Be Time-Varying
Three documented escalation windows produced clustering of large TRY moves:

1. **2018 Brunson crisis** — OFAC sanctions on Turkish ministers, followed by a
   doubling of steel and aluminium tariffs. TRY lost roughly 40% in weeks.
2. **2021 central bank shock** — the central bank governor who had raised rates was
   dismissed. TRY fell roughly 15% in a single session.
3. **2023 post-election uncertainty** — policy continuity concerns after the May 2023
   election produced a sustained jump cluster.

During these windows jump arrivals were measurably more frequent than during calm
periods. A constant λ would treat 2016–2017, a relatively quiet stretch, identically
to 2018. This is the structural departure from Case 1, where λ was constant.

## Jump Asymmetry
Sanctions and political shocks depreciate TRY (positive J in log-USD/TRY terms).
Relief rallies exist but are smaller and shorter-lived. μⱼ > 0 encodes this
depreciation bias.

## State Variable
X(t) = log(USD/TRY). An increase in X means lira depreciation — more lira per dollar.
This is the natural direction of stress. Starting value X₀ = log(38).

## Calibration Data
Daily USD/TRY from January 2000 to January 2026 (5,459 observations). No data
exclusion is required — TRY remained freely traded and market-clearing throughout the
sample, including during the escalation windows above.

Jump days are identified as daily moves exceeding 2σ of the return distribution and
removed before fitting the continuous parameters. The 2σ threshold is deliberately
lower than the 3σ used in Case 1: TRY's jump character is frequent moderate shocks
rather than rare catastrophic ones, and a 3σ filter would classify almost nothing as
a jump in this series.

## Geopolitical Signal s(t)
Binary signal encoding tension state:

- s(t) = 1 during documented escalation periods (2018 Brunson, 2021 governor
  dismissal, 2023 post-election)
- s(t) = 0 during calm periods

In simulation the signal switches stochastically via daily escalation and
de-escalation probabilities. In live deployment it would be driven by an NLP
sentiment score on geopolitical news. For calibration it is constructed from known
event dates.

## Known Limitation
Separate estimation of λ₀ and λ₁ from the data was attempted but produced λ₁ = 0,
because Yahoo Finance coverage within the tension windows is too sparse to identify
a distinct elevated intensity. λ₁ is therefore set at 3 × λ₀ as a stated prior rather
than an estimate. This is the weakest parameter in the case and is recorded here
deliberately: a transparent prior is preferable to a spurious estimate, but the
time-varying intensity that defines this case is assumed in magnitude, not measured.
