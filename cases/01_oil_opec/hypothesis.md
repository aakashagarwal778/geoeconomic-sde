# Case 1 — Hypothesis: Brent Crude Under OPEC+ Production Discipline

## Observation
Brent crude mean-reverts toward a $70–90 band over medium horizons.
The reversion is structural, not statistical — it is enforced by production decisions.

## Mechanism
**Floor (~$70–75):** OPEC+ fiscal breakevens cluster here. Below it, members cut
production at a rate proportional to the deviation. Small breach → modest cuts.
Large breach → coordinated emergency response.

**Ceiling (~$85–90):** US shale breakevens. Above it, rig counts rise and new supply
enters within 3–6 months, continuously capping upside.

The proportionality of both responses is what makes this mean-reverting, not just bounded.

## Why Volatility Is State-Dependent
Near $70 and $90, OPEC meetings and shale hedging data are watched closely — vol is
elevated. In the middle of the band, markets are calm — vol is low. Therefore σ is a
function of distance from θ, not a constant. This is the key nonlinearity.

## Jump Sources
Surprise OPEC+ decisions, tanker route disruptions, sudden China demand revisions.
Discrete, not continuous — require explicit jump component.

## State Variable
X(t) = log(Brent spot price). Log-price keeps values positive and makes
percentage moves the unit of analysis.