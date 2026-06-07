
# Case 2 — Hypothesis: USD/RUB Under Sanctions Shock

## Observation
USD/RUB exhibits slow mean-reversion toward a macro-fundamental equilibrium during
calm periods, punctuated by discrete, large, directional moves that cluster around
identifiable geopolitical events. The clustering is the key observation — jumps do
not arrive uniformly through time. They arrive in bursts when geopolitical tension
is elevated.

## Economic Argument
The rouble's equilibrium is anchored by Russia's current account surplus (oil and
gas export revenues) and Central Bank intervention. This creates a continuous
mean-reverting dynamic. The jumps come from a separate channel:

**Correspondent banking restrictions:** when a sanctioned bank loses dollar clearing
access, rouble demand collapses immediately and discontinuously. Discrete, not gradual.

**Reserve freezes:** freezing Central Bank reserves removes the mechanism enforcing
the equilibrium band. The mean-reversion force weakens or disappears during the shock.

**Expectation cascades:** market participants front-run anticipated sanctions, creating
self-reinforcing sell pressure that amplifies the initial move.

## Why Jump Intensity Must Be Time-Varying
A constant λ would treat the calm 2016–2019 period identically to 2014 or early 2022.
That is clearly wrong. Sanctions risk intensity tracks diplomatic escalation cycles,
OFAC designation announcements, and legislative calendars. When tension is elevated
λ is high. When tension subsides λ reverts to baseline. This is the structural
departure from Case 1 where λ was constant.

## Jump Asymmetry
Sanctions shocks depreciate the rouble (positive J in log-USD/RUB terms) more than
relief rallies appreciate it. μⱼ > 0 — the jump distribution has a positive mean,
encoding the directional bias of geopolitical stress on EM currencies.

## State Variable
X(t) = log(USD/RUB). An increase in X means rouble depreciation — more roubles per
dollar. This is the natural direction of stress.

## Calibration Data
Daily USD/RUB from 2000 to January 2022. Post-February 2022 data excluded —
mandatory export conversion rules and capital controls mean the post-invasion rate
is administratively managed, not market-clearing. The 2022 event is treated as a
scenario input to the jump component, not a calibration observation.

## Geopolitical Signal s(t)
Binary signal encoding tension state:
  s(t) = 1 during documented escalation periods (2014 Crimea, 2018 CAATSA, 2022 pre-invasion)
  s(t) = 0 during calm periods

In live deployment this signal would be driven by an NLP sentiment score on
geopolitical news. For calibration it is constructed from known event dates.
