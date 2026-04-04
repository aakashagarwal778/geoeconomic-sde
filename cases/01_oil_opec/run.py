"""
cases/01_oil_opec/run.py

Entry point for Case 1. Reads calibrated parameters from calibration/params.json.
Produces:
  1. results/paths.png          — Monte Carlo path fan
  2. results/prob_table.csv     — Scenario probability table
  3. results/distribution_comparison.png — GE SDE vs GBM distribution

Run from repo root:
    python cases/01_oil_opec/run.py
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kurtosis
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.euler_maruyama import solve

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "model", os.path.join(os.path.dirname(__file__), "model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
OilOPECModel = _mod.OilOPECModel

# ── Parameters ────────────────────────────────────────────────
PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'calibration', 'params.json')
with open(PARAMS_PATH) as f:
    raw = json.load(f)

PARAMS = {k: raw[k] for k in ["kappa","theta","sigma0","alpha","lam","mu_j","sigma_j"]}

calibrated = raw.get("calibrated_on")
if calibrated:
    print(f"MLE-calibrated parameters (fitted {calibrated}, n={raw.get('n_obs')} obs)")
else:
    print("Prior parameters — run calibration/estimate.py to fit from data")
print(f"  theta=${raw['theta_price']}  kappa={PARAMS['kappa']}  sigma0={PARAMS['sigma0']}  alpha={PARAMS['alpha']}")
print(f"  lambda={PARAMS['lam']}  mu_j={PARAMS['mu_j']}  sigma_j={PARAMS['sigma_j']}\n")

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), 'results')

X0=np.log(82.0); T=0.5; dt=1/252; N=10_000

model  = OilOPECModel(PARAMS)
paths  = solve(model, X0, T, dt, N)
prices = np.exp(paths)
n_steps= paths.shape[0]
t_axis = np.linspace(0, T*252, n_steps)

# ── 1. Path fan ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
for i in np.random.choice(N, 200, replace=False):
    ax.plot(t_axis, prices[:,i], color='#1a6b8a', alpha=0.06, linewidth=0.5)
pcts = np.percentile(prices, [5,25,50,75,95], axis=1)
ax.fill_between(t_axis, pcts[0], pcts[4], color='#1a6b8a', alpha=0.10, label='5th-95th percentile')
ax.fill_between(t_axis, pcts[1], pcts[3], color='#1a6b8a', alpha=0.22, label='25th-75th percentile')
ax.plot(t_axis, pcts[2], color='#1a6b8a', linewidth=1.8, label='Median')
ax.axhline(70,             color='#c0392b', linewidth=1.0, linestyle='--', alpha=0.8, label='OPEC floor $70')
ax.axhline(90,             color='#2471a3', linewidth=1.0, linestyle='--', alpha=0.8, label='Shale ceiling $90')
ax.axhline(raw['theta_price'], color='gray', linewidth=0.9, linestyle=':', label=f"theta=${raw['theta_price']}")
y_lo = np.percentile(prices, 1) * 0.92
y_hi = np.percentile(prices, 99) * 1.08
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel('Trading days from today', fontsize=10)
ax.set_ylabel('Brent crude price (USD)', fontsize=10)
label = "MLE calibrated" if calibrated else "prior estimates"
ax.set_title(f'Case 1 - Brent Crude Monte Carlo ({label})\nOU + state-dependent sigma + jumps - 10,000 paths - X0=$82 - theta=${raw["theta_price"]}')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'paths.png'), dpi=180, bbox_inches='tight')
plt.close()
print('Saved: results/paths.png')

# ── 2. Probability table ──────────────────────────────────────
rows = []
for lbl, day in [('30d',30),('60d',60),('90d',90)]:
    px = prices[min(day, n_steps-1)]
    rows.append({'horizon': lbl,
                 'P(< $70)': f"{(px<70).mean():.1%}",
                 'P(> $90)': f"{(px>90).mean():.1%}",
                 'P(in band)': f"{((px>=70)&(px<=90)).mean():.1%}",
                 'median': f"${np.median(px):.1f}",
                 '5th pct': f"${np.percentile(px,5):.1f}",
                 '95th pct': f"${np.percentile(px,95):.1f}"})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS, 'prob_table.csv'), index=False)
print('Saved: results/prob_table.csv')

# ── 3. Distribution comparison: GE SDE vs GBM ────────────────
class GBMModel:
    def __init__(self, sigma):
        self.sigma = sigma
    def drift(self, X, t):
        return np.zeros_like(X)
    def diffusion(self, X, t):
        return np.full_like(X, self.sigma)
    def jump(self, X, t, rng, dt):
        return np.zeros_like(X)

N2 = 50_000; DAY = 90
sde_px = np.exp(solve(OilOPECModel(PARAMS),             X0, T, dt, N2, seed=42)[DAY])
gbm_px = np.exp(solve(GBMModel(sigma=PARAMS["sigma0"]), X0, T, dt, N2, seed=42)[DAY])

sde_below = (sde_px < 70).mean()
sde_above = (sde_px > 90).mean()
gbm_below = (gbm_px < 70).mean()
gbm_above = (gbm_px > 90).mean()
sde_ekurt = kurtosis(sde_px)
gbm_ekurt = kurtosis(gbm_px)

x     = np.linspace(28, 172, 1000)
sde_d = gaussian_kde(sde_px, bw_method=0.10)(x)
gbm_d = gaussian_kde(gbm_px, bw_method=0.10)(x)
tail  = (x < 70) | (x > 90)

fig, ax = plt.subplots(figsize=(11, 5.4))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(x, gbm_d, color='#888888', linewidth=1.6, zorder=3, label='GBM (Black-Scholes)')
ax.fill_between(x, gbm_d, color='#888888', alpha=0.15, zorder=2)
ax.plot(x, sde_d, color='#1a5c7a', linewidth=1.8, zorder=4, label='GE SDE')
ax.fill_between(x, sde_d, color='#1a5c7a', alpha=0.12, zorder=2)
ax.fill_between(x, gbm_d, sde_d,
                where=tail & (sde_d > gbm_d),
                color='#b03a2e', alpha=0.55, zorder=5,
                label='GE SDE excess tail mass')

ax.axvline(70, color='#b03a2e', linewidth=0.9, linestyle='--', alpha=0.7, zorder=6)
ax.axvline(90, color='#b03a2e', linewidth=0.9, linestyle='--', alpha=0.7, zorder=6)
ax.text(70.8, 0.0014, '$70', color='#b03a2e', fontsize=8.5)
ax.text(90.8, 0.0014, '$90', color='#b03a2e', fontsize=8.5)

ax.legend(fontsize=9, loc='upper left', frameon=True,
          framealpha=0.95, edgecolor='#dddddd')

col_labels = ['Model', 'P(<$70)', 'P(>$90)', 'Tail total', 'Exc. kurt.']
table_data = [
    ['GE SDE', f"{sde_below:.1%}", f"{sde_above:.1%}", f"{sde_below+sde_above:.1%}", f"{sde_ekurt:.2f}"],
    ['GBM',    f"{gbm_below:.1%}", f"{gbm_above:.1%}", f"{gbm_below+gbm_above:.1%}", f"{gbm_ekurt:.2f}"],
]
tbl = ax.table(cellText=table_data, colLabels=col_labels,
               bbox=[0.50, 0.74, 0.49, 0.22])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cccccc')
    cell.set_linewidth(0.5)
    cell.PAD = 0.08
    if r == 0:
        cell.set_facecolor('#efefef')
        cell.set_text_props(fontweight='bold', fontsize=8)
    elif c == 0:
        cell.set_facecolor('#f7f7f7')
        cell.set_text_props(fontweight='bold', fontsize=8.5)
    else:
        cell.set_facecolor('white')
        cell.set_text_props(fontsize=8.5)

ax.set_xlabel('Brent crude price at 90-day horizon (USD)', fontsize=10, color='#333333')
ax.set_ylabel('Probability density', fontsize=10, color='#333333')
ax.set_title('Price distribution at 90 days — GE SDE vs GBM\n'
             '50,000 paths   |   Starting price $82',
             fontsize=10.5, color='#222222', pad=10)
ax.set_xlim(28, 172)
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.tick_params(colors='#555555', labelsize=9)
ax.grid(axis='y', color='#eeeeee', linewidth=0.8, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'distribution_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print('Saved: results/distribution_comparison.png')