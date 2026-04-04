"""
cases/01_oil_opec/plot_distribution.py

Generates distribution comparison plot: GE SDE vs GBM at 90-day horizon.
Reads parameters from calibration/params.json.

Run from repo root:
    python cases/01_oil_opec/plot_distribution.py
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, kurtosis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.euler_maruyama import solve

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "model", os.path.join(os.path.dirname(__file__), "model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
OilOPECModel = _mod.OilOPECModel

PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'calibration', 'params.json')
with open(PARAMS_PATH) as f:
    raw = json.load(f)

PARAMS = {k: raw[k] for k in ["kappa","theta","sigma0","alpha","lam","mu_j","sigma_j"]}

class GBMModel:
    def __init__(self, sigma):
        self.sigma = sigma
    def drift(self, X, t):
        return np.zeros_like(X)
    def diffusion(self, X, t):
        return np.full_like(X, self.sigma)
    def jump(self, X, t, rng, dt):
        return np.zeros_like(X)

X0=np.log(82.0); T=0.5; dt=1/252; N=50_000; DAY=90

sde_px = np.exp(solve(OilOPECModel(PARAMS),         X0, T, dt, N, seed=42)[DAY])
gbm_px = np.exp(solve(GBMModel(sigma=PARAMS["sigma0"]), X0, T, dt, N, seed=42)[DAY])

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
os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), 'results', 'distribution_comparison.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
