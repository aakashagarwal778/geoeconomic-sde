import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.euler_maruyama import solve

import importlib.util, os as _os
_spec = importlib.util.spec_from_file_location(
    "model", _os.path.join(_os.path.dirname(__file__), "model.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
OilOPECModel = _mod.OilOPECModel

PARAMS = {
    "kappa"  : 2.1,
    "theta"  : np.log(78.0),
    "sigma0" : 0.18,
    "alpha"  : 0.40,
    "lam"    : 3.0,
    "mu_j"   : 0.0,
    "sigma_j": 0.06,
}

X0      = np.log(82.0)
T       = 0.5
dt      = 1 / 252
N_PATHS = 10_000

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), 'results')

model  = OilOPECModel(PARAMS)
paths  = solve(model, X0, T, dt, N_PATHS)
prices = np.exp(paths)
n_steps = paths.shape[0]
t_axis  = np.linspace(0, T * 252, n_steps)

fig, ax = plt.subplots(figsize=(11, 5))
sample_idx = np.random.choice(N_PATHS, 200, replace=False)
for i in sample_idx:
    ax.plot(t_axis, prices[:, i], color='#1a6b8a', alpha=0.06, linewidth=0.6)
pcts = np.percentile(prices, [5, 25, 50, 75, 95], axis=1)
ax.fill_between(t_axis, pcts[0], pcts[4], color='#1a6b8a', alpha=0.10, label='5–95th pct')
ax.fill_between(t_axis, pcts[1], pcts[3], color='#1a6b8a', alpha=0.20, label='25–75th pct')
ax.plot(t_axis, pcts[2], color='#1a6b8a', linewidth=1.8, label='Median')
ax.axhline(70, color='#e05c2a', linewidth=1.0, linestyle='--', alpha=0.7, label='OPEC floor ~$70')
ax.axhline(90, color='#2a8ae0', linewidth=1.0, linestyle='--', alpha=0.7, label='Shale ceiling ~$90')
ax.axhline(np.exp(PARAMS['theta']), color='gray', linewidth=0.8, linestyle=':', label='θ = $78')
ax.set_xlabel('Trading days from today')
ax.set_ylabel('Brent crude (USD)')
ax.set_title('Case 1 · Brent crude Monte Carlo\nOU + state-dependent σ + jumps · 10,000 paths · X₀=$82 · θ=$78')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'paths.png'), dpi=150)
plt.close()
print('Saved: results/paths.png')

rows = []
for label, day in [('30d', 30), ('60d', 60), ('90d', 90)]:
    px = prices[min(day, n_steps - 1)]
    rows.append({
        'horizon'   : label,
        'P(< $70)'  : f"{(px < 70).mean():.1%}",
        'P(> $90)'  : f"{(px > 90).mean():.1%}",
        'P(in band)': f"{((px >= 70) & (px <= 90)).mean():.1%}",
        'median'    : f"${np.median(px):.1f}",
        '5th pct'   : f"${np.percentile(px, 5):.1f}",
        '95th pct'  : f"${np.percentile(px, 95):.1f}",
    })
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS, 'prob_table.csv'), index=False)
print('Saved: results/prob_table.csv')