"""
cases/01_oil_opec/run.py
Reads calibrated parameters from calibration/params.json.
Falls back to prior values if calibration has not been run.
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

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

calibrated = raw.get("calibrated_on")
if calibrated:
    print(f"MLE-calibrated parameters (fitted {calibrated}, n={raw.get('n_obs')} obs)")
else:
    print("Prior parameters — run calibration/estimate.py to fit from data")

print(f"  theta=${raw['theta_price']}  kappa={PARAMS['kappa']}  sigma0={PARAMS['sigma0']}  alpha={PARAMS['alpha']}")
print(f"  lambda={PARAMS['lam']}  mu_j={PARAMS['mu_j']}  sigma_j={PARAMS['sigma_j']}\n")

X0=np.log(82.0); T=0.5; dt=1/252; N=10_000
os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), 'results')

model  = OilOPECModel(PARAMS)
paths  = solve(model, X0, T, dt, N)
prices = np.exp(paths)
n_steps= paths.shape[0]
t_axis = np.linspace(0, T*252, n_steps)

fig, ax = plt.subplots(figsize=(11,5))
for i in np.random.choice(N, 200, replace=False):
    ax.plot(t_axis, prices[:,i], color='#1a6b8a', alpha=0.06, linewidth=0.6)
pcts = np.percentile(prices, [5,25,50,75,95], axis=1)
ax.fill_between(t_axis, pcts[0], pcts[4], color='#1a6b8a', alpha=0.10, label='5-95th pct')
ax.fill_between(t_axis, pcts[1], pcts[3], color='#1a6b8a', alpha=0.20, label='25-75th pct')
ax.plot(t_axis, pcts[2], color='#1a6b8a', linewidth=1.8, label='Median')
ax.axhline(70, color='#e05c2a', linewidth=1.0, linestyle='--', alpha=0.7, label='OPEC floor ~$70')
ax.axhline(90, color='#2a8ae0', linewidth=1.0, linestyle='--', alpha=0.7, label='Shale ceiling ~$90')
ax.axhline(raw['theta_price'], color='gray', linewidth=0.8, linestyle=':', label=f"theta=${raw['theta_price']}")
label = "MLE calibrated" if calibrated else "prior estimates"
ax.set_xlabel('Trading days from today'); ax.set_ylabel('Brent crude (USD)')
ax.set_title(f'Case 1 - Brent Crude Monte Carlo ({label})\nOU + state-dependent sigma + jumps - 10,000 paths - X0=$82 - theta=${raw["theta_price"]}')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); plt.tight_layout()
plt.savefig(os.path.join(RESULTS,'paths.png'), dpi=150); plt.close()
print('Saved: results/paths.png')

rows=[]
for label, day in [('30d',30),('60d',60),('90d',90)]:
    px = prices[min(day, n_steps-1)]
    rows.append({'horizon':label,
                 'P(< $70)':f"{(px<70).mean():.1%}",
                 'P(> $90)':f"{(px>90).mean():.1%}",
                 'P(in band)':f"{((px>=70)&(px<=90)).mean():.1%}",
                 'median':f"${np.median(px):.1f}",
                 '5th pct':f"${np.percentile(px,5):.1f}",
                 '95th pct':f"${np.percentile(px,95):.1f}"})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS,'prob_table.csv'), index=False)
print('Saved: results/prob_table.csv')