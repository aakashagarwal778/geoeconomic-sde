"""
cases/02_em_fx_sanctions/run.py

Entry point for Case 2. Reads calibrated parameters from
calibration/params_case02.json. Falls back to prior values if
calibration has not been run.

Produces:
  1. results/paths.png                 -- Monte Carlo path fan (calm vs tension)
  2. results/prob_table.csv            -- Scenario probability table
  3. results/distribution_comparison.png -- Calm vs tension distribution

Run calibration first:
    python calibration/estimate_case02.py

Then run from repo root:
    python cases/02_em_fx_sanctions/run.py
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.euler_maruyama import solve

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "model", os.path.join(os.path.dirname(__file__), "model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
USDRUBModel    = _mod.USDRUBModel
SignalGenerator = _mod.SignalGenerator

# ── Parameters ─────────────────────────────────────────────────
PARAMS_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                           'calibration', 'params_case02.json')

PRIOR = {
    "mu": 0.30, "sigma": 0.20, "lam0": 2.0, "lam1": 8.0,
    "mu_j": 0.04, "sigma_j": 0.08,
    "calibrated_on": None, "n_obs": None
}

if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH) as f:
        raw = json.load(f)
else:
    raw = PRIOR
    print("params_case02.json not found — using prior values")

PARAMS = {k: raw[k] for k in ["mu","sigma","lam0","lam1","mu_j","sigma_j"]}

calibrated = raw.get("calibrated_on")
if calibrated:
    print(f"MLE-calibrated parameters (fitted {calibrated}, n={raw.get('n_obs')} obs)")
else:
    print("Prior parameters — run calibration/estimate_case02.py to fit from data")
print(f"  mu={PARAMS['mu']:.4f} ({PARAMS['mu']*100:.1f}%/yr)  sigma={PARAMS['sigma']}")
print(f"  lam0={PARAMS['lam0']}  lam1={PARAMS['lam1']}  mu_j={PARAMS['mu_j']}  sigma_j={PARAMS['sigma_j']}\n")

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
RESULTS = os.path.join(os.path.dirname(__file__), 'results')

X0 = np.log(38.0)
T  = 1.0
dt = 1/252
N  = 10_000
DAY = 252

# ── Simulate calm and tension regimes ──────────────────────────
signal_calm    = SignalGenerator(mode="simulate", p_escalate=0.01,
                                 p_deescalate=0.20, initial_state=0)
signal_tension = SignalGenerator(mode="simulate", p_escalate=0.01,
                                 p_deescalate=0.05, initial_state=1)

prices_calm    = np.exp(solve(USDRUBModel(PARAMS, signal_calm),    X0, T, dt, N))
prices_tension = np.exp(solve(USDRUBModel(PARAMS, signal_tension), X0, T, dt, N))

n_steps = prices_calm.shape[0]
t_axis  = np.linspace(0, T*252, n_steps)

thr_hi = np.exp(X0) * 1.05  # 5% depreciation from starting rate
thr_lo = np.exp(X0) * 0.95  # 5% appreciation from starting rate

# ── 1. Path fan ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, prices, title, color in [
    (ax1, prices_calm,    "Calm regime  (s=0,  λ=λ₀)", '#1a6b8a'),
    (ax2, prices_tension, "Tension regime  (s=1,  λ=λ₁)", '#b03a2e'),
]:
    for i in np.random.choice(N, 150, replace=False):
        ax.plot(t_axis, prices[:,i], color=color, alpha=0.05, linewidth=0.5)
    pcts = np.percentile(prices, [5,25,50,75,95], axis=1)
    ax.fill_between(t_axis, pcts[0], pcts[4], color=color, alpha=0.10)
    ax.fill_between(t_axis, pcts[1], pcts[3], color=color, alpha=0.22)
    ax.plot(t_axis, pcts[2], color=color, linewidth=1.8, label='Median')
    ax.axhline(np.exp(X0), color='gray', linewidth=0.9,
               linestyle=':', label=f"X0 = {np.exp(X0):.0f}")
    ax.set_ylim(np.percentile(prices,1)*0.92, np.percentile(prices,99)*1.08)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Trading days from today', fontsize=9.5)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.set_ylabel('USD/TRY', fontsize=9.5)
label = "MLE calibrated" if calibrated else "prior estimates"
fig.suptitle(f'Case 2 — USD/TRY Monte Carlo ({label})\n'
             f'10,000 paths  |  X0={np.exp(X0):.0f}  |  mu={PARAMS["mu"]*100:.1f}%/yr',
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'paths.png'), dpi=180, bbox_inches='tight')
plt.close()
print('Saved: results/paths.png')

# ── 2. Probability table ───────────────────────────────────────
rows = []
for regime, prices in [('calm', prices_calm), ('tension', prices_tension)]:
    for day in [30, 60, 90]:
        px = prices[min(day, n_steps-1)]
        rows.append({
            'scenario'  : f"{regime} {day}d",
            'P(dep>5%)': f"{(px > thr_hi).mean():.1%}",
            'P(app>5%)': f"{(px < thr_lo).mean():.1%}",
            'median'    : f"{np.median(px):.1f}",
            '5th pct'   : f"{np.percentile(px,5):.1f}",
            '95th pct'  : f"{np.percentile(px,95):.1f}",
        })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(os.path.join(RESULTS, 'prob_table.csv'), index=False)
print('Saved: results/prob_table.csv')

# ── 3. Distribution comparison — calm vs tension at 90d ───────
px_calm    = prices_calm[DAY]
px_tension = prices_tension[DAY]

x_lo = min(px_calm.min(), px_tension.min()) * 0.85
x_hi = max(px_calm.max(), px_tension.max()) * 1.05
x_hi = min(x_hi, np.percentile(px_tension, 99) * 1.10)
x    = np.linspace(x_lo, x_hi, 800)

kde_c = gaussian_kde(px_calm,    bw_method=0.12)(x)
kde_t = gaussian_kde(px_tension, bw_method=0.12)(x)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.plot(x, kde_c, color='#1a6b8a', linewidth=1.8, label='Calm regime (s=0)')
ax.fill_between(x, kde_c, color='#1a6b8a', alpha=0.15)
ax.plot(x, kde_t, color='#b03a2e', linewidth=1.8, label='Tension regime (s=1)')
ax.fill_between(x, kde_t, color='#b03a2e', alpha=0.15)
# no theta line for GBM drift model

col_labels = ['Regime', 'P(dep>5%)', 'P(app>5%)', 'Median', '95th pct']
c90 = prices_calm[DAY]; t90 = prices_tension[DAY]
table_data = [
    ['Calm',    f"{(c90>thr_hi).mean():.1%}", f"{(c90<thr_lo).mean():.1%}",
     f"{np.median(c90):.1f}", f"{np.percentile(c90,95):.1f}"],
    ['Tension', f"{(t90>thr_hi).mean():.1%}", f"{(t90<thr_lo).mean():.1%}",
     f"{np.median(t90):.1f}", f"{np.percentile(t90,95):.1f}"],
]
tbl = ax.table(cellText=table_data, colLabels=col_labels,
               bbox=[0.48, 0.74, 0.51, 0.22])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cccccc'); cell.set_linewidth(0.5); cell.PAD = 0.08
    if r == 0:
        cell.set_facecolor('#efefef')
        cell.set_text_props(fontweight='bold', fontsize=8)
    elif c == 0:
        cell.set_facecolor('#f7f7f7')
        cell.set_text_props(fontweight='bold', fontsize=8.5)
    else:
        cell.set_facecolor('white')

ax.set_xlabel('USD/TRY at 1-year horizon', fontsize=10)
ax.set_ylabel('Probability density', fontsize=10)
ax.set_title('USD/TRY distribution at 1-year horizon — Calm vs Tension regime\n'
             f'10,000 paths  |  Starting rate {np.exp(X0):.0f}  |  1-year horizon', fontsize=10.5)
ax.legend(fontsize=9, loc='upper right', frameon=True,
          framealpha=0.95, edgecolor='#dddddd')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')
ax.grid(axis='y', color='#eeeeee', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'distribution_comparison.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print('Saved: results/distribution_comparison.png')