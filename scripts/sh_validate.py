"""Validation: is the 'under-dispersion' flag real, or an artifact of the
permutation dispersion test? We run the IDENTICAL test on synthetic iid data
generated from the known model (crash = max(1, round(0.97/v, 2)), v~U(0,1))
mapped onto the REAL per-hour round structure. If iid data also flags, the
test is biased and the 'finding' is spurious.

Also checks the OUTCOME (multiplier) sequence for autocorrelation directly --
if outcomes are serially independent, hourly under-dispersion cannot be a real
dynamical effect.
"""
import os, sqlite3
from datetime import datetime
import numpy as np
from scipy import stats as sp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "data", "sportyhero_seeds.db")


def epoch_ms(ts):
    try:
        return datetime.fromisoformat(ts).timestamp() * 1000.0
    except Exception:
        return None


con = sqlite3.connect(DB)
rows = con.execute("SELECT created_at, start_time, house_coefficient "
                   "FROM rounds ORDER BY round_id ASC").fetchall()
con.close()
t = []; hc = []
for ca, st, h in rows:
    e = epoch_ms(ca or st)
    if e is not None:
        t.append(e); hc.append(float(h))
t = np.array(t); hc = np.array(hc)

# replicate full-hour bucketing from sh_temporal
hour_bucket = (t // 3600_000).astype(np.int64)
hours = np.unique(hour_bucket)
full = hours[1:-1]
hmap = {h: i for i, h in enumerate(sorted(full.tolist()))}
hour_idx = np.array([hmap.get(h, -1) for h in hour_bucket])
mask = hour_idx >= 0
idx = hour_idx[mask]
hcm = hc[mask]
H = len(full)
N = len(idx)
print(f"full-hour rounds N={N} across H={H} hours")


def perm_p_for_indicator(ind, n_perm=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    counts = np.bincount(idx, weights=ind, minlength=H)
    D_obs = counts.var(ddof=1) / counts.mean()
    Ds = np.empty(n_perm)
    for k in range(n_perm):
        c = np.bincount(idx, weights=rng.permutation(ind), minlength=H)
        Ds[k] = c.var(ddof=1) / c.mean()
    p = 2 * min((Ds <= D_obs).mean(), (Ds >= D_obs).mean())
    return D_obs, min(p, 1.0), Ds


# --- 1. reproduce the real-data perm-p for >=2x and >=5x ---
print("\n=== REAL DATA ===")
for X in (2, 5):
    ind = (hcm >= X).astype(float)
    D, p, _ = perm_p_for_indicator(ind)
    print(f">= {X}x: D_obs={D:.3f} perm_p={p:.4f}")

# --- 2. OUTCOME autocorrelation (are multipliers serially independent?) ---
print("\n=== OUTCOME-SEQUENCE AUTOCORRELATION (full 10k) ===")
lg = np.log(hc)
lgm = lg - lg.mean()
den = np.sum(lgm * lgm)
band = 1.96 / np.sqrt(len(hc))
sigc = 0
for lag in range(1, 31):
    a = np.sum(lgm[:-lag] * lgm[lag:]) / den
    if abs(a) > band:
        sigc += 1
print(f"log-multiplier ACF lags 1-30: {sigc} exceed +/-{band:.4f} (expect ~1.5)")
for X in (2, 5):
    ind = (hc >= X).astype(float)
    im = ind - ind.mean()
    a1 = np.sum(im[:-1] * im[1:]) / np.sum(im * im)
    print(f">= {X}x indicator lag-1 autocorr = {a1:+.4f} (|crit|={band:.4f})")

# --- 3. SYNTHETIC iid CONTROL: does the test false-flag on iid data? ---
print("\n=== SYNTHETIC iid CONTROL (same hour structure) ===")
n_synth = 300
rng = np.random.default_rng(12345)
for X in (2, 5):
    p_marg = float(np.mean(hcm >= X))
    ps = np.empty(n_synth)
    for s in range(n_synth):
        v = rng.random(N)
        raw = 0.97 / v
        hc_syn = np.maximum(1.0, np.floor(raw * 100 + 0.5) / 100)
        ind = (hc_syn >= X).astype(float)
        # cheaper: 400 perms for the sweep
        _, ps[s], _ = perm_p_for_indicator(ind, n_perm=400,
                                            rng=np.random.default_rng(1000 + s))
    frac_flag = np.mean(ps < 0.01)
    print(f">= {X}x (p={p_marg:.3f}): synthetic iid perm_p  "
          f"mean={ps.mean():.3f} median={np.median(ps):.3f} "
          f"frac<0.01={frac_flag:.3f}  (calibrated => ~0.01, mean ~0.5)")
print("\nIf frac<0.01 is ~0.01 and mean ~0.5, the test is calibrated and the REAL")
print("flag is a true signal. If synthetic iid ALSO flags often, the test is biased.")
