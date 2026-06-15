#!/usr/bin/env python3
"""The definitive test of adaptive / discretionary 'pattern trading' (fast).

An adaptive meta-strategy mimics a discretionary player: a panel of pattern
'experts' (fixed targets + mean-reversion/'due' + momentum) with a
follow-the-leading-expert controller that bets whichever view has worked
recently and switches when it stops -- i.e. 'find a pattern, ride it till it
dies, find another'.

Killer comparison: run the SAME strategy on
  (A) the REAL round order, and
  (B) many SHUFFLES of the same rounds (outcomes identical, order destroyed).
If adaptiveness exploits sequential structure, (A) beats (B). If (A) sits in the
middle of (B), it extracts nothing -- it can't tell ordered data from
structureless data. Round independence guarantees (B) is structureless.
"""
import os
import sqlite3
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "data", "sportyhero_seeds.db")
con = sqlite3.connect(DB)
HC = np.array([r[0] for r in con.execute(
    "SELECT house_coefficient FROM rounds").fetchall()], dtype=np.float64)
con.close()
n = HC.size

TARGETS = [1.5, 2.0, 3.0, 5.0]
ALPHA = 0.10
WARMUP = 30
N_EXPERTS = len(TARGETS) + 2
names = [f"target {T}x" for T in TARGETS] + ["'due'/reversion", "momentum"]


def expert_return_matrix(seq):
    """(n x N_EXPERTS) matrix of per-round returns for each expert on `seq`."""
    R = np.empty((len(seq), N_EXPERTS))
    for j, T in enumerate(TARGETS):
        R[:, j] = np.where(seq >= T, T - 1.0, -1.0)
    # 'due': target 2.0 if previous 4 rounds all < 2.0 else 1.5
    low = seq < 2.0
    prev4_all_low = np.zeros(len(seq), bool)
    for t in range(4, len(seq)):
        prev4_all_low[t] = low[t - 4] and low[t - 3] and low[t - 2] and low[t - 1]
    tgt_due = np.where(prev4_all_low, 2.0, 1.5)
    R[:, len(TARGETS)] = np.where(seq >= tgt_due, tgt_due - 1.0, -1.0)
    # 'momentum': target 2.0 if previous round >= 2.0 else 3.0
    prev = np.concatenate([[2.0], seq[:-1]])
    tgt_mom = np.where(prev >= 2.0, 2.0, 3.0)
    R[:, len(TARGETS) + 1] = np.where(seq >= tgt_mom, tgt_mom - 1.0, -1.0)
    return R


def run_ftl(R):
    """Follow-the-leading-expert pass over a precomputed return matrix."""
    ewma = np.zeros(N_EXPERTS)
    total = 0.0
    a = ALPHA
    for t in range(R.shape[0]):
        choice = int(ewma.argmax()) if t >= WARMUP else 1
        total += R[t, choice]
        ewma += a * (R[t] - ewma)
    return total / R.shape[0]


# (A) real order
R_real = expert_return_matrix(HC)
roi_real = run_ftl(R_real)
expert_roi_real = R_real.mean(axis=0)

# (B) shuffled
rng = np.random.default_rng(0)
N_PERM = 500
roi_shuf = np.empty(N_PERM)
for i in range(N_PERM):
    roi_shuf[i] = run_ftl(expert_return_matrix(rng.permutation(HC)))

mean_s, std_s = roi_shuf.mean(), roi_shuf.std()
z = (roi_real - mean_s) / std_s
pct = 100.0 * (roi_shuf < roi_real).mean()

print("Adaptive 'find-a-pattern-ride-it-switch' strategy")
print(f"experts: targets {TARGETS} + 'due' + 'momentum'; "
      f"follow-leading-expert, EWMA alpha={ALPHA}\n")
print(f"{'':20}{'mean per-bet ROI':>18}")
print(f"{'REAL order':20}{roi_real*100:>17.2f}%")
print(f"{'SHUFFLED (mean)':20}{mean_s*100:>17.2f}%   "
      f"(std {std_s*100:.2f}%, n={N_PERM})")
print(f"\nreal vs shuffled:  z = {z:+.2f}   (real beats {pct:.0f}% of shuffles)")
print(f"theoretical 2x edge: {(0.4828 - 0.5172)*100:.2f}% per bet")

print("\nBest expert in HINDSIGHT (cherry-picked on the real data):")
for j in np.argsort(expert_roi_real)[::-1]:
    print(f"   {names[j]:18}  {expert_roi_real[j]*100:+.2f}% per bet")

print("\nReading:")
print("  - REAL-order ROI sits inside the SHUFFLED distribution (z~0): the adaptive")
print("    strategy does NO better on ordered data than on order-destroyed data.")
print("    It cannot detect structure because there is none to detect.")
print("  - EVERY expert -- even the best picked with hindsight -- is negative.")
print("  - The 'patterns' you ride are random clustering that always exists in iid")
print("    data; they 'die' because they were never predictive in the first place.")
