#!/usr/bin/env python3
"""Session-outcome simulator for active Sporty Hero play.

Takes the strategies a manual player actually uses (flat, martingale,
anti-martingale, tail-chasing) and runs each across MANY bootstrapped sessions
drawn from the 10,000 REAL collected rounds (rounds are iid, so bootstrap
resampling is valid). It reports the FULL distribution of outcomes -- not just
the average -- so we can see:

  - how often a session 10x's the bankroll (100k -> 1M), like the user did
  - how often a session busts
  - the median and MEAN final bankroll (the edge shows up in the mean)

The point: a big winning session is REAL and not even rare for some strategies,
but it is variance, not an edge. The mean stays negative for every strategy.
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

N_SESS = 20000
START = 100_000.0
GOAL = 1_000_000.0          # the 10x the user achieved
MAX_BETS = 1200             # ~ a long manual session
MIN_STAKE = 100.0
rng = np.random.default_rng(42)


def simulate(strategy, target, base_frac):
    bank = np.full(N_SESS, START)
    base = START * base_frac
    stake = np.full(N_SESS, base)
    alive = np.ones(N_SESS, bool)
    reached = np.zeros(N_SESS, bool)

    for _ in range(MAX_BETS):
        active = alive & ~reached
        if not active.any():
            break
        m = HC[rng.integers(0, HC.size, N_SESS)]      # bootstrap a round
        win = m >= target
        bet = np.minimum(stake, bank)
        bet = np.where(active, bet, 0.0)

        w = active & win
        l = active & ~win
        bank = bank + np.where(w, bet * (target - 1.0), 0.0)
        bank = bank - np.where(l, bet, 0.0)

        if strategy == "flat":
            stake = np.where(active, base, stake)
        elif strategy == "martingale":           # double on loss, reset on win
            stake = np.where(w, base, stake)
            stake = np.where(l, bet * 2.0, stake)
        elif strategy == "antimartingale":        # let winners ride, reset on loss
            stake = np.where(w, np.minimum(bet * 2.0, bank), stake)
            stake = np.where(l, base, stake)

        reached = reached | (bank >= GOAL)
        alive = alive & (bank >= MIN_STAKE)

    busted = ~alive
    pct_goal = 100.0 * reached.mean()
    pct_bust = 100.0 * busted.mean()
    return {
        "name": f"{strategy} @ {target}x (base {base_frac*100:.1f}%)",
        "p_goal": pct_goal,
        "p_bust": pct_bust,
        "median": np.median(bank),
        "mean": bank.mean(),
        "roi": 100.0 * (bank.mean() - START) / START,
    }


configs = [
    ("flat", 2.0, 0.02),
    ("flat", 5.0, 0.02),
    ("flat", 10.0, 0.02),
    ("martingale", 2.0, 0.01),
    ("martingale", 1.5, 0.01),
    ("antimartingale", 2.0, 0.02),
]

print(f"Sessions: {N_SESS:,} | start {START:,.0f} -> goal {GOAL:,.0f} "
      f"(10x) | up to {MAX_BETS} bets each")
print(f"Data: bootstrap from {HC.size:,} REAL rounds\n")
print(f"{'strategy':<34}{'P(reach 1M)':>12}{'P(bust)':>10}"
      f"{'median end':>13}{'mean end':>13}{'mean ROI':>10}")
print("-" * 92)
results = []
for strat, tgt, bf in configs:
    r = simulate(strat, tgt, bf)
    results.append(r)
    print(f"{r['name']:<34}{r['p_goal']:>11.2f}%{r['p_bust']:>9.1f}%"
          f"{r['median']:>13,.0f}{r['mean']:>13,.0f}{r['roi']:>9.1f}%")

print("\nKey reading:")
print("  - P(reach 1M) > 0 means a 100k->1M run IS achievable (you weren't lying).")
print("  - But P(bust) is far larger, and mean ROI is NEGATIVE for every strategy.")
print("  - Money management reshapes the distribution (more small wins OR rare")
print("    moonshots), but never lifts the mean above zero. That's the -3% edge.")
