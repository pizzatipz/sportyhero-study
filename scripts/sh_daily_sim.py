#!/usr/bin/env python3
"""Does 'hit a daily target, stop, repeat' beat the edge?

Tests the most disciplined version of the user's plan: each day, bring a fixed
buy-in, play the 'let it ride' style until either a DAILY PROFIT TARGET is hit
(then stop and bank it) or the day busts -- repeated over many days. Banks
profits, brings a fresh buy-in each day (steelman: assumes they can always
fund it). Run across many independent players on bootstrapped REAL rounds.

This is the optional-stopping question: can a stopping rule turn a negative-EV
game positive? Theory says no. We show it numerically with the real data.
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

N = 20000                 # independent players
BUYIN = 100_000.0
BASE = 0.02 * BUYIN       # base stake
TARGET = 2.0              # cashout multiplier
MAX_BETS_DAY = 400
MIN_STAKE = 100.0
rng = np.random.default_rng(7)


def play_days(n_days, tp_mult, deep_risk):
    """tp_mult: stop the day at bankroll >= tp_mult*BUYIN.
       deep_risk: if True, only other exit is ruin; else stop-loss at 0.5*BUYIN."""
    cum = np.zeros(N)              # cumulative net P&L across days
    green_days = np.zeros(N)
    for _ in range(n_days):
        bank = np.full(N, BUYIN)
        stake = np.full(N, BASE)
        done = np.zeros(N, bool)
        tp = tp_mult * BUYIN
        sl = 0.0 if deep_risk else 0.5 * BUYIN
        for _ in range(MAX_BETS_DAY):
            active = ~done
            if not active.any():
                break
            m = HC[rng.integers(0, HC.size, N)]
            win = m >= TARGET
            bet = np.where(active, np.minimum(stake, bank), 0.0)
            w = active & win
            l = active & ~win
            bank = bank + np.where(w, bet * (TARGET - 1.0), 0.0)
            bank = bank - np.where(l, bet, 0.0)
            stake = np.where(w, np.minimum(bet * 2.0, bank), stake)   # ride wins
            stake = np.where(l, BASE, stake)
            done = done | (bank >= tp) | (bank <= sl) | (bank < MIN_STAKE)
        cum += (bank - BUYIN)
        green_days += (bank > BUYIN)
    return cum, green_days


for label, tp, deep in [("modest target +20%, ride till goal-or-bust", 1.2, True),
                        ("ambitious +100% (2x), ride till goal-or-bust", 2.0, True),
                        ("symmetric +50% / -50% daily", 1.5, False)]:
    print("=" * 78)
    print(f"PLAN: {label}")
    print(f"{'after N days':>14}{'win-rate/day':>14}{'P(net up)':>11}"
          f"{'median net':>14}{'mean net':>14}")
    for nd in (10, 30, 100, 365):
        cum, green = play_days(nd, tp, deep)
        wr = 100.0 * green.mean() / nd
        p_up = 100.0 * (cum > 0).mean()
        print(f"{nd:>11} d {wr:>12.0f}%{p_up:>10.1f}%"
              f"{np.median(cum):>14,.0f}{cum.mean():>14,.0f}")
    print()

print("Reading:")
print("  - 'win-rate/day' can be HIGH -- most days you do hit the target and stop")
print("    feeling like a winner. This is why the plan feels like it works.")
print("  - But P(net up) FALLS as days accumulate, and mean net P&L is NEGATIVE")
print("    and grows more negative over time. The rare bust day gives back many")
print("    green days. No stopping rule beats a negative-EV game (optional")
print("    stopping theorem) -- it only delays and concentrates the loss.")
