"""
Cashout strategy backtesting for Sporty Hero crash data.

Strategies tested:
  1. Fixed multiplier — always cash out at Nx
  2. Martingale — double stake after a loss
  3. Anti-martingale — double stake after a win
  4. Streak-based — bet higher after N consecutive lows
  5. Kelly criterion — optimal sizing from empirical distribution
"""

import argparse
import sys

import numpy as np

from src.db import get_connection, init_db, get_all_crashes


# ── Helpers ──────────────────────────────────────────────────


def _load_data() -> np.ndarray:
    conn = get_connection()
    init_db(conn)
    values = get_all_crashes(conn)
    conn.close()
    if not values:
        print("No data. Run:  python -m src bot --rounds 200")
        sys.exit(1)
    return np.array(values)


def _section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _run_strategy(name: str, data: np.ndarray, stakes: list[float],
                  targets: list[float], base_stake: float) -> dict:
    """Run a strategy and return results."""
    total_wagered = sum(stakes)
    payouts = []
    wins = 0
    losses = 0

    for crash, stake, target in zip(data, stakes, targets):
        if crash >= target:
            payout = stake * target
            payouts.append(payout)
            wins += 1
        else:
            payouts.append(0.0)
            losses += 1

    total_payout = sum(payouts)
    profit = total_payout - total_wagered
    roi = (profit / total_wagered * 100) if total_wagered > 0 else 0

    # Track running balance for drawdown
    balance = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for payout, stake in zip(payouts, stakes):
        balance += payout - stake
        peak = max(peak, balance)
        drawdown = peak - balance
        max_drawdown = max(max_drawdown, drawdown)

    return {
        "name": name,
        "rounds": len(stakes),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(stakes) * 100 if stakes else 0,
        "total_wagered": total_wagered,
        "total_payout": total_payout,
        "profit": profit,
        "roi": roi,
        "max_drawdown": max_drawdown,
        "final_balance": balance,
    }


def _print_result(r: dict):
    color = "📈" if r["profit"] >= 0 else "📉"
    print(f"\n  {color} {r['name']}")
    print(f"    Rounds    : {r['rounds']}")
    print(f"    Wins      : {r['wins']}  ({r['win_rate']:.1f}%)")
    print(f"    Wagered   : {r['total_wagered']:>10.2f}")
    print(f"    Payout    : {r['total_payout']:>10.2f}")
    print(f"    Profit    : {r['profit']:>+10.2f}  (ROI: {r['roi']:>+.2f}%)")
    print(f"    Max DD    : {r['max_drawdown']:>10.2f}")
    print(f"    Final bal : {r['final_balance']:>+10.2f}")


# ── 1. Fixed multiplier ─────────────────────────────────────


def fixed_multiplier(data: np.ndarray, target: float, stake: float = 100.0) -> dict:
    """Always cash out at the same multiplier."""
    stakes = [stake] * len(data)
    targets = [target] * len(data)
    return _run_strategy(f"Fixed {target:.2f}x", data, stakes, targets, stake)


# ── 2. Martingale ───────────────────────────────────────────


def martingale(data: np.ndarray, target: float, base_stake: float = 100.0,
               max_stake: float = 10000.0) -> dict:
    """Double stake after each loss, reset after a win."""
    stakes = []
    targets_list = []
    current_stake = base_stake

    for crash in data:
        capped_stake = min(current_stake, max_stake)
        stakes.append(capped_stake)
        targets_list.append(target)
        if crash >= target:
            current_stake = base_stake  # win → reset
        else:
            current_stake *= 2  # loss → double

    return _run_strategy(f"Martingale {target:.2f}x", data, stakes, targets_list, base_stake)


# ── 3. Anti-martingale ──────────────────────────────────────


def anti_martingale(data: np.ndarray, target: float, base_stake: float = 100.0,
                    max_streak: int = 3) -> dict:
    """Double stake after a win, up to max_streak consecutive wins."""
    stakes = []
    targets_list = []
    current_stake = base_stake
    win_streak = 0

    for crash in data:
        stakes.append(current_stake)
        targets_list.append(target)
        if crash >= target:
            win_streak += 1
            if win_streak < max_streak:
                current_stake *= 2
            else:
                current_stake = base_stake
                win_streak = 0
        else:
            current_stake = base_stake
            win_streak = 0

    return _run_strategy(f"Anti-Martingale {target:.2f}x (max {max_streak})",
                         data, stakes, targets_list, base_stake)


# ── 4. Streak-based ─────────────────────────────────────────


def streak_based(data: np.ndarray, threshold: float = 2.0,
                 streak_trigger: int = 3, high_target: float = 2.0,
                 base_stake: float = 100.0, boost_mult: float = 3.0) -> dict:
    """
    After N consecutive crashes below threshold, bet bigger at high_target.
    Theory: if low streaks are followed by higher values, this profits.
    """
    stakes = []
    targets_list = []
    low_streak = 0

    for crash in data:
        if low_streak >= streak_trigger:
            stakes.append(base_stake * boost_mult)
            targets_list.append(high_target)
        else:
            stakes.append(base_stake)
            targets_list.append(high_target)

        if crash < threshold:
            low_streak += 1
        else:
            low_streak = 0

    return _run_strategy(
        f"Streak (after {streak_trigger}×<{threshold}x, {boost_mult}× stake)",
        data, stakes, targets_list, base_stake,
    )


# ── 5. Kelly criterion ─────────────────────────────────────


def kelly_criterion(data: np.ndarray, target: float, bankroll: float = 10000.0) -> dict:
    """
    Use Kelly formula to size bets optimally.
    f* = (p·b - q) / b  where p = win prob, b = net odds, q = 1-p
    """
    # Estimate win probability from data
    p = np.mean(data >= target)
    b = target - 1  # net payout per unit bet
    q = 1 - p

    if b <= 0:
        print(f"  Kelly: target {target}x gives b={b}, skipping")
        return _run_strategy(f"Kelly {target:.2f}x", data, [], [], 0)

    kelly_frac = (p * b - q) / b
    kelly_frac = max(0, min(kelly_frac, 0.25))  # cap at 25%

    print(f"\n  Kelly fraction for {target:.2f}x:  f* = {kelly_frac:.4f}")
    print(f"    Win prob  = {p:.4f}")
    print(f"    Net odds  = {b:.2f}")

    stakes = []
    targets_list = []
    current_bankroll = bankroll

    for crash in data:
        bet = max(1.0, current_bankroll * kelly_frac)
        stakes.append(bet)
        targets_list.append(target)
        if crash >= target:
            current_bankroll += bet * (target - 1)
        else:
            current_bankroll -= bet
        if current_bankroll <= 0:
            break

    # Pad if busted early
    while len(stakes) < len(data):
        stakes.append(0)
        targets_list.append(target)

    return _run_strategy(f"Kelly {target:.2f}x (f*={kelly_frac:.3f})",
                         data[:len(stakes)], stakes, targets_list, bankroll)


# ── Summary ─────────────────────────────────────────────────


def run_all(data: np.ndarray, base_stake: float = 100.0):
    """Run all strategies and print comparison table."""

    _section("STRATEGY BACKTESTING")
    print(f"  Data: {len(data)} rounds | Base stake: {base_stake}")

    results = []

    # Fixed multiplier at various targets
    _section("FIXED MULTIPLIER")
    for target in [1.5, 2.0, 3.0, 5.0, 10.0]:
        r = fixed_multiplier(data, target, base_stake)
        _print_result(r)
        results.append(r)

    # Martingale
    _section("MARTINGALE")
    for target in [1.5, 2.0]:
        r = martingale(data, target, base_stake)
        _print_result(r)
        results.append(r)

    # Anti-martingale
    _section("ANTI-MARTINGALE")
    for target in [2.0, 3.0]:
        r = anti_martingale(data, target, base_stake)
        _print_result(r)
        results.append(r)

    # Streak-based
    _section("STREAK-BASED")
    for trigger in [3, 5]:
        r = streak_based(data, streak_trigger=trigger, base_stake=base_stake)
        _print_result(r)
        results.append(r)

    # Kelly
    _section("KELLY CRITERION")
    for target in [1.5, 2.0, 3.0]:
        r = kelly_criterion(data, target)
        _print_result(r)
        results.append(r)

    # Comparison table
    _section("COMPARISON TABLE")
    print(f"\n  {'Strategy':<42} {'Win%':>6} {'ROI%':>8} {'Profit':>10} {'MaxDD':>10}")
    print(f"  {'─' * 80}")
    for r in sorted(results, key=lambda x: x["roi"], reverse=True):
        print(f"  {r['name']:<42} {r['win_rate']:>5.1f}% {r['roi']:>+7.2f}% "
              f"{r['profit']:>+10.2f} {r['max_drawdown']:>10.2f}")


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest cashout strategies")
    parser.add_argument("--stake", type=float, default=100.0,
                        help="Base stake per round (default: 100)")
    args = parser.parse_args()

    data = _load_data()
    run_all(data, base_stake=args.stake)

    _section("DONE")
    print(f"  Backtested {len(data)} rounds.\n")


if __name__ == "__main__":
    main()
