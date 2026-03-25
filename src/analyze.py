"""
Statistical analysis pipeline for Sporty Hero crash data.

Analyses performed:
  1. Descriptive statistics (mean, median, std, percentiles)
  2. Distribution fitting (exponential, log-normal)
  3. Streak / run analysis (consecutive lows/highs)
  4. Autocorrelation (are consecutive crashes correlated?)
  5. Frequency histogram + fitted PDF overlay
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import acf

from src.db import get_connection, init_db, get_all_crashes


# ── Helpers ──────────────────────────────────────────────────


def _load_data() -> np.ndarray:
    """Load crash values from DB as a numpy array."""
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


# ── 1. Descriptive stats ────────────────────────────────────


def descriptive(data: np.ndarray):
    _section("DESCRIPTIVE STATISTICS")
    n = len(data)
    print(f"  Rounds collected : {n}")
    print(f"  Mean             : {data.mean():.3f}x")
    print(f"  Median           : {np.median(data):.3f}x")
    print(f"  Std dev          : {data.std():.3f}")
    print(f"  Min / Max        : {data.min():.2f}x / {data.max():.2f}x")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p:<2}              : {np.percentile(data, p):.2f}x")

    # Threshold table
    print(f"\n  {'Threshold':>10}  {'Count':>6}  {'Pct':>7}")
    print(f"  {'─' * 28}")
    for t in [1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        below = int(np.sum(data < t))
        pct = below / n * 100
        print(f"  {'< ' + f'{t:.1f}x':>10}  {below:>6}  {pct:>6.1f}%")


# ── 2. Distribution fitting ─────────────────────────────────


def distribution_fit(data: np.ndarray):
    _section("DISTRIBUTION FITTING")

    # Shift data by 1 (crash values are >= 1.0)
    shifted = data - 1.0
    shifted = shifted[shifted > 0]  # drop exact 1.00 for log-fitting

    # Exponential fit on shifted data
    exp_loc, exp_scale = 0, shifted.mean()
    exp_ks, exp_p = sp_stats.kstest(shifted, "expon", args=(exp_loc, exp_scale))
    print(f"\n  Exponential  (on x-1):")
    print(f"    λ (rate)   = {1 / exp_scale:.4f}")
    print(f"    mean       = {exp_scale:.4f}")
    print(f"    KS stat    = {exp_ks:.4f}")
    print(f"    KS p-value = {exp_p:.4f}")
    print(f"    {'✅ Good fit' if exp_p > 0.05 else '❌ Poor fit'} (α=0.05)")

    # Log-normal fit
    if len(shifted) > 2:
        ln_shape, ln_loc, ln_scale = sp_stats.lognorm.fit(shifted, floc=0)
        ln_ks, ln_p = sp_stats.kstest(shifted, "lognorm", args=(ln_shape, ln_loc, ln_scale))
        print(f"\n  Log-normal   (on x-1):")
        print(f"    σ (shape)  = {ln_shape:.4f}")
        print(f"    scale      = {ln_scale:.4f}")
        print(f"    KS stat    = {ln_ks:.4f}")
        print(f"    KS p-value = {ln_p:.4f}")
        print(f"    {'✅ Good fit' if ln_p > 0.05 else '❌ Poor fit'} (α=0.05)")

    # Pareto fit
    if len(shifted) > 2:
        par_b, par_loc, par_scale = sp_stats.pareto.fit(shifted, floc=0)
        par_ks, par_p = sp_stats.kstest(shifted, "pareto", args=(par_b, par_loc, par_scale))
        print(f"\n  Pareto       (on x-1):")
        print(f"    α (shape)  = {par_b:.4f}")
        print(f"    scale      = {par_scale:.4f}")
        print(f"    KS stat    = {par_ks:.4f}")
        print(f"    KS p-value = {par_p:.4f}")
        print(f"    {'✅ Good fit' if par_p > 0.05 else '❌ Poor fit'} (α=0.05)")


# ── 3. Streak analysis ──────────────────────────────────────


def streak_analysis(data: np.ndarray, threshold: float = 2.0):
    _section(f"STREAK ANALYSIS  (threshold = {threshold:.1f}x)")

    below = data < threshold
    streaks_low = []
    streaks_high = []
    current = 0
    is_low = below[0]

    for b in below:
        if b == is_low:
            current += 1
        else:
            (streaks_low if is_low else streaks_high).append(current)
            current = 1
            is_low = b
    (streaks_low if is_low else streaks_high).append(current)

    for label, streaks in [("LOW  (<2x)", streaks_low), ("HIGH (≥2x)", streaks_high)]:
        if streaks:
            arr = np.array(streaks)
            print(f"\n  {label} streaks:")
            print(f"    Count      : {len(arr)}")
            print(f"    Mean len   : {arr.mean():.2f}")
            print(f"    Max len    : {arr.max()}")
            print(f"    Median len : {np.median(arr):.1f}")
        else:
            print(f"\n  {label} streaks: none")

    # Runs test for randomness
    n_low = int(np.sum(below))
    n_high = len(data) - n_low
    n_runs = len(streaks_low) + len(streaks_high)
    if n_low > 0 and n_high > 0:
        expected_runs = 1 + (2 * n_low * n_high) / (n_low + n_high)
        var_runs = ((2 * n_low * n_high) * (2 * n_low * n_high - n_low - n_high)) / \
                   (((n_low + n_high) ** 2) * (n_low + n_high - 1))
        if var_runs > 0:
            z_runs = (n_runs - expected_runs) / np.sqrt(var_runs)
            p_runs = 2 * sp_stats.norm.sf(abs(z_runs))
            print(f"\n  Wald-Wolfowitz runs test:")
            print(f"    Runs       = {n_runs}")
            print(f"    Expected   = {expected_runs:.1f}")
            print(f"    Z          = {z_runs:.3f}")
            print(f"    p-value    = {p_runs:.4f}")
            print(f"    {'✅ Consistent with random' if p_runs > 0.05 else '⚠️  Non-random pattern detected'}")


# ── 4. Autocorrelation ──────────────────────────────────────


def autocorrelation(data: np.ndarray, max_lag: int = 10):
    _section("AUTOCORRELATION")

    if len(data) < max_lag + 2:
        print("  Not enough data for autocorrelation analysis.")
        return

    nlags = min(max_lag, len(data) // 2 - 1)
    acf_vals, confint = acf(data, nlags=nlags, alpha=0.05)
    ci_width = (confint[1:, 1] - confint[1:, 0]) / 2

    print(f"\n  {'Lag':>4}  {'ACF':>8}  {'95% CI':>14}  {'Sig?':>5}")
    print(f"  {'─' * 36}")
    for i in range(1, nlags + 1):
        lo = acf_vals[i] - ci_width[i - 1]
        hi = acf_vals[i] + ci_width[i - 1]
        sig = "  *" if abs(acf_vals[i]) > ci_width[i - 1] else ""
        print(f"  {i:>4}  {acf_vals[i]:>+8.4f}  [{lo:>+6.3f}, {hi:>+6.3f}]{sig}")

    significant = sum(1 for i in range(1, nlags + 1) if abs(acf_vals[i]) > ci_width[i - 1])
    print(f"\n  Significant lags: {significant}/{nlags}")
    if significant == 0:
        print("  ✅ No autocorrelation — crashes appear independent")
    else:
        print("  ⚠️  Some autocorrelation detected — investigate further")


# ── 5. Visualization ────────────────────────────────────────


def plot_analysis(data: np.ndarray):
    """Generate analysis plots and save to data/analysis.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Sporty Hero Crash Analysis  (n={len(data)})", fontsize=14, fontweight="bold")

    # 1. Histogram
    ax = axes[0, 0]
    capped = np.clip(data, 0, np.percentile(data, 98))
    ax.hist(capped, bins=40, density=True, alpha=0.7, color="#2196F3", edgecolor="white")
    ax.set_title("Crash Value Distribution")
    ax.set_xlabel("Crash multiplier (x)")
    ax.set_ylabel("Density")

    # Overlay exponential fit
    shifted = data - 1.0
    shifted = shifted[shifted > 0]
    if len(shifted) > 0:
        lam = 1.0 / shifted.mean()
        x_fit = np.linspace(0, np.percentile(capped, 98) - 1, 200)
        ax.plot(x_fit + 1, sp_stats.expon.pdf(x_fit, scale=1 / lam),
                "r-", lw=2, label=f"Exp fit (λ={lam:.3f})")
        ax.legend()

    # 2. Time series
    ax = axes[0, 1]
    ax.plot(data, linewidth=0.5, alpha=0.8, color="#4CAF50")
    ax.axhline(y=data.mean(), color="red", linestyle="--", alpha=0.7, label=f"Mean={data.mean():.2f}x")
    ax.axhline(y=2.0, color="orange", linestyle=":", alpha=0.7, label="2.0x")
    ax.set_title("Crash Values Over Time")
    ax.set_xlabel("Round #")
    ax.set_ylabel("Crash multiplier (x)")
    ax.legend(fontsize=8)

    # 3. ACF plot
    ax = axes[1, 0]
    nlags = min(20, len(data) // 2 - 1)
    if nlags > 1:
        acf_vals, confint = acf(data, nlags=nlags, alpha=0.05)
        ax.bar(range(1, nlags + 1), acf_vals[1:], color="#FF9800", alpha=0.7)
        ci = (confint[1:, 1] - confint[1:, 0]) / 2
        ax.fill_between(range(1, nlags + 1), -ci, ci, alpha=0.2, color="blue")
        ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("Autocorrelation (ACF)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")

    # 4. CDF
    ax = axes[1, 1]
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, linewidth=1.5, color="#9C27B0")
    for t in [1.5, 2.0, 3.0, 5.0]:
        pct = np.sum(data < t) / len(data) * 100
        ax.axvline(x=t, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"{t}x\n({pct:.0f}%)", xy=(t, np.sum(data < t) / len(data)),
                    fontsize=7, ha="right")
    ax.set_title("Empirical CDF")
    ax.set_xlabel("Crash multiplier (x)")
    ax.set_ylabel("P(X < x)")
    ax.set_xlim(0, min(sorted_data[-1], np.percentile(data, 99) * 1.2))

    plt.tight_layout()
    from pathlib import Path
    out = Path(__file__).parent.parent / "data" / "analysis.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"\n📊 Plot saved to {out}")


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze Sporty Hero crash data")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Threshold for streak analysis (default: 2.0)")
    args = parser.parse_args()

    data = _load_data()

    descriptive(data)
    distribution_fit(data)
    streak_analysis(data, threshold=args.threshold)
    autocorrelation(data)

    if not args.no_plot:
        plot_analysis(data)

    _section("DONE")
    print(f"  Analyzed {len(data)} rounds.\n")


if __name__ == "__main__":
    main()
