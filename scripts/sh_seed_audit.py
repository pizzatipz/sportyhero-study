#!/usr/bin/env python3
"""Sporty Hero — 48-bit serverSeed PRNG audit (Phase 2A).

Reads ground-truth rounds collected by sh_seed_collect.py and subjects the
operator's revealed 48-bit serverSeed sequence to a battery of randomness and
predictability tests. The serverSeed is the ONLY remaining attack surface: if
it is produced by a weak/predictable PRNG, the seed sequence could be predicted.

Sections
    0. Integrity recap + house-edge confirmation (formula holds at scale)
    1. Uniformity (hex digit chi-square, per-bit balance, 48-bit KS vs uniform)
    2. Collisions / entropy
    3. Sequential structure (autocorrelation, successive diffs, lag-1 corr)
    4. Time-seeding (does the clock predict the seed?)
    5. NIST SP 800-22 core battery on the 48-bit-per-seed bitstream (+ control)
    6. PRNG recovery attempts (LCG, counter/monotonic, linear-complexity / LFSR)
    7. Client-seed dependency analysis (the second barrier to exploitation)

Usage:
    python scripts/sh_seed_audit.py [--db PATH] [--no-plot]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import stats as sp

import nist

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO, "data", "sportyhero_seeds.db")
DOCS = os.path.join(REPO, "docs")
TWO32 = 2 ** 32
TWO48 = 2 ** 48
HOUSE = 0.97

SUMMARY: dict = {}


def hr(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def flag(p: float, alpha: float = 0.01) -> str:
    if p != p:  # nan
        return "(n/a)"
    return "  <-- FLAG" if p < alpha else ""


# --------------------------------------------------------------------------- #
# Load                                                                        #
# --------------------------------------------------------------------------- #
def load(db_path: str):
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT round_id, created_at, start_time, server_seed, server_seed_int, "
        "client_seeds, client_seed_count, decimal, house_coefficient, verified "
        "FROM rounds ORDER BY round_id ASC"
    ).fetchall()
    con.close()
    cols = ["round_id", "created_at", "start_time", "server_seed", "seed_int",
            "client_seeds", "cs_count", "decimal", "hc", "verified"]
    data = [dict(zip(cols, r)) for r in rows]
    return data


# --------------------------------------------------------------------------- #
# 0. Integrity + house edge                                                   #
# --------------------------------------------------------------------------- #
def section_integrity(data):
    hr("0. INTEGRITY RECAP + HOUSE-EDGE CONFIRMATION")
    n = len(data)
    verified = sum(d["verified"] for d in data)
    hc = np.array([d["hc"] for d in data], dtype=float)
    busts = int(np.sum(hc <= 1.0))
    # A round shows 1.00x (instant bust / no win) whenever the raw value rounds
    # below the smallest winning step 1.01x, i.e. raw < 1.005. For raw = 0.97/U
    # that is P = 1 - 0.97/1.005 = 3.4826% -- the same 3% house-edge mechanism
    # expressed as a point mass.
    bust_theory = 1 - 0.97 / 1.005
    print(f"rounds              : {n}")
    print(f"hash-verified       : {verified}/{n} ({100*verified/n:.2f}%)")
    print(f"instant-bust (1.00x): {busts} ({100*busts/n:.3f}%)   "
          f"theoretical = {100*bust_theory:.3f}% (=1-0.97/1.005)")
    p_bust = sp.binomtest(busts, n, bust_theory).pvalue if n else float("nan")
    print(f"  binomial p vs theory: {p_bust:.4f}{flag(p_bust)}")
    print(f"mean multiplier     : {hc.mean():.3f}x   median = {np.median(hc):.2f}x")
    print("\nHouse-edge check  P(crash>=X)*X  (theory = 0.97 for all X):")
    for X in (1.5, 2, 5, 10, 20, 50, 100):
        obs = np.mean(hc >= X)
        print(f"  X={X:6.1f}  P(>=X)={obs*100:6.3f}%  P*X={obs*X:.4f}  "
              f"(theory {0.97/X*100:6.3f}%)")
    SUMMARY["n"] = n
    SUMMARY["verified"] = verified
    SUMMARY["bust_rate"] = busts / n
    SUMMARY["bust_p_vs_3pct"] = float(p_bust)
    SUMMARY["mean_multiplier"] = float(hc.mean())


# --------------------------------------------------------------------------- #
# 1. Uniformity                                                               #
# --------------------------------------------------------------------------- #
def section_uniformity(data):
    hr("1. SEED UNIFORMITY (48-bit operator commitment)")
    seeds = [d["server_seed"] for d in data]
    ints = np.array([d["seed_int"] for d in data], dtype=np.uint64)
    n = len(ints)

    # hex digit frequency (overall)
    allhex = "".join(seeds)
    counts = np.array([allhex.count(c) for c in "0123456789abcdef"], dtype=float)
    chi2, p_hex = sp.chisquare(counts)
    print(f"hex-digit chi-square (overall, {len(allhex)} chars): "
          f"chi2={chi2:.1f} p={p_hex:.4f}{flag(p_hex)}")

    # per-position hex frequency
    pos_flags = 0
    for pos in range(12):
        col = [s[pos] for s in seeds]
        c = np.array([col.count(ch) for ch in "0123456789abcdef"], dtype=float)
        _, pp = sp.chisquare(c)
        if pp < 0.01:
            pos_flags += 1
    print(f"per-position hex chi-square: {pos_flags}/12 positions flag at p<0.01 "
          f"(expect ~0)")

    # per-bit balance (48 bits)
    bitmat = ((ints[:, None] >> np.arange(48, dtype=np.uint64)[::-1]) & 1).astype(np.int8)
    ones = bitmat.sum(axis=0)
    # binomial test per bit
    bit_p = np.array([sp.binomtest(int(o), n, 0.5).pvalue for o in ones])
    worst = int(np.argmin(bit_p))
    print(f"per-bit balance: {int((bit_p<0.01).sum())}/48 bits flag at p<0.01 "
          f"(expect ~0). worst bit#{worst}: {ones[worst]}/{n} ones, p={bit_p[worst]:.4f}")

    # 48-bit integer uniformity vs Uniform(0,1)
    u = ints.astype(np.float64) / TWO48
    ks, p_ks = sp.kstest(u, "uniform")
    print(f"48-bit int KS vs Uniform(0,1): D={ks:.4f} p={p_ks:.4f}{flag(p_ks)}")

    SUMMARY["uniformity"] = {
        "hex_chi2_p": float(p_hex),
        "perpos_flags": pos_flags,
        "perbit_flags": int((bit_p < 0.01).sum()),
        "int_ks_p": float(p_ks),
    }
    return ints, bitmat


# --------------------------------------------------------------------------- #
# 2. Collisions                                                               #
# --------------------------------------------------------------------------- #
def section_collisions(data):
    hr("2. COLLISIONS / ENTROPY")
    ints = [d["seed_int"] for d in data]
    n = len(ints)
    uniq = len(set(ints))
    dups = n - uniq
    expected = n * (n - 1) / 2 / TWO48
    print(f"unique seeds   : {uniq}/{n}")
    print(f"duplicates     : {dups}   (birthday expectation in 2^48 ~ {expected:.4f})")
    if dups > 0:
        from collections import Counter
        c = Counter(ints)
        rep = [v for v, k in c.items() if k > 1][:5]
        print(f"  !! repeated seed values (entropy < 48 bits?): {[hex(x) for x in rep]}")
    SUMMARY["collisions"] = {"unique": uniq, "dups": dups, "expected": float(expected)}


# --------------------------------------------------------------------------- #
# 3. Sequential structure                                                     #
# --------------------------------------------------------------------------- #
def section_sequential(data, do_plot):
    hr("3. SEQUENTIAL STRUCTURE (is seed[i+1] predictable from history?)")
    ints = np.array([d["seed_int"] for d in data], dtype=np.int64)
    n = len(ints)
    u = ints.astype(np.float64) / TWO48

    # autocorrelation of u at lags 1..30
    um = u - u.mean()
    denom = np.sum(um * um)
    band = 1.96 / math.sqrt(n)
    sig = []
    acf_vals = []
    for lag in range(1, 31):
        a = np.sum(um[:-lag] * um[lag:]) / denom
        acf_vals.append(a)
        if abs(a) > band:
            sig.append((lag, a))
    print(f"autocorrelation lags 1-30: {len(sig)} exceed +/-{band:.4f} "
          f"(expect ~{0.05*30:.1f} false positives)")
    for lag, a in sig[:8]:
        print(f"    lag {lag}: r={a:+.4f}")

    # lag-1 Spearman & Pearson
    sr, sp_p = sp.spearmanr(u[:-1], u[1:])
    pr, pp = sp.pearsonr(u[:-1], u[1:])
    print(f"lag-1 Spearman rho={sr:+.4f} p={sp_p:.4f}{flag(sp_p)}")
    print(f"lag-1 Pearson   r ={pr:+.4f} p={pp:.4f}{flag(pp)}")

    # successive diffs mod 2^48
    diffs = np.diff(ints.astype(object))
    diffs_mod = np.array([int(d) % TWO48 for d in diffs], dtype=np.float64)
    print(f"successive diff mod 2^48: mean={diffs_mod.mean()/TWO48*100:.2f}% of range "
          f"(expect ~50%), std={diffs_mod.std()/TWO48*100:.2f}%")
    small = int(np.sum(np.minimum(diffs_mod, TWO48 - diffs_mod) < TWO48 * 1e-4))
    print(f"  near-constant/counter check: {small} diffs within 0.01% of 0 "
          f"(counter/LCG tell; expect ~0)")

    # Ljung-Box on u (manual)
    q = n * (n + 2) * sum((acf_vals[k] ** 2) / (n - (k + 1)) for k in range(10))
    p_lb = sp.chi2.sf(q, 10)
    print(f"Ljung-Box(10) on seed sequence: Q={q:.2f} p={p_lb:.4f}{flag(p_lb)}")

    if do_plot:
        _plot_sequential(u, acf_vals, band)

    SUMMARY["sequential"] = {
        "acf_significant": len(sig),
        "lag1_spearman_p": float(sp_p),
        "lag1_pearson_p": float(pp),
        "ljungbox10_p": float(p_lb),
        "counter_diffs": small,
    }


def _plot_sequential(u, acf_vals, band):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].bar(range(1, len(acf_vals) + 1), acf_vals, color="steelblue")
        ax[0].axhline(band, color="r", ls="--", lw=0.8)
        ax[0].axhline(-band, color="r", ls="--", lw=0.8)
        ax[0].set_title("Seed sequence autocorrelation")
        ax[0].set_xlabel("lag"); ax[0].set_ylabel("ACF")
        ax[1].scatter(u[:-1], u[1:], s=3, alpha=0.3)
        ax[1].set_title("Lag-1 scatter (seed_i vs seed_{i+1})")
        ax[1].set_xlabel("u[i]"); ax[1].set_ylabel("u[i+1]")
        os.makedirs(DOCS, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "seed_sequential.png"), dpi=110)
        plt.close(fig)
        print("  [saved docs/seed_sequential.png]")
    except Exception as e:
        print(f"  (plot skipped: {e})")


# --------------------------------------------------------------------------- #
# 4. Time-seeding                                                             #
# --------------------------------------------------------------------------- #
def _epoch_ms(ts: str):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp() * 1000.0
    except Exception:
        return None


def section_time(data):
    hr("4. TIME-SEEDING (does the clock predict the seed?)")
    pairs = [(d, _epoch_ms(d["created_at"] or d["start_time"])) for d in data]
    pairs = [(d, t) for d, t in pairs if t is not None]
    if len(pairs) < 50:
        print("  insufficient timestamped rows; skipping")
        return
    ints = np.array([d["seed_int"] for d, _ in pairs], dtype=np.float64)
    dec = np.array([d["decimal"] for d, _ in pairs], dtype=np.float64)
    t = np.array([tt for _, tt in pairs], dtype=np.float64)
    span_h = (t.max() - t.min()) / 3600_000
    print(f"window: {len(pairs)} rounds spanning {span_h:.1f} h")

    # --- seed vs absolute time, WITH a provably-independent control ---
    # `decimal` is a SHA-512 output: independent of time by construction. If it
    # shows the same correlation as the seed, the signal is a sampling artifact.
    sr, sp_p = sp.spearmanr(ints, t)
    cr, cp = sp.spearmanr(dec, t)
    lo, hi = _boot_spearman_ci(ints, t)
    print(f"seed    vs time : Spearman rho={sr:+.4f} p={sp_p:.4f}  "
          f"95%CI[{lo:+.3f},{hi:+.3f}]{flag(sp_p)}")
    print(f"CONTROL decimal vs time : rho={cr:+.4f} p={cp:.4f}  "
          f"(SHA-512 output; MUST be ~0 if methodology is clean)")
    # split-half robustness: a real effect appears in both halves
    h = len(ints) // 2
    s1 = sp.spearmanr(ints[:h], t[:h]).statistic
    s2 = sp.spearmanr(ints[h:], t[h:]).statistic
    print(f"split-half seed-vs-time: first={s1:+.4f} second={s2:+.4f} "
          f"(real effect => same sign & magnitude in both)")

    # seed vs time-of-day components (meaningful once span > ~1 day)
    for name, mod in (("ms-of-day", 86400_000), ("ms-of-hour", 3600_000),
                      ("ms-of-minute", 60_000)):
        comp = np.mod(t, mod)
        phase = 2 * np.pi * comp / mod
        rc = max(abs(np.corrcoef(ints, np.sin(phase))[0, 1]),
                 abs(np.corrcoef(ints, np.cos(phase))[0, 1]))
        rcc = max(abs(np.corrcoef(dec, np.sin(phase))[0, 1]),
                  abs(np.corrcoef(dec, np.cos(phase))[0, 1]))
        print(f"seed vs {name:12s}: circ|r|={rc:.4f}  (control decimal={rcc:.4f})")

    # mutual information with coarse bins + shuffled baseline (de-biased)
    seed_bin = (np.array([d["seed_int"] for d, _ in pairs]) >> 44).astype(int)  # top 4 bits
    hod = (np.mod(t, 86400_000) // 3600_000).astype(int)                        # hour-of-day
    mi = _mutual_info(hod, seed_bin)
    rng = np.random.default_rng(0)
    base = np.mean([_mutual_info(hod, rng.permutation(seed_bin)) for _ in range(20)])
    print(f"MI(hour-of-day ; seed top-4-bits)={mi:.4f} nats  "
          f"shuffled-baseline={base:.4f}  excess={mi-base:+.4f}")

    artifact = abs(cr) > 0.5 * abs(sr) and abs(sr) > 0
    SUMMARY["time_seeding"] = {
        "seed_vs_time_rho": float(sr), "seed_vs_time_p": float(sp_p),
        "control_decimal_rho": float(cr), "control_decimal_p": float(cp),
        "split_half": [float(s1), float(s2)],
        "mi_excess": float(mi - base),
        "likely_artifact": bool(artifact),
    }
    if sp_p < 0.01:
        if abs(cr) > 0.5 * abs(sr):
            print("  => control shows similar correlation: SAMPLING ARTIFACT, not a seed flaw.")
        else:
            print("  => control is clean but seed correlates: investigate further.")


def _boot_spearman_ci(x, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = sp.spearmanr(x[idx], y[idx]).statistic
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def _mutual_info(a, b):
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    ca = defaultdict(int); cb = defaultdict(int); cab = defaultdict(int)
    for x, y in zip(a, b):
        ca[x] += 1; cb[y] += 1; cab[(x, y)] += 1
    mi = 0.0
    for (x, y), nxy in cab.items():
        pxy = nxy / n
        mi += pxy * math.log(pxy / (ca[x] / n * cb[y] / n))
    return mi


# --------------------------------------------------------------------------- #
# 5. NIST battery                                                             #
# --------------------------------------------------------------------------- #
def seeds_to_bits(data):
    ints = np.array([d["seed_int"] for d in data], dtype=np.uint64)
    bits = ((ints[:, None] >> np.arange(48, dtype=np.uint64)[::-1]) & 1).astype(np.int8)
    return bits.ravel()


def section_nist(data):
    hr("5. NIST SP 800-22 CORE BATTERY")
    bits = seeds_to_bits(data)
    print(f"serverSeed bitstream: {bits.size} bits (= {len(data)} x 48)")
    if bits.size < 1000:
        print("  too few bits; skipping")
        return
    res = nist.run_battery(bits)
    fails = 0
    for k, v in res.items():
        if k == "n_bits":
            continue
        f = flag(v)
        if f:
            fails += 1
        print(f"  {k:30s} p={v:.4f}{f}")
    print(f"\n  -> {fails} test(s) flag at p<0.01 "
          f"({len(res)-1} run; ~{0.01*(len(res)-1):.2f} expected by chance)")

    # control: the SHA-512-derived `decimal` (top 32 bits) MUST look random
    dec = np.array([d["decimal"] for d in data], dtype=np.uint64)
    dbits = ((dec[:, None] >> np.arange(32, dtype=np.uint64)[::-1]) & 1).astype(np.int8).ravel()
    cres = nist.run_battery(dbits)
    cfails = sum(1 for k, v in cres.items() if k != "n_bits" and v < 0.01)
    print(f"  [control] SHA-512 decimal bitstream ({dbits.size} bits): {cfails} flags "
          f"(sanity: hash output should pass)")

    SUMMARY["nist_seed_fails"] = fails
    SUMMARY["nist_control_fails"] = cfails
    SUMMARY["nist_detail"] = {k: float(v) for k, v in res.items() if k != "n_bits"}


# --------------------------------------------------------------------------- #
# 6. PRNG recovery                                                            #
# --------------------------------------------------------------------------- #
def _egcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def lcg_recover(ints):
    """Attempt full-width LCG recovery: s_{i+1} = (a s_i + c) mod m."""
    s = [int(x) for x in ints]
    if len(s) < 10:
        return None
    diffs = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    zeroes = [diffs[i + 2] * diffs[i] - diffs[i + 1] ** 2 for i in range(len(diffs) - 2)]
    m = 0
    for z in zeroes:
        m = math.gcd(m, abs(z))
    if m < 2:
        return {"recovered": False, "modulus": m}
    # recover a, c
    a = c = None
    for i in range(len(diffs) - 1):
        g, inv, _ = _egcd(diffs[i] % m, m)
        if g == 1:
            a = (diffs[i + 1] * inv) % m
            c = (s[i + 1] - a * s[i]) % m
            break
    if a is None:
        return {"recovered": False, "modulus": m}
    # validate predictions on the sequence
    ok = sum(1 for i in range(len(s) - 1) if (a * s[i] + c) % m == s[i + 1])
    return {"recovered": ok == len(s) - 1, "modulus": m, "a": a, "c": c,
            "predict_ok": ok, "predict_total": len(s) - 1}


def berlekamp_massey_gf2(bitseq):
    n = len(bitseq)
    c = np.zeros(n, dtype=np.int8); b = np.zeros(n, dtype=np.int8)
    c[0] = b[0] = 1
    L = 0; mm = -1
    for i in range(n):
        d = bitseq[i]
        for j in range(1, L + 1):
            d ^= c[j] & bitseq[i - j]
        if d == 1:
            t = c.copy()
            shift = i - mm
            c[shift:] ^= b[:n - shift]
            if 2 * L <= i:
                L = i + 1 - L; mm = i; b = t
    return L


def section_recovery(data):
    hr("6. PRNG RECOVERY ATTEMPTS")
    ints = [d["seed_int"] for d in data]
    n = len(ints)

    # counter / monotonic
    diffs = np.diff(np.array(ints, dtype=np.int64))
    mono_up = int(np.sum(diffs > 0)); mono_dn = int(np.sum(diffs < 0))
    print(f"monotonic check: {mono_up} up / {mono_dn} down steps "
          f"(counter would be ~all one direction)")

    # LCG
    lcg = lcg_recover(ints)
    if lcg is None:
        print("LCG recovery: insufficient data")
    elif lcg["recovered"]:
        print(f"LCG recovery: *** RECOVERED *** m={lcg['modulus']} a={lcg['a']} "
              f"c={lcg['c']}  ({lcg['predict_ok']}/{lcg['predict_total']} predicted)")
    else:
        print(f"LCG recovery: not an LCG (modulus gcd={lcg['modulus']}, "
              f"predictions {lcg.get('predict_ok','-')}/{lcg.get('predict_total','-')})")

    # linear complexity on a prefix (detect small-state LFSR / xorshift)
    bits = seeds_to_bits(data)
    prefix = bits[: min(4000, bits.size)]
    L = berlekamp_massey_gf2(prefix.astype(np.int8))
    print(f"linear complexity (Berlekamp-Massey on first {prefix.size} bits): "
          f"L={L} (expect ~{prefix.size//2}; a small fixed L => LFSR/xorshift)")
    lfsr = L < prefix.size * 0.4

    SUMMARY["recovery"] = {
        "lcg_recovered": bool(lcg and lcg.get("recovered")),
        "lcg_modulus": int(lcg["modulus"]) if lcg else None,
        "linear_complexity": int(L),
        "lc_prefix": int(prefix.size),
        "lfsr_suspected": bool(lfsr),
    }


# --------------------------------------------------------------------------- #
# 7. Client seeds                                                             #
# --------------------------------------------------------------------------- #
def section_clientseeds(data):
    hr("7. CLIENT-SEED DEPENDENCY (second barrier to exploitation)")
    counts = [d["cs_count"] for d in data]
    print(f"bettors/round: min={min(counts)} mean={np.mean(counts):.2f} max={max(counts)}")
    uid_seeds = defaultdict(set)
    rounds_with = 0
    for d in data:
        try:
            cs = json.loads(d["client_seeds"])
        except Exception:
            continue
        rounds_with += 1
    # are clientSeeds stable per uid? need uid; re-load raw
    print(f"note: outcome hash mixes ALL bettors' clientSeeds (revealed post-round).")
    print(f"      Even a predicted serverSeed needs the full ordered clientSeed set,")
    print(f"      so single-seed prediction is necessary but NOT sufficient to bet.")
    SUMMARY["clientseeds"] = {"min": int(min(counts)), "mean": float(np.mean(counts)),
                              "max": int(max(counts))}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    data = load(args.db)
    if len(data) < 50:
        print(f"Only {len(data)} rounds in DB; collect more first.")
        return
    print(f"Loaded {len(data)} rounds (ids {data[0]['round_id']}..{data[-1]['round_id']})")

    section_integrity(data)
    section_uniformity(data)
    section_collisions(data)
    section_sequential(data, not args.no_plot)
    section_time(data)
    section_nist(data)
    section_recovery(data)
    section_clientseeds(data)

    out = os.path.join(REPO, "data", "seed_audit_summary.json")
    with open(out, "w") as f:
        json.dump(SUMMARY, f, indent=2)
    hr("VERDICT")
    _verdict()
    print(f"\n[summary JSON -> {out}]")


def _verdict():
    s = SUMMARY
    flags = []
    if s.get("uniformity", {}).get("int_ks_p", 1) < 0.01:
        flags.append("seed not uniform")
    if s.get("collisions", {}).get("dups", 0) > 0:
        flags.append("seed collisions (low entropy)")
    if s.get("sequential", {}).get("lag1_spearman_p", 1) < 0.01:
        flags.append("lag-1 correlation")
    if s.get("sequential", {}).get("ljungbox10_p", 1) < 0.01:
        flags.append("serial dependence")
    ts = s.get("time_seeding", {})
    if ts.get("seed_vs_time_p", 1) < 0.01 and not ts.get("likely_artifact"):
        flags.append("time correlation (control-confirmed)")
    if s.get("nist_seed_fails", 0) > 2:
        flags.append(f"{s['nist_seed_fails']} NIST failures")
    if s.get("recovery", {}).get("lcg_recovered"):
        flags.append("LCG RECOVERED")
    if s.get("recovery", {}).get("lfsr_suspected"):
        flags.append("low linear complexity (LFSR)")
    if flags:
        print("POTENTIAL STRUCTURE DETECTED:")
        for f in flags:
            print(f"   !! {f}")
        print("\n-> Escalate: attempt held-out seed prediction and quantify edge.")
    else:
        print("No exploitable structure in the serverSeed sequence.")
        print("The 48-bit operator commitment behaves as a CSPRNG draw:")
        print("uniform, collision-free, serially independent, clock-independent,")
        print("passes NIST, and is not an LCG/LFSR. No prediction is possible.")
    SUMMARY["flags"] = flags


if __name__ == "__main__":
    main()
