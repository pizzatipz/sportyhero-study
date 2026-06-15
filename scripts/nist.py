"""NIST SP 800-22 (rev. 1a) statistical test battery — compact NumPy/SciPy port.

Implements the well-powered core tests appropriate for a ~5x10^5-bit stream:
    monobit (frequency), block frequency, runs, longest-run-of-ones,
    binary matrix rank, discrete Fourier transform (spectral), serial,
    approximate entropy, and cumulative sums (forward/backward).

Each function takes a NumPy uint8 array of bits (0/1) and returns a p-value
(or a tuple of p-values). A p-value < 0.01 is the conventional NIST failure
threshold for a single sequence. Multiple-testing caveats apply when running
the whole battery — interpret with a Bonferroni / FDR lens.

References: NIST SP 800-22 rev1a, sections 2.1-2.16.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc, gammaincc


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _as_bits(bits) -> np.ndarray:
    b = np.asarray(bits, dtype=np.int8).ravel()
    if b.size == 0:
        raise ValueError("empty bit array")
    return b


# --------------------------------------------------------------------------- #
# 2.1 Frequency (Monobit)                                                     #
# --------------------------------------------------------------------------- #
def monobit(bits) -> float:
    b = _as_bits(bits)
    s = np.sum(2 * b - 1)
    s_obs = abs(s) / np.sqrt(b.size)
    return float(erfc(s_obs / np.sqrt(2)))


# --------------------------------------------------------------------------- #
# 2.2 Frequency within a Block                                                #
# --------------------------------------------------------------------------- #
def block_frequency(bits, M: int = 128) -> float:
    b = _as_bits(bits)
    N = b.size // M
    if N == 0:
        return float("nan")
    blocks = b[: N * M].reshape(N, M)
    pi = blocks.mean(axis=1)
    chi2 = 4.0 * M * np.sum((pi - 0.5) ** 2)
    return float(gammaincc(N / 2.0, chi2 / 2.0))


# --------------------------------------------------------------------------- #
# 2.3 Runs                                                                    #
# --------------------------------------------------------------------------- #
def runs(bits) -> float:
    b = _as_bits(bits)
    n = b.size
    pi = b.mean()
    if abs(pi - 0.5) >= (2.0 / np.sqrt(n)):
        return 0.0  # monobit precondition fails
    vn = 1 + np.sum(b[1:] != b[:-1])
    num = abs(vn - 2.0 * n * pi * (1 - pi))
    den = 2.0 * np.sqrt(2.0 * n) * pi * (1 - pi)
    return float(erfc(num / den))


# --------------------------------------------------------------------------- #
# 2.4 Longest Run of Ones in a Block                                          #
# --------------------------------------------------------------------------- #
def longest_run_ones(bits) -> float:
    b = _as_bits(bits)
    n = b.size
    if n < 128:
        return float("nan")
    if n < 6272:
        M, K, V = 8, 3, [1, 2, 3, 4]
        pi = [0.2148, 0.3672, 0.2305, 0.1875]
    elif n < 750000:
        M, K, V = 128, 5, [4, 5, 6, 7, 8, 9]
        pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    else:
        M, K, V = 10000, 6, [10, 11, 12, 13, 14, 15, 16]
        pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    N = n // M
    blocks = b[: N * M].reshape(N, M)
    # longest run of ones per block
    def longest(block):
        best = cur = 0
        for x in block:
            cur = cur + 1 if x else 0
            if cur > best:
                best = cur
        return best
    longs = np.array([longest(blk) for blk in blocks])
    # bucket into classes <=V[0], V[1], ..., >=V[-1]
    counts = np.zeros(K + 1)
    lo, hi = V[0], V[-1]
    clamped = np.clip(longs, lo, hi)
    for idx, val in enumerate(range(lo, hi + 1)):
        counts[idx] = np.sum(clamped == val)
    chi2 = np.sum((counts - N * np.array(pi)) ** 2 / (N * np.array(pi)))
    return float(gammaincc(K / 2.0, chi2 / 2.0))


# --------------------------------------------------------------------------- #
# 2.5 Binary Matrix Rank                                                      #
# --------------------------------------------------------------------------- #
def _gf2_rank(matrix: np.ndarray) -> int:
    m = matrix.copy().astype(np.uint8)
    rows, cols = m.shape
    rank = 0
    pr = 0
    for pc in range(cols):
        # find pivot
        pivot = -1
        for r in range(pr, rows):
            if m[r, pc]:
                pivot = r
                break
        if pivot == -1:
            continue
        m[[pr, pivot]] = m[[pivot, pr]]
        for r in range(rows):
            if r != pr and m[r, pc]:
                m[r] ^= m[pr]
        pr += 1
        rank += 1
        if pr == rows:
            break
    return rank


def binary_matrix_rank(bits, M: int = 32, Q: int = 32) -> float:
    b = _as_bits(bits)
    n = b.size
    N = n // (M * Q)
    if N < 38:
        return float("nan")
    fm = fm1 = rest = 0
    full = min(M, Q)
    for i in range(N):
        blk = b[i * M * Q:(i + 1) * M * Q].reshape(M, Q)
        r = _gf2_rank(blk)
        if r == full:
            fm += 1
        elif r == full - 1:
            fm1 += 1
        else:
            rest += 1
    # theoretical probabilities for 32x32
    p_full, p_full1, p_rest = 0.2888, 0.5776, 0.1336
    chi2 = ((fm - p_full * N) ** 2 / (p_full * N)
            + (fm1 - p_full1 * N) ** 2 / (p_full1 * N)
            + (rest - p_rest * N) ** 2 / (p_rest * N))
    return float(np.exp(-chi2 / 2.0))


# --------------------------------------------------------------------------- #
# 2.6 Discrete Fourier Transform (Spectral)                                   #
# --------------------------------------------------------------------------- #
def dft_spectral(bits) -> float:
    b = _as_bits(bits)
    n = b.size
    x = 2 * b - 1
    mags = np.abs(np.fft.rfft(x))[: n // 2]
    T = np.sqrt(np.log(1.0 / 0.05) * n)
    n0 = 0.95 * n / 2.0
    n1 = np.sum(mags < T)
    d = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4.0)
    return float(erfc(abs(d) / np.sqrt(2)))


# --------------------------------------------------------------------------- #
# 2.11 Serial                                                                 #
# --------------------------------------------------------------------------- #
def _psi2(b: np.ndarray, m: int) -> float:
    n = b.size
    if m == 0:
        return 0.0
    ext = np.concatenate([b, b[: m - 1]])  # wraparound
    # build m-bit pattern indices
    idx = np.zeros(n, dtype=np.int64)
    for j in range(m):
        idx = (idx << 1) | ext[j:j + n]
    counts = np.bincount(idx, minlength=1 << m)
    return (np.sum(counts.astype(float) ** 2) * (1 << m) / n) - n


def serial(bits, m: int = 12):
    b = _as_bits(bits)
    p0 = _psi2(b, m)
    p1 = _psi2(b, m - 1)
    p2 = _psi2(b, m - 2)
    d1 = p0 - p1
    d2 = p0 - 2 * p1 + p2
    pv1 = float(gammaincc(2 ** (m - 2), d1 / 2.0))
    pv2 = float(gammaincc(2 ** (m - 3), d2 / 2.0))
    return pv1, pv2


# --------------------------------------------------------------------------- #
# 2.12 Approximate Entropy                                                    #
# --------------------------------------------------------------------------- #
def _phi(b: np.ndarray, m: int) -> float:
    n = b.size
    if m == 0:
        return 0.0
    ext = np.concatenate([b, b[: m - 1]])
    idx = np.zeros(n, dtype=np.int64)
    for j in range(m):
        idx = (idx << 1) | ext[j:j + n]
    counts = np.bincount(idx, minlength=1 << m).astype(float)
    c = counts / n
    nz = c[c > 0]
    return float(np.sum(nz * np.log(nz)))


def approximate_entropy(bits, m: int = 10) -> float:
    b = _as_bits(bits)
    n = b.size
    apen = _phi(b, m) - _phi(b, m + 1)
    chi2 = 2.0 * n * (np.log(2) - apen)
    return float(gammaincc(2 ** (m - 1), chi2 / 2.0))


# --------------------------------------------------------------------------- #
# 2.13 Cumulative Sums                                                        #
# --------------------------------------------------------------------------- #
def _normcdf(x):
    from scipy.special import ndtr
    return ndtr(x)


def cumulative_sums(bits):
    b = _as_bits(bits)
    n = b.size
    x = 2 * b - 1

    def pval(z):
        if z == 0:
            return 1.0
        k1 = np.arange((-n // z + 1) // 4, (n // z - 1) // 4 + 1)
        s1 = np.sum(_normcdf((4 * k1 + 1) * z / np.sqrt(n))
                    - _normcdf((4 * k1 - 1) * z / np.sqrt(n)))
        k2 = np.arange((-n // z - 3) // 4, (n // z - 1) // 4 + 1)
        s2 = np.sum(_normcdf((4 * k2 + 3) * z / np.sqrt(n))
                    - _normcdf((4 * k2 + 1) * z / np.sqrt(n)))
        return float(1.0 - s1 + s2)

    csum = np.cumsum(x)
    z_fwd = int(np.max(np.abs(csum)))
    csum_b = np.cumsum(x[::-1])
    z_bwd = int(np.max(np.abs(csum_b)))
    return pval(z_fwd), pval(z_bwd)


# --------------------------------------------------------------------------- #
# Battery runner                                                              #
# --------------------------------------------------------------------------- #
def run_battery(bits) -> dict:
    b = _as_bits(bits)
    out = {}
    out["n_bits"] = int(b.size)
    out["monobit"] = monobit(b)
    out["block_frequency(M=128)"] = block_frequency(b, 128)
    out["runs"] = runs(b)
    out["longest_run_ones"] = longest_run_ones(b)
    out["binary_matrix_rank(32x32)"] = binary_matrix_rank(b, 32, 32)
    out["dft_spectral"] = dft_spectral(b)
    s1, s2 = serial(b, 12)
    out["serial_1(m=12)"] = s1
    out["serial_2(m=12)"] = s2
    out["approximate_entropy(m=10)"] = approximate_entropy(b, 10)
    c1, c2 = cumulative_sums(b)
    out["cusum_forward"] = c1
    out["cusum_backward"] = c2
    return out
